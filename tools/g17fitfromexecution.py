#!/usr/bin/env python3
"""Name what an executed G17 instruction COMPUTES, by eliminating every competing candidate.

WHY THIS EXISTS. isa/ holds 613 isolated-dispatch records whose plan supplies inputs and whose
result supplies the words the hardware returned, over 403 distinct opcodes - and 526 of them carry
NO `expect` field. The map reports those as `executed_unchecked`, correctly: running is not
checking. But an absent expectation is not absent evidence. It is the reader's job to supply the
hypothesis, and a record with inputs and outputs can be CHECKED by computing every candidate
function and keeping only those that reproduce every case.

WHAT MAKES THIS DIFFERENT FROM FITTING. A fit reports the best candidate; this reports how many
candidates SURVIVE. One survivor of sixteen is a measurement. Two survivors is a statement that
these inputs cannot separate them, and it is recorded as such rather than resolved by preference.
Zero survivors is the most interesting outcome of all - the function is outside the class offered.

    ledger/g17-the-system-register-is-byte1-and-byte3.toml is the shape of the hazard this avoids:
    a coarse key that answers confidently because it cannot see the distinction.

NO CANDIDATE HAS A FREE PARAMETER, DELIBERATELY. `src + 4` is what op10279's control computes and
this tool does NOT fit it, because a candidate with a fitted constant can absorb any single
column - "one survivor" would then mean "one FAMILY survived, at one of its infinitely many
settings", which is a different and much weaker claim. Records whose function needs a constant are
reported as no-fit, and the positive control being among them is the honest cost.

THE CLASS IS THE LIMIT AND IT IS PUBLISHED. Uniqueness holds only within the candidate library
below. A function reading the operands differently, consulting a modifier, or of an arity other
than the record's is not excluded by four points, and the output says so per record.

BOTH INTERPRETATIONS ARE TRIED AND REPORTED APART. The same 32 bits are a uint32 and a float32,
and a record does not say which. Integer and float candidates are evaluated separately; a record
where each interpretation yields its own unique survivor is reported as INTERPRETATION-AMBIGUOUS
rather than silently resolved, because that is a real limit of the inputs chosen.

    python3 tools/g17fitfromexecution.py            # the census to stdout
    python3 tools/g17fitfromexecution.py --write    # isa/g17-execution-fits.json
    python3 tools/g17fitfromexecution.py --op 13460 # one opcode, with its surviving candidates
"""
import argparse
import collections
import json
import math
import functools
import itertools
import operator
import random
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g17auth

ROOT = Path(__file__).resolve().parents[1]
ISA = ROOT / 'isa'
sys.path.insert(0, str(ROOT))
from agxforge.g17.model import decode as decode_instruction        # noqa: E402

M32 = 0xFFFFFFFF


def _f(u):
    return struct.unpack('<f', struct.pack('<I', int(u) & M32))[0]


def _u(f):
    try:
        return struct.unpack('<I', struct.pack('<f', float(f)))[0]
    except (OverflowError, ValueError):
        return None


def _s(u):
    u = int(u) & M32
    return u - 0x100000000 if u & 0x80000000 else u


def _sat(x, lo=0.0, hi=1.0):
    return min(max(x, lo), hi)


# THE SIXTEEN two-input bitwise functions, complete by construction rather than by a list someone
# curated: every function of two bits is a 4-bit truth table, and there are exactly sixteen.
def _bitwise(table):
    def fn(a, b):
        out = 0
        for bit in range(32):
            out |= table[((a >> bit) & 1) << 1 | ((b >> bit) & 1)] << bit
        return out & M32
    return fn


INT1 = {
    # MSB OF THE LOW 16 BITS, WITH ZERO GIVING 0xFFFF. op9990 is named `msb`, has a 16-bit
    # destination, and the library's 32-bit `msb` misses 13 of 31 cases on its widest record.
    # Every miss is explained by the source being read as its LOW HALF: 0x0ABCDEF1 gives 15 (the
    # msb of 0xDEF1) where a 32-bit read gives 27, and 0x80000001 gives 0 where a 32-bit read
    # gives 31. A low half of zero returns 0xFFFF, the conventional "no bit set" answer, observed
    # at four inputs whose low half is zero. Same shape as op3341's second operand: a 16-bit
    # destination reading a 16-bit LANE of a 32-bit register, which no 32-bit candidate expresses.
    'msb_lo16': lambda a: (0xFFFF if (a & 0xFFFF) == 0 else ((a & 0xFFFF).bit_length() - 1)),
    'identity': lambda a: a & M32, 'not': lambda a: ~a & M32,
    'neg': lambda a: -a & M32, 'abs_s32': lambda a: abs(_s(a)) & M32,
    'zero': lambda a: 0, 'ones': lambda a: M32,
    'shl1': lambda a: (a << 1) & M32, 'shr1': lambda a: (a & M32) >> 1,
    'sar1': lambda a: (_s(a) >> 1) & M32,
    'byteswap': lambda a: int.from_bytes((a & M32).to_bytes(4, 'little'), 'big'),
    'popcount': lambda a: bin(a & M32).count('1'),
    # TWO MORE MISSING INTERPRETATIONS, both named by Apple's table and neither expressible.
    # `msb` returns the INDEX of the highest set bit and encodes "no bit set" as all-ones: op9986
    # returns 0xFFFFFFFF for zero and 13 for 0x3C00, and the 16-bit-destination forms return
    # 0xFFFF, which is the same -1 truncated. That sentinel is a measurement, not part of the
    # name - `clz` is already in this library and is a different function of the same quantity.
    #
    # `reverse` is a bit reversal, and BOTH WIDTHS ARE OFFERED RATHER THAN CHOSEN, because a
    # 32-bit reversal and a 16-bit reversal landed in the high half agree on every input whose
    # top half is zero - which is most of the sweep. Letting the census eliminate one is the only
    # way the answer is the data's rather than mine; if both survive it reports ambiguous, which
    # is the truth about these inputs.
    'msb': lambda a: (int(a).bit_length() - 1) & M32 if int(a) & M32 else M32,
    'bitreverse': lambda a: int('{:032b}'.format(int(a) & M32)[::-1], 2),
    'bitreverse_b16': lambda a: int('{:016b}'.format(int(a) & 0xFFFF)[::-1], 2) << 16,
    'clz': lambda a: 32 - (a & M32).bit_length(),
}
INT2 = {
    'add': lambda a, b: (a + b) & M32, 'sub': lambda a, b: (a - b) & M32,
    'rsub': lambda a, b: (b - a) & M32, 'mul': lambda a, b: (a * b) & M32,
    'umin': lambda a, b: min(a & M32, b & M32), 'umax': lambda a, b: max(a & M32, b & M32),
    'smin': lambda a, b: min(_s(a), _s(b)) & M32, 'smax': lambda a, b: max(_s(a), _s(b)) & M32,
    # THE SHIFTS SATURATE; THEY DO NOT WRAP MODULO 32. These masked the amount with `& 31`,
    # which is the C convention and NOT this hardware's:
    # ledger/g17-addr16-turns-a-count-into-a-mask.toml measured op621 and wrote it down in 2026-09
    # - "src >= 32 -> 0, the shift SATURATES; it does not wrap modulo 32" - and 33 and 63 return 0
    # and 0 there rather than the 15 and 1 a modulo shift gives.
    #
    # So the library contradicted a measurement this repository already held, and the cost was
    # visible: op14392 `shl` matched 9 of 10 cases, missing only the pair whose shift amount is
    # 0x5A5A5A5A. The hardware returned 0 and the masked candidate predicted 0xC4000000, so a
    # correctly-named shift read as "no candidate fits". Fixing this applies an existing
    # measurement rather than fitting a new one.
    #
    # The wrapping forms are KEPT under their own names, because which convention an opcode uses
    # is a question and not an assumption - `shl_wrap` may yet be the right reading somewhere.
    # THE AMOUNT IS A NARROW FIELD AND THEN IT SATURATES. Measured 2026-09-18 on op14392 `shl`,
    # op17014 `shr` and op16807 `asr`, 10 of 10 cases each:
    #
    #   amount 0x4000 -> the source passes through, so the high bits of b are DROPPED
    #   amount 0x5A5A5A5A -> 0, so what survives the drop is >= 32 and saturates
    #
    # HOW WIDE THE FIELD IS, IS NOT PINNED BY THIS DATA and `& 0xFF` is a representative choice,
    # not a measurement. 0x4000 must lose its bit 14, so the mask is at most 14 bits; 0x5A must
    # keep its bit 6 to reach 32, so the mask is at least 7. Anything in that range fits every
    # case I have. Writing 8 and calling it measured would be the mistake this file keeps making.
    #
    # EVIDENCE IS THIN AND THE CASE SET IS THE REASON: 8 of my 10 pairs have a second operand
    # whose low byte is ZERO, because they are half bit patterns chosen to separate float
    # candidates. For a shift family that means eight pass-through cases and two informative ones.
    # The scheduling class told me these were integer; it could not tell me my inputs were wrong
    # for them.
    # SEVEN BITS, NOW PINNED. I wrote `& 0xFF` earlier and said explicitly that the width was
    # between 7 and 14 bits and that 8 was representative rather than measured. The batch with
    # eight amounts above 32 settled it: amount 0x80 returns the SOURCE UNCHANGED, which only a
    # mask that drops bit 7 produces, while 0x64 and 0x41 return zero. Fitted across shl, shr and
    # sar: 5 bits gives 8-11 of 18, 6 bits gives 16 of 18, SEVEN gives 18 of 18, and 8 or 9 bits
    # give 17 of 18 - failing on exactly the 0x80 case. Uniquely determined in both directions.
    #
    # Then it saturates: an amount of 32 through 127 returns zero, or an all-ones sign fill for
    # the arithmetic shift. So the full reading is `amount = b & 0x7F`, then shift if the amount
    # is below the word width and saturate otherwise.
    'shl': lambda a, b: (a << (b & 0x7F)) & M32 if (b & 0x7F) < 32 else 0,
    'shr': lambda a, b: (a & M32) >> (b & 0x7F) if (b & 0x7F) < 32 else 0,
    'sar': lambda a, b: (_s(a) >> (b & 0x7F)) & M32 if (b & 0x7F) < 32
    else (M32 if _s(a) < 0 else 0),
    # the wrapping variants stay, now refuted for these six forms at margin 8-11
    'shl_wrap': lambda a, b: (a << (b & 31)) & M32,
    'shr_wrap': lambda a, b: (a & M32) >> (b & 31),
    'sar_wrap': lambda a, b: (_s(a) >> (b & 31)) & M32,
    'rotl': lambda a, b: ((a << (b & 31)) | ((a & M32) >> ((32 - (b & 31)) & 31))) & M32,
}
for _i in range(16):
    _t = [(_i >> _k) & 1 for _k in range(4)]
    INT2['bitwise_%X' % _i] = _bitwise(_t)
# SHIFT-ADD, HIGH PRODUCT AND THE OTHER REAL FORMS THE FIRST LIBRARY LACKED. Added after reading
# the no-fit pile rather than guessing: sweep/D10282 returns a + 8192 with its second operand held
# at 4096, which is `a + 2b` - a shift-add, and one this backend's own corpus contains. Each shift
# amount is a SEPARATE candidate with no free parameter, which is the distinction that matters: a
# candidate carrying a fitted constant could absorb any single column, and then "one survivor"
# would mean "one family at one of its infinitely many settings".
#
# EXPANDING THE LIBRARY CANNOT MANUFACTURE A CLAIM, and that is worth stating because it is the
# reason this is safe to do late. A new candidate can only turn a unique survivor into an ambiguous
# pair, or name a record that previously had none. It can never make an already-named form名 change
# its function, and the test asserts exactly that against the committed artifact.
SHIFTADD = {
    'add_2b': lambda a, b: (a + 2 * b) & M32, 'add_4b': lambda a, b: (a + 4 * b) & M32,
    'add_8b': lambda a, b: (a + 8 * b) & M32, 'sub_2b': lambda a, b: (a - 2 * b) & M32,
    'a2_add_b': lambda a, b: (2 * a + b) & M32, 'a4_add_b': lambda a, b: (4 * a + b) & M32,
    'mulhi_u': lambda a, b: ((a * b) >> 32) & M32,
    'mulhi_s': lambda a, b: ((_s(a) * _s(b)) >> 32) & M32,
    'avg_u': lambda a, b: ((a + b) >> 1) & M32,
    'absdiff_u': lambda a, b: abs((a & M32) - (b & M32)) & M32,
}
INT2.update(SHIFTADD)
INT3 = {
    'madd': lambda a, b, c: (a * b + c) & M32,
    'msub': lambda a, b, c: (a * b - c) & M32,
    'add3': lambda a, b, c: (a + b + c) & M32,
    'select_a_gt_b': lambda a, b, c: c,          # placeholder, replaced for arity 4
}
# ROUNDING MUST PRESERVE THE SIGN OF ZERO, and getting this wrong is the third instance of one
# mistake in this repository: math.ceil and math.floor return INTEGERS, so ceil(-0.3) is 0 and the
# sign is gone, while IEEE-754 roundToIntegral gives -0.0 and so does this hardware. Written
# straight it produced a FALSE NEGATIVE - op3802, whose ceil I had already confirmed by hand, came
# back as "no candidate fits". A false negative is the safe direction and still wrong.
def _stated_configuration(op, plan):
    """The operands a record CHOSE, as (operand, value): its `imms`, and - for a record authored on
    its own TEMPLATE bytes at the authoring table's width - every raw operand the template carries
    at a value other than the table witness's.

    A TEMPLATE IS A CHOICE AS MUCH AS AN IMMS KEY IS. isa/g17-execution-compiledunary's op1062 ran
    the compiler's own fsat instance (source modifier 2) and fitted fsat, beside three witness-
    authored records (modifier 0) that fitted fsat_neg. Keyed by `imms` alone they were one
    configuration with two functions, and g17questions' null control rightly flagged op1062 as an
    opcode carrying two. Read from the template, the compiled instance states (3, 2) and is its own
    configuration. Only the template the PLAN supplies is read - not the executed bytes, whose
    lifetimes the liveness pass rewrites on every record and would fragment every group."""
    stated = {int(k): int(v) for k, v in (plan.get('imms') or {}).items()}
    tmpl = plan.get('bytes')
    if tmpl and plan.get('author') in ('fieldmap', 'assembler') and not plan.get('program'):
        try:
            raw = bytes.fromhex(tmpl)
            if len(raw) == g17auth.length(op):
                fm = g17auth.fields(op)
                mine = g17auth.decode(op, raw)
                base = g17auth.decode(op, bytes.fromhex(g17auth.record(op)['witness']))
                for i, (dom, _pos) in fm.items():
                    if dom == 'raw' and i not in stated and mine.get(i) != base.get(i):
                        stated[i] = int(mine.get(i))
        except Exception:
            pass
    return sorted(stated.items())


def _integral(value, mode):
    # NaN AND INFINITY PROPAGATE. They did not, and that made ceil/floor/trunc predict 0 for a NaN
    # input where the hardware returns the canonical quiet NaN - so the census rejected three
    # names that were correct on 15 of 16 cases. IEEE fixes this a priori; it is a bug in the
    # model, not a value fitted to the measurement that exposed it.
    if isinstance(value, float):
        if value != value:
            return float('nan')
        if value in (float('inf'), float('-inf')):
            return value
    import math
    out = {'ceil': math.ceil, 'floor': math.floor, 'trunc': math.trunc}[mode](value)
    out = float(out)
    if out == 0.0 and math.copysign(1.0, value) < 0:
        out = -0.0
    return out


def _rne(x):
    """Round to nearest, TIES TO EVEN, on a Python float. Not `round()` in spirit but in fact -
    Python's built-in already breaks ties to even, and this spells it out because the tie rule is
    the whole content of the two candidates below and must not read as incidental."""
    if x != x or math.isinf(x):
        return x
    floor = math.floor(x)
    frac = x - floor
    if frac > 0.5:
        return floor + 1
    if frac < 0.5:
        return floor
    return floor if floor % 2 == 0 else floor + 1


def _frint(word):
    """f32 round-to-nearest-even, NaN canonicalised, and a zero result KEEPING ITS SIGN.

    The sign clause is not decoration: -0.5 rounds to a zero whose sign the hardware preserves
    (observed 0x80000000), and a first version of this candidate written as a one-line lambda
    dropped it and returned +0.0. It fitted four of the form's five records anyway, because only
    one case in fifty-nine asks - which is exactly the shape that makes a clause look incidental.
    """
    x = _f(word)
    if x != x:
        return 0x7FC00000
    if math.isinf(x):
        return _u(x)
    rounded = _rne(x)
    if rounded == 0 and (word & 0x80000000):
        return 0x80000000
    return _u(rounded)


def _hsat_nan_and_negzero_to_zero(word):
    """Clamp a half to [0, 1], with NaN and negative zero both answering POSITIVE zero."""
    x = _h(word)
    if x != x:
        return 0x0000
    clamped = min(1.0, max(0.0, x))
    return 0x0000 if clamped == 0 else _uh(clamped)


def _hrint(word):
    """The f16 counterpart. THE TIE DIRECTION IS NOT MEASURED FOR THIS FORM.

    op3775 is named `rint.f16` and its records separate rounding from truncation at exactly two
    cases, 1.5 -> 2.0 and -1.5 -> -2.0 - and ties-to-even and ties-away-from-zero AGREE on both,
    because 2 is even and away-from-zero also gives 2. So the direction here is INHERITED from
    op3770, where it is measured, and that is corroboration from a sibling rather than evidence
    for this form. The inputs that would settle it are +-2.5 and +-0.5 in half: even gives 2.0
    and -0.0, away-from-zero gives 3.0 and -1.0.
    """
    x = _h(word)
    if x != x:
        return 0x7E00
    if math.isinf(x):
        return _uh(x)
    rounded = _rne(x)
    if rounded == 0 and (word & 0x8000):
        return 0x8000
    return _uh(rounded)


FLOAT1 = {
    'fneg': lambda a: _u(-_f(a)), 'fabs': lambda a: _u(abs(_f(a))),
    'fceil': lambda a: _u(_integral(_f(a), 'ceil')),
    'ftrunc': lambda a: _u(_integral(_f(a), 'trunc')),
    'ffloor': lambda a: _u(_integral(_f(a), 'floor')),
    # THE FLOOR THIS HARDWARE PERFORMS ON A DENORMAL INPUT, measured on op3786 at 13
    # preregistered cases (isa/g17-execution-denorm-results.json, plan 3fab5f60, control exact).
    # `ffloor` is mathematically right and wrong about the machine: at 0x80000001, the smallest
    # negative denormal, it returns -1.0 and the hardware returns -0.0. The input is flushed to
    # zero with its sign kept, and THEN floored.
    #
    # Three DIFFERENT negative denormals discriminate - minimum, maximum and mid-range - and all
    # three returned -0.0. The control that could have refuted it did not: 0x80800000, the
    # smallest negative NORMAL, floored to -1.0 rather than -0.0, so the rule is specific to
    # denormals and is not "anything tiny becomes zero". Positive denormals measure nothing here,
    # because floor(+tiny) is +0.0 under either rule.
    #
    # f32 ONLY, deliberately. The rule was measured at f32; `3791/10` is promoted on the HALF
    # `hfloor`, and a half twin could turn that determination into an ambiguity on evidence from
    # a different opcode at a different width. No promoted form fits `ffloor` itself, so this
    # addition can retract nothing.
    # ROUND TO NEAREST, TIES TO EVEN - the function Apple's table calls `rint`, which this
    # library did not have at all. op3770 is named `rint` and read as `ffloor_denormal_to_zero`
    # for that reason: floor and rint agree on every input its narrow record asked, and the wide
    # 31-case record then REFUTED floor at four cases, which is what the census reports as a
    # record of a form refuting its own fit. The four are 1.5 -> 2.0, -2.5 -> -2.0, -0.5 -> -0.0
    # and 3.9 -> 4.0.
    #
    # THE TIE DIRECTION IS MEASURED HERE, and that is worth stating because it usually is not:
    # -2.5 -> -2.0 and -0.5 -> -0.0 separate ties-to-EVEN from ties-away-from-zero, which would
    # give -3.0 and -1.0. A negative zero result keeps its sign, also measured, at -0.5.
    'frint': lambda a: _frint(a),
    'ffloor_denormal_to_zero': lambda a: (
        (int(a) & 0x80000000) if ((int(a) & 0x7F800000) == 0 and (int(a) & 0x007FFFFF) != 0)
        else _u(_integral(_f(a), 'floor'))),
    'fsat': lambda a: _u(_sat(_f(a))), 'fsat_neg': lambda a: _u(_sat(-_f(a))),
    'fhalf': lambda a: _u(_f(a) * 0.5), 'fdouble': lambda a: _u(_f(a) * 2.0),
    'fsquare': lambda a: _u(_f(a) * _f(a)),
    'frcp': lambda a: _u(1.0 / _f(a)) if _f(a) != 0 else None,
    'ffract': lambda a: _u(_f(a) - _integral(_f(a), 'floor')),
}
FLOAT2 = {
    'fadd': lambda a, b: _u(_f(a) + _f(b)), 'fsub': lambda a, b: _u(_f(a) - _f(b)),
    'fmul': lambda a, b: _u(_f(a) * _f(b)),
    'fmin': lambda a, b: _u(min(_f(a), _f(b))), 'fmax': lambda a, b: _u(max(_f(a), _f(b))),
}
FLOAT3 = {
    'ffma': lambda a, b, c: _u(_f(a) * _f(b) + _f(c)),
    'ffms': lambda a, b, c: _u(_f(a) * _f(b) - _f(c)),
    'fadd3': lambda a, b, c: _u(_f(a) + _f(b) + _f(c)),
}
# THE SILICON'S OWN CONVENTIONS, BESIDE THE PURE ONES RATHER THAN INSTEAD OF THEM.
#
# A batch of sixteen cases dispatched at op3290 `fmul` (isa/g17-execution-arithseparate.json,
# preregistered in its own commit) missed the library's `fmul` on exactly six cases, and all six
# are one of two behaviours:
#
#     pi * 0x00010001  -> 0x00000000    an f32 INPUT DENORMAL flushed to a zero of its own sign
#     0x0000FFFF * -1  -> 0x80000000    the same, with the sign coming from the other operand
#     0xFFFFFFFF * 0.3 -> 0x7FC00000    a NaN operand answered with the CANONICAL quiet NaN
#
# Both are already measured facts of this hardware - the input flush by op3338 and op3341's own
# probes, the canonical NaN by the arity-1 confirm batch - and neither was expressible in the
# arithmetic library, so every form that saw a denormal or a NaN read as "no candidate fits".
# That is the raising-probe defect with the raised hypothesis being a CONVENTION rather than a
# function: six misses out of sixteen for the name written on the opcode.
#
# THEY ARE ADDED AS RIVALS, NOT AS A REPAIR. A pure `fmul` and a flushing one differ on inputs
# this hardware admits, so a record holding no denormal and no NaN genuinely cannot choose between
# them and must report both - which is the honest outcome and the reason the pair is not merged.
# The batch above separates them on six of its sixteen cases, so where it runs, the population
# decides.
def _ftz_qnan(fn, nargs, flush_result=False):
    """Input-denormal flush and canonical NaN, optionally flushing a subnormal RESULT too.

    THE RESULT FLUSH IS A PEER REPORT VERIFIED HERE, NOT ADOPTED FROM IT. The TensorOps lane
    measured that fmul, fadd, fsub and fma flush subnormal INPUTS AND subnormal RESULTS to a zero
    of the same sign, that normal results stay IEEE, and that none of four compile modes changes
    it (docs/g17-tensorops-accelerator-recon.md section 138 part 5). My own candidates flushed
    only the input, so a form whose product underflows would have missed and been filed as
    fitting nothing.

    It is added as a RIVAL rather than folded into the existing wrapper, for the same reason every
    convention here is: the two differ only when a result is subnormal, so a record with no such
    case cannot choose between them, and merging would assert an elimination this lane never
    performed on evidence it did not gather. Where the population separates them it decides; where
    it does not, both survive and the entry says so.
    """
    def wrapped(*args):
        flushed = [_f(_flush_f32_input_denormal(int(a) & M32)) for a in args[:nargs]]
        try:
            value = fn(*flushed)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
        if value != value:
            return 0x7FC00000
        if any(_f(int(a) & M32) != _f(int(a) & M32) for a in args[:nargs]):
            return 0x7FC00000
        bits = _u(value)
        if flush_result and bits is not None:
            bits = _flush_f32_input_denormal(bits)
        return bits
    return wrapped


_FTZ2 = {'fadd': lambda a, b: a + b, 'fsub': lambda a, b: a - b,
         'fmul': lambda a, b: a * b, 'fmin': min, 'fmax': max}
_FTZ3 = {'ffma': lambda a, b, c: a * b + c, 'ffms': lambda a, b, c: a * b - c,
         'fadd3': lambda a, b, c: a + b + c}
for _n, _fn in _FTZ2.items():
    FLOAT2['%s_ftz_qnan' % _n] = _ftz_qnan(_fn, 2)
    # NOT FOR min AND max, where the result flush cannot change anything: the result IS one of
    # the inputs, and the inputs were flushed on the way in, so it can never be subnormal. The
    # twin would be the same function under a second name, which no form computing it could
    # escape. The duplicate audit named both pairs.
    if _n not in ('fmin', 'fmax'):
        FLOAT2['%s_ftz_qnan_ftzout' % _n] = _ftz_qnan(_fn, 2, flush_result=True)
for _n, _fn in _FTZ3.items():
    FLOAT3['%s_ftz_qnan' % _n] = _ftz_qnan(_fn, 3)
    FLOAT3['%s_ftz_qnan_ftzout' % _n] = _ftz_qnan(_fn, 3, flush_result=True)


# THREE-OPERAND RIVALS, ADDED BECAUSE THE BAR REFUSED SIX FORMS FOR HAVING TOO FEW.
#
# The integer arity-3 table held FOUR functions, one of which the LIBRARY excludes, so a unique
# survivor there had eliminated two candidates. The bar asks for eight, and its own text says why:
# "a unique survivor of three candidates eliminated two, and of twenty-eight eliminated
# twenty-seven; the count of competitors is part of the claim". Six madd-family forms fit
# `integer.madd`, `integer.msub` and `float.ffma` uniquely on sixteen preregistered cases with the
# control exact, and were refused - correctly - for exactly that reason.
#
# ADDING RIVALS MAKES THE TEST HARDER, WHICH IS WHY THIS IS NOT FITTING TO THE ANSWER. Every one
# below is a three-operand function a GPU instruction set plausibly has, chosen without looking at
# what the refused forms returned; if `madd` still survives them uniquely, the claim is stronger
# than it was, and if one of them TIES, the form was never determined and the census will say so.
# The duplicate audit runs over them like any other candidate: a function already in the table
# under another name would make both unpromotable forever.
INT3.update({
    'msub_rev': lambda a, b, c: (int(c) - int(a) * int(b)) & M32,
    'mul3': lambda a, b, c: (int(a) * int(b) * int(c)) & M32,
    'shift_add': lambda a, b, c: ((int(a) << (int(b) & 31)) + int(c)) & M32,
    'shift_sub': lambda a, b, c: ((int(a) << (int(b) & 31)) - int(c)) & M32,
    'add_shift': lambda a, b, c: ((int(a) + int(b)) << (int(c) & 31)) & M32,
    'madd_lo16': lambda a, b, c: (int(a) * (int(b) & 0xFFFF) + int(c)) & M32,
    'mad_hi': lambda a, b, c: (((int(a) * int(b)) >> 32) + int(c)) & M32,
    'bitfield_extract': lambda a, b, c: (int(a) >> (int(b) & 31)) & ((1 << (int(c) & 31)) - 1),
    'clamp_i': lambda a, b, c: max(min(int(a), int(c)), int(b)) & M32,
    'xor3': lambda a, b, c: (int(a) ^ int(b) ^ int(c)) & M32,
    'and_or': lambda a, b, c: ((int(a) & int(b)) | int(c)) & M32,
    'funnel_left': lambda a, b, c: (((int(a) << (int(c) & 31))
                                     | (int(b) >> (32 - (int(c) & 31)) if (int(c) & 31) else 0))
                                    & M32),
})
FLOAT3.update({
    'fmul3': lambda a, b, c: _u(_f(a) * _f(b) * _f(c)),
    'ffma_rev': lambda a, b, c: _u(_f(c) * _f(a) + _f(b)),
    'fnma': lambda a, b, c: _u(_f(c) - _f(a) * _f(b)),
    'fclamp': lambda a, b, c: _u(max(min(_f(a), _f(c)), _f(b))),
})


# THE CONDITIONAL-SELECT SHAPE, WHICH THE THREE-OPERAND LIBRARY DID NOT HOLD AT ALL.
#
# Forty-eight untouched forms are named `csel` and not one of them has a unique fit, which reads
# as forty-eight unknown instructions and is really one missing shape: a select whose PREDICATE is
# a property of one source and whose other two operands are the arms. The library's arity-3
# functions were all arithmetic, so a select could only ever be reported as fitting nothing.
#
# The predicate is taken over the FIRST source alone, because that is what the declared operand
# classes support: `11395/14` reads (GPR16, GPR32, GPR32), a narrow predicate source and two
# 32-bit arms, and its authored operand sweep showed the arm flipping with operand 2 while both
# arms were held distinct. A two-source comparison is a different shape and is not assumed here.
#
# EVERY ARM ORDER IS OFFERED IN BOTH DIRECTIONS, because which arm a taken branch returns is
# exactly what is unknown, and offering only one direction would make half the population refute
# a predicate that is right with its arms swapped - the failure the csel census already reports
# as "its records REFUTE every predicate offered".
def _sel3(test, swap=False):
    def fn(a, b, c):
        taken = test(int(a) & M32)
        return (int(c) if taken else int(b)) if swap else (int(b) if taken else int(c))
    return fn


# NO TEST HERE MAY BE ANOTHER'S COMPLEMENT, because every test is offered with both arm orders
# and a complement with swapped arms is the SAME FUNCTION. `sel_zero` was in this table and is
# exactly `sel_nonzero_swapped`; the duplicate audit caught it, and a library holding one function
# twice can never name it - every form computing it would have two survivors forever. The audit is
# the reason this is a comment and not a shipped defect.
_SEL3_TESTS = {
    'nonzero': lambda a: a != 0,
    'negative': lambda a: bool(a & 0x80000000),
    'lsb': lambda a: bool(a & 1),
    'low16_nonzero': lambda a: bool(a & 0xFFFF),
    'half_negative': lambda a: bool(a & 0x8000),
    'all_ones': lambda a: (a & M32) == M32,
}
for _name, _test in _SEL3_TESTS.items():
    INT3['sel_%s' % _name] = _sel3(_test)
    INT3['sel_%s_swapped' % _name] = _sel3(_test, swap=True)


SELECT4 = {
    'sel_a_gt_b': lambda a, b, x, y: x if a > b else y,
    'sel_a_lt_b': lambda a, b, x, y: x if a < b else y,
    'sel_a_eq_b': lambda a, b, x, y: x if a == b else y,
    'sel_a_ne_b': lambda a, b, x, y: x if a != b else y,
    'sel_s_gt': lambda a, b, x, y: x if _s(a) > _s(b) else y,
    # THE BIT TEST, which op11375's code 1 is (ledger g17-csel-code-one-is-a-bit-test). Without it
    # a record on equal nonzero pairs and unequal disjoint ones fitted `a == b` uniquely, because
    # the two agree there; with it such a record is ambiguous, which is what it is.
    'sel_a_and_b_nonzero': lambda a, b, x, y: x if (int(a) & int(b)) != 0 else y,
    'sel_a_and_b_zero': lambda a, b, x, y: x if (int(a) & int(b)) == 0 else y,
    'sel_always_x': lambda a, b, x, y: x, 'sel_always_y': lambda a, b, x, y: y,
}

# HALF PRECISION, A WHOLE INTERPRETATION THE LIBRARY DID NOT HAVE. The arity-1 splitting batch
# (plan 49ade47b, results 6cb21223) dispatched 58 ambiguous opcodes and 32 returned a function
# outside all twenty-four candidates - with 0x3C00, 0x7C00 and 0xFC00 recurring in the outputs,
# which are half 1.0, +inf and -inf. A census offering only integer and single-precision readings
# cannot fit an fp16 opcode at all, and reports it as "no candidate fits" however many records it
# is given: the missing thing was an INTERPRETATION, not more data.
#
# The set mirrors the single-precision one in half, plus the two width conversions, and is
# deliberately NOT tailored to any observed vector. op3299 came back as exactly 1/4 of op1004 on
# every non-zero case and there is no `h2f_quarter` here, because a candidate invented to fit one
# observation is an absorbing fit wearing a name - both opcodes have a free immediate their record
# never wrote (128 and 16), which is the likelier home for a factor of four and is a question for
# a dispatch rather than for this table.
def _h(u):
    """Low sixteen bits read as an IEEE half. struct, not arithmetic, so -0.0, the subnormals and
    both infinities survive - this project has put a signed-zero bug into these helpers twice."""
    return struct.unpack('<e', struct.pack('<H', int(u) & 0xFFFF))[0]


H_INF, H_NINF, H_QNAN = 0x7C00, 0xFC00, 0x7E00


def _uh(x):
    """A Python float back to half bits, with IEEE specials as VALUES rather than as None.

    None means "this candidate cannot answer here" and the census skips it - which was wrong for
    the transcendentals, because the hardware does answer: exp2 of 16 overflows half and returns
    an infinity, log2 of 0 returns -inf, log2 of a negative returns a NaN. Predicting None there
    would report "no candidate fits" for an opcode that behaved exactly as its name says, which is
    the a-raising-probe-is-not-a-negative-result defect one level in.

    The NaN is 0x7E00, the positive quiet pattern, because that is what this path was MEASURED to
    produce today: four opcodes canonicalise a half NaN to 0x7E00 and negative zero to +0
    (isa/g17-execution-arity1confirm-results.json). So this is a measurement carried into the
    model, not an IEEE assumption.
    """
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if x != x:
        return H_QNAN
    if x == float('inf'):
        return H_INF
    if x == float('-inf'):
        return H_NINF
    try:
        return struct.unpack('<H', struct.pack('<e', x))[0]
    except (OverflowError, ValueError):
        # out of half's range in the direction its sign says
        return H_NINF if x < 0 else H_INF


def _log2(x):
    if x > 0:
        return math.log2(x)
    return float('-inf') if x == 0 else float('nan')


def _rsqrt(x):
    if x > 0:
        return 1.0 / math.sqrt(x)
    if x == 0:
        # THE SIGN OF ZERO SURVIVES THE RECIPROCAL: rsqrt(-0.0) is -inf, not +inf. IEEE says so
        # and the hardware agrees; my model did not, and it cost rsqrt.f16 its sixteenth case.
        return math.copysign(float('inf'), x)
    return float('nan')


def _half_unary(fn):
    def call(a):
        try:
            return _uh(fn(_h(a)))
        except (OverflowError, ValueError, ZeroDivisionError):
            return None
    return call


HALF1 = {
    # ROUND TO NEAREST IN HALF PRECISION, the f16 sibling of `frint`. op3775 is named `rint.f16`
    # and read as `htrunc`, because truncation and rounding agree on every case its 16-case record
    # asked; the 31-case record refutes truncation at +-1.5. See `_hrint` for the clause its
    # population does NOT measure.
    'hrint': lambda a: _hrint(a),
    # SATURATE WITH NaN -> +0 AND NEGATIVE ZERO -> +0. op767 is named `fadd.imm.sat.f16` and read
    # as `hsat`; its two wider records refute plain `hsat` at exactly the inputs where a saturate
    # has a choice. A half NaN returns 0x0000 rather than the canonical 0x7E00, and a half -0.0
    # returns 0x0000 rather than 0x8000 - so the clamp's lower bound is a POSITIVE zero and the
    # NaN answer matches op766's saturating add rather than op3338's multiply. A TWIN, not an edit
    # to the shared clamp, for the reason op766 needed one: the NaN answer is not uniform across
    # this hardware, and pushing it into the helper would rewrite every other saturating candidate
    # on the strength of one opcode.
    'hsat_nan_and_negzero_to_zero': lambda a: _hsat_nan_and_negzero_to_zero(a),
    # the two width conversions, which are what an fp16 pipeline mostly does
    'h2f': lambda a: _u(_h(a)),
    'f2h': lambda a: _uh(_f(a)),
    'h2f_hi': lambda a: _u(struct.unpack('<e', struct.pack('<H', (int(a) >> 16) & 0xFFFF))[0]),
    # the single-precision set, in half
    'habs': _half_unary(abs),
    'hneg': _half_unary(lambda x: -x),
    'hhalf': _half_unary(lambda x: x * 0.5),
    'hdouble': _half_unary(lambda x: x * 2.0),
    'hsquare': _half_unary(lambda x: x * x),
    'hrcp': _half_unary(lambda x: 1.0 / x if x else None),
    'hsat': _half_unary(lambda x: _sat(x)),
    'hsat_neg': _half_unary(lambda x: _sat(x, -1.0, 0.0)),
    'hfloor': _half_unary(lambda x: _integral(x, 'floor')),
    'hceil': _half_unary(lambda x: _integral(x, 'ceil')),
    'htrunc': _half_unary(lambda x: _integral(x, 'trunc')),
    'hfract': _half_unary(lambda x: x - _integral(x, 'floor')),
    # THE TRANSCENDENTALS, ADDED BEFORE THE DISPATCH THAT TESTS THEM AND NOT AFTER.
    # Apple's table names ceil.f16, exp2.f16, log2.f16 and rsqrt.f16, and the source-reach batch
    # made the compiler emit all four. Three of them had NO candidate here, so a probe would have
    # returned "no candidate fits" and the honest-looking reading - the name does not hold - would
    # have been a statement about my library. That exact defect has already cost this project four
    # forms called unrecovered because the function name did not exist
    # (a-raising-probe-is-not-a-negative-result). A name cannot be tested against a set that
    # excludes it.
    # The specials follow IEEE because the hardware was measured to: log2(0) = -inf,
    # log2(negative) = NaN, rsqrt(0) = +inf, and exp2 overflowing half = +inf.
    'hexp2': _half_unary(lambda x: 2.0 ** x if x < 17 else float('inf')),
    # A SEPARATE CANDIDATE, NOT AN EDIT TO hexp2. op1277 returned 0x7BFF - half's largest finite
    # value, 65504 - where exp2(16) = 65536 overflows and IEEE says infinity. That is a hardware
    # behaviour measured on 2026-09-17, so it gets its own name and competes; folding it into
    # hexp2 would let the fit absorb the finding and nothing would record that the unit saturates.
    'hexp2_sat': _half_unary(lambda x: min(2.0 ** x, 65504.0) if x < 1e4 else 65504.0),
    'hlog2': _half_unary(lambda x: _log2(x)),
    'hsqrt': _half_unary(lambda x: math.sqrt(x) if x >= 0 else float('nan')),
    'hrsqrt': _half_unary(lambda x: _rsqrt(x)),
}


# MASKS AND EXTENSIONS, WHICH THE INTEGER SET DID NOT HAVE EITHER. op408 was measured as
# `a & 0xFFEF` and op10284 as a sign extension of the low half, both on this hardware and both
# recorded in tools/g17isamap.py - yet neither function was offered by this library, so any
# opcode computing one could only ever read "no candidate fits". Worse, their ABSENCE let a half
# candidate win uncontested: `htrunc` was the unique survivor for op3775 over the arity-1
# splitting batch, and `a & 0xFFFC` reproduces all four of those values exactly. The two differ
# on any half below 1.0 - htrunc(0.5) is 0, the mask leaves 0x3800 alone - so the fit was unique
# only because the competitor was missing.
MASK1 = {
    'trunc8': lambda a: int(a) & 0xFF,
    'trunc12': lambda a: int(a) & 0xFFF,
    'trunc16': lambda a: int(a) & 0xFFFF,
    'trunc20': lambda a: int(a) & 0xFFFFF,
    'trunc24': lambda a: int(a) & 0xFFFFFF,
    'sext8': lambda a: (int(a) & 0xFF) - 0x100 & M32 if int(a) & 0x80 else int(a) & 0xFF,
    'sext16': lambda a: (int(a) & 0xFFFF) - 0x10000 & M32 if int(a) & 0x8000 else int(a) & 0xFFFF,
    'mask_ffef': lambda a: int(a) & 0xFFEF,
    'clear_low1': lambda a: int(a) & (M32 ^ 0x1),
    'clear_low2': lambda a: int(a) & (M32 ^ 0x3),
    'clear_low3': lambda a: int(a) & (M32 ^ 0x7),
    'trunc16_clear_low1': lambda a: int(a) & 0xFFFE,
    'trunc16_clear_low2': lambda a: int(a) & 0xFFFC,
    'trunc16_clear_low3': lambda a: int(a) & 0xFFF8,
}


def _flush_f32_input_denormal(word):
    """An f32 operand that is denormal reads as a zero carrying its own sign.

    MEASURED, not assumed, and measured separately for each form that uses it: op3338's
    E3338.round and op3341's C3341.denorm each probe both operands, a negative denormal so
    the sign is checked rather than inherited, the largest denormal, and the smallest NORMAL
    value as the control immediately above the boundary - so a rule that flushes at the wrong
    threshold is caught instead of being averaged away.
    """
    word &= 0xFFFFFFFF
    if (word & 0x7F800000) == 0 and (word & 0x007FFFFF) != 0:
        return word & 0x80000000
    return word


def _flush_bf16_input_denormal(half):
    """The bf16 counterpart, for an operand read out of a 16-bit lane."""
    half &= 0xFFFF
    if (half & 0x7F80) == 0 and (half & 0x007F) != 0:
        return half & 0x8000
    return half


def _f32_word_of(product):
    """The f32 bit pattern of a Python float, with overflow handled explicitly."""
    if math.isinf(product):
        return 0xFF800000 if product < 0 else 0x7F800000
    try:
        return struct.unpack('<I', struct.pack('<f', product))[0]
    except OverflowError:
        # THE HARNESS BUG THIS FUNCTION EXISTS TO NOT REPEAT: packing an out-of-range double
        # as f32 raises, and a bare `except: return None` turns an overflow into a miss or,
        # worse, into a zero that agrees with a zero observation by accident.
        return 0xFF800000 if product < 0 else 0x7F800000


def _bf16_of_f32_word(word):
    """Round an f32 pattern to bf16, round-to-nearest-EVEN, flushing a denormal result.

    The rounding is measured, not a default. Both tie parities are probed - a tie whose bf16
    mantissa is even must stay put and one whose mantissa is odd must step up - which separates
    round-half-even from round-half-up. The earlier version of this function TRUNCATED, and the
    44 retained cases of op3338 could not tell the two apart: not one of them had a product whose
    discarded bits reached half a bfloat16 ULP, so truncation was the fitting code's default
    rather than an observation. E3338.round refutes truncation directly, on three cases.
    """
    if (word & 0x7F800000) == 0x7F800000 and (word & 0x007FFFFF) != 0:
        return 0x7FC0
    half = (word >> 16) & 0xFFFF
    low = word & 0xFFFF
    if low > 0x8000 or (low == 0x8000 and (half & 1)):
        half = (half + 1) & 0xFFFF
    if (half & 0x7F80) == 0 and (half & 0x007F) != 0:
        half &= 0x8000
    return half


def _bf16_of_f32_product(a, b):
    """op3338: bf16 of the product of two f32 operands, input and output denormals flushed."""
    x = _f(_flush_f32_input_denormal(a))
    y = _f(_flush_f32_input_denormal(b))
    if x != x or y != y:
        return 0x7FC0
    product = x * y
    if product != product:
        return 0x7FC0
    return _bf16_of_f32_word(_f32_word_of(product))


def _bf16_of_f32_times_low_lane(a, b):
    """op3341: bf16 of (f32 operand a) x (operand b's LOW 16 bits read as a bf16).

    The two forms carry the same Apple name, `fmul`, and the same length, and differ only in
    how the second operand is read - b's low lane here, the whole word in op3338. That is a
    difference a name cannot express, which is why the census keeps them as separate forms.

    Which lane b supplies is measured, not inferred from the operand class: the probes place
    b's two halves 2^25 apart and then REVERSE which half is large, so a rule reading the wrong
    half, an absorbing rule, and a monotone rule each fail at least one. A third probe puts
    opposite signs in the two halves, so the lane is confirmed by the result's sign as well as
    its magnitude.
    """
    x = _f(_flush_f32_input_denormal(a))
    y = _f(_flush_bf16_input_denormal(b & 0xFFFF) << 16)
    if x != x or y != y:
        return 0x7FC0
    product = x * y
    if product != product:
        return 0x7FC0
    return _bf16_of_f32_word(_f32_word_of(product))


def _half_binary(fn):
    def call(a, b):
        try:
            return _uh(fn(_h(a), _h(b)))
        except (OverflowError, ValueError, ZeroDivisionError):
            return None
    return call


# ARITY-2 HALF, added for the same reason as the unary set and BEFORE the dispatch that needs it:
# `fadd.sat.f16` and `fmul.f32.f16.to.f16` are names about to be tested, and a name cannot be
# tested against a library that excludes it. HALF2 also carries the MIXED form the second name
# describes - an f32 source times an f16 source, narrowed to f16 - because a set holding only
# both-half candidates would report "no candidate fits" for a mixed-precision opcode and that
# would read as the name failing.
HALF2 = {
    'hadd': _half_binary(lambda a, b: a + b),
    'hsub': _half_binary(lambda a, b: a - b),
    'hmul': _half_binary(lambda a, b: a * b),
    'hmax': _half_binary(max),
    'hmin': _half_binary(min),
    'hadd_sat': _half_binary(lambda a, b: _sat(a + b)),
    # THE SATURATE THIS HARDWARE ACTUALLY PERFORMS, measured on op766 at 12 preregistered cases
    # (isa/g17-execution-satrule-results.json, plan 46dcdc46, run c3620786, control exact).
    # `_sat` is IEEE-shaped: it preserves the sign of zero and lets a NaN through, which `_uh`
    # then canonicalises to 0x7E00. The hardware does NEITHER - (-0.0)+(-0.0) returns +0.0, and
    # three distinct NaN payloads (0xFFFF, 0x7FFF, 0x7E00) all return +0.0. Four of the twelve
    # cases discriminate the two and the observation matched this one at every one, while eight
    # controls agreed under both: 0.5+0.25 unclamped, a subnormal pair surviving, both infinities
    # clamping.
    #
    # ADDED AS A TWIN RATHER THAN A CHANGE TO `_sat`, deliberately. The rule was measured on an
    # ADD at f16; applying it to `hmul_sat`, `hsat`, `fsat` or the integer saturates would
    # generalise one opcode's behaviour across a family, which is the error this file has caught
    # three times. `hmul_sat` in particular carries the promotion of 854/10 on a single record,
    # and a twin of it could turn that determination into an ambiguity on evidence from a
    # different opcode. No promoted form fits `hadd_sat`, so this addition can retract nothing.
    'hadd_sat_nan_to_zero': _half_binary(
        lambda a, b: 0.0 if (a + b) != (a + b) else _sat(a + b) + 0.0),
    'hmul_sat': _half_binary(lambda a, b: _sat(a * b)),
    'hadd_sat_signed': _half_binary(lambda a, b: _sat(a + b, -1.0, 1.0)),
    'f32_x_f16_to_f16': lambda a, b: _uh(_f(a) * _h(b)),
    # A BFLOAT16 MULTIPLY - an interpretation this library did not hold at all, and op3338 needed
    # it. The result is the HIGH HALF of the f32 product, not an IEEE half narrowing: at
    # (0xFFFFFFFF, 0x4) the hardware returns 0x7FC0, which is the top sixteen bits of the f32
    # quiet NaN 0x7FC00000, where every half candidate returns the canonicalised half NaN 0x7E00.
    #
    # Three clauses, and the last two were fitted from t3338's own misses before being confirmed
    # OUT OF SAMPLE at 13 new cases (isa/g17-execution-bf16mul-results.json, plan committed
    # first, control exact):
    #   the bf16 high half of the f32 product;
    #   NaN canonicalised to the f32 quiet NaN - holds at a signalling NaN, a negative quiet NaN,
    #   a payload the records never carried, and 0*inf, which IEEE generates rather than receives;
    #   a denormal result flushed to zero with its sign kept.
    # OVERFLOW was pure prediction - no retained record of this opcode overflows f32 - and the
    # two overflow cases returned 0x7F80 and 0xFF80, the bf16 of both infinities, as predicted.
    #
    # In the half table because the result is sixteen bits and because `f32_x_f16_to_f16` and
    # `hmul_f32_operands` already sit here while reading f32 operands. It is NOT an IEEE half and
    # the name says so.
    'bf16_mul_of_f32_operands': lambda a, b: _bf16_of_f32_product(a, b),
    'bf16_mul_of_f32_by_low_lane': lambda a, b: _bf16_of_f32_times_low_lane(a, b),
    'f32_plus_f16_to_f16': lambda a, b: _uh(_f(a) + _h(b)),
    'hmul_f32_operands': lambda a, b: _uh(_f(a) * _f(b)),
}


# MIXED-WIDTH INTEGER ARITY-2, DRIVEN BY THE OPERAND CLASSES AND NOT BY AN OBSERVATION.
# op11667 and op11668 are both named `sub` and differ in one declared class: op11667's second
# source is GPR32, op11668's is GPR16. A 4-case record had fitted op11668/12 as plain `sub`, and
# the 10-case record refuted it on the single pair whose second operand has nonzero high bits -
# 0x0ABCDEF1 - 0x5A5A5A5A came back 0x0ABC8497, not 0xB0628497. `a - (b & 0xFFFF)` fits all ten
# exactly, and g17auth.operand_classes says operand 4 IS GPR16, so this is the DECLARED width
# rather than a curve through the data. Adding the family generally, because any opcode pairing a
# 32-bit first source with a 16-bit second one has the same reading available and the library
# could not express it at all.
MIXED2 = {
    # SATURATING ARITHMETIC WAS A WHOLE MISSING INTERPRETATION, not a missing fit. Apple's table
    # names `subsat` and `addsat` and the arity-2 integer library held neither, so six `subsat`
    # records read as "no candidate fits" for an opcode behaving exactly as its name says - the
    # fifth time this file has been extended for that reason after fp16, masks, mixed precision
    # and the 32-minus-16 subtract. Both signs are offered because the name does not say which,
    # and the unsigned one is what survived: op11624 and op11637 eliminate `subsat_s` and the
    # reversed reading.
    #
    # THESE WILL MAKE SOME EXISTING FITS AMBIGUOUS AND THAT IS CORRECT. A saturating subtract
    # equals a plain one on every case that does not saturate, so any record whose inputs never
    # reach the bound cannot tell them apart and should stop claiming it can.
    'subsat_u': lambda a, b: max(int(a) - int(b), 0) & M32,
    'subsat_s': lambda a, b: (max(min(_s(a) - _s(b), 0x7FFFFFFF), -0x80000000)) & M32,
    'addsat_u': lambda a, b: min(int(a) + int(b), M32),
    'addsat_s': lambda a, b: (max(min(_s(a) + _s(b), 0x7FFFFFFF), -0x80000000)) & M32,
    'subsat_u_b16': lambda a, b: max((int(a) & 0xFFFF) - (int(b) & 0xFFFF), 0),
    'addsat_u_b16': lambda a, b: min((int(a) & 0xFFFF) + (int(b) & 0xFFFF), 0xFFFF),
    'sub_b16': lambda a, b: (int(a) - (int(b) & 0xFFFF)) & M32,
    'add_b16': lambda a, b: (int(a) + (int(b) & 0xFFFF)) & M32,
    # BOTH OPERANDS NARROWED, AND THE SUM NOT TRUNCATED - the interpretation op10286 needed and
    # the library did not hold. `add_b16` narrows only the SECOND operand and keeps the first
    # one's high bits, so at (0x12345678, 0x0F0F0F0F) it gives 0x12346587 where the hardware
    # returns 0x6587. Narrowing both and keeping the carry gives 0x6587, and it fits EVERY case
    # of all four of op10286's records - 34 cases across the residue, shiftpairs, sweep and widen
    # batches - while no other candidate in this library computes it.
    #
    # This is the defect class "a form called unrecovered because the function name did not
    # exist", and op10286's Apple name is `add`, which agrees: it is an add, of two sixteen-bit
    # sources, into a destination wide enough to hold the carry.
    'add_widen_b16': lambda a, b: ((int(a) & 0xFFFF) + (int(b) & 0xFFFF)) & M32,
    'and_b16': lambda a, b: int(a) & (int(b) & 0xFFFF),
    'or_b16': lambda a, b: int(a) | (int(b) & 0xFFFF),
    'xor_b16': lambda a, b: int(a) ^ (int(b) & 0xFFFF),
    'sub_b16_sext': lambda a, b: (int(a) - ((((int(b) & 0xFFFF) ^ 0x8000) - 0x8000))) & M32,
    'add_b16_sext': lambda a, b: (int(a) + ((((int(b) & 0xFFFF) ^ 0x8000) - 0x8000))) & M32,
}


# THE BFLOAT ROUNDING SHAPE, WHICH THE SIXTEEN-BIT UNARY TABLE DID NOT HOLD.
#
# Four rounding forms - 3785 rint, 3789 and 3801 floor, 3833 trunc - declare GPR16 sources and
# have non-degenerate records from a preregistered batch, and NOTHING in the library fit them.
# Read as bfloats they fit their own names: `3789/10` floor misses zero of fourteen into an f32
# destination and `3801/10` misses zero of fourteen into a bfloat one. The half readings miss
# three to seven. This is the same defect as the transcendental width assumption two commits ago -
# `GPR16` fixes a register's WIDTH, not its format - and it had cost four more forms.
#
# THE SIGN OF ZERO IS THE WHOLE REMAINING DIFFERENCE for trunc and rint, and this project has put
# that bug into these helpers three times. `math.trunc(-0.0078)` returns the INT 0, which packs to
# +0.0, and the hardware returns 0x8000. An int-valued result cannot carry a zero's sign, so it
# has to be taken from the input. Both conventions are offered as RIVALS rather than the measured
# one alone: the population separates them on three of fourteen cases, so the choice is made by
# the records and not by this comment.
def _bf_round(fn, keep_sign, flush=False):
    def wrapped(u):
        half = _flush_bf16_input_denormal(int(u) & 0xFFFF) if flush else (int(u) & 0xFFFF)
        word = half << 16
        x = _f(word)
        if x != x:
            return 0x7FC0
        if math.isinf(x):
            return 0xFF80 if x < 0 else 0x7F80
        try:
            y = float(fn(x))
        except (ValueError, OverflowError):
            return None
        bits = _bf16_of_f32_word(_f32_word_of(y))
        if keep_sign and (bits & 0x7FFF) == 0:
            bits = (bits & 0x7FFF) | (int(u) & 0x8000)
        return bits
    return wrapped


def _bf_round_wide(fn, keep_sign, flush=False):
    """The same, into a THIRTY-TWO bit destination - op3789's shape."""
    def wrapped(u):
        half = _flush_bf16_input_denormal(int(u) & 0xFFFF) if flush else (int(u) & 0xFFFF)
        word = half << 16
        x = _f(word)
        if x != x:
            return 0x7FC00000
        if math.isinf(x):
            return 0xFF800000 if x < 0 else 0x7F800000
        try:
            y = float(fn(x))
        except (ValueError, OverflowError):
            return None
        bits = _f32_word_of(y)
        if keep_sign and (bits & 0x7FFFFFFF) == 0:
            bits = (bits & 0x7FFFFFFF) | ((int(u) & 0x8000) << 16)
        return bits
    return wrapped


def _half_even(x):
    floor = math.floor(x)
    diff = x - floor
    if diff > 0.5:
        return floor + 1
    if diff < 0.5:
        return floor
    return floor if floor % 2 == 0 else floor + 1


_BF_ROUNDS = {'floor': math.floor, 'ceil': math.ceil, 'trunc': math.trunc, 'rint': _half_even}
# THE INPUT-DENORMAL FLUSH, AGAIN, AND ONE CASE FOUND IT. `3789/10` and `3801/10` fit
# `bf16_floor` on every case of two batches and were refuted by a single input: 0x8001 is a
# NEGATIVE bfloat denormal, floor of it is -1.0, and the hardware returns -0. This silicon flushes
# an input denormal to a zero of its own sign - measured by op3338 and op3341's probes, and now by
# these - and floor of -0 is -0. The flushing variant is a RIVAL beside the IEEE one, not a
# replacement: they differ only on denormal inputs, so a batch without one cannot choose, and the
# census reports both rather than taking the flattering reading.
for _name, _fn in _BF_ROUNDS.items():
    HALF1['bf16_%s' % _name] = _bf_round(_fn, True)
    HALF1['bf16_%s_to_f32' % _name] = _bf_round_wide(_fn, True)
    # THE FLUSHING TWIN EXISTS ONLY WHERE THE FLUSH CHANGES THE ANSWER, and for two of these four
    # it cannot: truncation toward zero of ANY denormal is a zero carrying the input's sign, which
    # is exactly what flushing the input produces, and round-to-nearest of a denormal is the same.
    # Only floor and ceil move a denormal to a NON-zero - floor(-9e-41) is -1.0 - so only they
    # have two functions here. Offering the other two twins put one function in the library twice,
    # which the duplicate audit caught and which no form computing it could ever escape.
    if _name in ('floor', 'ceil'):
        HALF1['bf16_%s_ftz' % _name] = _bf_round(_fn, True, flush=True)
        HALF1['bf16_%s_to_f32_ftz' % _name] = _bf_round_wide(_fn, True, flush=True)
    # THE UNSIGNED-ZERO TWIN EXISTS ONLY WHERE IT IS A DIFFERENT FUNCTION. `floor` never returns a
    # zero from a negative input - floor(-0.0078) is -1 - so its two sign conventions agree
    # everywhere and offering both would put one function in the library twice, which no form
    # computing it could ever escape. ceil, trunc and rint each do return a zero from a negative
    # input, so for them the twin is a real rival and the records choose.
    if _name != 'floor':
        HALF1['bf16_%s_plus_zero' % _name] = _bf_round(_fn, False)
        HALF1['bf16_%s_to_f32_plus_zero' % _name] = _bf_round_wide(_fn, False)


# FIXED SHIFTS BEYOND ONE, COMPLETING A SHAPE THE LIBRARY ALREADY HAD AT k=1.
#
# `shl1` and `shr1` have always been here, so a one-argument instruction that shifts by a CONSTANT
# is a shape this census already believed in - it just stopped at one. Two forms were reported as
# "no candidate of the published library fits" for want of the rest: `16808/12`, named `asr`,
# returns 0x4049 -> 0x1012, which is a shift right by TWO, and `10280/12`, named `add`, returns
# 0x1000 -> 0x10000, a shift left by FOUR. Both names are about something other than what the
# form computes, which is worth knowing on its own.
#
# The amounts are the ones an address calculation or a field extract plausibly uses. Extending an
# existing family to its natural range is a different act from inventing a shape to fit a result:
# every one of these was already implied by `shl1`, and adding them raises the competitor count
# for every one-argument integer form rather than lowering the bar for these two.
for _k in (2, 3, 4, 5, 6, 7, 8, 12, 16, 24):
    INT1['shl%d' % _k] = (lambda k: lambda v: (int(v) << k) & M32)(_k)
    INT1['shr%d' % _k] = (lambda k: lambda v: (int(v) & M32) >> k)(_k)
    INT1['asr%d' % _k] = (lambda k: lambda v: (_s(int(v)) >> k) & M32)(_k)
    # AND THE NARROW ONES, because both forms that wanted this shape declare a GPR16 SOURCE.
    # `16808/12` returns 0x3FFF from 0xFFFFFFFF - fourteen bits, not thirty - so it shifted the
    # LOW SIXTEEN and not the word; `10280/12` returns 0xFFFF0 from the same input. The library
    # has carried this narrowing convention since `add_b16`, and reading the full word was why
    # both looked like "no candidate fits" on every record that reached past sixteen bits, while
    # fitting perfectly on the ones that did not.
    #
    # ONLY BELOW SIXTEEN, because at or above it the narrow variant stops being a distinct
    # function: a sixteen-bit value shifted right by sixteen is always zero (`shr16_b16` IS
    # `zero`), shifted left by sixteen it keeps nothing the wide shift would have kept
    # (`shl16` IS `shl16_b16`), and an arithmetic shift past the sign bit is all sign bits
    # whatever the distance. The duplicate audit named all four groups; each is an identity, so
    # the repair is to stop generating them rather than to widen the probe.
    if _k < 16:
        INT1['shl%d_b16' % _k] = (lambda k: lambda v: ((int(v) & 0xFFFF) << k) & M32)(_k)
        INT1['shr%d_b16' % _k] = (lambda k: lambda v: (int(v) & 0xFFFF) >> k)(_k)
        INT1['asr%d_b16' % _k] = (lambda k: lambda v:
                                  ((((int(v) & 0xFFFF) ^ 0x8000) - 0x8000) >> k) & M32)(_k)


LIBRARY = {1: dict(integer=dict(INT1, **MASK1), float=FLOAT1, half=HALF1), 2: dict(integer=dict(INT2, **MIXED2), float=FLOAT2, half=HALF2),
           3: dict(integer={k: v for k, v in INT3.items() if k != 'select_a_gt_b'},
                   float=FLOAT3),
           4: dict(integer=SELECT4, float={})}


# A LIBRARY HOLDING ONE FUNCTION TWICE CAN NEVER NAME IT. `fidentity` (a float move) and
# `identity` (an integer move) are the same bit function under two names, so every record of an
# instruction that simply moves its operand had at least two survivors and could NEVER be
# promoted - a uniqueness test that one true answer is structurally unable to pass. Found by
# evaluating every candidate against a diverse vector and grouping by behaviour, and fixed by
# deleting the duplicate rather than by special-casing the comparison.
#
# THE CHECK IS BUILT IN so a candidate added later cannot quietly reintroduce the problem. It is
# also a lesson in its own right: the FIRST version of this audit took the first 64 tuples of
# itertools.product, which holds the first operand at V[0], and reported the four-source selects
# as five duplicate groups. They are all distinct. The probe vector was degenerate, in exactly the
# way the census itself is built to detect.
DEDUP_SEED = 20260917
# NEGATIVE DENORMALS ARE IN HERE NOW, AND THEIR ABSENCE WAS A BLIND SPOT. The set held
# 0x00000001 - a POSITIVE f32 denormal - and 0x80000000, which is negative zero and not a
# denormal at all. Positive denormals cannot separate a denormal-flushing operation from an IEEE
# one, because floor(+tiny) is +0.0 under both; only a NEGATIVE denormal does, where IEEE floor
# gives -1.0 and a flushing one gives -0.0. So the duplicate detector reported `ffloor` and
# `ffloor_denormal_to_zero` as the same function while op3786's hardware distinguishes them at
# three different negative denormals (isa/g17-execution-denorm-results.json).
#
# Widening a probe set can only SPLIT groups, never merge them, so this makes the detector
# strictly more discriminating - for every candidate pair, not just the one that exposed it.
DEDUP_PROBE = (0x00000000, 0x00000001, 0x3F800000, 0xBF800000, 0x40490FDB, 0xC0490FDB,
               0x7FFFFFFF, 0x80000000, 0x0000FFFF, 0xFFFF0000, 0x12345678, 0xDEADBEEF,
               0x00001000, 0x40000000, 0x3DCCCCCD, 0xBDCCCCCD,
               0x80000001, 0x807FFFFF, 0x007FFFFF, 0x80800000,
               # ROUNDING TIES, ADDED BECAUSE THE PROBE COULD NOT SEPARATE `rint` FROM `trunc`.
               # Every value above is an integer, a NaN, a denormal, or has a fraction below a
               # half, so round-to-nearest and truncation agree on all twenty and the duplicate
               # detector would have called a newly added `frint` a duplicate of `ftrunc` -
               # correctly, on that probe. Repair the probe, not the guard. The f32 group carries
               # both tie parities and both signs, so half-to-EVEN is separable from
               # half-away-from-zero as well as from truncation, and two values with a fraction
               # ABOVE a half separate rounding from floor.
               0x3FC00000, 0xBFC00000, 0x40200000, 0xC0200000, 0x3F000000, 0xBF000000,
               # AND THE SAME TIES IN THE LOW SIXTEEN BITS, because a BFLOAT candidate reads them
               # there and every f32 tie above has low bits of zero - so `0x3FC00000` is 1.5 to an
               # f32 reader and 0.0 to a bfloat one. The detector duly called `bf16_rint` a
               # duplicate of `bf16_trunc`: on that probe they ARE the same function, since no
               # value it carries has a bfloat fraction at all. Repair the probe, not the guard -
               # a second time, for a second float format.
               0x00003FC0, 0x0000BFC0, 0x00004020, 0x0000C020, 0x00003F00, 0x0000BF00,
               0x00003FA0, 0x0000BFA0, 0x00004040, 0x0000C040,
               # AND A NEGATIVE BFLOAT DENORMAL, which is what separates a flushing rounder from
               # an IEEE one. The probe carried 0x80000001, whose low half is a POSITIVE bfloat
               # denormal - floor of it and floor of +0 are both +0, so the two agreed and the
               # detector called them duplicates. The sign is the whole distinction: floor of a
               # negative denormal is -1.0, floor of the -0 it flushes to is -0.
               0x00008001, 0x00008040, 0x0000807F,
               # A NEAR-CANCELLATION PAIR, because a SUM of two normals can only become subnormal
               # by cancelling, and without one the result-flush twin of `fadd` is the plain one:
               # the detector called them duplicates and it was right about the probe. These two
               # differ by about 1e-45, so their difference is subnormal while both are normal.
               0x00800001, 0x00800002, 0x80800001, 0x00800000,
               0x4079999A, 0xBE99999A,
               # and the same ties in the LOW HALF, because a half candidate reads these words
               # there and a word that is an f32 tie has a low half of zero.
               0x00003E00, 0x0000BE00, 0x00003800, 0x0000B800, 0x00004480)


def duplicate_candidates(samples=400):
    """Groups of candidates of the same arity that agree on every sampled tuple. Should be empty."""
    import random
    rnd = random.Random(DEDUP_SEED)
    dupes = {}
    for arity, kinds in sorted(LIBRARY.items()):
        combos = [tuple(rnd.choice(DEDUP_PROBE) for _ in range(arity)) for _ in range(samples)]
        signatures = collections.defaultdict(list)
        for kind, library in kinds.items():
            for name, fn in library.items():
                out, ok = [], True
                for combo in combos:
                    try:
                        value = fn(*combo)
                    except Exception:
                        value = None
                    if value is None:
                        ok = False
                        break
                    out.append(int(value) & M32)
                signatures[tuple(out) if ok else ('undefined', kind, name)].append(
                    '%s:%s' % (kind, name))
        for names in signatures.values():
            if len(names) > 1 and not str(names[0]).startswith('undefined'):
                dupes.setdefault(arity, []).append(sorted(names))
    return dupes


def load_pairs():
    plans, results = {}, {}
    for path in sorted(ISA.glob('g17-execution-*.json')):
        name = path.name
        try:
            recs = json.loads(path.read_text())
        except Exception:
            continue
        recs = recs if isinstance(recs, list) else recs.get('results', list(recs.values()))
        is_result = name.endswith('-results.json')
        stem = name[:-len('-results.json')] if is_result else name[:-len('.json')]
        for record in recs:
            if isinstance(record, dict) and record.get('id'):
                (results if is_result else plans).setdefault(stem, {})[record['id']] = record
    return plans, results


def widths_of(res, op):
    out = set()
    for hexed in (res.get('decoded') or {}).get('encoded') or []:
        try:
            raw = bytes.fromhex(hexed)
        except Exception:
            continue
        for ins in decode_instruction(raw, 0):
            if ins.opcode is not None and ins.opcode.id == op:
                out.add(len(ins.raw))
    return sorted(out)


MIN_CASES, MIN_RUNS, MIN_COMPETITORS = 4, 3, 8
# How many candidates a unique survivor actually beat, per (arity, interpretation). Derived from
# the library rather than restated, so adding a candidate moves the bar's denominator.
COMPETITORS = {(arity, kind): len(lib) for arity, kinds in LIBRARY.items()
               for kind, lib in kinds.items()}


def _kind_of(row):
    for kind, names in row['survivors'].items():
        if len(names) == 1:
            return kind
    return None


def survivors(cases, values, arity):
    """Which candidates of each interpretation reproduce EVERY case. Both are returned."""
    found = {}
    for kind, library in (LIBRARY.get(arity) or {}).items():
        keep = []
        for name, fn in library.items():
            try:
                got = [fn(*[int(x) & M32 for x in case]) for case in cases]
            except Exception:
                continue
            if any(g is None for g in got):
                continue
            if [int(g) & M32 for g in got] == [int(v) & M32 for v in values]:
                keep.append(name)
        found[kind] = sorted(keep)
    return found


def unwritten_immediates(op, plan):
    """Non-register operands this record did NOT state, minus the ones it cannot state.

    Operand 1 is excluded because every opcode in this table has it free, so including it makes
    the answer "all of them" and the census measures its own frame - 208 of 208 opcodes, which is
    what the first version of this reported. Lifetime carriers are excluded because a record
    cannot state one: the compiler's liveness pass rewrites it into the finished bytes after the
    encoder has honoured the request, so a value asked for there is discarded without a word
    (tools/g17oracle.py refuses such a record now).

    What is left is the operands a record COULD have written and did not, which therefore hold
    whatever the authoring witness carries.
    """
    dsts, srcs = g17auth.register_operands(op)
    owned = {c for s in srcs for c in (g17auth.lifetime_operand(op, s),) if c is not None}
    stated = {int(k) for k in (plan.get('imms') or {})}
    return sorted(set(g17auth.fields(op)) - set(dsts) - set(srcs) - owned - stated - {1})


def asked_outside_the_domain(rows):
    """Constant-output records whose constant is ABSORBING and whose bounds were never written.

    This is op612's signature, and op612 is why the class exists. It returned 0 for all four of
    its inputs, no candidate in the library is constant at 0, elimination reported nothing left,
    and the record was written down as an unreachable. It is a four-bit range predicate whose
    window is operands 4 and 5, both left at the witness's values, and every input chosen was far
    above the window - so zero was the CORRECT answer four times. Re-probed inside the window it
    determined in one batch, and its structural twin op621 in one more.

    A constant output and a dead instrument give the same vector, and so does a correct answer
    asked outside the subject's domain. What separates the third from the first two is cheap: move
    an unwritten immediate and see whether the answer moves. This names the records where that is
    worth one dispatch, and it is a CANDIDATE SET, not a set of findings - the shape is necessary
    for the op612 story and nowhere near sufficient.

    ABSORBING means all-zero or all-ones. A constant at some other value is excluded here not
    because it cannot have this cause but because it is not the shape a range check produces
    outside its range, and a criterion that admits everything ranks nothing.
    """
    out = {}
    for row in rows:
        if not row['outputs_constant_across_cases'] or not row['unwritten_immediates']:
            continue
        if row['constant_value'] not in ('0x00000000', '0xFFFFFFFF'):
            continue
        out.setdefault('op%d' % row['op'], []).append(
            dict(stem=row['stem'], id=row['id'], constant=row['constant_value'],
                 unwritten_immediates=row['unwritten_immediates'], cases=row['cases'],
                 destination_file=destination_file(row['op'])))
    return out


def destination_file(op):
    """The register class this opcode's destination is declared in, or None if unknown.

    Carried because the class above is NOT homogeneous without it, and an aggregate over mixed
    causes ranks nothing. A record whose destination is FLAGR can return a robust zero because the
    flag file is not visible to a GPR store, which this repository measured on 2026-09-07 and is a
    COMPETING explanation, not the one this class is about. Splitting them is also what refutes
    part of that measurement: its headline groups op612 with op10369 as opcodes that "write
    somewhere a GPR store cannot read", and op612's destination is a GPR16 out of which this
    campaign read 24 varying values. op10369's is a FLAGR and that half stands.
    """
    try:
        dsts, _srcs = g17auth.register_operands(op)
        return g17auth.operand_classes(op)[dsts[0]] if dsts else None
    except Exception:
        return None


def _by_destination(flagged):
    """The class split by destination register file, with a count of records and opcodes.

    Not a presentation choice: 21 of the 94 records write FLAGR, whose constant output has a cause
    already measured here, and reporting one number over both populations is the defect this file
    has hit before - an aggregate over a population that is not homogeneous.
    """
    out = {}
    for op, recs in flagged.items():
        for rec in recs:
            key = rec['destination_file'] or 'unknown'
            row = out.setdefault(key, dict(records=0, opcodes=set()))
            row['records'] += 1
            row['opcodes'].add(op)
    return {k: dict(records=v['records'], opcodes=len(v['opcodes']),
                    which=sorted(v['opcodes']),
                    competing_cause=('the flag file is not visible to a GPR store - '
                                     'ledger/g17-what-a-gpr-store-cannot-see.toml'
                                     if k == 'FLAGR' else None))
            for k, v in sorted(out.items())}


WIDTH_PROBE_SEED = 20260918


def candidate_operand_widths(arity, kind, trials=64):
    """For each candidate, which operands does it read at 16 bits rather than 32?

    MEASURED, not declared. A candidate reads operand i at sixteen bits exactly when changing the
    HIGH half of that operand never changes its answer - so this probes each candidate with random
    words, flips the high half of one operand at a time, and records whether anything moved. A
    hand-written table of "which candidates are half functions" would drift from the library the
    moment a candidate is added, and the library grew four times in one day.

    The destination width is measured the same way: a candidate whose output never has a nonzero
    high half writes sixteen bits.
    """
    rng = random.Random(WIDTH_PROBE_SEED)
    lib = (LIBRARY.get(arity) or {}).get(kind) or {}
    out = {}
    for name, fn in sorted(lib.items()):
        reads_high = [False] * arity
        writes_high = False
        answered = 0
        for _ in range(trials):
            base = [rng.getrandbits(32) for _ in range(arity)]
            try:
                got = fn(*base)
            except Exception:
                continue
            if got is None:
                continue
            answered += 1
            if (int(got) >> 16) & 0xFFFF:
                writes_high = True
            for i in range(arity):
                alt = list(base)
                alt[i] ^= (rng.getrandbits(16) or 1) << 16
                try:
                    other = fn(*alt)
                except Exception:
                    continue
                if other is not None and int(other) != int(got):
                    reads_high[i] = True
        out[name] = dict(
            answered=answered,
            reads_operand_high_half=reads_high,
            writes_high_half=writes_high,
            operand_widths=[32 if h else 16 for h in reads_high],
            destination_width=32 if writes_high else 16)
    return out


def shapes_the_library_cannot_express(rows):
    """no-fit records whose DECLARED operand widths no candidate of that arity reads.

    This is the question that has mattered all day: is a "no candidate fits" a gap in my knowledge
    or a gap in my LIBRARY? Four times today it was the library - fp16 had no interpretation at
    all, masks had none, mixed precision had none, and a 32-bit-minus-16-bit subtract had none,
    and each time the census reported "no candidate fits" for an opcode that behaved exactly as
    its name said. So the shapes are enumerated and checked against the widths the candidates
    actually read.

    A shape being expressible does NOT mean the opcode is understood - it means the failure to fit
    is not explained by operand widths, which is the only thing this function can settle.
    """
    widths = {}
    for arity, kinds in LIBRARY.items():
        for kind in kinds:
            if LIBRARY[arity][kind]:
                widths[(arity, kind)] = candidate_operand_widths(arity, kind)
    per_shape = collections.defaultdict(lambda: dict(records=0, opcodes=set()))
    for row in rows:
        if row['verdict'] != 'no candidate fits':
            continue
        if row['outputs_constant_across_cases'] or row['outputs_equal_an_input_column']:
            continue
        try:
            classes = g17auth.operand_classes(row['op'])
            dsts, srcs = g17auth.register_operands(row['op'])
        except Exception:
            continue
        if not dsts:
            continue
        def declared(cls):
            return 16 if cls and 'GPR16' in cls else 32 if cls else None
        want_src = [declared(classes[i]) for i in srcs]
        want_dst = declared(classes[dsts[0]])
        if any(w is None for w in want_src) or want_dst is None:
            continue
        key = '%d <- %s' % (want_dst, ','.join(str(w) for w in want_src))
        bucket = per_shape[key]
        bucket['records'] += 1
        bucket['opcodes'].add('op%d' % row['op'])
        arity = len(want_src)
        expressible = False
        for (a, _kind), table in widths.items():
            if a != arity:
                continue
            for _name, sig in table.items():
                if sig['operand_widths'] == want_src and sig['destination_width'] == want_dst:
                    expressible = True
                    break
            if expressible:
                break
        bucket['a_candidate_reads_these_widths'] = expressible
    return {k: dict(records=v['records'], opcodes=len(v['opcodes']),
                    which=sorted(v['opcodes'])[:10],
                    a_candidate_reads_these_widths=v.get('a_candidate_reads_these_widths'))
            for k, v in sorted(per_shape.items(),
                               key=lambda kv: -kv[1]['records'])}


# APPLE'S NAME TOKENS AND WHAT THEY MEAN, DECLARED RATHER THAN FUZZY-MATCHED. A substring test
# would claim agreement it cannot justify - `bitwise_6` and `xor` share no characters, and `hadd`
# and `fadd.f16` share only "add". So the correspondence is written out, and a fitted function
# with no entry here scores `unknown_correspondence` instead of a guess.
NAME_MEANS = {
    'hadd': ('fadd', 16), 'hsub': ('fsub', 16), 'hmul': ('fmul', 16),
    'hmul_sat': ('fmul.sat', 16), 'hadd_sat': ('fadd.sat', 16),
    'bf16_mul_of_f32_operands': ('fmul', 16),
    'bf16_mul_of_f32_by_low_lane': ('fmul', 16),
    'hadd_sat_nan_to_zero': ('fadd.sat', 16),
    'hmax': ('fmax', 16), 'hmin': ('fmin', 16),
    'hceil': ('ceil', 16), 'hfloor': ('floor', 16), 'htrunc': ('trunc', 16),
    'hexp2': ('exp2', 16), 'hlog2': ('log2', 16), 'hrsqrt': ('rsqrt', 16),
    'hsqrt': ('sqrt', 16), 'habs': ('fabs', 16), 'hneg': ('fneg', 16),
    'hsat': ('sat', 16), 'h2f': ('cvt', 32), 'f2h': ('cvt', 16),
    'f32_x_f16_to_f16': ('fmul', 16), 'f32_plus_f16_to_f16': ('fadd', 16),
    'hmul_f32_operands': ('fmul', 16),
    'sub': ('sub', 32), 'add': ('add', 32), 'fmul': ('fmul', 32), 'fadd': ('fadd', 32),
    'fsub': ('fsub', 32), 'sub_b16': ('sub', 32), 'add_b16': ('add', 32),
    'add_widen_b16': ('add', 32),
    'subsat_u': ('subsat', 32), 'subsat_s': ('subsat', 32), 'addsat_u': ('addsat', 32),
    'addsat_s': ('addsat', 32), 'subsat_u_b16': ('subsat', 16), 'addsat_u_b16': ('addsat', 16),
    'sar1': ('asr', 32), 'shr1': ('shr', 32), 'shl1': ('shl', 32),
    'trunc16': ('trunc', 32), 'sext16': ('sext', 32), 'identity': ('mov', 32),
    # the bitwise family is indexed by its truth table, so the correspondence is arithmetic
    'bitwise_0': ('zero', 32), 'bitwise_1': ('nor', 32), 'bitwise_6': ('xor', 32),
    'bitwise_7': ('nand', 32), 'bitwise_8': ('and', 32), 'bitwise_9': ('xnor', 32),
    'bitwise_14': ('or', 32), 'bitwise_15': ('ones', 32),
    'bitwise_A': ('mov.b', 32), 'bitwise_C': ('mov.a', 32),
    # THE f32 UNARY FAMILY HAD NO CORRESPONDENCE AT ALL, so every fit in it scored
    # `unknown_correspondence` and nothing could notice a fit CONTRADICTING the table's own name.
    # op3801 is named `floor` and was explained as `fhalf`; op3785 `rint` and op3833 `trunc` were
    # explained as `fhalf` too - three operations that cannot all be a multiply by one half.
    'ffloor': ('floor', 32), 'ffloor_denormal_to_zero': ('floor', 32),
    'frint': ('rint', 32), 'hrint': ('rint.f16', 16),
    'msb_lo16': ('msb', 16), 'hsat_nan_and_negzero_to_zero': ('fadd.imm.sat.f16', 16),
    'fceil': ('ceil', 32), 'ftrunc': ('trunc', 32),
    'fabs': ('fabs', 32), 'fneg': ('fneg', 32), 'fsat': ('sat', 32),
    'frcp': ('recip', 32), 'ffract': ('fract', 32), 'fsquare': ('square', 32),
    'fdouble': ('double', 32), 'fhalf': ('half', 32),
    'hsquare': ('square', 16), 'hdouble': ('double', 16), 'hhalf': ('half', 16),
    'hfract': ('fract', 16), 'hrcp': ('recip', 16),
    'clz': ('clz', 32), 'popcount': ('popcount', 32), 'byteswap': ('byteswap', 32),
    'msb': ('msb', 32), 'bitreverse': ('reverse', 32), 'bitreverse_b16': ('reverse', 16),
    'neg': ('neg', 32), 'not': ('not', 32), 'abs_s32': ('abs', 32),
    'mul': ('mul', 32), 'shl': ('shl', 32), 'shr': ('shr', 32), 'sar': ('asr', 32),
    'smax': ('smax', 32), 'smin': ('smin', 32), 'umax': ('umax', 32), 'umin': ('umin', 32),
    'mulhi_s': ('mulhi', 32), 'mulhi_u': ('mulhi', 32), 'rotl': ('rotl', 32),
    'rsub': ('rsub', 32), 'absdiff_u': ('absdiff', 32), 'avg_u': ('avg', 32),
}


# OPERATIONS THAT CANNOT AGREE, so if they do the PROBE is degenerate and no fit from it counts.
# Each pair is mutually exclusive by definition over any input set that varies at all: a maximum
# and a minimum agree only when everything being reduced is equal, and a sum and a product agree
# only in the same degenerate case. The tokens are matched on Apple's own names.
CANNOT_AGREE = (
    ('fmax', 'fmin'), ('smax', 'smin'), ('umax', 'umin'),
    ('sum', 'product'), ('max', 'min'), ('shuffle_up1', 'shuffle_down1'),
    # ADDED 2026-09-18 after it caught something. An ARITHMETIC right shift and a LOGICAL one
    # differ on every negative input, so `asr` and `shr` cannot compute the same function - and
    # op16819 `asr` and op17045 `shr`, both declaring GPR16 for the destination and both sources,
    # returned BYTE-IDENTICAL vectors across all 17 cases and both fit a 16-bit logical shift at
    # 17 of 17. The 32-bit pair does NOT do this: op16807 `asr` and op17014 `shr` differ, so the
    # defect is specific to the narrow forms and is most likely an operand these records do not
    # write. Neither narrow form is determined.
    ('asr', 'shr'), ('sar', 'shr'),
)


def probe_degeneracy(rows, universe, results_by_id):
    """Pairs of forms whose Apple names cannot describe the same function, that returned the same
    values anyway - which means the PROBE could not tell them apart.

    This exists because it caught four promoted determinations. op16860 `simd.fmax.f16` and
    op16868 `simd.fmin.f16` returned BYTE-IDENTICAL vectors in two separate batches, as did
    `simd.sum.f16` against `simd.product.f16`. A maximum and a minimum agree only when every lane
    holds the same value, so the harness was feeding all lanes identically and every cross-lane
    operation was returning its input - which the library read as a low-sixteen-bit truncation, at
    16 of 16 inputs, with a committed expectation that held.

    A passing prediction is not evidence when the probe cannot distinguish the hypothesis from its
    opposite. This is the a-test-a-degenerate-hypothesis-passes defect, and the fourth instance,
    so it gets a detector rather than another note.
    """
    by_name = {}
    by_op = {}
    for row in rows:
        nm = (universe.get('op%d' % row['op']) or {}).get('name')
        if nm and (row['stem'], row['id']) in results_by_id:
            by_name.setdefault((row['stem'], nm), []).append((row['stem'], row['id']))
            by_op[(row['stem'], row['id'])] = row['op']

    def _shape(op):
        """The declared widths of a form, so a comparison is between comparable things.

        THE SEPARATION CLAIM AND THE DEGENERACY CLAIM NEED DIFFERENT CARE HERE, and the asymmetry
        is the reason this exists. Both detectors key on Apple's NAME, so `asr` and `shr` match
        whichever opcodes carry those tokens - and this table has a 16-bit asr and a 32-bit shr.
        For degeneracy that is conservative: two records agreeing across different widths still
        says the probe is not discriminating. For SEPARATION it is the opposite, because two
        records of different FORMS returning different values is exactly what different forms do,
        and reading it as "the complement is separated" would lift a block on no evidence.
        """
        try:
            classes = g17auth.operand_classes(op)
            dsts, srcs = g17auth.register_operands(op)
            return (tuple(16 if 'GPR16' in (classes[i] or '') else 32 for i in dsts),
                    tuple(16 if 'GPR16' in (classes[i] or '') else 32 for i in srcs))
        except Exception:
            return None
    flagged = []
    for (stem, nm), ids in sorted(by_name.items()):
        for a, b in CANNOT_AGREE:
            if a not in nm:
                continue
            other = nm.replace(a, b)
            mate = by_name.get((stem, other))
            if not mate:
                continue
            for i in ids:
                for j in mate:
                    if results_by_id[i] == results_by_id[j]:
                        flagged.append(dict(
                            stem=stem, a=dict(id=i[1], name=nm), b=dict(id=j[1], name=other),
                            identical_values=True,
                            means=('%s and %s cannot compute the same function, so this probe '
                                   'cannot distinguish them and no fit from either record may be '
                                   'promoted' % (nm, other))))
    # THE INVERSE DETECTOR, WITHOUT WHICH A LIFTED BLOCK CANNOT BE SEEN. `pairs` above says a
    # probe could not tell two opcodes apart; nothing said when one finally could. The
    # lane-varying probe of 2026-09-18 makes op16860 `simd.fmax.f16` return 0x3C1F and op16868
    # `simd.fmin.f16` return 0x3C00 AT THE SAME INPUTS, which is the observation the whole
    # cross-lane class was blocked on - and with only the degeneracy detector those records would
    # have been logged and the forms would still read "blocked on a probe carrying per-lane
    # variation", because the old uniform records still pair and the new ones fit no candidate.
    #
    # SAME INPUTS IS THE WHOLE CLAIM. Two records of complementary opcodes at DIFFERENT inputs
    # returning different values says nothing at all, so the case lists are compared and only
    # matching ones count. That is not a hypothetical: the batch that produced this evidence gave
    # its maximum records a positive base and its minimum records a negative one, so none of its
    # own pairs is comparable, and a separate two-record batch had to be run at one base.
    plans = _plans_by_stem_and_id()
    separated = []
    for (stem, nm), ids in sorted(by_name.items()):
        for a, b in CANNOT_AGREE:
            if a not in nm:
                continue
            other = nm.replace(a, b)
            for i in ids:
                for j in by_name.get((stem, other), []):
                    if results_by_id[i] == results_by_id[j]:
                        continue
                    pi, pj = plans.get(i) or {}, plans.get(j) or {}
                    if not pi.get('cases') or pi.get('cases') != pj.get('cases'):
                        continue
                    oi, oj = by_op.get(i), by_op.get(j)
                    if oi is None or oj is None or _shape(oi) is None:
                        continue
                    if _shape(oi) != _shape(oj):
                        continue
                    separated.append(dict(
                        stem=stem, a=dict(id=i[1], op=oi, name=nm, values=results_by_id[i]),
                        b=dict(id=j[1], op=oj, name=other, values=results_by_id[j]),
                        declared_shape='dst%s <- srcs%s' % _shape(oi),
                        means=('%s and %s returned DIFFERENT values at the same inputs, so an '
                               'observation distinguishing them exists and neither is blocked on '
                               'the complement cause any more' % (nm, other))))
    return dict(pairs=flagged, separated_at_the_same_inputs=separated,
                separated_means=('the inverse of `pairs`: complementary opcodes a probe CAN tell '
                                 'apart, compared only where the case lists are identical AND the '
                                 'declared operand widths match. Different inputs returning '
                                 'different values proves nothing, and neither does a 16-bit form '
                                 'differing from a 32-bit one - that is what different forms do. A '
                                 'form appearing here is no longer blocked on the complement, '
                                 'whatever its older uniform-lane records still show'),
                checked_pairs=len(CANNOT_AGREE),
                means=('forms whose Apple names are mutually exclusive that returned identical '
                       'values. A non-empty list means a probe is DEGENERATE for those forms - '
                       'for the cross-lane family it means every lane held the same value - and a '
                       'prediction that held on such a probe is not evidence'))


def _values_by_id():
    """Retained values keyed by (STEM, id), because an id alone is not unique.

    `CONTROL.op10279` appears in a dozen plan files - it is the mandatory control - so a dict keyed
    on the id alone holds whichever batch was read last and silently answers for all of them. That
    is how a scratch analysis came to pair one batch's inputs with another batch's outputs and
    report 39 records of a proven `src + 4` control as "not a constant offset". The census itself
    keys rows by (stem, id) and was never wrong; this helper was.
    """
    out = {}
    for path in sorted(ISA.glob('g17-execution-*-results.json')):
        stem = path.name[:-len('-results.json')]
        try:
            rows = json.loads(path.read_text())
        except Exception:
            continue
        for row in (rows if isinstance(rows, list) else rows.get('results') or []):
            if isinstance(row, dict) and row.get('id') and row.get('values') is not None:
                out[(stem, row['id'])] = [int(v) & M32 for v in row['values']]
    return out


def _plans_by_stem_and_id():
    out = {}
    for path in sorted(ISA.glob('g17-execution-*.json')):
        if path.name.endswith('-results.json'):
            continue
        try:
            rows = json.loads(path.read_text())
        except Exception:
            continue
        for row in (rows if isinstance(rows, list) else []):
            if isinstance(row, dict) and row.get('id') and row.get('cases'):
                out[(path.name[:-len('.json')], row['id'])] = row
    return out


def _encoded_by_id():
    """The emitted bytes per (stem, id), from the results files - the census rows do not carry
    them, and reading `row['encoded']` returned None for every record while the law it fed
    reported zero matches. A reader that answers empty is the quietest kind of wrong."""
    out = {}
    for path in sorted(ISA.glob('g17-execution-*-results.json')):
        stem = path.name[:-len('-results.json')]
        try:
            rows = json.loads(path.read_text())
        except Exception:
            continue
        for row in (rows if isinstance(rows, list) else rows.get('results') or []):
            enc = ((row.get('decoded') or {}).get('encoded') or [None])
            if isinstance(row, dict) and row.get('id') and enc and enc[0]:
                out[(stem, row['id'])] = enc[0]
    return out


def e3m4(byte):
    """An 8-bit minifloat: 1 sign, 3 exponent, 4 mantissa, bias 3, subnormals, no reserved exponent.

    Measured by the TensorOps recon lane on op9751's immediate field and confirmed on hardware at
    7 of 7 discriminating bytes including e=111, which is what rules out an inf/nan encoding. It is
    in THIS file because it has since appeared on two more opcodes in two more families - op1048's
    `fadd.imm.f16`, where writing 0x37 returned the f32 encoding of 1.4375, and op1000's
    `fadd.imm`, whose immediate 4 decodes to 0.0625 and whose measured delta is exactly +0.0625.
    Three opcodes in three families makes it look like the ISA's general encoding for a small
    float immediate rather than a property of one instruction.
    """
    byte = int(byte) & 0xFF
    sign, exponent, mantissa = (byte >> 7) & 1, (byte >> 4) & 7, byte & 15
    if exponent == 0:
        value = (mantissa / 16.0) * 2.0 ** (1 - 3)
    else:
        value = (1.0 + mantissa / 16.0) * 2.0 ** (exponent - 3)
    return -value if sign else value


def _addr16_window(row, plan, encoded):
    """(lo, w) from the operands the record's own bytes carry, or None if they cannot be read."""
    enc = encoded.get((row['stem'], row['id']))
    if not enc:
        return None
    try:
        for inst in decode_instruction(bytes.fromhex(enc), 0):
            vals = [v for k, v in inst.values if k != 'reg'] if hasattr(inst, 'values') else []
            if len(vals) >= 2:
                return int(vals[-2]), int(vals[-1])
    except Exception:
        return None
    return None


# THE TRANSCENDENTALS THE EXACT-MATCH CENSUS CAN NEVER PROMOTE, MEASURED IN ULPS INSTEAD.
# `exp2`, `log2`, `rsqrt` and `recip` are approximated in hardware, so bit-exact elimination
# reports "no candidate fits" for every one of them however right the name is. An accuracy
# TOLERANCE is not the answer - a 1e-6 slack once accepted a truncating conversion 1.2e-7 away -
# so nothing here is accepted by a threshold I chose. The distance is measured and published, and
# the reader is told which forms sit at zero or one unit in the last place and which do not.
TRANSCENDENTAL = {
    'exp2': lambda x: 2.0 ** x,
    'log2': lambda x: math.log2(x),
    'sqrt': lambda x: math.sqrt(x),
    'rsqrt': lambda x: 1.0 / math.sqrt(x),
    'recip': lambda x: 1.0 / x,
}


def _ulp_order(bits, width):
    """Float encodings mapped to a monotone integer line, so a distance is meaningful.

    Two float words differ by "one ULP" when their encodings are adjacent - but the sign-magnitude
    layout makes -0 and +0 far apart as integers, so the negative half is reflected first.
    """
    sign = 1 << (width - 1)
    return bits if not (bits & sign) else sign - (bits & (sign - 1)) - 1


def _bf(u):
    """Low sixteen bits read as a bfloat16 - the HIGH half of an f32, zero-extended."""
    return _f((int(u) & 0xFFFF) << 16)


def _ub(x):
    """A Python float back to bfloat16 bits, round-to-nearest-even, specials as VALUES.

    The counterpart of `_uh` for the other 16-bit float format, and it returns the specials for
    the same reason: None means "this candidate cannot answer" and would report a miss for an
    opcode that answered exactly as its name says.
    """
    if x != x:
        return 0x7FC0
    if math.isinf(x):
        return 0xFF80 if x < 0 else 0x7F80
    return _bf16_of_f32_word(_f32_word_of(float(x)))


def _flush_f16_input_denormal(half):
    """The half counterpart of the f32 and bf16 input flushes."""
    half &= 0xFFFF
    if (half & 0x7C00) == 0 and (half & 0x03FF) != 0:
        return half & 0x8000
    return half


# HOW A SOURCE WORD BECOMES A NUMBER, AND EVERY WAY IT PLAUSIBLY COULD. Two assumptions were
# baked into this instrument and each one produced a confident negative result:
#
#   * A `GPR16` operand class says the register is sixteen bits WIDE. It does not say which float
#     lives there. Reading every 16-bit source as a half put six forms between 1,460 and
#     326,708,319 ULP from the function named on their own opcode, published as "a different
#     operation rather than an inaccurate one". They are bfloat16 forms and they reproduce their
#     names exactly.
#   * The hardware FLUSHES an input denormal to a zero of the same sign - measured, on this
#     silicon, by op3338 and op3341's own probes. Three f32 forms were fed half bit patterns in a
#     32-bit slot, which are denormals as f32; scoring them against an un-flushed reference put
#     `2570/log2` 1,014,667,179 ULP from `log2`. Under the flush it is within one ULP.
#
# Both are now READINGS rather than assumptions: each is scored and the one that fits is
# recorded. Choosing among readings is a free parameter, so two readings that predict the same
# bits on every case of a form are ONE hypothesis and are collapsed before the choice is made -
# otherwise a form whose cases contain no denormal would be refused as "ambiguous" for failing to
# separate two readings that are not separable by anything.
SOURCE_READINGS_16 = (
    ('f16', _h),
    ('f16-ftz', lambda u: _h(_flush_f16_input_denormal(int(u) & 0xFFFF))),
    ('bf16', _bf),
    ('bf16-ftz', lambda u: _bf(_flush_bf16_input_denormal(int(u) & 0xFFFF))),
)
SOURCE_READINGS_32 = (
    ('f32', _f),
    ('f32-ftz', lambda u: _f(_flush_f32_input_denormal(int(u) & M32))),
)
DEST_READINGS_16 = (('f16', _uh), ('bf16', _ub))
DEST_READINGS_32 = (('f32', _u),)
SOURCE_READINGS = dict(SOURCE_READINGS_16 + SOURCE_READINGS_32)
DEST_READINGS = dict(DEST_READINGS_16 + DEST_READINGS_32)


def _dest_float(bits, dname, width):
    """The value a destination bit pattern holds, for measuring how close a reference sits to it."""
    import struct
    if width == 32:
        return struct.unpack('<f', struct.pack('<I', bits & M32))[0]
    if dname == 'bf16':
        return struct.unpack('<f', struct.pack('<I', (bits & 0xFFFF) << 16))[0]
    import numpy as np
    return float(np.array([bits & 0xFFFF], dtype=np.uint16).view(np.float16)[0])


# A CASE THAT CANNOT TELL AN ACCURATE IMPLEMENTATION FROM A SLOPPY ONE. When the real result lies
# within 1/64 of a unit in the last place of a representable value - an exact power for exp2 of an
# integer, 1.0 for exp2 of a tiny input, 0.5 for rsqrt(4) - every implementation within half a unit
# returns those exact bits, so an exact answer there says nothing about rounding. exp2 read 39 of 46
# exact; 38 of its 46 interior cases were of this kind, and on the other eight it is 1 of 8.
TRIVIAL_MARGIN_ULP = 1.0 / 64


def _score_transcendental(records, token, unpack, pack, width, plans, values, dname=None):
    """One (source reading, destination reading) scored over a form's records."""
    largest_finite = 0x7BFF if width == 16 else 0x7F7FFFFF
    infinity = 0x7C00 if width == 16 else 0x7F800000
    mask = (1 << width) - 1
    got = dict(interior_cases=0, interior_exact=0, interior_max_ulp=0,
               interior_nontrivial_cases=0, interior_nontrivial_exact=0,
               overflow_cases=0, overflow_returned_the_largest_finite=0,
               overflow_returned_infinity=0, overflow_returned_something_else=0,
               skipped_no_real_reference=0)
    # WHAT THIS READING PREDICTS, case by case, so two readings can be compared as HYPOTHESES
    # rather than as scores. Identical prediction vectors mean the cases cannot tell them apart.
    predictions = []
    for stem, rid in records:
        plan, out = plans.get((stem, rid)), values.get((stem, rid))
        if not (plan and out):
            continue
        for case, value in zip(plan['cases'], out):
            x = unpack(int(case[0]) & M32)
            observed = int(value) & mask
            try:
                y = TRANSCENDENTAL[token](x)
            except Exception:
                got['skipped_no_real_reference'] += 1
                predictions.append(None)
                continue
            if y != y:
                got['skipped_no_real_reference'] += 1
                predictions.append(None)
                continue
            try:
                reference = pack(y)
            except Exception:
                reference = None
            if reference is None:
                got['skipped_no_real_reference'] += 1
                predictions.append(None)
                continue
            reference &= mask
            predictions.append(reference)
            if reference in (infinity, infinity | (1 << (width - 1))):
                got['overflow_cases'] += 1
                if observed in (largest_finite, largest_finite | (1 << (width - 1))):
                    got['overflow_returned_the_largest_finite'] += 1
                elif observed in (infinity, infinity | (1 << (width - 1))):
                    got['overflow_returned_infinity'] += 1
                else:
                    got['overflow_returned_something_else'] += 1
                continue
            distance = abs(_ulp_order(reference, width) - _ulp_order(observed, width))
            got['interior_cases'] += 1
            got['interior_exact'] += 1 if distance == 0 else 0
            try:
                here = _dest_float(reference, dname or ('f32' if width == 32 else 'f16'), width)
                step = abs(_dest_float(reference + 1, dname or ('f32' if width == 32 else 'f16'), width) - here)
                trivial = step > 0 and abs(float(y) - here) < TRIVIAL_MARGIN_ULP * step
            except Exception:
                trivial = False
            if not trivial:
                got['interior_nontrivial_cases'] += 1
                got['interior_nontrivial_exact'] += 1 if distance == 0 else 0
            got['interior_max_ulp'] = max(got['interior_max_ulp'], distance)
    got['predictions'] = tuple(predictions)
    return got


def _choose_transcendental_reading(scored):
    """Which (source, destination) reading an entry reports, given every reading's score.

    A READING WITH NO INTERIOR CASES SCORES ZERO ULP, which is the best possible number produced
    by measuring nothing - so readings that clear the case bar are ranked ahead of ones that do
    not, and only then by distance. Without the first key op3979's empty f16 reading outranks a
    sixteen-case bf16 fit, and the published format would be the one no case tested.

    Split out of `transcendental_accuracy` because no form in today's corpus has one reading
    below the case bar and another above it: the rule is therefore unexercised by the artifact
    and a guard written against the artifact could not fail. It is tested directly instead.
    """
    return min(scored, key=lambda s: (0 if s['interior_cases'] >= MIN_CASES else 1,
                                      s['interior_max_ulp'], -s['interior_cases']))


def _transcendental_verdict(entry):
    """The sentence an entry earns, given its interior score and what its readings identified.

    Split out because the guard on the REFUTING branch could otherwise only be written against
    the artifact, and the artifact stopped containing a refutation the day the readings were
    repaired - so the test went red for the instrument becoming right. The branch is exercised
    here directly instead.
    """
    if entry['interior_cases'] >= MIN_CASES and entry['interior_max_ulp'] <= 1:
        return ('reproduces the correctly-rounded value of its own name on every interior case '
                'to at most one unit in the last place (%d of %d exact; %d of the %d cases that can '
                'tell rounding apart), reading its source as %s and its destination as %s%s'
                % (entry['interior_exact'], entry['interior_cases'],
                   entry.get('interior_nontrivial_exact', 0), entry.get('interior_nontrivial_cases', 0),
                   '/'.join(entry.get('source_format') or ()) or entry.get('source_reading'),
                   '/'.join(entry.get('destination_format') or ())
                   or entry.get('destination_reading'),
                   '' if entry.get('input_denormal_flush_determined') else
                   '; whether it flushes an input denormal is NOT determined by these records'))
    if entry['interior_cases'] >= MIN_CASES:
        return ('the named function does NOT reproduce these outputs under any width reading: '
                '%d of %d interior cases are exact and the worst is %d ULP away, which is a '
                'different operation rather than an inaccurate one'
                % (entry['interior_exact'], entry['interior_cases'], entry['interior_max_ulp']))
    return ('too few interior cases (%d, the bar is %d) to say anything'
            % (entry['interior_cases'], MIN_CASES))


def _peer_reported_conventions():
    """Conventions another lane measured, with what THIS lane's records can say about each.

    KEPT SEPARATE FROM LOCAL EVIDENCE ON PURPOSE. A peer measurement is not this census's
    measurement, and folding one in would make a column mean two instruments with different
    populations and different failure modes - the same reason `d3_peer_reported` is not summed
    into `d3_any`. What belongs here is the claim, its source, and the honest answer to "can my
    own records tell?"
    """
    flushing = [name for table in (FLOAT2, FLOAT3) for name in table if name.endswith('_ftzout')]
    return {
        'subnormal RESULTS are flushed, not only subnormal inputs': dict(
            reported_by='the TensorOps lane',
            source=('docs/g17-tensorops-accelerator-recon.md section 138 part 5, commit 3892f646 '
                    'on linker/g17-tensorops-recon, with results/g17-tensorops-recon-v1/'
                    'alu_flush.log and mxprobe_extreme_*.log'),
            claim=('fmul, fadd, fsub and fma flush subnormal INPUTS and subnormal RESULTS to a '
                   'zero of the same sign; normal results stay IEEE; none of four compile modes '
                   '- denormals enabled or disabled, fast math on or off - changes it, and the '
                   'matrix unit keeps gradual underflow'),
            tested_here=('added as RIVAL candidates (%s) beside the input-only flush, which '
                         'differ from it exactly when a result is subnormal' % ', '.join(
                             sorted(flushing))),
            what_my_records_say=('NOTHING EITHER WAY. No form of this census fits a '
                                 'result-flushing variant, and none is refuted by one: no '
                                 'retained record has a case whose result underflows, so the two '
                                 'conventions predict the same bits everywhere I can look. The '
                                 'candidates stay because a future batch with such a case would '
                                 'separate them, and the claim stays attributed because this '
                                 'lane did not measure it'),
            corroborates=('the input flush, which this lane measured independently on op3290 - '
                          'six misses of sixteen for fmul against its own name, all of them a '
                          'flushed denormal or a canonicalised NaN')),
    }


def transcendental_accuracy(rows, universe):
    """Per form: how far the hardware is from the correctly-rounded value of its OWN name.

    THE OVERFLOW CASES ARE SEPARATED FROM THE INTERIOR ONES, because at the overflow boundary the
    difference is SEMANTIC and not an accuracy question at all - and the two are easy to confuse
    in exactly the flattering direction. The largest finite half is 0x7BFF and half infinity is
    0x7C00, which are ADJACENT encodings, so an instruction that saturates on overflow scores one
    ULP against an IEEE reference and reads as a rounding difference. op1277 `exp2.f16` returns
    65504 where 2**16 overflows, and that is a saturating opcode rather than an inaccurate one.

    So overflow cases are counted apart, with what the hardware returned there, and the accuracy
    claim covers only the interior.

    THE SIXTEEN-BIT READING IS MEASURED, NOT ASSUMED. `GPR16` fixes the register's WIDTH and says
    nothing about the format, so each 16-bit side is scored as a half AND as a bfloat and the
    reading that fits is recorded on the entry. Choosing among readings is a free parameter, so
    it is fenced: a form is promotable only when EXACTLY ONE reading meets the accuracy bar. If
    both readings fit, the records do not identify the format and the form says so instead of
    taking the flattering one.

    THIS FEEDS NO DENOMINATOR. D3 is produced by exact-match elimination over a published
    candidate library, and a form agreeing to one ULP has not satisfied that instrument. Mixing an
    approximate criterion into the same column would make the denominator mean two different
    things, which is the reason the five D3 columns are exclusive in the first place. The evidence
    is real and it is of a different kind, so it is reported in its own field.
    """
    out = {}
    values, plans = _values_by_id(), _plans_by_stem_and_id()
    for row in rows:
        if row['arity'] != 1:
            continue
        name = (universe.get('op%d' % row['op']) or {}).get('name') or ''
        token = name.split('.')[0]
        if token not in TRANSCENDENTAL:
            continue
        key = (row['stem'], row['id'])
        got, plan = values.get(key), plans.get(key)
        if not got or not plan or len(got) != len(plan['cases']):
            continue
        try:
            classes = g17auth.operand_classes(row['op'])
            dsts, srcs = g17auth.register_operands(row['op'])
        except Exception:
            continue
        src_16 = 'GPR16' in (classes[srcs[0]] or '')
        dst_16 = 'GPR16' in (classes[dsts[0]] or '')
        entry = out.setdefault('%d/%s' % (row['op'], name), dict(
            name=name, token=token,
            destination_width=16 if dst_16 else 32,
            source_width=16 if src_16 else 32,
            source_reading=None, destination_reading=None,
            interior_cases=0, interior_exact=0, interior_max_ulp=0,
            interior_nontrivial_cases=0, interior_nontrivial_exact=0,
            overflow_cases=0, overflow_returned_the_largest_finite=0,
            overflow_returned_infinity=0, overflow_returned_something_else=0,
            skipped_no_real_reference=0, readings=[], records=[]))
        if list(key) not in entry['records']:
            entry['records'].append(list(key))
    for entry in out.values():
        token = entry['token']
        width = entry['destination_width']
        sources = SOURCE_READINGS_16 if entry['source_width'] == 16 else SOURCE_READINGS_32
        dests = DEST_READINGS_16 if width == 16 else DEST_READINGS_32
        records = [tuple(r) for r in entry['records']]
        scored = []
        for sname, unpack in sources:
            for dname, pack in dests:
                got = _score_transcendental(records, token, unpack, pack, width, plans, values, dname)
                got.update(source_reading=sname, destination_reading=dname)
                scored.append(got)
        # TWO READINGS THAT PREDICT THE SAME BITS EVERYWHERE ARE ONE HYPOTHESIS. Collapsing them
        # before the ambiguity gate is what keeps the gate about the DATA: a form whose cases hold
        # no denormal cannot separate a flushing reading from a non-flushing one, and refusing it
        # for that would be counting the instrument's own options as rival explanations.
        classes = {}
        for got in scored:
            classes.setdefault(got['predictions'], []).append(got)
        entry['readings'] = [dict(
            reading='|'.join('%s->%s' % (g['source_reading'], g['destination_reading'])
                             for g in group),
            source_reading=group[0]['source_reading'],
            destination_reading=group[0]['destination_reading'],
            interior_cases=group[0]['interior_cases'],
            interior_exact=group[0]['interior_exact'],
            interior_nontrivial_cases=group[0]['interior_nontrivial_cases'],
            interior_nontrivial_exact=group[0]['interior_nontrivial_exact'],
            interior_max_ulp=group[0]['interior_max_ulp'],
            indistinguishable_readings=len(group)) for group in classes.values()]
        entry['readings'].sort(key=lambda r: r['reading'])
        scored = [group[0] for group in classes.values()]
        labels = {id(group[0]): '|'.join('%s->%s' % (g['source_reading'], g['destination_reading'])
                                         for g in group) for group in classes.values()}
        fitting = [s for s in scored
                   if s['interior_cases'] >= MIN_CASES and s['interior_max_ulp'] <= 1]
        # The reading REPORTED is the best-scoring one; the reading ACCEPTED needs `fitting` to
        # hold exactly one member, which is checked at the promotion gate below. Reporting the
        # best and promoting only the unique one keeps the ambiguous case visible rather than
        # silently discarding it.
        chosen = _choose_transcendental_reading(scored)
        entry.update({k: v for k, v in chosen.items() if k != 'predictions'})
        entry['reading'] = labels[id(chosen)]
        entry['readings_that_fit'] = sorted(labels[id(s)] for s in fitting)
        # An ambiguous form can still have determined ONE of its two sides: op1287's four cases
        # fit under either source reading but only ever with a bfloat destination, and that is a
        # fact the all-or-nothing gate below would otherwise throw away.
        # The FORMAT, with the flushing variant folded in: `bf16` and `bf16-ftz` are the same
        # format read two ways, so a form that cannot separate them has still named its format.
        fmt = lambda label, side: {part.split('->')[side].split('-ftz')[0]
                                   for part in label.split('|')}
        src_formats = set().union(*[fmt(labels[id(s)], 0) for s in fitting]) if fitting else set()
        dst_formats = set().union(*[fmt(labels[id(s)], 1) for s in fitting]) if fitting else set()
        entry['source_reading_determined'] = len(src_formats) == 1
        entry['destination_reading_determined'] = len(dst_formats) == 1
        entry['source_format'] = sorted(src_formats)
        entry['destination_format'] = sorted(dst_formats)
        # WHETHER THE OPERATION FLUSHES ITS INPUT DENORMAL IS A SEPARATE, FINER FACT than which
        # float its source is, and it is separated because the two gates need different answers.
        # `3662/recip` fits under a flushing and a non-flushing f32 reading that differ ONLY at a
        # denormal input, where both predictions land in the overflow bucket the accuracy claim
        # already excludes - so the records identify the FORMAT and do not identify the flush.
        # Folding the flush into the format gate would refuse a form for failing to answer a
        # question the published claim never asks.
        flushes = {('-ftz' in part.split('->')[0])
                   for s in fitting for part in labels[id(s)].split('|')}
        entry['input_denormal_flush_determined'] = len(flushes) == 1
        entry['input_denormal_flush'] = (sorted(flushes)[0] if len(flushes) == 1 else None)
        entry['verdict'] = _transcendental_verdict(entry)
    # ACCURACY IS NOT ELIMINATION, and this is the difference between a measurement and a claim.
    # A distance to ONE reference says the hardware is close to that function; it does not say no
    # OTHER named operation is equally close. So every rival in TRANSCENDENTAL is measured against
    # the same records, and a form is promotable only when the named operation is the sole
    # reference within a ULP everywhere. Without this the column would assert elimination it had
    # never performed - the same gap the ULP instrument's own note warns about when it says it
    # feeds no denominator.
    # TWO OPCODES WITH ONE NAME ARE NOT ONE INSTRUCTION UNTIL SOMETHING SAYS SO. Apple names
    # both op3850 and op3978 `rsqrt`; both reproduce 1/sqrt(x) to one ULP on the same twelve
    # interior cases, so the ACCURACY question cannot tell them apart and this table reported
    # them identically. They are not identical: on a flushed-to-zero input op3850 returns +inf,
    # which is IEEE rsqrt(0), and op3978 returns 1.0. The interior agreed and the boundary did
    # not, which is where a same-named pair will always separate if it separates at all - so the
    # comparison is made here, on the inputs the two actually share, and recorded on both.
    same_name = collections.defaultdict(list)
    for key, entry in out.items():
        same_name[entry['name']].append(key)

    def _outputs(entry):
        seen = {}
        for record in (tuple(r) for r in entry['records']):
            plan, got = plans.get(record), values.get(record)
            if not plan or not got or len(got) != len(plan['cases']):
                continue
            for case, value in zip(plan['cases'], got):
                source = case[0] if isinstance(case, list) else case
                seen.setdefault(source, value)
        return seen

    for name, keys in sorted(same_name.items()):
        if len(keys) < 2:
            continue
        outputs = {key: _outputs(out[key]) for key in keys}
        for a, b in itertools.combinations(sorted(keys), 2):
            # ONLY LIKE FOR LIKE. The first version of this pass compared raw output WORDS
            # between forms of different widths and reported all 32 pairs "separated" - but
            # 1272/exp2 returning 0x3F800000 and 1276/exp2 returning 0x00003C00 for the same
            # input word are both exp2(0) = 1, one as an f32 and one as a half. That comparison
            # measures the width convention, not the function, which is the same defect as
            # reading correct bytes through the wrong dtype.
            if (out[a]['source_width'], out[a]['destination_width']) != \
                    (out[b]['source_width'], out[b]['destination_width']):
                continue
            shared = sorted(set(outputs[a]) & set(outputs[b]))
            differ = [x for x in shared if outputs[a][x] != outputs[b][x]]
            for near, far in ((a, b), (b, a)):
                out[near].setdefault('same_named_siblings', {})[far] = dict(
                    inputs_shared=len(shared), inputs_differing=len(differ),
                    examples=[dict(input='0x%08X' % x, mine='0x%08X' % outputs[near][x],
                                   theirs='0x%08X' % outputs[far][x]) for x in differ[:3]],
                    separated=bool(differ),
                    both_widths='%d->%d' % (out[near]['source_width'],
                                            out[near]['destination_width']),
                    means=('two forms Apple gives the same name AND the same source and '
                           'destination widths, compared on the inputs they share. SEPARATED '
                           'means they are different instructions whatever the name says; not '
                           'separated means these records cannot tell, which is not the same as '
                           'their being one instruction. Pairs of different widths are not '
                           'compared at all: the same output word means different values to a '
                           '16-bit reader and a 32-bit one, so a difference there would be the '
                           'width convention rather than the function'))
    for key, entry in out.items():
        entry['rivals_within_one_ulp'] = []
        if entry['interior_cases'] < MIN_CASES or entry['interior_max_ulp'] > 1:
            entry['promotable'] = False
            entry['why_not_promotable'] = 'it does not meet the accuracy bar'
            continue
        if not (entry['source_reading_determined'] and entry['destination_reading_determined']):
            entry['promotable'] = False
            entry['why_not_promotable'] = (
                'the name fits under more than one float FORMAT (source %s, destination %s), so '
                'these records do not say what the operands denote and the accuracy is not '
                'evidence about either reading'
                % ('/'.join(entry['source_format']), '/'.join(entry['destination_format'])))
            continue
        own = (entry.get('name') or '').split('.')[0]
        width = entry.get('destination_width') or 32
        mask = (1 << width) - 1
        unpack = SOURCE_READINGS[entry['source_reading']]
        pack = DEST_READINGS[entry['destination_reading']]
        for rival, fn in sorted(TRANSCENDENTAL.items()):
            if rival == own:
                continue
            agrees, compared = True, 0
            for stem, rid in (entry.get('records') or []):
                plan = _plans_by_stem_and_id().get((stem, rid))
                got = _values_by_id().get((stem, rid))
                if not (plan and got):
                    continue
                for case, value in zip(plan['cases'], got):
                    source = unpack(int(case[0]) & M32)
                    if source != source or math.isinf(source):
                        continue
                    try:
                        want = fn(source)
                    except (ValueError, ZeroDivisionError, OverflowError):
                        continue
                    try:
                        bits = pack(want)
                    except (OverflowError, ValueError):
                        continue
                    # `_u` and `_uh` RETURN None rather than raising when the value will not
                    # convert, so an except clause alone lets a None through to the arithmetic.
                    if bits is None:
                        continue
                    compared += 1
                    if abs(_ulp_order(int(value) & mask, width)
                           - _ulp_order(bits & mask, width)) > 1:
                        agrees = False
                        break
                if not agrees:
                    break
            if agrees and compared >= MIN_CASES:
                entry['rivals_within_one_ulp'].append(rival)
        entry['promotable'] = not entry['rivals_within_one_ulp']
        entry['why_not_promotable'] = ('' if entry['promotable'] else
                                       'another named operation is also within one ULP on these '
                                       'records, so the accuracy does not identify which')
    return dict(sorted(out.items()))


# THE OPERATIONS WHOSE SECOND OPERAND IS AN IMMEDIATE, AND WHAT THE NAME TOKEN MEANS. Every
# entry is a function of (value, k) where k comes out of the record's OWN EMITTED BYTES - it is
# never solved from the output, so none of these has a free parameter to absorb an observation
# with.
IMMEDIATE_OPS = {
    'mul': lambda v, k: (v * k) & M32,
    'madd': lambda v, k: (v * k) & M32,
    'add': lambda v, k: (v + k) & M32,
    'sub': lambda v, k: (v - k) & M32,
    'and': lambda v, k: v & k,
    'or': lambda v, k: v | k,
    'xor': lambda v, k: v ^ k,
    'shl': lambda v, k: (v << (k & 0x7F)) & M32 if (k & 0x7F) < 32 else 0,
    'shr': lambda v, k: (v >> (k & 0x7F)) if (k & 0x7F) < 32 else 0,
}


def _shift_half(token, suffix, value, k, signed_bits):
    """`.hi` and `.lo` name the half of a WIDENED shift, which is a measurement and not a guess.

    op14310 `shl.hi` with k=1 returns 1 for exactly the inputs whose bit 31 is set (6 of 31 in one
    record, 2 of 4 in another) and op16778 `shr.lo` with k=1 returns 0x80000000 for exactly the
    ODD inputs (3 of 31, and the predictor is right on all 31). So the two suffixes are the bits
    that leave the register at the top and at the bottom respectively, the bottom half
    left-justified - a 64-bit shift whose other half this destination holds.
    """
    k &= 0x7F
    if token == 'shl' and suffix == 'hi':
        return (value << k) >> 32
    if token == 'shr' and suffix == 'lo':
        return (value << (32 - k)) & M32 if 0 < k < 32 else 0
    if token in ('asr', 'sar'):
        wide = value - (1 << signed_bits) if value >> (signed_bits - 1) else value
        return (wide >> k) & M32 if k < 32 else (M32 if wide < 0 else 0)
    return None


# THE PREDICATES A CONDITIONAL SELECT MIGHT BE TESTING, eliminated exactly as the candidate
# library is eliminated against values - because for this family the comparison is enumerated into
# the OPCODE, so identifying an opcode means identifying a predicate rather than a function.
def _predicates():
    S = lambda v: v - (1 << 32) if v >> 31 else v
    return {
        'a == 0': lambda a, b: a == 0,
        'a != 0': lambda a, b: a != 0,
        'b == 0': lambda a, b: b == 0,
        'b != 0': lambda a, b: b != 0,
        'a == b': lambda a, b: a == b,
        'a != b': lambda a, b: a != b,
        'a <u b': lambda a, b: a < b,
        'a <=u b': lambda a, b: a <= b,
        'a >u b': lambda a, b: a > b,
        'a >=u b': lambda a, b: a >= b,
        'a <s b': lambda a, b: S(a) < S(b),
        'a <=s b': lambda a, b: S(a) <= S(b),
        'a >s b': lambda a, b: S(a) > S(b),
        'a >=s b': lambda a, b: S(a) >= S(b),
        'a <s 0': lambda a, b: S(a) < 0,
        'a >=s 0': lambda a, b: S(a) >= 0,
        'b <s 0': lambda a, b: S(b) < 0,
        'b >=s 0': lambda a, b: S(b) >= 0,
        'a is odd': lambda a, b: bool(a & 1),
        'a all ones': lambda a, b: a == M32,
        'f32(a) < f32(b)': lambda a, b: _f(a) < _f(b),
        'f32(a) == f32(b)': lambda a, b: _f(a) == _f(b),
        'f32(a) > f32(b)': lambda a, b: _f(a) > _f(b),
        'f32(a) < 0': lambda a, b: _f(a) < 0,
        'f16(a) < f16(b)': lambda a, b: _h(a) < _h(b),
        'f16(a) > f16(b)': lambda a, b: _h(a) > _h(b),
        # THE FAMILY IS COMPLETED SYMMETRICALLY, NOT EXTENDED BY THE ONE THAT FITS. f32 equality
        # was here and f16 equality was not, which is an asymmetry in this library rather than a
        # fact about the ISA - and op11421 needs exactly the missing one: its output is nonzero
        # whenever the halves differ EXCEPT at a = b = 0xFFFF, which as a HALF is a NaN, and a NaN
        # compares unequal to itself. Adding only `f16 ==` would have been shaping the library to
        # that record, so every comparison now exists at both precisions in both directions.
        'f16(a) == f16(b)': lambda a, b: _h(a) == _h(b),
        'f16(a) != f16(b)': lambda a, b: _h(a) != _h(b),
        'f16(a) <= f16(b)': lambda a, b: _h(a) <= _h(b),
        'f16(a) >= f16(b)': lambda a, b: _h(a) >= _h(b),
        'f32(a) != f32(b)': lambda a, b: _f(a) != _f(b),
        'f32(a) <= f32(b)': lambda a, b: _f(a) <= _f(b),
        'f32(a) >= f32(b)': lambda a, b: _f(a) >= _f(b),
    }


def _has_register_destination(opcode):
    """Whether the form writes a register this harness can read back."""
    try:
        dsts, _srcs = g17auth.register_operands(opcode)
        return bool(dsts)
    except Exception:
        return False


def _declared_source_widths(opcode):
    """[16|32, ...] for an opcode's register sources, or None if the table cannot say."""
    try:
        classes = g17auth.operand_classes(opcode)
        _dsts, srcs = g17auth.register_operands(opcode)
        return [16 if 'GPR16' in (classes[i] or '') else 32 for i in srcs]
    except Exception:
        return None


def _note_width(entry, row):
    """Record the emitted instruction width of a row on the form entry it contributes to.

    THE WIDTH WAS ALWAYS ONE LEVEL DOWN. The candidate sixth D3 column is held partly because
    "the predicate field aggregates per opcode-and-name; it does not carry the instruction
    LENGTH" - which is true of the AGGREGATE and not of the rows it is built from, every one of
    which carries `widths`. Four of the fourteen candidate opcodes have two lengths in the map
    (op11372, op11412, op11462, op11492) and each was dispatched at exactly one of them,
    unanimously across dozens of records and several independent batches. Measured twice and in
    agreement: this field, and decoding `decoded.encoded` out of the retained results - 10, 10,
    14, 14.

    Kept as a LIST and not a scalar. A form entry that ever sees two widths is an opcode-level
    fact wearing a form's name, which is the error this map has caught three times, and it has to
    be visible rather than collapsed by taking the first element.
    """
    seen = set(entry.setdefault('emitted_widths', []))
    seen.update(row.get('widths') or [])
    entry['emitted_widths'] = sorted(seen)


def csel_boolean_comparisons(rows, universe):
    """Conditional-select opcodes that return a BOOLEAN, with the predicate eliminated against it.

    THE BRANCH-PATTERN MODEL WAS THE WRONG MODEL and the else-values said so: they came back as 0
    and 1, not as a large inherited constant. op11372's eighteen outputs are 1,0,0,1,1,0,... - all
    zeros and ones. These are COMPARISON instructions producing a flag, and the earlier a/b/K
    classification was an artifact of an input set that happens to contain 0 and 1 as operand
    values, so a boolean output coincidentally matched an operand.

    Applied to the boolean directly the predicate library works as intended:

        op11372 csel.reg   (a == b) ? 1 : 0              unique of 26, true at 4 of 18
        op11381 csel.reg   (a == b) ? 1.0f : 0.0f        the same predicate, float result
        op11492 csel.reg   (a & 0xFFFF) <u b ? 1 : 0     unique once the widths are applied

    EACH SOURCE IS READ AT ITS DECLARED WIDTH, which is what op11492 needed. Its sources are
    declared (GPR16, GPR32) and the raw-32-bit library explained nothing; masking the first to
    sixteen bits makes an unsigned less-than fit all eighteen cases. Third time today that the
    declared operand classes turned a no-fit into a determination.

    The true-value is recognised rather than assumed - integer 1, f32 1.0, f16 1.0, or all-ones at
    either width - because two opcodes here compute the same predicate and encode the answer
    differently, and calling one of them wrong would lose that pairing.
    """
    values, plans = _values_by_id(), _plans_by_stem_and_id()
    library = _predicates()
    true_values = {1: 'integer 1', 0x3F800000: 'f32 1.0', 0x3C00: 'f16 1.0',
                   0xFFFF: 'all-ones at 16 bits', M32: 'all-ones at 32 bits'}
    out, summary = {}, collections.Counter()
    for row in rows:
        name = (universe.get('op%d' % row['op']) or {}).get('name') or ''
        # `cmp` AND `fcmp` WERE EXCLUDED BY A NAME FILTER THAT LISTED EVERY FAMILY BUT THEIRS.
        # This field eliminates predicates against a two-valued output and the filter read
        # csel/clamp/fselect - leaving out the instructions Apple literally names COMPARE, which
        # are the most likely flag producers in the table. Four of them have two-valued records
        # with trues this field already recognises: 0x3F800000, 0x3C00 and integer 1.
        if not name.startswith(('csel', 'clamp', 'fselect', 'cmp', 'fcmp')) \
                or row['arity'] != 2:
            continue
        key = (row['stem'], row['id'])
        got, plan = values.get(key), plans.get(key)
        if (not got or not plan or len(got) != len(plan['cases'])
                or row.get('distinct_cases', row['cases']) < MIN_CASES):
            continue
        try:
            classes = g17auth.operand_classes(row['op'])
            dsts, srcs = g17auth.register_operands(row['op'])
            dmask = 0xFFFF if 'GPR16' in (classes[dsts[0]] or '') else M32
            smasks = [0xFFFF if 'GPR16' in (classes[i] or '') else M32 for i in srcs]
        except Exception:
            continue
        outputs = [int(v) & dmask for v in got]
        distinct = set(outputs)
        # A RECORD THAT IS NOT TWO-VALUED REFUTES THE BOOLEAN MODEL FOR THIS FORM, and merely
        # counting it in the summary let a determination stand that a later record contradicts.
        # op11456 was "determined" as `a == 0` on two ten-case records whose outputs happened to
        # be two-valued, while its eighteen-case record returns 5, 7 and 65535 - so this opcode
        # does not produce a flag at all, and the model does not apply to it.
        #
        # THIRD INSTANCE OF THIS EXACT DEFECT TODAY. The function census needed the same veto,
        # the branch-pattern field needed it, and now the boolean field. A record that contradicts
        # the model has to outrank one that fits it, and writing the fitting path first makes the
        # contradicting path an afterthought every time.
        # A TRUE-VALUE I DO NOT RECOGNISE IS NOT A REASON TO DISCARD THE RECORD, and adding the
        # word to the list would be shaping the recogniser to the observation - the thing this
        # file refuses everywhere else. op9787 `fcmp.cc` and op11472 `cmp` return 0x30 as their
        # true, which is none of integer 1, f32 1.0, f16 1.0 or all-ones.
        #
        # No list is actually needed: a two-valued output has only two possible polarities, so
        # BOTH are tried and whichever yields predicates is reported. `a == b` returning 1/0 and
        # `a != b` returning 0/1 are the same instruction under complementary naming, which is why
        # a polarity-free search is the honest form of this question rather than a looser one.
        if len(distinct) == 2 and (distinct - {0} if 0 in distinct else distinct):
            pair = sorted(distinct)
            if 0 not in distinct or pair[1] not in true_values:
                readings = []
                for true_word in pair:
                    want = [o == true_word for o in outputs]
                    inputs = [(int(c[0]) & smasks[0], int(c[1]) & smasks[1])
                              for c in plan['cases']]
                    survivors = sorted(n for n, fn in library.items()
                                       if all(bool(fn(a, b)) == w
                                              for (a, b), w in zip(inputs, want)))
                    if survivors:
                        readings.append(dict(true_word='0x%X' % true_word,
                                             predicates=survivors, true_at=sum(want)))
                entry = out.setdefault('%d/%s' % (row['op'], name), dict(
                    records=[], readings=[], refuted_by=[], unrecognised_true=[],
                    competitors=len(library)))
                _note_width(entry, row)
                if list(key) not in entry['records']:
                    entry['records'].append(list(key))
                # setdefault ON THE LIST, not on the entry: an entry created earlier by the
                # refutation path has no such key, and `out.setdefault` hands back that older
                # dict unchanged - a default that only applies when the container is new.
                entry.setdefault('unrecognised_true', []).append(dict(
                    stem=row['stem'], id=row['id'],
                    values=['0x%X' % v for v in pair], readings=readings,
                    means=('two-valued with a true-value this field does not recognise, so BOTH '
                           'polarities were eliminated against rather than the record dropped. '
                           'A predicate list under one polarity and its complement under the '
                           'other describe the same instruction')))
                summary['two-valued with an unrecognised true - both polarities tried'] += 1
                continue
        if len(distinct) != 2 or 0 not in distinct or (distinct - {0}).pop() not in true_values:
            # TWO FILTERS ON THE VETO, BOTH OF WHICH THIS FILE ALREADY LEARNED FOR FUNCTIONS, and
            # without them the first version of this vetoed every form including the four it had
            # just determined.
            #
            # A record with FEWER THAN TWO distinct outputs measured nothing and cannot refute a
            # model - that is the degenerate case, not a contradiction. And only the SAME OPERAND
            # CONFIGURATION may refute: a record that states a different immediate is a different
            # configuration measuring a different instruction, which is the exact rule the
            # function census needed after c1004.imm16 vetoed op1004's base form.
            if len(distinct) < 2:
                summary['constant output - measured nothing, cannot refute'] += 1
                continue
            summary['more than two distinct outputs - REFUTES the boolean model'] += 1
            _refuted = out.setdefault('%d/%s' % (row['op'], name), dict(
                records=[], readings=[], refuted_by=[], competitors=len(library)))
            _note_width(_refuted, row)
            _refuted['refuted_by'].append(dict(
                stem=row['stem'], id=row['id'], distinct_outputs=len(distinct),
                stated_immediates=row.get('stated_immediates'),
                sample=['0x%X' % v for v in sorted(distinct)[:6]],
                means=('this record has %d distinct outputs, so the form does not produce a '
                       'two-valued flag and no predicate over a boolean can describe it'
                       % len(distinct))))
            continue
        true_word = (distinct - {0}).pop()
        want = [o == true_word for o in outputs]
        inputs = [(int(c[0]) & smasks[0], int(c[1]) & smasks[1]) for c in plan['cases']]
        survivors = sorted(n for n, fn in library.items()
                           if all(bool(fn(a, b)) == w for (a, b), w in zip(inputs, want)))
        entry = out.setdefault('%d/%s' % (row['op'], name), dict(
            records=[], readings=[], refuted_by=[], competitors=len(library)))
        _note_width(entry, row)
        if list(key) not in entry['records']:
            entry['records'].append(list(key))
        reading = dict(true_value=true_values[true_word], predicates=survivors,
                       cases=len(want), true_at=sum(want),
                       stated_immediates=row.get('stated_immediates'),
                       source_widths=[32 if m == M32 else 16 for m in smasks],
                       determined=len(survivors) == 1)
        if reading not in entry['readings']:
            entry['readings'].append(reading)
        summary['boolean output, predicate determined' if len(survivors) == 1
                else 'boolean output, NO predicate in the library explains it' if not survivors
                else 'boolean output, these inputs separate no single predicate'] += 1
    for key, entry in out.items():
        # only a refutation at the SAME operand configuration as a supporting record counts
        configs = {json.dumps(r.get('stated_immediates') or []) for r in entry['readings']}
        relevant = [r for r in entry['refuted_by']
                    if not configs or json.dumps(r.get('stated_immediates') or []) in configs]
        entry['refuted_by_at_the_same_configuration'] = relevant
        if relevant:
            entry['per_form'] = dict(
                predicates=[], records=len(entry['records']),
                true_at_fewest_cases=0, true_at_most_cases=0,
                state='a record of this form is not two-valued, which refutes the boolean model',
                means=('%d record(s) of this form return more than two distinct values, so it '
                       'does not produce a flag and nothing here is determined however well a '
                       'two-valued record fitted' % len(relevant)))
            summary['form: boolean model refuted by its own record'] += 1
            continue
        sets = [set(r['predicates']) for r in entry['readings']]
        common = set.intersection(*sets) if sets else set()
        # BOTH MARGINS, because the form's evidence is the INTERSECTION across its records and
        # the weakest one alone understates it. A form with four records, one pinning the
        # predicate at a single case and another at eleven, is not a one-case determination - and
        # reporting only the minimum is the same defect the fit margins had this morning, where
        # keeping the minimum made an improvement unobservable by construction.
        # AN AMBIGUITY MAY BE AN IN-PRINCIPLE ONE, and that is a different report from "probe
        # harder". op11502's two survivors are `a == b` and `f32(a) == f32(b)`, and its sources are
        # declared 16 bits: over [0, 0xFFFF] no two distinct patterns map to the same f32, so
        # those predicates agree on EVERY input the declared widths admit. No batch can separate
        # them there, and telling a reader to find better inputs would be telling them to do
        # something impossible.
        #
        # Checked by SAMPLING the declared domain, not by proof, and labelled as sampled - a
        # sampled equivalence is evidence that no separating input exists, never a demonstration.
        # THE UNRECOGNISED-TRUE READINGS MUST REACH per_form OR THEY ARE BURIED. op11472 `cmp` is
        # determined - `a <u b` when its output is 0x0 and `a >=u b` when it is 0x30, pinned at 3
        # and 15 of 18 cases - and those are exact complements, which is one instruction read two
        # ways rather than two findings. A determination sitting only in a side list is a
        # determination nobody reads.
        if not entry['readings'] and entry.get('unrecognised_true'):
            per_polarity = collections.defaultdict(list)
            for record in entry['unrecognised_true']:
                for reading in record['readings']:
                    per_polarity[reading['true_word']].append(
                        (set(reading['predicates']), reading['true_at']))
            verdicts = {}
            for word, seen in sorted(per_polarity.items()):
                common = set.intersection(*[p for p, _w in seen])
                verdicts[word] = dict(
                    predicates=sorted(common),
                    true_at_most_cases=max(w for _p, w in seen),
                    true_at_fewest_cases=min(w for _p, w in seen),
                    state='determined' if len(common) == 1 else
                          'no predicate survives every record' if not common else
                          'these inputs cannot separate %d predicates' % len(common))
            # NO POLARITY YIELDING ANYTHING IS ITS OWN OUTCOME, and taking min() over an empty
            # set of verdicts crashed on it. op11421, op9718, op9787 and op9813 are two-valued
            # with an unrecognised true AND no predicate of the 26 explains either polarity -
            # which says the library is missing the operation, not that the record is bad.
            if not verdicts:
                entry['per_form'] = dict(
                    predicates=[], records=len(entry['records']),
                    true_at_fewest_cases=0, true_at_most_cases=0, per_true_value={},
                    state='two-valued, unrecognised true, and NO predicate explains either '
                          'polarity',
                    means=('both polarities were eliminated against all %d predicates and none '
                           'survived either, so the library is missing this operation rather '
                           'than the inputs being weak' % len(library)))
                summary['form: unrecognised true, no predicate explains either polarity'] += 1
                continue
            determined = [w for w, v in verdicts.items() if v['state'] == 'determined']
            entry['per_form'] = dict(
                predicates=sorted({v['predicates'][0] for w, v in verdicts.items()
                                   if v['state'] == 'determined'}),
                records=len(entry['records']),
                true_at_fewest_cases=min(v['true_at_fewest_cases'] for v in verdicts.values()),
                true_at_most_cases=max(v['true_at_most_cases'] for v in verdicts.values()),
                per_true_value=verdicts,
                state=('determined at an unrecognised true-value' if determined else
                       'two-valued with an unrecognised true, no predicate determined'),
                means=('the true-value is not one this field recognises, so BOTH polarities were '
                       'eliminated against. %s'
                       % ('; '.join('%s -> %s' % (w, verdicts[w]['predicates'][0])
                                    for w in determined)
                          if determined else 'neither polarity yields a single predicate')))
            summary['form: determined at an unrecognised true-value' if determined
                    else 'form: unrecognised true, not determined'] += 1
            continue
        weights = [r['true_at'] for r in entry['readings']]
        equivalent = None
        if len(common) > 1:
            widths = entry['readings'][0].get('source_widths') or [32, 32]
            hi = [(1 << w) - 1 for w in widths]
            rng = random.Random(DEDUP_SEED)
            probes = [(rng.randint(0, hi[0]), rng.randint(0, hi[1])) for _ in range(20000)]
            probes += [(0, 0), (0, hi[1]), (hi[0], 0), (hi[0], hi[1]), (0x8000, 0),
                       (0, 0x8000), (0x80000000 & hi[0], 0), (0, 0x80000000 & hi[1])]
            fns = [library[n] for n in sorted(common)]
            equivalent = all(len({bool(f(a, b)) for f in fns}) == 1 for a, b in probes)
        entry['per_form'] = dict(
            predicates=sorted(common), records=len(entry['records']),
            true_at_fewest_cases=min(weights, default=0),
            true_at_most_cases=max(weights, default=0),
            predicates_equivalent_over_the_declared_domain=equivalent,
            state=('determined' if len(common) == 1 else
                   'records refute every predicate offered' if not common else
                   'INDISTINGUISHABLE at these declared widths - %d predicates agree on every '
                   'sampled input the widths admit, so no batch can separate them' % len(common)
                   if equivalent else
                   'these inputs cannot separate %d predicates' % len(common)),
            means=('%s, eliminated from %d predicates over %d record(s); the strongest record '
                   'pins it at %d of its cases and the weakest at %d'
                   % (sorted(common)[0] if len(common) == 1 else 'not determined',
                      entry['competitors'], len(entry['records']),
                      max(weights, default=0), min(weights, default=0))))
    # THE DETERMINED FORMS MAKE A GRID, NOT A LIST, and the grid is the finding. Ten forms
    # resolve to three predicates across three source-width combinations and three result
    # encodings, which is the family enumerating (predicate x widths x result type) into opcode
    # numbers - the same structure the earlier grouping inferred from opcodes disagreeing, now
    # with the axes named.
    #
    # The integer/float PAIRS sit nine apart: 11372/11381, 11382/11391, 11402/11411, all `a == b`.
    # That offset is reported as an OBSERVATION over three pairs and not as a rule, because
    # op11492 `a <u b` at integer plus nine is op11501, which measures `a >=u b` at f16 - so
    # whatever the offset means it is not "add nine to change the result type" in general. Three
    # agreeing pairs and one counterexample is exactly the population that produces a false rule.
    grid = collections.defaultdict(dict)
    for key, entry in out.items():
        per = entry.get('per_form') or {}
        if per.get('state') != 'determined':
            continue
        reading = entry['readings'][0]
        grid['%s -> %s' % (per['predicates'][0], reading['true_value'])][
            'sources %s' % (reading['source_widths'],)] = 'op%s' % key.split('/')[0]
    # SPLIT BY WHETHER THE PREDICATE IS THE SAME, because my first version of this matched cells
    # without checking and then described all four pairs as "differing only in result encoding" -
    # which is false of the fourth, where `a <u b` pairs with `a >=u b`. A list that mixes the
    # agreements with the counterexample under the agreements' label is how a false rule gets
    # published with its own refutation sitting inside it.
    same_predicate, different_predicate = [], []
    for label, row in grid.items():
        predicate = label.split(' -> ')[0]
        for cell, name in row.items():
            for other_label, other_row in grid.items():
                if other_label == label:
                    continue
                twin = other_row.get(cell)
                if not twin or int(twin[2:]) - int(name[2:]) != 9:
                    continue
                text = ('%s -> %s at %s becomes %s -> %s'
                        % (name, label.split(' -> ')[1], cell, twin,
                           other_label.split(' -> ')[1]))
                (same_predicate if other_label.split(' -> ')[0] == predicate
                 else different_predicate).append(text)
    pairs = sorted(set(same_predicate))
    return dict(forms=dict(sorted(out.items())), summary=dict(sorted(summary.items())),
                predicates_offered=len(library),
                grid={k: dict(sorted(v.items())) for k, v in sorted(grid.items())},
                grid_means=('the determined forms arranged by (predicate -> result encoding) '
                            'against declared source widths. The family enumerates these into '
                            'opcode numbers, which is why 232 opcodes share four names and why '
                            'operand 2 holds at most one bit'),
                grid_has_no_16_16_column=dict(
                    # DERIVED FROM THE OPCODE, NOT FROM THE READINGS, because a refuted entry has
                    # no readings to carry widths - it was created by the refutation path - and
                    # reading them there returned None for every one of them, so the count came
                    # out zero while twelve forms sat in the report. A field asking the wrong
                    # object answers emptily rather than loudly.
                    refuted_as_not_two_valued=len([
                        1 for key, entry in out.items()
                        if entry.get('refuted_by_at_the_same_configuration')
                        and _declared_source_widths(int(key.split('/')[0])) == [16, 16]]),
                    # COMPUTED, AND SCOPED TO THE SAME POPULATION AS THE COUNT BESIDE IT. I first
                    # wrote 22 here as a literal from a hand count, and computing it returned 37 -
                    # because my hand count required EXACTLY TWO register sources while the
                    # computation accepted any all-16-bit source list, including one- and
                    # three-source opcodes. Two numbers over two populations printed side by side
                    # is the defect this file has now fixed four times; the filter is explicit.
                    opcodes_that_could_fill_a_16_16_cell=len([
                        1 for opcode, record in (universe or {}).items()
                        if ((record or {}).get('name') or '').startswith(
                            ('csel', 'clamp', 'fselect'))
                        and _declared_source_widths(int(opcode[2:])) == [16, 16]
                        and _has_register_destination(int(opcode[2:]))]),
                    could_fill_means=('csel-family opcodes with exactly two 16-bit sources AND a '
                                      'register destination - the population that could appear in '
                                      'a (16,16) grid cell at all. My hand count said 22 and the '
                                      'first computed version said 37; the difference is fifteen '
                                      'opcodes with no register destination, which can never fill '
                                      'a cell whatever they compute. Two numbers over two '
                                      'populations printed side by side is the defect this file '
                                      'has now fixed four times, so the filter is written out'),
                    means=('THE GRID DOES NOT EXTEND TO (16,16) AND THAT IS MEASURED, not '
                           'unprobed. 22 opcodes of this family declare two 16-bit sources and '
                           'twelve of them are refuted outright - their records return more than '
                           'two distinct values, so they are not two-valued comparisons at all. '
                           'The absent column is a named difference in what those opcodes do, '
                           'which is a different statement from a cell nobody has dispatched, and '
                           'the one that would otherwise be assumed from a systematic-looking '
                           'enumeration')),
                opcode_offsets_observed=pairs,
                opcode_offsets_counterexamples=sorted(set(different_predicate)),
                opcode_offsets_means=(
                    'pairs of determined forms NINE apart in opcode number. `observed` holds the '
                    'ones where only the result encoding changes and the predicate is the same; '
                    '`counterexamples` holds the ones nine apart whose PREDICATE also changes, '
                    'and it is not empty - op11492 `a <u b` returning integer 1 plus nine is '
                    'op11501, which measures `a >=u b` returning f16 1.0. So the offset does not '
                    'mean "change the result type", and this is an observation over three pairs '
                    'rather than a rule. My first version of this field matched cells without '
                    'comparing predicates and filed all four under the agreements\' label, which '
                    'is how a false rule ships with its own refutation inside it'))


def csel_predicate_elimination(rows, universe):
    """Which PREDICATE each conditional-select opcode is testing, by elimination over its branches.

    The comparison in this family is carried by the opcode, not by an operand value - 232 opcodes
    and at most one recovered bit at operand 2, with opcodes of one shape on one input list
    returning five different answers. So determining one of them means naming a predicate, and a
    record that exercises BOTH branches has already measured that predicate at every one of its
    inputs: each case either returned a source or returned the else-value, and which one it was is
    the condition's truth value there.

    46 arity-2 records have such a mixed pattern. The pattern becomes a boolean vector and the
    predicate library is eliminated against it exactly as the candidate library is eliminated
    against values, with the same discipline: the competitor count is part of the claim, a tie is
    reported as a tie rather than resolved by preference, and the inputs that would split a tie
    are what a future batch should carry.

    WHAT THIS CANNOT DO. It cannot say what the two branches ARE - only when each is taken. The
    else-value here is an operand the records do not write, so "returned the else-value" is
    identified by exclusion: the output matched neither source. And a record whose every case
    takes one branch measures no predicate at all, which is 41 of the 87 arity-2 records in this
    family and is exactly what made these opcodes look like the identity.
    """
    values, plans = _values_by_id(), _plans_by_stem_and_id()
    library = _predicates()
    out, summary = {}, collections.Counter()
    for row in rows:
        name = (universe.get('op%d' % row['op']) or {}).get('name') or ''
        if not name.startswith(('csel', 'clamp', 'fselect')) or row['arity'] != 2:
            continue
        key = (row['stem'], row['id'])
        got, plan = values.get(key), plans.get(key)
        if (not got or not plan or len(got) != len(plan['cases'])
                or row.get('distinct_cases', row['cases']) < MIN_CASES):
            continue
        try:
            classes = g17auth.operand_classes(row['op'])
            dsts, _srcs = g17auth.register_operands(row['op'])
            dmask = 0xFFFF if 'GPR16' in (classes[dsts[0]] or '') else M32
        except Exception:
            dmask = M32
        pattern, inputs = [], []
        for case, value in zip(plan['cases'], got):
            a, b = int(case[0]) & M32, int(case[1]) & M32
            o = int(value) & dmask
            pattern.append('b' if o == (b & dmask) else 'a' if o == (a & dmask) else 'K')
            inputs.append((a, b))
        if len(set(pattern)) < 2:
            summary['record: one branch only - measures no predicate'] += 1
            continue
        summary['record: both branches exercised'] += 1
        # the MINORITY outcome is the one the predicate picks out; both polarities are tried,
        # since a predicate and its negation are different claims about the same opcode
        # A RECORD NO PREDICATE EXPLAINS MUST BE ABLE TO VETO, AND IT COULD NOT. The first
        # version of this `continue`d past an empty survivor set, so a record that REFUTES every
        # predicate contributed nothing and the form kept whatever a weaker record had
        # "determined" - the identical defect this file already fixed for functions, re-made in a
        # new field two hundred lines away.
        #
        # It fired immediately. Eighteen separating inputs on op9724 give the pattern
        # babKKKKKKbKabbKbKb and NOTHING in the library explains it at any polarity; the ten-case
        # batch had "determined" `a != 0` there. So the earlier determinations were artifacts of
        # inputs with one or two minority cases, and the refuting records were being dropped.
        entry = out.setdefault('%d/%s' % (row['op'], name), dict(
            records=[], readings=[], refuted_by=[], competitors=len(library)))
        if list(key) not in entry['records']:
            entry['records'].append(list(key))
        explained = False
        for taken in sorted(set(pattern)):
            want = [p == taken for p in pattern]
            survivors = sorted(n for n, fn in library.items()
                               if all(bool(fn(a, b)) == w for (a, b), w in zip(inputs, want)))
            if not survivors:
                continue
            explained = True
            if list(key) not in entry['records']:
                entry['records'].append(list(key))
            # HOW MANY CASES THE PREDICATE IS TRUE AT, because that is this field's margin. A
            # pattern whose minority branch is taken ONCE pins the predicate at a single input,
            # and "a == 0" surviving there means only that no other predicate in the library is
            # true at exactly that one position - which is elimination, but thin. Five minority
            # cases is a different claim from one, and a reader cannot tell them apart from the
            # predicate name.
            reading = dict(pattern=''.join(pattern), selects=taken, predicates=survivors,
                           cases=len(pattern), true_at=sum(1 for p in pattern if p == taken),
                           determined=len(survivors) == 1)
            # DEDUPED, because the residue's own redundancy reaches this field too: two batches
            # dispatching the same program on the same inputs produce the same reading twice, and
            # counting it twice would make one measurement look like agreement between two.
            if reading not in entry['readings']:
                entry['readings'].append(reading)
        if not explained:
            entry['refuted_by'].append(dict(
                stem=row['stem'], id=row['id'], pattern=''.join(pattern), cases=len(pattern),
                outcomes=len(set(pattern)),
                means=('no predicate of the %d offered explains this record at ANY polarity. With '
                       '%d distinct outcomes the instruction is not a two-way select at all - it '
                       'chooses among its first source, its second source and an else-value - and '
                       'a binary-condition library cannot express that whatever inputs it is given'
                       % (len(library), len(set(pattern))))))
    # AGGREGATED PER FORM BY INTERSECTION, NOT LISTED PER RECORD. op11372 determines `a != 0` on
    # one record and `b != 0` on another: each record is internally consistent and the FORM is not
    # determined, because two records give two different answers - the same disagreement the
    # function census handles by refusing to promote. Correlated inputs are how that happens (a
    # record whose a is nonzero exactly when its b is), and preferring the first answer would
    # publish one of two contradictory claims.
    for key, entry in out.items():
        verdict = {}
        if entry['refuted_by']:
            # THE VETO. A record refuting every predicate outranks one that fitted a weak pattern,
            # for the same reason a later refuting measurement outranks an earlier weaker
            # confirmation in the function census.
            entry['per_form'] = dict(all=dict(
                predicates=[], records=len(entry['records']), true_at_fewest_cases=0,
                true_at_most_cases=0,
                state='records refute every predicate offered',
                means=('%d record(s) of this form are explained by NO predicate in the library at '
                       'any polarity, so nothing here is determined however well a narrower '
                       'record fitted. The patterns have three outcomes: the model is wrong, not '
                       'merely under-determined' % len(entry['refuted_by']))))
            summary['form: records refute every predicate offered'] += 1
            continue
        for taken in sorted({r['selects'] for r in entry['readings']}):
            sets = [set(r['predicates']) for r in entry['readings'] if r['selects'] == taken]
            common = set.intersection(*sets) if sets else set()
            weights = [r['true_at'] for r in entry['readings'] if r['selects'] == taken]
            verdict[taken] = dict(
                predicates=sorted(common), records=len(sets),
                true_at_fewest_cases=min(weights) if weights else 0,
                true_at_most_cases=max(weights) if weights else 0,
                state=('determined' if len(common) == 1 else
                       'records disagree - no predicate survives all of them' if not common else
                       'these inputs cannot separate %d predicates' % len(common)),
                means=('this form returns %s exactly when %s' % (
                    {'a': 'its first source', 'b': 'its second source'}.get(
                        taken, 'a value matching neither source'),
                    sorted(common)[0] if len(common) == 1
                    else 'one of several predicates these inputs do not separate' if common
                    else 'no single predicate explains every record')
                       + ('; pinned at only %d of %d cases, so read it beside that number'
                          % (min(weights), max(r['cases'] for r in entry['readings']))
                          if weights and min(weights) <= 1 else '')))
        entry['per_form'] = verdict
        states = {v['state'] for v in verdict.values()}
        summary['form: predicate determined' if 'determined' in states
                else 'form: records disagree' if any('disagree' in x for x in states)
                else 'form: both branches seen, predicate not determined'] += 1
    return dict(forms=dict(sorted(out.items())), summary=dict(sorted(summary.items())),
                predicates_offered=len(library))


def cross_lane_forms_this_harness_cannot_reach(universe):
    """Cross-lane opcodes the lane-varying probe CANNOT determine, with the structural reason.

    THE PREFIX REDUCTIONS ARE UNREACHABLE FOR A REASON, not for want of trying, and the reason is
    a fact this session measured rather than a guess. A prefix reduction gives lane L the fold over
    lanes 0..L, so every lane holds a different answer and all of them write the one output slot.
    The calibration batch established WHICH store lands: op14022 `quad.shuffle_down1` and op14175
    `simd.shuffle_xor1` both have offsets their names fix at 1 and both returned base + 1, which
    puts the winning lane at ZERO.

    Lane zero's prefix reduction is the fold over lanes 0..0 - its own value. So every prefix
    opcode in this table reads back as the IDENTITY through this harness, for every input, no
    matter what it computes. That is why they sit in the disqualified report as identity-looking,
    and no choice of inputs changes it: the missing capability is a lane-indexed DESTINATION.

    The shuffles are the partial case. Lane zero reads one specific other lane, so the value
    identifies that lane and therefore the opcode's offset - op14283 `simd.shuffle_down` returned
    base + 4, naming its inherited offset operand - but the rest of the permutation is invisible.
    """
    # TWO POPULATIONS, BOTH REPORTED, because the first version of this counted every prefix and
    # shuffle opcode in the table - 95 and 79 - while the claim is about the ones the lane-zero
    # argument actually binds: a form whose destination this harness cannot read is unreachable
    # for a DIFFERENT reason, and a multi-source form may not take the lane-varying probe at all.
    # My own earlier scan of the same question said 49 and 32. Fifth instance of a two-population
    # comparison in this file, so both numbers are published with their filters in the names.
    groups = collections.defaultdict(lambda: dict(all=0, single_source_readable_dst=0))
    for key, record in (universe or {}).items():
        name = (record or {}).get('name') or ''
        if not name.startswith(('simd.', 'quad.')):
            continue
        token = name.split('.')[1] if name.count('.') >= 1 else ''
        kind = 'prefix' if token.startswith('prefix') else 'shuffle' if 'shuffle' in token else None
        if not kind:
            continue
        groups[kind]['all'] += 1
        try:
            opcode = int(key[2:])
            classes = g17auth.operand_classes(opcode)
            dsts, srcs = g17auth.register_operands(opcode)
            reachable = (len(srcs) == 1 and bool(dsts)
                         and 'FLAGR' not in (classes[dsts[0]] or ''))
        except Exception:
            reachable = False
        if reachable:
            groups[kind]['single_source_readable_dst'] += 1
    return dict(
        prefix_opcodes_in_the_table=groups['prefix']['all'],
        prefix_opcodes_the_lane_zero_argument_binds=(
            groups['prefix']['single_source_readable_dst']),
        shuffle_opcodes_in_the_table=groups['shuffle']['all'],
        shuffle_opcodes_the_lane_zero_argument_binds=(
            groups['shuffle']['single_source_readable_dst']),
        two_populations_means=('the first count is every opcode of that shape in the table; the '
                               'second is those with ONE register source and a destination this '
                               'harness can read, which is the population the lane-zero argument '
                               'binds. A FLAGR destination is unreachable for a different reason '
                               'and a multi-source form may not take the lane-varying probe at '
                               'all, so quoting the first number as the unreachable set would '
                               'overstate it by roughly a factor of two'),
        winning_lane=0,
        winning_lane_basis=('op14022 quad.shuffle_down1 and op14175 simd.shuffle_xor1 both carry '
                            'a MEASURED immediate of 1 in their retained records, and both '
                            'returned base + 1 - so the store that lands is lane zero\'s, '
                            'established without assuming any opcode\'s semantics. This read '
                            '"offsets their NAMES fix at 1", which is RETRACTED: the name does '
                            'not pin the immediate, and a compiled quad_shuffle_down(v, 2) emits '
                            'op14022 with operand 4 = 2 and shifts by two. The conclusion is '
                            'unaffected because the records\' immediate was measured rather than '
                            'read off the name, but the sentence asserting it was wrong'),
        prefix_means=('a prefix reduction gives lane L the fold over lanes 0..L, and lane ZERO\'s '
                      'is the fold over lane 0 alone - its own value. So every prefix opcode reads '
                      'back as the IDENTITY through this harness for every input, whatever it '
                      'computes, and no choice of inputs changes that. The missing capability is a '
                      'lane-indexed DESTINATION, not better inputs'),
        shuffle_means=('lane zero reads one specific other lane, so the value identifies that '
                       'lane and therefore the offset - simd.shuffle_down returned base + 4, '
                       'naming its inherited offset - but the rest of the permutation is '
                       'invisible to a single-slot read-back'))


# THE CONSTANT A RECORD RETURNED, CLASSIFIED - because "constant output" covers two opposite
# situations and the goal asks for these to become determinations or NAMED unreachables.
RECOGNISABLE_CONSTANTS = {
    0x00000001: 'integer 1', 0x3F800000: 'f32 1.0', 0x3C00: 'f16 1.0',
    0x7F800000: 'f32 +infinity', 0x7C00: 'f16 +infinity',
    0xFF800000: 'f32 -infinity', 0x7FC00000: 'f32 quiet NaN', 0x7E00: 'f16 quiet NaN',
    0x3F000000: 'f32 0.5', 0x40000000: 'f32 2.0', 0x3800: 'f16 0.5', 0x4000: 'f16 2.0',
}


def duplicate_candidates_at_a_declared_width(samples=3000):
    """Candidate pairs that are the SAME FUNCTION once the operands are masked to a form's width.

    `duplicate_candidates` samples full 32-bit words and reports nothing, which is correct and
    incomplete: it is WIDTH-BLIND. The `_b16` variants exist precisely because a form may declare
    a 16-bit operand, and at such a form `add` and `add_b16` are the same function - the mask the
    variant applies is already applied by the declaration. Same for `bitwise_E` against `or_b16`,
    `bitwise_6` against `xor_b16`, `and_b16` against `bitwise_8`, and `bitreverse` against
    `bitreverse_b16`.

    THAT MATTERS BECAUSE THE CENSUS CALLS THE COLLISION AMBIGUOUS, which is a claim about the ISA,
    when the true statement is about this library. 45 surviving pairs across the ambiguous
    population are of this kind: not instructions nobody can separate, but two names for one
    function at that form's declared width. No dispatch can ever split them and none should be
    designed to.
    """
    rng = random.Random(DEDUP_SEED)
    out = {}
    for arity, kinds in sorted(LIBRARY.items()):
        for widths in sorted({w for w in ((16,) * arity, (32,) * arity,
                                          tuple([32] + [16] * (arity - 1)) if arity > 1 else None,
                                          tuple([16] + [32] * (arity - 1)) if arity > 1 else None)
                              if w}):
            # STRATIFIED, BECAUSE UNIFORM SAMPLING OVER 32 BITS CANNOT PRODUCE A SMALL VALUE,
            # and that is exactly the case the cross-width pairs need. `bitwise_A(a,b) = b`
            # against `umax(a,b)` differs only when the WIDE operand is the larger one, and a
            # uniform 32-bit draw lands below a 16-bit value with probability about 1.5e-5 - so
            # 3000 uniform samples miss it almost surely and the pair reads as identical. That
            # would have published a FALSE library identity and dismissed a real ambiguity as
            # unsplittable, which is the population-that-cannot-produce-a-false-positive defect
            # with the roles reversed.
            #
            # So each operand is drawn from a MIXTURE: its own full width, the low 16 bits only,
            # and small integers - which puts the wide operand below the narrow one often.
            masks = [(1 << w) - 1 for w in widths]

            def draw(mask):
                pick = rng.random()
                if pick < 0.34:
                    return rng.randint(0, mask)
                if pick < 0.67:
                    return rng.randint(0, min(mask, 0xFFFF))
                return rng.randint(0, min(mask, 64))

            probes = [tuple(draw(m) for m in masks) for _ in range(samples)]
            probes += [tuple(0 for _ in masks), tuple(m for m in masks),
                       tuple(1 if i else m for i, m in enumerate(masks)),
                       tuple(m if i else 1 for i, m in enumerate(masks))]
            signatures = collections.defaultdict(list)
            for kind, table in kinds.items():
                for name, fn in table.items():
                    words, ok = [], True
                    for probe in probes:
                        try:
                            value = fn(*probe)
                        except Exception:
                            ok = False
                            break
                        if value is None:
                            ok = False
                            break
                        words.append(int(value) & M32)
                    if ok:
                        signatures[(kind, tuple(words))].append(name)
            collisions = sorted(sorted(v) for v in signatures.values() if len(v) > 1)
            if collisions:
                out['arity %d, sources %s' % (arity, list(widths))] = collisions
    return dict(collisions=out,
                means=('candidate pairs that compute the SAME function once operands are masked '
                       'to a declared width. `library_duplicate_groups` samples full 32-bit words '
                       'and is empty, correctly and incompletely - it is width-blind, and the '
                       '_b16 variants exist because forms declare 16-bit operands. Where the '
                       'census reports such a pair as AMBIGUOUS the honest statement is that this '
                       'library has two names for one function at that width: no dispatch can '
                       'split them and none should be designed to'))


# HOW EACH INSTRUMENT FAILS, WRITTEN OUT RATHER THAN INFERRED. Root's instruction is to keep
# every instrument's failure mode published, and I had never audited whether that was true. A
# heuristic audit - "does this field have a `means` nearby" - gave three different answers as I
# refined it, because a companion key with a slightly different stem, a prose string nested one
# level deeper, and an EMPTY dict with nowhere to put prose all read as absences. An automatic
# check cannot substitute for the author saying how the thing breaks.
#
# So it is a table, and the guard asserts every published field appears in it: a new field cannot
# ship without its author writing down how it misleads.
INSTRUMENT_FAILURE_MODES = {
    'peer_reported_conventions': (
        'holds claims ANOTHER lane measured, with what this one can say about each. The way it '
        'misleads is by proximity: a reader skimming a census of local measurements finds a '
        'sentence stated as confidently as the rest and carries it away without the attribution. '
        'Every entry therefore names its reporter, its source document and commit, and answers '
        '"can my own records tell?" - and for the subnormal-result flush the answer is NOTHING '
        'EITHER WAY, which is the entry a reader is most likely to misread as agreement. It is '
        'not summed into any denominator, for the same reason d3_peer_reported is not'),
    'records': 'a COUNT of plan/result pairs, inflated by redundant batches - 40% of the residue '
               'is the same programs measured twice; see records_that_add_no_information',
    'summary': 'verdict tallies over RECORDS, not forms or opcodes: op10279 contributes fifty of '
               'them because it is the mandatory control in every batch',
    'candidate_library': 'the whole scope of every uniqueness claim here. A function of another '
                         'arity, reading the operands differently, or depending on a modifier is '
                         'not excluded by anything in this file',
    'library_duplicate_groups': 'WIDTH-BLIND: it samples full 32-bit words, so it reports nothing '
                                'while `add` and `add_b16` are the same function at a form '
                                'declaring 16-bit operands. See '
                                'library_duplicates_at_a_declared_width',
    'library_dedup_probe': 'the sample the duplicate check used; a pair differing only outside '
                           'these words reads as a duplicate and a pair differing only at a '
                           'width it does not sample reads as distinct',
    'forms_promotable': 'promotion means one candidate of the published library survived at the '
                        'operand configuration dispatched - not that the instruction cannot be '
                        'something the library does not contain',
    'forms_uniquely_fitted': 'PRE-BAR: includes forms below MIN_CASES, MIN_RUNS or '
                             'MIN_COMPETITORS, so it is larger than forms_promotable and must '
                             'not be read as a determination count',
    'forms_rejected_by_the_bar': 'a tally of rejection REASONS, one per form, so it cannot be '
                                 'summed with the promotable count to get the population',
    'forms_blocked_by_a_complementary_opcode': 'a SUBSET of the disqualified report, and it is '
                                               'empty when every complement cause has been '
                                               'answered - an empty dict here and a broken '
                                               'detector look identical, so read '
                                               'probe_degeneracy.pairs beside it',
    'inseparable_pairs': 'the TOP 40 of over 1,500 pairs, so a pair missing from it may be '
                         'separated or may merely rank below the cap; and it counts co-survivals '
                         'per RECORD, which overstates the per-form question by an order of '
                         'magnitude. See inseparable_pairs_cap and named_pairs_per_form',
    'fit_margin_distribution': 'the BEST margin per form. An all-large distribution would mean '
                               'the library holds no near-neighbours, which is a sparse library '
                               'rather than strong evidence',
    'fit_margin_distribution_worst': 'the WORST margin per form; a form determined by an '
                                     'intersection across records is understated by it',
    'shapes_the_library_cannot_express': 'a shape being expressible does NOT mean the opcode is '
                                         'understood - only that the failure to fit is not about '
                                         'operand widths',
    'asked_outside_the_domain_by_destination': 'splits the candidate set by destination because a '
                                               'FLAGR write returns a robust zero for a reason '
                                               'unrelated to the domain; those entries are not '
                                               'the cheap dispatches the others are',
    'the_control_gate': 'a batch is gated when its OWN control computes src + 4 at its own '
                        'inputs. Rows from batches carrying no control are KEPT rather than '
                        'voided, so `forms_with_at_least_one_gated_record` is the licensing '
                        'question and `forms_resting_entirely_on_gated_records` the strict one - '
                        'the gap between them is forms whose original record was ungated and '
                        'whose re-dispatch was not, which is a history rather than a doubt. '
                        'HOW THE LAW-LEVEL READING MISLEADS: asked of the OPCODES a law '
                        'covers it answers zero unlicensed laws, because an opcode can '
                        'carry a gated row that a different law placed - so it is asked of '
                        'the records each law names in `family_laws.laws[*].explains`, and '
                        'a law whose `explains` list is empty is skipped rather than '
                        'counted as licensed',
    'fit_margins': 'its field-level text is under the SINGULAR `fit_margin_means`, which a '
                   'reader looking for `fit_margins_means` will not find. The margin itself is '
                   'the distance to the nearest LIBRARY rival, so it says nothing about a '
                   'function the library does not contain',
    'forms_where_records_refute_the_fit': 'carries its explanation PER ENTRY, so an empty dict '
                                          'publishes no text at all - and an empty dict here '
                                          'means either that no record refutes a fit or that the '
                                          'refutation path has stopped running',
    'named_pairs_per_form': 'carries its explanation per entry for the same reason, and answers '
                            'only about the three pairs root named - the other 1,500-odd pairs '
                            'are not in it',
    'cross_lane_forms_this_harness_cannot_reach': 'the prefix argument binds only forms with one '
                                                  'source and a readable destination; the '
                                                  'whole-table count is about twice that, and '
                                                  'quoting it would overstate the unreachable set',
}


def the_control_gate(rows, promotable, laws=None):
    """Which batches reproduced CONTROL.op10279, and which determinations rest on ones that did not.

    The standing goal says the control must reproduce its retained result "before any new record is
    trusted". I included it in every batch I ran, by discipline - and nothing ENFORCED it, so the
    clause was a habit rather than a gate. Measured: 746 census rows come from a batch whose
    control reproduced src + 4 at its own inputs, 492 come from a batch carrying no control at
    all, and ONE batch's control genuinely failed.

    MY FIRST MEASUREMENT OF THIS WAS WRONG AND SAID 439 FAILURES. It compared every control
    against the literal vector [4100, 4101, 4356, 8196], which is the 4-case batch's answer - and
    other batches run the control at DIFFERENT inputs, where src + 4 gives [4, 5, 6, 8, ...]. A
    control is valid when it computes src + 4 at ITS OWN inputs, not when it matches one
    remembered vector. Third time today that my measuring instrument needed fixing before its
    verdict could be trusted.

    THE UNGATED ROWS ARE NOT DELETED, and that is a judgement worth stating. Most come from
    batches that predate the control discipline, and voiding 492 records for a procedural reason
    would discard real evidence to make a number look better. So the promotable count is published
    BOTH ways - over all records, and over gated records only - which is the same rule as never
    collapsing the three denominators: report them apart and let the reader choose the standard.
    """
    M = M32
    plans = _plans_by_stem_and_id()
    values = _values_by_id()
    # BUILT FROM THE RESULT FILES, NOT FROM `_values_by_id`, and that distinction is the whole
    # reason this field works. `_values_by_id` skips any record whose `values` is null - which is
    # exactly what a control that DID NOT FINISH has - so a state map built from it cannot see the
    # one batch whose control failed, and the field meant to record that failure came back empty.
    # A traversal whose unknown branch returns nothing answers "clean" by construction.
    state = {}
    for path in sorted(ISA.glob('g17-execution-*-results.json')):
        stem = path.name[:-len('-results.json')]
        try:
            records = json.loads(path.read_text())
        except Exception:
            continue
        records = records if isinstance(records, list) else records.get('results') or []
        control = next((r for r in records
                        if isinstance(r, dict) and r.get('id') == 'CONTROL.op10279'), None)
        if control is None:
            continue
        got = control.get('values')
        cases = (plans.get((stem, 'CONTROL.op10279')) or {}).get('cases') or []
        if not got or not cases or len(cases) != len(got):
            state[stem] = 'FAILED'
            continue
        state[stem] = ('passing' if all(((int(c[0]) + 4) & M) == (int(v) & M)
                                        for c, v in zip(cases, got))
                       else 'FAILED')
    # THE SAME QUESTION ONE LEVEL OUT, AND THE COARSE VERSION GAVE THE FLATTERING ANSWER.
    # A family law is the census's OTHER output, and it deserves the same licence as a
    # determination. Asked at OPCODE granularity - "is any row of any opcode this law covers
    # gated?" - the answer was a clean zero laws unlicensed, which is not the question: a law is
    # licensed when the RECORDS IT EXPLAINS were licensed, and an opcode can easily have a gated
    # row that some other law placed. Asked exactly, once `family_laws` began naming its records,
    # one law rests entirely on ungated records.
    stems = {row['stem'] for row in rows}
    for stem in stems:
        state.setdefault(stem, 'absent')
    counts = collections.Counter(state[row['stem']] for row in rows)
    # `ANY UNGATED` WAS THE WRONG PREDICATE AND IT HID THE WORK THAT FIXED IT. A determination
    # needs ONE supporting record that a control licensed - that record alone establishes the fit -
    # so a form is only ungated when NONE of its records comes from a gated batch. Under `any` a
    # form kept counting as ungated after a gated re-dispatch was added beside its original, which
    # is how 21 successful re-dispatches moved the count by zero.
    #
    # Both readings are published, because they answer different questions: how many
    # determinations are licensed at all, and how many rest ENTIRELY on licensed records.
    ungated = sorted(k for k, entry in promotable.items()
                     if not any(state.get(r['stem']) == 'passing' for r in entry['records']))
    partly = sorted(k for k, entry in promotable.items()
                    if any(state.get(r['stem']) == 'passing' for r in entry['records'])
                    and any(state.get(r['stem']) != 'passing' for r in entry['records']))
    return dict(
        batches={k: v for k, v in sorted(state.items()) if k in stems},
        rows_by_control_state=dict(sorted(counts.items())),
        batches_whose_control_failed=sorted(k for k, v in state.items()
                                            if v == 'FAILED' and k in stems),
        # A BATCH WHOSE CONTROL DID NOT FINISH CONTRIBUTES NOTHING, and that is worth publishing
        # rather than leaving as an empty list to be read as "no failures ever happened".
        # g17-execution-storebit's control reports `did-not-finish` on three of three runs, and so
        # does every other record in that file - so the census sees zero rows from it. The gate
        # held there by construction, not by my vigilance.
        batches_whose_control_failed_and_contribute_no_rows=sorted(
            k for k, v in state.items() if v == 'FAILED' and k not in stems),
        forms_promotable_total=len(promotable),
        forms_with_at_least_one_gated_record=len(promotable) - len(ungated),
        forms_resting_entirely_on_gated_records=len(promotable) - len(ungated) - len(partly),
        forms_with_no_gated_record_at_all=ungated,
        forms_mixing_gated_and_ungated_records=partly,
        laws_resting_entirely_on_ungated_records=sorted(
            name for name, law in (laws or {}).items()
            if law['explains'] and not any(state.get(stem) == 'passing'
                                           for stem, _ in law['explains'])),
        laws_resting_entirely_on_ungated_records_that_count_as_coverage=sorted(
            name for name, law in (laws or {}).items()
            if law['explains'] and _law_is_tested(name)
            and not any(state.get(stem) == 'passing' for stem, _ in law['explains'])),
        laws_measured=len(laws or {}),
        laws_asked_at=('the records each law EXPLAINS, named by (batch, id) in '
                       'family_laws.laws[*].explains - not the rows of the opcodes it covers. '
                       'The opcode-granularity reading of this same question returns zero '
                       'unlicensed laws, because an opcode can carry a gated row that a '
                       'different law placed'),
        means=('a batch is GATED when its own CONTROL.op10279 record computes src + 4 at the '
               'inputs that batch gave it - not when it matches one remembered vector, which is '
               'what my first version of this compared against and why it reported 439 failures '
               'instead of one. Ungated rows are kept rather than deleted: most predate the '
               'control discipline and voiding them for a procedural reason would discard real '
               'evidence, so the promotable count is published BOTH ways and the reader chooses '
               'the standard. `forms_resting_on_an_ungated_record` names every one'))


def instrument_failure_modes(published):
    """The table above, checked against what this census actually publishes."""
    others = sorted(k for k in published
                    if k not in INSTRUMENT_FAILURE_MODES
                    and not k.endswith(('_means', '_basis', '_why', '_scope'))
                    and k not in ('rows', 'scope', 'bar'))
    # VERIFIED, NOT ASSUMED. The first version of this listed these as "carrying their own means
    # text" without checking, and three of eighteen did not: `fit_margins` keeps its text under
    # the SINGULAR `fit_margin_means`, and two more carry text per ENTRY, which publishes nothing
    # at all when the dict is empty. All three are in the table now.
    without = sorted(k for k in others
                     if k + '_means' not in published
                     and not (isinstance(published.get(k), dict)
                              and 'means' in published[k]))
    return dict(failure_modes=dict(sorted(INSTRUMENT_FAILURE_MODES.items())),
                fields_with_their_own_means_text=[k for k in others if k not in without],
                fields_with_NO_failure_mode_anywhere=without,
                means=('how each instrument MISLEADS, written by hand rather than inferred. A '
                       'heuristic audit of "does this field have a means nearby" gave three '
                       'different answers as I refined it - a companion key with a different '
                       'stem, prose nested one level deeper, and an EMPTY dict with nowhere to '
                       'put prose all read as absences - so the table is explicit and a guard '
                       'requires every published field to be either in it or to carry its own '
                       '_means text. A new field cannot ship without its author saying how it '
                       'breaks'))


def the_input_that_would_split_it(rows):
    """For every ambiguous record: a CONCRETE input separating its survivors, or why none exists.

    The goal this map serves asks that anything staying ambiguous be recorded as unreachable
    "with the inputs that would split it", and until now it was recorded only as ambiguous. 53
    records meet the bar, are not degenerate, and are named by nothing - and 31 of them have
    exactly TWO survivors, which is one input away from a determination.

    So the survivors are searched against each other over the domain the DECLARED operand classes
    admit, and the first input where they disagree is published. That input is the deliverable: a
    case to add to the next batch, not an instruction to think harder.

    WHEN NO SUCH INPUT IS FOUND the report says so differently, because the two situations need
    different work. A sampled equivalence over the declared domain is evidence that no separating
    input EXISTS - as for a 16-bit form where integer equality and f16 equality coincide on every
    pattern the width admits - and telling a reader to find better inputs there would be telling
    them to do something impossible. It is evidence and not proof, and the field says which.
    """
    out, summary = {}, collections.Counter()
    excluded = collections.Counter()
    rng = random.Random(DEDUP_SEED)
    for row in rows:
        # INTERPRETATION-AMBIGUOUS ROWS ARE IN NOW. They were excluded by an exact match on
        # 'ambiguous', and they are the rows whose ambiguity is a NARROW DESTINATION - exactly
        # the ones most in need of an input that separates their survivors. Ten rows, three of
        # them at a narrow destination with more than one candidate.
        if row['verdict'] not in ('ambiguous', 'interpretation-ambiguous'):
            continue
        if row.get('distinct_cases', row['cases']) < MIN_CASES:
            excluded['%s: fewer than %d cases' % (row['verdict'], MIN_CASES)] += 1
            continue
        # A DEGENERATE RECORD IS EXCLUDED, AND THAT IS WHY WIDENING THE VERDICT FILTER ADDED
        # NOTHING. Every one of the ten interpretation-ambiguous rows is disqualified here -
        # nine because the output equals an input column, one because it is constant - so they
        # get no separating input, correctly: a record that measured nothing does not need a
        # better pair of candidates to compare, it needs NON-DEGENERATE INPUTS, which is a new
        # dispatch rather than a search over this one. The counts are published so that "the
        # splitter covers this class now" cannot be read as "the class gained entries".
        if (row['outputs_constant_across_cases'] or row['outputs_equal_an_input_column']
                or row['outputs_equal_an_input_column_at_the_destination_width']):
            excluded['%s: the record is degenerate, so it measured nothing to split'
                     % row['verdict']] += 1
            continue
        pairs = []
        for kind, names in row['survivors'].items():
            if len(names) < 2:
                continue
            table = (LIBRARY.get(row['arity']) or {}).get(kind) or {}
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    if names[i] in table and names[j] in table:
                        pairs.append((kind, names[i], names[j]))
        if not pairs:
            continue
        widths = _declared_source_widths(row['op']) or [32] * row['arity']
        masks = [(1 << w) - 1 for w in widths[:row['arity']]] or [M32]
        probes = [tuple(rng.randint(0, m) for m in masks) for _ in range(4000)]
        probes += [tuple(v & m for m in masks) for v in
                   (0, 1, 2, 0x7FFF, 0x8000, 0xFFFF, 0x7FFFFFFF, 0x80000000, M32,
                    0x3F800000, 0xBF800000, 0x7F800000, 0x3C00, 0xFFFF0000)]
        # SEPARATION IS JUDGED AT THE DESTINATION WIDTH, WHICH IS THE ONLY WIDTH ANYONE CAN
        # OBSERVE. This compared `(x & M32) == (y & M32)` regardless of where the result lands,
        # so for a form writing a 16-bit destination it could report an input whose two outputs
        # differ only in bits nobody reads. Measured on the published artifact before the fix:
        # 11 of 52 records sit at a 16-bit destination, and 14 of their 56 separating inputs do
        # NOT separate once masked - a quarter of them. Those inputs are the deliverable, so a
        # batch built from one comes back with identical values and reads as "still ambiguous",
        # which is how a masked degeneracy gets mislabelled unreachable.
        dst_width = row.get('destination_width') or 32
        dst_mask = (1 << dst_width) - 1
        found = []
        for kind, a, b in pairs:
            table = LIBRARY[row['arity']][kind]
            fa, fb = table[a], table[b]
            witness = differs_untruncated = None
            for probe in probes:
                try:
                    x, y = fa(*probe), fb(*probe)
                except Exception:
                    continue
                if x is None or y is None:
                    continue
                if (int(x) & M32) != (int(y) & M32) and differs_untruncated is None:
                    differs_untruncated = ['0x%X' % v for v in probe]
                if (int(x) & dst_mask) == (int(y) & dst_mask):
                    continue
                witness = dict(inputs=['0x%X' % v for v in probe],
                               gives=['0x%X' % (int(x) & dst_mask),
                                      '0x%X' % (int(y) & dst_mask)])
                break
            # THREE OUTCOMES, NOT TWO, because "no input splits them" now covers two situations
            # that need different work. Two candidates may be one function at this declared
            # width - nothing to do, ever - or genuinely different functions that this
            # DESTINATION cannot show apart, which is a fact about the form and not the library.
            if witness:
                state = 'one input splits them at the destination width'
            elif differs_untruncated:
                state = ('DIFFERENT FUNCTIONS THAT THIS DESTINATION CANNOT SHOW APART: they '
                         'disagree at full width but agree in every bit the %d-bit destination '
                         'keeps, so no dispatch of THIS form can separate them. A wider '
                         'destination of the same opcode could' % dst_width)
            else:
                state = ('SAME FUNCTION at this declared width - not an ISA ambiguity but two '
                         'names in this library for one function once the operands are masked, '
                         'so no dispatch can split them and none should be designed to')
            found.append(dict(interpretation=kind, candidates=[a, b], separating_input=witness,
                              destination_width=dst_width,
                              differs_before_truncation=differs_untruncated, state=state))
        key = '%d/%s' % (row['op'], row['id'])
        out[key] = dict(stem=row['stem'], cases=row['cases'], source_widths=widths,
                        pairs=found)
        for entry in found:
            # THREE STATES COUNTED, not two. The summary keyed on whether a witness exists, so
            # "different functions this destination cannot show apart" was being tallied as
            # "indistinguishable over the declared domain" - which says the library holds one
            # function twice, a claim about the LIBRARY rather than about the form.
            if entry['separating_input']:
                summary['a named input splits them at the destination width'] += 1
            elif entry['differs_before_truncation']:
                summary['different functions this destination cannot show apart'] += 1
            else:
                summary['indistinguishable over the declared domain'] += 1
    return dict(records=dict(sorted(out.items())), summary=dict(sorted(summary.items())),
                rows_excluded=dict(sorted(excluded.items())),
                means=('per ambiguous record, a CONCRETE input separating each pair of surviving '
                       'candidates, or a statement that the pair is equivalent over every sampled '
                       'input the declared widths admit. The first is a case to add to the next '
                       'batch; the second means no batch can help and asking for better inputs '
                       'would be asking for the impossible. Sampled, not proved, and labelled so'))


def constant_output_causes(rows, universe):
    """Constant-output records grouped by what the constant TELLS you, not that there was one.

    "The output was constant" covers two opposite situations and collapsing them loses the more
    useful half. An ABSORBING constant - zero or all-ones - is what a dead instrument, an
    undelivered operand, a released register and a correct answer outside the domain all look
    like; it carries almost no information. A constant that is a RECOGNISABLE VALUE - f32 1.0,
    integer 1, an infinity - proves the instruction ran and computed something, so the probe
    reached it and what failed is the INPUT RANGE.

    Measured here: of the constant-output records that meet the case bar and are named by no other
    instrument, 38 return zero while 13 return f32 1.0, 9 return integer 1, 4 return f16 1.0 and 3
    return f32 +infinity. Those last groups are not mysteries about the ISA - they are operations
    whose interesting range these inputs do not span, and saying so is a different instruction to
    the next person than "constant output".

    The unwritten-immediate count is carried per class because it names the LEVER: a record with
    an unwritten immediate has an operand to move, and one without has only its inputs.
    """
    values = _values_by_id()
    named_elsewhere = set(asked_outside_the_domain(rows))
    # A LATER BATCH CAN ANSWER AN EARLIER RECORD, AND THIS CLASS COULD NOT SEE IT. Records are
    # retained rather than deleted, which is right, so a form re-dispatched with better inputs
    # keeps its old constant-output row and the class keeps counting it. Nineteen records here
    # read "constant at an unrecognised value 0x1000" and every one is the instruction returning
    # an operand that g17-execution-sweep pinned at 4096; a later gated batch moved that operand
    # and all nineteen came back varying. Without this the class reports nineteen open questions
    # that are closed, which is the same overstatement as counting a duplicate case as evidence.
    # THE TEST IS WHETHER THE OUTPUT VARIED, NOT WHETHER THE LATER RECORD WOULD BE PROMOTABLE.
    # My first version also excluded records whose output equals an input column, and that
    # excluded the ANSWER: thirteen of these nineteen resolve to "returns operand N", which is
    # degenerate for FITTING and is a complete explanation of why the earlier output was
    # constant. Asking the promotion question here measured the wrong thing and reported five of
    # the nineteen as still open.
    informative = set()
    for other in rows:
        if (other['outputs_constant_across_cases']
                or other.get('distinct_cases', other['cases']) < MIN_CASES):
            continue
        informative.add((other['op'], tuple(other.get('widths') or [])))
    groups = collections.defaultdict(lambda: dict(records=0, opcodes=set(),
                                                  with_an_unwritten_immediate=0))
    for row in rows:
        if not row['outputs_constant_across_cases'] or row['cases'] < MIN_CASES:
            continue
        if 'op%d' % row['op'] in named_elsewhere:
            continue
        word = values.get((row['stem'], row['id']))
        word = (int(word[0]) & M32) if word else None
        if word is None:
            continue
        if word in (0, M32, 0xFFFF):
            label = ('absorbing constant 0x%X - indistinguishable from a dead instrument, an '
                     'undelivered operand or a correct answer outside the domain' % word)
        elif word in RECOGNISABLE_CONSTANTS:
            label = ('constant at %s - the instruction RAN and computed a meaningful value, so '
                     'what these inputs miss is its interesting range'
                     % RECOGNISABLE_CONSTANTS[word])
        elif word & 0xFFFF in RECOGNISABLE_CONSTANTS and not word >> 16:
            label = ('constant at %s - the instruction RAN and computed a meaningful value, so '
                     'what these inputs miss is its interesting range'
                     % RECOGNISABLE_CONSTANTS[word & 0xFFFF])
        else:
            label = 'constant at an unrecognised value - see `values` for the words'
        bucket = groups[label]
        bucket['records'] += 1
        bucket['opcodes'].add('op%d' % row['op'])
        bucket['with_an_unwritten_immediate'] += 1 if row['unwritten_immediates'] else 0
        bucket.setdefault('values', collections.Counter())['0x%X' % word] += 1
        if (row['op'], tuple(row.get('widths') or [])) in informative:
            bucket['superseded_by_a_later_informative_record'] = (
                bucket.get('superseded_by_a_later_informative_record', 0) + 1)
    return dict(
        classes={k: dict(records=v['records'], opcodes=len(v['opcodes']),
                         with_an_unwritten_immediate=v['with_an_unwritten_immediate'],
                         superseded_by_a_later_informative_record=v.get(
                             'superseded_by_a_later_informative_record', 0),
                         still_open=v['records'] - v.get(
                             'superseded_by_a_later_informative_record', 0),
                         values=dict(v.get('values') or {}))
                 for k, v in sorted(groups.items())},
        means=('constant-output records that meet the case bar and that '
               '`asked_outside_the_domain` does not already name, grouped by what the constant '
               'says. An ABSORBING constant carries almost no information - a dead instrument, an '
               'undelivered operand and a correct answer outside the domain all produce it. A '
               'RECOGNISABLE one proves the instruction ran, so the failure is the input range '
               'rather than the probe, and that is a different instruction to whoever picks it '
               'up. `with_an_unwritten_immediate` names the LEVER: an operand to move, or only '
               'the inputs. `superseded_by_a_later_informative_record` counts the records of this '
               'class whose FORM has since been dispatched non-degenerately somewhere else, and '
               '`still_open` is the class size minus those - the number a reader should treat as '
               'open questions. Records are kept rather than deleted, so without this split the '
               'class keeps counting questions that later evidence has already answered'))


def named_pairs_per_form(rows):
    """For the pairs root named: how many FORMS are genuinely unseparated, not how many records.

    A PER-RECORD STATISTIC SUMMED ACROSS RECORDS IS NOT A PER-FORM CLAIM, and reading it as one
    overstates the work by an order of magnitude. `inseparable_pairs` counts co-survivals one
    record at a time, so a form with one thin four-case record and one rich eighteen-case record
    keeps contributing to the count while being fully separated by the rich one. The raw numbers
    are 61, 19 and 21 records; the forms where the pair co-survives in EVERY record of the form
    are 8, 1 and 2.

    That is the same form-versus-opcode discipline this map applies everywhere, one level over -
    and the pair counts had it backwards: they are the denominator of a question nobody asked.
    """
    pairs = [('abs_s32', 'identity'), ('bitwise_A', 'umin'), ('bitwise_C', 'rotl')]
    by_form = {pair: collections.defaultdict(list) for pair in pairs}
    for row in rows:
        surviving = set()
        for names in row['survivors'].values():
            surviving |= set(names)
        for width in row['widths'] or []:
            for pair in pairs:
                by_form[pair]['%d/%d' % (row['op'], width)].append(set(pair) <= surviving)
    out = {}
    for pair in pairs:
        forms = by_form[pair]
        always = sorted(k for k, v in forms.items() if v and all(v))
        some = sorted(k for k, v in forms.items() if any(v) and not all(v))
        out[' / '.join(pair)] = dict(
            forms_unseparated_by_any_record=always,
            forms_a_richer_record_already_separates=len(some),
            means=('%d form(s) have this pair co-surviving in EVERY record of the form, which is '
                   'the number that means "not separated". %d more contribute to the raw record '
                   'count only because a thin record of theirs cannot separate what a richer one '
                   'does' % (len(always), len(some))))
    return out


def records_that_add_no_information(rows):
    """Unexplained records whose emitted bytes AND input list match another record's.

    THE RESIDUE COUNT CAN BE INFLATED BY MY OWN DISPATCHES, and I inflated it this session. A
    sweep of sixteen "condition codes" on op9754 emitted only TWO distinct programs, because the
    operand it swept has one recovered bit - so six of the eight records that ran are byte-identical
    to another record on identical inputs. They measure the same program twice and cannot explain
    anything the other does not, yet each one counts once in `records_remaining`.

    A residue that grows when a redundant batch runs is a residue that cannot be read as remaining
    work. Duplicates are counted apart so the number means what it says.

    Same-bytes-same-inputs is a strict test: two records of one opcode at DIFFERENT inputs are two
    measurements, and two records with different bytes are two programs however similar their
    names. Only the pair that is identical in both is redundant.
    """
    encoded, plans, values = _encoded_by_id(), _plans_by_stem_and_id(), _values_by_id()
    seen, duplicates, disagree = {}, collections.defaultdict(list), []
    for row in rows:
        if row['verdict'] != 'no candidate fits':
            continue
        key = (row['stem'], row['id'])
        blob, plan = encoded.get(key), plans.get(key)
        if not blob or not plan or not plan.get('cases'):
            continue
        signature = (blob, json.dumps(plan['cases']))
        if signature in seen:
            first = seen[signature]
            duplicates[first[0]].append('%s/%s' % key)
            # AND THE QUESTION THAT MAKES THIS FIELD WORTH HAVING. A byte-identical program on an
            # identical input list returning DIFFERENT values is not redundancy - it is
            # non-determinism or a harness fault, and it would invalidate every three-run
            # agreement check in this artifact. So the duplicates are compared rather than merely
            # counted, and 116 of them agreeing is a reproducibility statement across batches that
            # no single batch can make.
            if values.get(key) is not None and values.get(first[1]) is not None \
                    and values[key] != values[first[1]]:
                disagree.append(dict(a=first[0], b='%s/%s' % key,
                                     a_values=values[first[1]], b_values=values[key]))
        else:
            seen[signature] = ('%s/%s' % key, key)
    total = sum(len(v) for v in duplicates.values())
    return dict(
        redundant_records=total,
        distinct_programs_behind_them=len(duplicates),
        duplicates_that_disagree=disagree,
        duplicates_that_disagree_means=(
            'byte-identical programs on identical inputs that returned DIFFERENT values. This '
            'must be empty: a non-empty list is non-determinism or a harness fault, and it would '
            'invalidate every three-run agreement check in this artifact rather than being a '
            'finding about any opcode'),
        groups={k: sorted(v) for k, v in sorted(duplicates.items())},
        means=('records with NO candidate fit whose emitted bytes and input list both match an '
               'earlier record: the same program measured again. They cannot explain anything the '
               'original does not, and each still counts once in records_remaining - so a '
               'redundant batch makes the residue look larger. My own condition sweep put six '
               'here: sixteen requested codes, two distinct programs, because the operand being '
               'swept holds one recovered bit. Subtract these before reading the residue as '
               'remaining SEMANTIC work.\n\n'
               'THEY ARE NOT WASTE, and calling them "adds no information" would be the wrong '
               'word for them. A second identical measurement that AGREES is reproducibility '
               'evidence across batches, which no single batch can produce - so what is reported '
               'is both: how much of the residue is redundant for EXPLANATION, and whether any '
               'pair disagrees, which would be a far more serious finding than anything about an '
               'opcode'))


def the_condition_is_selected_by_the_opcode(rows, universe):
    """csel-family opcodes at IDENTICAL inputs, grouped, with how many distinct answers they give.

    THE FAMILY HAS NO MULTI-BIT CONDITION FIELD TO RECOVER, and that reframes what "unknown" means
    for 232 opcodes. `g17auth.carriers(op, 2)` returns at most ONE bit for every csel, clamp and
    fselect opcode in the table - 122 with zero bits, 101 with one, none with more - and I first
    read that as a decode-side gap, because this file's own csel law calls operand 2 "cond".

    It is not a gap. These records were already in the artifact and only needed grouping: six
    opcodes of the same declared shape, dispatched on the same six inputs in the same batch,
    return FIVE DISTINCT output vectors. Four more do the same one group over. So the comparison
    is enumerated into the OPCODE - which is what a 232-opcode family with a one-bit operand looks
    like - and the single recovered bit is a modifier on it, measured on op9754 to invert the
    branch sense.

    NO DISPATCH PRODUCED THIS. It is the lever that was already built: I had designed a batch to
    sweep an operand that turns out to hold one bit, and the answer was sitting in retained values
    that nothing had grouped by (batch, input list, declared shape).

    AGREEMENT INSIDE A GROUP IS REPORTED TOO, and it is the ambiguous half: two different opcodes
    returning identical vectors either compute the same condition or are not separated by these
    inputs, and this function cannot tell which. Those pairs are named rather than counted as
    evidence either way.
    """
    values, plans = _values_by_id(), _plans_by_stem_and_id()
    groups = collections.defaultdict(list)
    for row in rows:
        name = (universe.get('op%d' % row['op']) or {}).get('name') or ''
        if not name.startswith(('csel', 'clamp', 'fselect')):
            continue
        key = (row['stem'], row['id'])
        got, plan = values.get(key), plans.get(key)
        if not got or not plan or not plan.get('cases'):
            continue
        try:
            classes = g17auth.operand_classes(row['op'])
            dsts, srcs = g17auth.register_operands(row['op'])
            shape = ('%d <- %s'
                     % (16 if 'GPR16' in (classes[dsts[0]] or '') else 32,
                        ','.join(str(16 if 'GPR16' in (classes[i] or '') else 32) for i in srcs)))
        except Exception:
            continue
        groups[(row['stem'], json.dumps(plan['cases']), shape)].append(
            (row['op'], name, tuple(int(v) & M32 for v in got)))
    out = {}
    for (stem, cases, shape), members in sorted(groups.items()):
        opcodes = {m[0] for m in members}
        if len(opcodes) < 2:
            continue
        vectors = collections.defaultdict(set)
        for op, name, vector in members:
            vectors[vector].add('op%d %s' % (op, name))
        agreeing = sorted(sorted(who) for who in vectors.values() if len(who) > 1)
        out['%s / %s / %d inputs' % (stem, shape, len(json.loads(cases)))] = dict(
            opcodes=len(opcodes), distinct_output_vectors=len(vectors),
            opcodes_that_agree_with_another=agreeing,
            selection_is_opcode_borne=len(vectors) > 1,
            means=('%d opcodes of one declared shape on one input list returned %d different '
                   'answers, so what they compute differs by OPCODE rather than by an operand '
                   'value' % (len(opcodes), len(vectors))))
    return out


# THE CROSS-LANE REDUCTIONS, whose variation is across LANES and not across cases.
# THE TOKEN FIXES BOTH THE REDUCTION AND THE INTERPRETATION, and getting the second from the
# first is the whole point of the signed/unsigned pair: `smax` and `umax` are the same reduction
# over different readings of the same 32 bits, and over any lane set that does not cross the sign
# boundary they compute the same function. Measured 2026-09-18 over lanes straddling it - smax
# returned 0x7FFFFFFF where umax returned 0x80000001 at identical inputs - so the distinction
# Apple's name asserts is now an observation.
CROSS_LANE = {
    'fmax': (max, 'float'), 'fmin': (min, 'float'),
    'smax': (max, 'signed'), 'smin': (min, 'signed'),
    'umax': (max, 'unsigned'), 'umin': (min, 'unsigned'),
    # THE INTEGER FOLDS, WHICH ARE EXACTLY PREDICTABLE WHERE THEIR FLOAT SIBLINGS ARE NOT. A
    # float sum over 32 lanes depends on the hardware's reduction tree, which is why
    # simd.sum.f16 and simd.product.f16 are carried as characterisation with no expected value.
    # An integer sum, and, or or xor is order-INDEPENDENT, so the answer follows from the lane
    # set alone - and all ten records of five opcodes matched at two bases each.
    'sum': (lambda v: sum(v) & M32, 'unsigned'),
    'and': (lambda v: functools.reduce(operator.and_, v), 'unsigned'),
    'or': (lambda v: functools.reduce(operator.or_, v), 'unsigned'),
    'xor': (lambda v: functools.reduce(operator.xor, v), 'unsigned'),
}


def cross_lane_reduction_law(row, plan, got, name):
    """A reduction over the 32 (or 4) lane values the record's own `lane_varying` declares.

    THE VARIATION IS ON AN AXIS THE DEGENERACY FILTERS DO NOT LOOK AT, and that is why this law
    needs its own entry rather than a candidate. A lane-varying record carries ONE case, so
    `outputs constant across cases` is trivially true of it and every filter in this file would
    drop it - while the information is in the 32 lane values, which no case column shows. Asking
    whether the outputs vary across cases is the wrong question for a record whose whole point is
    that they vary across lanes.

    What keeps that from being an excuse: the value is PREDICTED. Lane L holds base + step*L, the
    name says which extreme to take, the declared classes say in what precision, and the answer
    either matches or does not. op16860 `simd.fmax.f16` returns 0x3C1F and op16868 `simd.fmin.f16`
    returns 0x3C00 over the same 32 lanes - predicted before the dispatch, and the observation the
    whole cross-lane class had been blocked on.

    SUM AND PRODUCT ARE NOT HERE. A float sum over 32 lanes depends on the hardware's reduction
    tree, which is not known, so no exact value can be predicted and this law does not pretend
    otherwise - those records are carried as characterisation with their measured values.
    """
    lane = (plan or {}).get('lane_varying')
    token = name.split('.')[1] if name.count('.') >= 1 else ''
    if not lane or token not in CROSS_LANE or len(got) != 1:
        return None
    threads = int(plan.get('threads') or 0)
    if threads < 2 or 0 not in set(lane.get('sources') or ()):
        return None
    try:
        classes = g17auth.operand_classes(row['op'])
        dsts, srcs = g17auth.register_operands(row['op'])
        src_16 = 'GPR16' in (classes[srcs[0]] or '')
        dst_16 = 'GPR16' in (classes[dsts[0]] or '')
    except Exception:
        return None
    base = int(plan['cases'][0][0]) & M32
    step = int(lane.get('step', 1))
    smask = 0xFFFF if src_16 else M32
    reduce_with, interpretation = CROSS_LANE[token]
    dmask = 0xFFFF if dst_16 else M32
    values = [(base + step * L) & smask for L in range(threads)]
    try:
        # the integer folds take the whole lane list rather than a key-comparison
        if token in ('sum', 'and', 'or', 'xor'):
            want = reduce_with(values)
        elif interpretation == 'float':
            unpack, pack = (_h if src_16 else _f), (_uh if dst_16 else _u)
            want = pack(reduce_with([unpack(v) for v in values]))
        elif interpretation == 'signed':
            bits = 16 if src_16 else 32
            want = reduce_with(values, key=lambda v: v - (1 << bits) if v >> (bits - 1) else v)
        else:
            want = reduce_with(values)
    except Exception:
        return None
    if want is None or (want & dmask) != (int(got[0]) & dmask):
        return None
    return ('cross-lane reduction law: the named extreme over the lane values the record\'s own '
            'lane_varying declares, PREDICTED from the name and the declared precision - the one '
            'axis this file\'s degeneracy filters cannot see, since such a record carries a '
            'single case',
            '%s over %d lanes of 0x%X + L -> 0x%X, read as %s'
            % (name, threads, base, want & dmask, interpretation))


def cross_lane_rivals(row, plan, got, name):
    """Which OTHER reductions in CROSS_LANE also predict this record, and is it lane zero's value.

    A matched record is not the same thing as a discriminating one. Over lanes base + L an `fmin`
    of positive floats returns lane zero's own value, which is what the identity returns and what
    `umin`, `smin` and `and` return too - the exact shape of the uniform-lane defect, one axis
    over. And over positive floats `fmax`, `smax`, `umax` and `or` all give the top lane. So
    each record is asked of every other token with the same precision, and the law publishes how
    many of its records NO rival reduction and not the identity would also have produced.
    """
    lane = (plan or {}).get('lane_varying') or {}
    parts = name.split('.')
    if len(parts) < 2 or parts[1] not in CROSS_LANE:
        return None
    rivals = sorted(t for t in CROSS_LANE if t != parts[1]
                    and cross_lane_reduction_law(row, plan, got,
                                                 '.'.join([parts[0], t] + parts[2:])))
    try:
        classes = g17auth.operand_classes(row['op'])
        dsts, _srcs = g17auth.register_operands(row['op'])
        dmask = 0xFFFF if 'GPR16' in (classes[dsts[0]] or '') else M32
    except Exception:
        dmask = M32
    base = int(plan['cases'][0][0]) & M32
    lane_zero = (int(got[0]) & dmask) == (base & dmask) and int(lane.get('step', 1)) != 0
    return dict(rivals=rivals, lane_zero=lane_zero)


def _discrimination_fields(split, records):
    """The discriminating denominator beside a law's raw record count, when the law has one.

    Only the cross-lane reduction law computes it today; every other law publishes no such key
    rather than a zero, because a zero would read as "nothing discriminates"."""
    if not split:
        return {}
    return dict(
        records_no_rival_matches=split['alone'],
        records_a_rival_reduction_also_matches=split['rival'],
        records_returning_lane_zeros_own_value=split['lane_zero'],
        opcodes_with_a_record_no_rival_matches=sorted(split['alone_opcodes']),
        discrimination_means=(
            '%d records, of which %d are matched by NO other reduction in CROSS_LANE and are not '
            'lane zero\'s own value. %d return lane zero\'s value, which the identity and every '
            'min-shaped rival also return; %d are matched by another reduction as well - over '
            'positive floats fmax, smax, umax and or all give the top lane. The raw count is '
            'what the law explains; only the first number is what it TESTS'
            % (records, split['alone'], split['lane_zero'], split['rival'])))


def lane_select_law(row, plan, got, name, encoded):
    """A shuffle read at LANE ZERO, whose source lane is an immediate in the record's own bytes.

    The companion to `cross_lane_reduction_law`, for the other half of the cross-lane family. A
    reduction is lane-independent, so the single slot this harness reads gives its answer whatever
    lane wins. A shuffle is not: the 32 lanes hold 32 different results. What makes it readable at
    all is that the store which lands is LANE ZERO's - established, not assumed, by op14022
    `quad.shuffle_down1` and op14175 `simd.shuffle_xor1`, two opcodes whose Apple names fix their
    offsets at 1 and which both returned base + 1, so the winning lane was identified without
    assuming any shuffle's semantics.

    THE IMMEDIATE IS READ OUT OF THE EMITTED BYTES, NOT OUT OF THE PLAN. A plan that both asks for
    an immediate and predicts the answer from it cannot fail; decoding operand 4 from the
    instruction that actually ran makes the prediction independent of the record's own request, and
    catches the case where the authoring table placed a different value than the plan asked for -
    which is the same discipline as reading a mask out of a record's bytes rather than solving it
    from the output.

    WHAT THIS LAW CANNOT SEE, stated because the name says `xor`: at lane zero an XOR of the lane
    index and an ADDITION to it agree, since `0 ^ k == 0 + k`. So this law establishes that operand
    4 selects a SOURCE LANE and that an out-of-range selection yields zero; it does not establish
    the permutation, and cannot. Separating `shuffle_xor` from `shuffle_down` needs every lane read,
    which needs a thread-indexed destination this harness does not build. The recon lane measured it
    that way and their result stays in the peer column.

    Operand 4 is named here as the lane parameter rather than searched for, so the choice is
    falsifiable: it is the position that carries 1 in every corpus instance of the four opcodes
    whose Apple names end in `1`, and the position whose observed range stops at the largest power
    of two each lane group can address - `{1,2}` for the 4-lane quad form over 56 instances,
    `{1,4,8,16}` for the 32-lane simd form over 66. A record whose operand 4 is not a raw immediate
    is refused rather than guessed at.
    """
    if not (plan and got and encoded):
        return None
    # THE NAME ONLY SELECTS THIS LAW; IT NEVER SUPPLIES THE PREDICTED VALUE, which comes from the
    # immediate in the bytes and the lane step. The census does not always carry a name on the row
    # - it is None for every record of this opcode - so the declared name is looked up here rather
    # than depending on a caller's variable that is empty exactly when the law would apply.
    declared = name
    if not declared:
        try:
            declared = g17auth.record(int(row['op'])).get('name') or ''
        except (KeyError, TypeError, ValueError):
            return None
    if 'shuffle' not in declared:
        return None
    lane = plan.get('lane_varying')
    threads = int(plan.get('threads') or 1)
    if not lane or threads < 2 or 0 not in set(lane.get('sources') or ()):
        return None
    if len(plan.get('cases') or ()) != 1 or len(got) != 1:
        return None
    # THE BYTES ARE KEYED BY (stem, id) - the same lookup every other byte-reading law uses. My
    # first version indexed the dict as if it were this record's own list and let a blanket
    # `except` turn the KeyError into "the law does not apply", so the law reported zero matches
    # and looked like a negative result. `_encoded_by_id` above already records that shape: a
    # reader that answers empty is the quietest kind of wrong. So nothing is caught broadly here.
    e_bytes = encoded.get((row['stem'], row['id']))
    if not e_bytes:
        return None
    mapped = g17auth.fields(int(row['op']))
    if 4 not in mapped or mapped[4][0] != 'raw':
        return None
    selector = g17auth.decode(int(row['op']), bytes.fromhex(e_bytes))[4]
    base = int(plan['cases'][0][0])
    step = int(lane.get('step', 1))
    want = (base + step * selector) & M32 if selector < threads else 0
    dmask = (1 << (row.get('destination_width') or 32)) - 1
    if (want & dmask) != (int(got[0]) & dmask):
        return None
    return ('lane-select law: the value lane zero receives, PREDICTED from the source-lane '
            'immediate decoded out of the record\'s own bytes and the lane step its lane_varying '
            'declares, with an out-of-range lane yielding zero. Establishes that the immediate '
            'selects a source lane; does NOT establish the permutation, because at lane zero an '
            'XOR of the lane index and an addition to it agree',
            '%s with operand 4 = %d over %d lanes of 0x%X + 0x%X*L -> 0x%X'
            % (declared, selector, threads, base, step, want & dmask))


def funnel_shift_law(row, plan, got, name, encoded):
    """A two-source shift whose amount is an immediate in its own bytes, and the two are different.

    `funnel.shr` is the textbook one: 9 records of op697, op724 and op727 are exactly
    ((second source << 32) | first source) >> k, and the reading is constant while the DECLARED
    widths vary across the three opcodes (32/32 -> 32, 32/32 -> 16, 32/16 -> 16), so the
    concatenation is 32-bit regardless of what the classes say the operands are.

    `funnel.shl` IS NOT ITS MIRROR, and assuming it was cost two wrong readings. The standard
    left funnel puts the low word's TOP k bits into the result; op14403 and op14439 put its
    BOTTOM k bits there:

        dst = (second source << k) | (first source & ((1 << k) - 1))

    17 of 17 cases on each, at k=1 and k=2, with the mask tracking k (a fixed mask of 3 predicts
    0xB where the k=1 opcode returns 9). My first pass called the two opcodes incompatible - one
    fitting a 16-bit concatenation and the other a 32-bit one - on the strength of five cases
    each; the fifth case of seventeen is where both of those readings die.

    THE AMOUNT WAS WITNESSED AT ONE POINT AND IS NOW WITNESSED AT TWO, WHICH CHANGED THE CLAIM.
    Every retained record of op697, op724 and op727 carried k=27, so "shifts by operand 6" and
    "shifts by 27" were the same statement and this docstring said so. op697 dispatched at k=4
    follows the operand, so the amount IS operand 6.

    The left form's bottom-k-bits reading was witnessed at k=1 and k=2, where it cannot be told
    from a fixed 1- or 2-bit mask nor from the standard funnel taking the low word's TOP k bits.
    op14403 at k=5 separates all three and the bottom-k reading is the one that holds. Both
    amounts remain published per record, because a field rule extrapolated past its witnessed
    points is how a six-bit immediate came to be read as a byte - and 27, 4 and 5 are still three
    points, not a proof for all 127.
    """
    e_bytes = encoded.get((row['stem'], row['id']))
    if not e_bytes or row['arity'] != 2 or len(plan['cases']) != len(got):
        return None
    try:
        operands = list(next(decode_instruction(bytes.fromhex(e_bytes), 0)).values)
        classes = g17auth.operand_classes(row['op'])
        dsts, srcs = g17auth.register_operands(row['op'])
    except Exception:
        return None
    if not dsts or len(srcs) != 2:
        return None
    dmask = 0xFFFF if 'GPR16' in (classes[dsts[0]] or '') else M32
    smask = [0xFFFF if 'GPR16' in (classes[i] or '') else M32 for i in srcs]
    right = name.endswith('shr')
    fits = {}
    for index in (row.get('unwritten_immediates') or []):
        if index >= len(operands) or operands[index][0] != 'imm':
            continue
        k = operands[index][1] & 0x7F
        if not 0 < k < 32:
            continue

        def predict(a, b, k=k):
            if right:
                return (((b << 32) | a) >> k) & M32
            return ((b << k) | (a & ((1 << k) - 1))) & M32

        if all((predict(int(c[0]) & smask[0], int(c[1]) & smask[1]) & dmask)
               == (int(g) & dmask) for c, g in zip(plan['cases'], got)):
            fits[index] = k
    if len(fits) != 1:
        return None
    index, k = next(iter(fits.items()))
    if right:
        return ('funnel shift law: ((second source << 32) | first source) >> k, the amount read '
                'from the record\'s own bytes and the 32-bit concatenation holding across three '
                'different declared width combinations',
                'operand %d = %d' % (index, k))
    return ('funnel shift law, LEFT IS NOT THE MIRROR OF RIGHT: (second source << k) | (first '
            'source & ((1 << k) - 1)) - the low word\'s BOTTOM k bits fill the vacated bits, '
            'not its top k bits',
            'operand %d = %d' % (index, k))


def immediate_operand_law(row, plan, got, name, encoded):
    """A one-register-source operation whose other operand is an IMMEDIATE IN ITS OWN BYTES.

    THE WHOLE POINT IS THAT NOTHING HERE IS FITTED. The additive-immediate law above solves the
    constant from `output - input`; this one reads the constant out of the record's emitted
    operands and then predicts every output with it. A solved constant always exists, so that law
    can only ever report that SOME number was added; a predicted one can be wrong, and being able
    to be wrong is what makes agreement evidence. 53 records of 19 opcodes agree, and the single
    record where no arm held turned out to be the reversed form below rather than a failure.

    Three things are predicted rather than searched, each from structure that is already published:

      the OPERATION, from the table's name token - `mul`, `and`, `or`, `xor`, `shl`, `shr`, `asr`;
      the WIDTHS, from the declared operand classes, source and destination separately. op10831
        `mul` IRGPR32 <- GPR16 is (src & 0xFFFF) * 7 and op10858 GPR16 <- GPR16 is
        ((src & 0xFFFF) * 7 + 3) & 0xFFFF. Both read as no-fit until the classes were applied, and
        all 53 agreeing records agree in the arm the classes predict;
      the ROLE of the immediate, from the OPERAND ORDER. A free immediate positioned AFTER the
        register source is the shift amount; one positioned BEFORE it is the shifted VALUE and the
        register supplies the amount. op14389 `shl` is `1 << (src & 0x7F)`, which is why it
        returns 1 for every input whose low seven bits are zero and 0 for every input where they
        reach 32. Across all 32 shift records the order predicts the role 32 times, no record
        admits both roles, and no record admits neither.

    A record where two different carriers or roles both fit is reported as NOT determined rather
    than resolved by preference - the same rule the narrow-destination law uses when masking
    destroys its discrimination.
    """
    e_bytes = encoded.get((row['stem'], row['id']))
    if not e_bytes:
        return None
    try:
        operands = list(next(decode_instruction(bytes.fromhex(e_bytes), 0)).values)
        classes = g17auth.operand_classes(row['op'])
        dsts, srcs = g17auth.register_operands(row['op'])
    except Exception:
        return None
    if not dsts or not srcs or len(plan['cases']) != len(got):
        return None
    # A STATED OPERAND IS A BETTER CARRIER THAN AN INHERITED ONE, NOT A DISQUALIFIED ONE.
    # `unwritten_immediates` lists operands the record did NOT write, which is the right set for
    # asking what a record inherited - and using it as the carrier list excluded the one record
    # of op10279 that MOVED the operand to 8 and returned src + 8. That record is the strongest
    # evidence in the whole family: the constant was chosen, the emitted bytes carry the chosen
    # value, and the output followed. An inherited constant only shows the operand is READ; a
    # moved one shows it is the operand.
    #
    # The value always comes from the decoded bytes, never from the plan's request, because an
    # operand the compiler owns accepts a value and discards it after encoding.
    stated = {i for i, _v in (row.get('stated_immediates') or [])}
    free = [i for i in sorted(set(row.get('unwritten_immediates') or []) | stated)
            if i < len(operands) and operands[i][0] == 'imm']
    if not free:
        return None
    token = name.split('.')[0]
    suffix = name.split('.')[-1]
    src_16 = 'GPR16' in (classes[srcs[0]] or '')
    dst_16 = 'GPR16' in (classes[dsts[0]] or '')
    smask, dmask = (0xFFFF if src_16 else M32), (0xFFFF if dst_16 else M32)
    inputs = [int(c[0]) & smask for c in plan['cases']]
    outputs = [int(g) & dmask for g in got]
    if len(set(outputs)) < 2:
        return None

    def agrees(fn):
        try:
            want = [fn(v) for v in inputs]
        except Exception:
            return False
        return all(w is not None and (int(w) & dmask) == o for w, o in zip(want, outputs))

    fits = {}
    # THE FLOAT IMMEDIATE, WITH THE TWO PRECISIONS TAKEN SEPARATELY. The older E3M4 block below
    # tries (f32 in, f32 out) and (f16 in, f16 out) as COUPLED pairs, which is exactly the defect
    # that made op3314 `fmul.imm.f32.to.f16` read as a no-fit: its source is f32 and its
    # destination f16, and no coupled pair can express that. The classes declare both ends, so
    # they are read independently here.
    FLOAT_IMM = {'fadd': lambda x, k: x + k, 'fsub': lambda x, k: x - k,
                 'fmul': lambda x, k: x * k, 'fdiv': lambda x, k: x / k if k else None}
    if token in FLOAT_IMM:
        # THE MULTIPLICATIVE FLOAT IMMEDIATE, WHICH IS THE E3M4 MINIFLOAT AGAIN - a fourth family
        # after op9751, op1048 and op1000, and the first MULTIPLICATIVE one. op3314
        # `fmul.imm.f32.to.f16` carries 16 in its immediate, e3m4(16) is exactly 0.25, and 1.0f,
        # 2.0f, 4.0f and 8.0f return the halves 0.25, 0.5, 1.0 and 2.0. The two precisions come
        # from the two declared classes and are NOT assumed equal - reading both ends in one
        # space is what made this record look like a no-fit to my first pass.
        raw = [int(c[0]) & M32 for c in plan['cases']]
        unpack, pack = (_h if src_16 else _f), (_uh if dst_16 else _u)
        saturating = '.sat' in name
        # THE FAMILY IS AFFINE, AND READING IT AS ONE ROLE AT A TIME MISSED THE OTHER HALF.
        # These opcodes carry TWO E3M4 immediates in fixed operand positions - a multiplier and
        # an addend - so `fadd.imm` is not always an addition: op3299 named `fadd.imm` multiplies
        # by e3m4(operand 2) = 0.25 and adds nothing, while op1000 adds e3m4(operand 4) = 0.0625
        # and multiplies by nothing, and op2200 uses operand 4 as the multiplier and operand 5 as
        # the addend. One affine form covers all of them and NAMES which operand took which role,
        # which a per-token function cannot do.
        #
        # A UNIT MULTIPLIER AND A ZERO ADDEND ARE THE IDENTITY. op1004, op767 and op1048 all
        # carry 0x80 there, which is E3M4 negative zero, so their records measure the conversion
        # and cannot test the addition at all - said in the law's own text rather than counted as
        # if the addition had been observed.
        #
        # Flush-to-zero was tried as an arm and is NOT claimed: every record that fits with
        # denormals flushed also fits without, so these observations do not separate the two.
        #
        # THE MULTIPLIER ROLE IS A MULTIPLY, AND IT TOOK A DISPATCH TO SAY SO. Every multiplier
        # witnessed by the retained records was plus or minus a POWER OF TWO - 0.25, 0.5, -0.5,
        # 1 - over which a multiply cannot be told from an adjustment of the exponent field with
        # a sign flip. So this comment carried the caveat until the batch in
        # isa/g17-execution-discriminate.json ran: e3m4(0x31) = 1.0625 has a mantissa no exponent
        # field can express, and op3299, op3298 and op3314 all returned input * 1.0625 at five
        # inputs, across f16->f32, f32->f32 and f32->f16. e3m4(0x21) = 0.53125 confirms it at a
        # second point where the exponent moves too. The exponent-only reading is refuted on
        # every discriminating case. The ADDEND role never had the ambiguity: adding 0.0625 is
        # not an exponent operation.
        #
        # The carrier positions are READ, not predicted. Unlike the integer branch, where operand
        # order settles which operand is the second one, the multiplier sits at operand 2 for
        # op3298/op3299/op3314 and at operand 4 for op2200, whose addend is at operand 5 while
        # op1000's addend is at operand 4. So this arm searches assignments and relies on the
        # uniqueness check below to refuse a record where more than one of them fits.
        assignments = [(m, a) for m in [None] + free for a in [None] + free
                       if not (m is None and a is None) and m != a]
        for multiplier, addend in ([] if len(raw) < MIN_CASES else assignments):
            km = e3m4(operands[multiplier][1]) if multiplier is not None else 1.0
            ka = e3m4(operands[addend][1]) if addend is not None else 0.0
            if km == 1.0 and ka == 0.0:
                continue
            try:
                want = []
                for v in raw:
                    y = unpack(v) * km + ka
                    want.append(pack(min(max(y, 0.0), 1.0) if saturating else y))
            except Exception:
                continue
            if not all(w is not None and (w & dmask) == (int(g) & dmask)
                       for w, g in zip(want, got)):
                continue
            untested = []
            if multiplier is not None and km == 1.0:
                untested.append('the multiplier in operand %d is 1, which the record cannot test'
                                % multiplier)
            if addend is not None and ka == 0.0:
                untested.append('the addend in operand %d is zero, which the record cannot test'
                                % addend)
            fits[('affine', token, suffix, multiplier, addend)] = (
                '%s as e3m4: input * %+g (operand %s) %+g (operand %s), f%d source and f%d '
                'destination%s' % (token, km, multiplier, ka, addend,
                                   16 if src_16 else 32, 16 if dst_16 else 32,
                                   '; ' + ' and '.join(untested) if untested else ''))
    elif token in IMMEDIATE_OPS or token in ('asr', 'sar'):
        signed_bits = 16 if src_16 else 32
        for index in free:
            k = operands[index][1] & M32
            # ORDER DECIDES THE ROLE ONLY WHERE THE OPERATION CARES, and treating it as universal
            # excluded this repository's own mandatory control. op10279 `add` carries the literal
            # 4 in operand 2 while its register source is operand 3, so the immediate PRECEDES
            # the register - and for an addition that is the same instruction either way. 57
            # records of the control were left on the solved-delta law, which could say a 4 was
            # added but not that operand 2 is where the 4 lives.
            #
            # A subtraction and a shift are not commutative and keep the positional reading:
            # op11665 is named `sub.rev` and its immediate precedes the register, which is the
            # vendor's own name agreeing with the order.
            after = index > srcs[0] or token in ('add', 'mul', 'madd', 'and', 'or', 'xor')
            if after:
                if token in ('asr', 'sar') or suffix in ('hi', 'lo'):
                    fn = (lambda v, k=k: _shift_half(token, suffix, v, k, signed_bits))
                    if _shift_half(token, suffix, 1, k & 0x7F, signed_bits) is None:
                        continue
                else:
                    fn = (lambda v, k=k, g=IMMEDIATE_OPS[token]: g(v, k))
                if agrees(fn):
                    fits[('second operand', token, suffix, index, None)] = (
                        '%s%s by operand %d = 0x%X (%s); %d-bit source, %d-bit destination'
                        % (token, '.' + suffix if suffix in ('hi', 'lo') else '', index, k,
                           ('the record WROTE this operand and the emitted bytes carry the '
                            'written value' if index in stated else
                            'the operation is commutative, so the immediate is the second '
                            'operand wherever it sits') if index < srcs[0] else
                           ('the record WROTE this operand and the emitted bytes carry the '
                            'written value' if index in stated else
                            'the immediate FOLLOWS the register, so it is the second operand'),
                           16 if src_16 else 32, 16 if dst_16 else 32))
            elif token == 'sub':
                # THE REVERSED SUBTRACT, and Apple's own suffix agrees with the operand order:
                # op11665 is named `sub.rev` and its free immediate PRECEDES the register source,
                # so the constant is the minuend and the register the subtrahend.
                if agrees(lambda v, k=k: (k - v) & M32):
                    fits[('reversed operand', token, suffix, index, None)] = (
                        'operand %d = 0x%X MINUS the register (the immediate PRECEDES the '
                        'register, so the constant is the minuend); %d-bit source, %d-bit '
                        'destination' % (index, k, 16 if src_16 else 32, 16 if dst_16 else 32))
            elif token in ('shl', 'shr', 'asr', 'sar'):
                # THE REVERSED FORM. The immediate sits BEFORE the register source, so it is the
                # value being shifted and the register is the amount.
                value = k
                if token in ('asr', 'sar') or suffix in ('hi', 'lo'):
                    fn = (lambda v, value=value:
                          _shift_half(token, suffix, value, v, signed_bits))
                else:
                    fn = (lambda v, value=value, g=IMMEDIATE_OPS[token]: g(value, v))
                if agrees(fn):
                    fits[('shifted value', token, suffix, index, None)] = (
                        '%s of operand %d = 0x%X BY the register (the immediate PRECEDES the '
                        'register, so the register is the shift amount)' % (token, index, k))
    if token in ('mul', 'madd'):
        # TWO CONSTANTS, AND THE SECOND ONE IS OFTEN ZERO, WHICH IS NOT A MEASUREMENT OF IT.
        # op10822 is src * 13 + 4 with both constants in its own operands; op10831 carries 7 and
        # 0, where the addend is the additive identity and `src * 7` fits identically. An addend
        # of zero is reported as UNTESTED rather than claimed, for the same reason the csel law
        # refuses a record whose else-value is zero.
        for a in free:
            ka = operands[a][1] & M32
            for b in free:
                if b == a:
                    continue
                kb = operands[b][1] & M32
                if not agrees(lambda v, ka=ka, kb=kb: (v * ka + kb) & M32):
                    continue
                if ka == 1 and kb == 0:
                    continue
                if kb == 0:
                    # THE SAME READING, NOT A COMPETING ONE. `src * 7` and `src * 7 + 0` are one
                    # function, and keying this dictionary on the SENTENCE rather than on the
                    # semantics made them look like two operand assignments in disagreement -
                    # which reported op10831's three records as CARRIER NOT DETERMINED when the
                    # carrier was never in doubt. The key is the reading; the sentence is how it
                    # is published. A detector that compares prose finds conflicts in synonyms.
                    fits.setdefault(('second operand', token, suffix, a, None),
                                    '%s by operand %d = 0x%X, with operand %d = 0 as an addend '
                                    'the record CANNOT test (adding zero is the identity)'
                                    % (token, a, ka, b))
                else:
                    fits[('second operand', token, suffix, a, b)] = (
                        '%s by operand %d = 0x%X then add operand %d = 0x%X, BOTH constants '
                        'read from the record\'s own bytes' % (token, a, ka, b, kb))
    if not fits:
        return None
    if len(fits) > 1:
        return ('immediate operand law, CARRIER NOT DETERMINED: the operation and the widths hold '
                'but more than one operand assignment predicts every output',
                ' | '.join(sorted(fits.values()))[:220])
    return ('immediate operand law: the constant is READ FROM THE RECORD\'S OWN BYTES and the '
            'widths from the declared operand classes, so nothing is solved from the output',
            sorted(fits.values())[0])


# EVERY UPPERCASE QUALIFIER A LAW TITLE MAY CARRY, AND WHETHER ITS RECORDS TEST THE LAW.
# A BLOCKLIST OF WEAKNESS WORDS FAILS OPEN, WHICH IS HOW THE THIRD ONE GOT THROUGH. The predicate
# below used to list `WEAKER` and `DEGENERATE`; `NOT DETERMINED` was invented later and silently
# counted as coverage. Sharing one list fixed the disagreement between fields and fixed nothing
# about the next word - `CONTRADICTS THE TABLE NAME` was added an hour later and the guard written
# expressly to catch this stayed green, because it only compared two copies of the same list.
#
# So the vocabulary is CLOSED. A title carrying a qualifier that is not registered here raises
# rather than defaulting, which makes the classification a decision the author has to make at the
# moment they name a law instead of a property of a list they did not think to update.
LAW_QUALIFIERS = {
    'WEAKER EVIDENCE': False,
    'DEGENERATE EVIDENCE': False,
    'NOT DETERMINED': False,
    'CARRIER NOT DETERMINED': False,
    'CONTRADICTS THE TABLE NAME': False,
    # not a weakness - part of the finding, and registered so it says so
    'LEFT IS NOT THE MIRROR OF RIGHT': True,
}


def _law_is_tested(law_name):
    """Whether a law's records TEST it, read off the qualifiers its own title carries."""
    prefix = law_name.split(':', 1)[0]
    qualifiers = [q.strip() for q in re.findall(r'[A-Z]{2}[A-Z ]*', prefix)]
    for qualifier in qualifiers:
        if qualifier not in LAW_QUALIFIERS:
            raise AssertionError(
                'the law title %r carries the qualifier %r, which is not registered in '
                'LAW_QUALIFIERS. Decide whether its records TEST the law and add it there - a '
                'title that names its own doubt has twice been counted as coverage because the '
                'predicate did not know the word.' % (law_name, qualifier))
    return all(LAW_QUALIFIERS[q] for q in qualifiers)


def family_laws(rows, universe):
    """no-fit records a FAMILY law explains, predicted from the table name plus operand classes.

    Root's milestone, 2026-09-18: explain as many no-fit records as possible with family-level
    interpretations before dispatching anything, and only dispatch the residue. A family law here
    must be predictable from structure - the name and the declared classes - and then checked
    against the retained values. It is not a curve fitted per opcode.

    THE FIRST LAW WAS THE ADDITIVE-IMMEDIATE ONE, and it now explains NOTHING - deliberately.
    An opcode named `add` with exactly ONE register source has its second addend in an IMMEDIATE
    operand, so `output - input` is a single constant across every case. That reading held on 63
    records, but it SOLVES the constant from the output, so all it can ever report is that some
    number was added. `immediate_operand_law` reads the same constant out of the record's emitted
    bytes and predicts the outputs with it, which also names the OPERAND, so it is tried first and
    has taken every record this law used to hold. The code stays as the fallback for a record
    whose encoding cannot be decoded, and its record count reads zero in the artifact rather than
    being hidden - a law displaced by a stronger one should be visibly empty, not deleted.

    What that displacement bought is legible in the artifact: 56 records of op10279 read `add by
    operand 2 = 0x4, the operation is commutative`, and beside them ONE record reads `add by
    operand 2 = 0x8, the record WROTE this operand and the emitted bytes carry the written
    value`. Same form, operand moved, output following - which is what turns this repository's
    positive control from a description into a measurement of which operand carries the addend.

    The census reports all of these as "no candidate fits" because the library holds no candidate
    parameterised on an immediate, and it should keep doing so - a candidate per observed constant
    would fit anything. The laws are recorded here instead, where the constant is reported rather
    than absorbed.
    """
    values = _values_by_id()
    plans = _plans_by_stem_and_id()
    encoded = _encoded_by_id()
    # THE RECORDS ARE NAMED NOW, NOT JUST COUNTED. A law said "63 records" without saying which,
    # so nothing downstream could ask a question about its evidence - including whether a control
    # licensed it. Checking that at OPCODE granularity, which was all the artifact allowed, gave
    # a weaker answer than the same question about determinations: "some row of this opcode was
    # gated" is not "the record this law explains was gated".
    laws = collections.defaultdict(lambda: dict(records=0, opcodes=collections.Counter(),
                                                constants=collections.Counter(),
                                                explains=[]))
    unexplained = []
    for row in rows:
        key = (row['stem'], row['id'])
        got, plan = values.get(key), plans.get(key)
        name = (universe.get('op%d' % row['op']) or {}).get('name') or ''
        placed = None
        # CHECKED BEFORE THE VERDICT AND DEGENERACY FILTERS BOTH, and each exclusion was costing
        # real records. The degeneracy filters ask the wrong axis: a lane-varying record carries
        # one case, so "outputs constant across cases" is true of it by construction while its 32
        # lane values are what vary. And the VERDICT filter excluded it whenever the library
        # happened to fit that single case - which is not rare, because over one input a
        # 26-candidate arity-1 library has a real chance: quad.umax returned 0x80000001 from
        # 0x7FFFFFFE and `not` of 0x7FFFFFFE IS 0x80000001. Three of eight opcodes dropped out of
        # this law for that reason rather than because anything failed.
        #
        # So a lane-varying record is read here whatever its verdict, and when the library also
        # fits it the competing reading is NAMED rather than silently preferred either way. What
        # settles it is a second base: no unary function of the base survives both.
        if got and plan:
            _cross = (cross_lane_reduction_law(row, plan, got, name)
                      or lane_select_law(row, plan, got, name, encoded))
            if _cross:
                placed, _constant = _cross
                _split = cross_lane_rivals(row, plan, got, name)
                if _split is not None:
                    _d = laws[placed].setdefault('discrimination', dict(
                        lane_zero=0, rival=0, alone=0, alone_opcodes=set()))
                    if _split['lane_zero']:
                        _d['lane_zero'] += 1
                    elif _split['rivals']:
                        _d['rival'] += 1
                    else:
                        _d['alone'] += 1
                        _d['alone_opcodes'].add('op%d' % row['op'])
                if row['verdict'] == 'unique':
                    _rival = sorted({v[0] for v in row['survivors'].values() if len(v) == 1})
                    _constant += ('; the library ALSO fits this single case with %s, which a '
                                  'second base rules out' % (_rival[0] if _rival else '?'))
                laws[placed]['constants'][_constant] += 1
        if placed:
            laws[placed]['records'] += 1
            laws[placed]['opcodes']['op%d' % row['op']] += 1
            laws[placed]['explains'].append([row['stem'], row['id']])
            continue
        if row['verdict'] != 'no candidate fits':
            continue
        if (row['outputs_constant_across_cases']
                or row['outputs_equal_an_input_column']):
            continue
        if got and plan and len(plan['cases']) == len(got) and row['arity'] == 1:
            deltas = {(g - c[0]) & M32 for c, g in zip(plan['cases'], got)}
            xors = {(g ^ c[0]) & M32 for c, g in zip(plan['cases'], got)}
            # THE PREDICTED READING IS TRIED BEFORE THE SOLVED ONE, which is the order the E3M4
            # law already established: a constant read out of the record's own bytes says WHERE
            # the number came from, while a constant solved from the output only says that some
            # number was involved. This moves op17770 and op17773 off the solved-XOR law onto the
            # predicted mask; the solved law is kept, because it still catches a record whose
            # encoding does not surrender the operand.
            _imm = immediate_operand_law(row, plan, got, name, encoded)
            if _imm:
                placed, _constant = _imm
                laws[placed]['constants'][_constant] += 1
            elif len(deltas) == 1 and name.split('.')[0] in ('add', 'sub'):
                placed = 'additive immediate: output - input is one constant'
                laws[placed]['constants']['+0x%X' % sorted(deltas)[0]] += 1
            elif name.split('.')[0] in ('fadd', 'fsub') and '.imm' in name:
                # FIRST: IS THE IMMEDIATE AN E3M4 MINIFLOAT? Tried before the solve-for-k law,
                # because this reading PREDICTS the constant from the encoding instead of
                # recovering it from the output - so when it holds it explains where the number
                # came from, and the solve-for-k law only ever says that some number was added.
                #
                # Which operand carries it is not assumed. `unwritten_immediates` already lists
                # the non-register operands this record did not state - operand 1 and the lifetime
                # carriers excluded - and each is tried, so a hit also NAMES the carrier.
                e_bytes = encoded.get((row['stem'], row['id']))
                if e_bytes and not placed:
                    try:
                        operands = list(next(decode_instruction(
                            bytes.fromhex(e_bytes), 0)).values)
                    except Exception:
                        operands = []
                    for index in (row.get('unwritten_immediates') or []):
                        if index >= len(operands) or operands[index][0] != 'imm':
                            continue
                        k = e3m4(operands[index][1])
                        if not k:
                            continue
                        for space, unpack, pack in (('f32', _f, _u), ('f16', _h, _uh)):
                            try:
                                want = [pack(unpack(c[0]) + k) for c in plan['cases']]
                            except Exception:
                                continue
                            wide = 0xFFFF if space == 'f16' else M32
                            if want and all(w is not None and (g & wide) == (w & wide)
                                            for w, g in zip(want, got)):
                                placed = ('E3M4 minifloat immediate: output == input + '
                                          'e3m4(operand) in %s, the constant PREDICTED from the '
                                          'encoding rather than recovered from the output' % space)
                                laws[placed]['constants'][
                                    'operand %d = 0x%02X -> %+g' % (index,
                                                                    operands[index][1] & 0xFF, k)] += 1
                                break
                        if placed:
                            break
                # THE SAME LAW IN FLOAT, AND "THE DELTA IS CONSTANT" IS THE WRONG TEST FOR IT.
                # Float addition ROUNDS: 1e16 + 0.0625 is 1e16, so a record with a large-magnitude
                # case has a delta of zero there and a delta of 0.0625 elsewhere, and a
                # constant-delta test reads that as the law failing. It failed on op1000, whose
                # every small case shows exactly +0.0625.
                #
                # So the constant is SOLVED from one case and then the addition is REDONE in the
                # target precision for every case. That is the law as stated - output equals input
                # plus k - rather than a proxy for it.
                for space, unpack, pack in ((('f32', _f, _u), ('f16', _h, _uh))
                                            if not placed else ()):
                    try:
                        k = unpack(got[0]) - unpack(plan['cases'][0][0])
                        want = [pack(unpack(c[0]) + k) for c in plan['cases']]
                    except Exception:
                        continue
                    if k and want and all(
                            w is not None and (g & (0xFFFF if space == 'f16' else M32))
                            == (w & (0xFFFF if space == 'f16' else M32))
                            for w, g in zip(want, got)):
                        placed = ('additive float immediate: output == input + k in %s, k solved '
                                  'from one case and re-added in that precision' % space)
                        laws[placed]['constants']['%+g' % k] += 1
                        break
                if not placed:
                    # THE WEAKER TEST, KEPT RATHER THAN DISCARDED. Replacing constant-delta with
                    # solve-and-re-add was a net LOSS - it gained nothing and dropped two f16
                    # records that constant-delta explains, so "the better law" made the artifact
                    # explain less. Both are published, and the weaker one says so in its own
                    # name: a constant delta is consistent with an additive immediate and does not
                    # demonstrate the addition was done in that precision.
                    for space, unpack in (('f32', _f), ('f16', _h)):
                        try:
                            ds = {round(unpack(g) - unpack(c[0]), 6)
                                  for c, g in zip(plan['cases'], got)}
                        except Exception:
                            continue
                        if len(ds) == 1 and sorted(ds)[0]:
                            placed = ('additive float immediate, WEAKER EVIDENCE: the delta is '
                                      'one constant in %s but the addition was not reproduced '
                                      'in that precision' % space)
                            laws[placed]['constants']['%+g' % sorted(ds)[0]] += 1
                            break
            elif len(xors) == 1 and sorted(xors)[0] and name.split('.')[0] in ('xor', 'or', 'and'):
                placed = 'bitwise immediate: output XOR input is one constant'
                laws[placed]['constants']['^0x%X' % sorted(xors)[0]] += 1
            elif len(xors) == 1 and sorted(xors)[0] and not name:
                # THE SAME CHECK WITH NOTHING TO CORROBORATE IT, AND LABELLED AS SUCH. op17782 is
                # an opcode the table names nothing for, and its output is exactly its input with
                # bit 0 toggled across every case. For a NAMED opcode the law is the name plus a
                # check; here there is no name, so it is a one-parameter fit and nothing
                # independent agrees with it.
                #
                # This is deliberately NOT done by adding an `xor1` candidate to the library. A
                # candidate added because one record looks like it would let that record be
                # PROMOTED on a post-hoc fit, and the fp16 and mask families were added because a
                # whole interpretation was missing - which is a different thing from shaping the
                # library to an observation.
                placed = ('bitwise immediate, WEAKER EVIDENCE: output XOR input is one constant '
                          'but the table names this opcode nothing, so no name corroborates it')
                laws[placed]['constants']['^0x%X' % sorted(xors)[0]] += 1
        if not placed and got and plan and name.startswith('funnel'):
            _funnel = funnel_shift_law(row, plan, got, name, encoded)
            if _funnel:
                placed, _constant = _funnel
                laws[placed]['constants'][_constant] += 1
        if not placed and row['arity'] == 1 and got and plan and name.startswith(
                ('cvt.f2i', 'cvt.i2f', 'f2i', 'i2f')):
            # THE CONVERSION LAW, PREDICTED ENTIRELY FROM THE DECLARED CLASSES. The name gives the
            # direction and the operand classes give both widths, so nothing here is fitted:
            #
            #   cvt.i2f  GPR32 -> 32-bit dst    0x1000     -> 0x45800000  = f32(4096)
            #   cvt.i2f  GPR32 -> GPR16 dst     0x1000     -> 0x6C00      = half(4096)
            #   cvt.f2i  GPR32 -> 32-bit dst    1.0f       -> 1
            #   cvt.f2i  GPR16 -> 32-bit dst    0x5A5A     -> 0xCB        = int(half 203.25)
            #
            # AND f2i IS UNSIGNED-SATURATING, which is a measurement and not part of the name:
            # -1.5f returns 0, where a truncate-toward-zero would return -1. So the conversion
            # clamps at zero rather than wrapping, and every negative and every NaN in these
            # records returns 0.
            try:
                classes = g17auth.operand_classes(row['op'])
                dsts, srcs = g17auth.register_operands(row['op'])
                src_w = 16 if 'GPR16' in (classes[srcs[0]] or '') else 32
                dst_w = 16 if 'GPR16' in (classes[dsts[0]] or '') else 32
            except Exception:
                src_w = dst_w = None
            if src_w and len(plan['cases']) == len(got):
                def _conv(word):
                    if name.startswith(('cvt.f2i', 'f2i')):
                        v = _h(word) if src_w == 16 else _f(word)
                        if v != v or v <= 0:          # NaN and negatives clamp to zero
                            return 0
                        return min(int(v), 0xFFFF if dst_w == 16 else M32)
                    src = int(word) & (0xFFFF if src_w == 16 else M32)
                    return (_uh(float(src)) if dst_w == 16 else _u(float(src)))
                try:
                    want = [_conv(c[0]) for c in plan['cases']]
                except Exception:
                    want = None
                if want is not None and all(
                        w is not None and (g & (0xFFFF if dst_w == 16 else M32)) == (w & (
                            0xFFFF if dst_w == 16 else M32))
                        for w, g in zip(want, got)):
                    placed = ('conversion law: direction from the name, both widths from the '
                              'declared operand classes; f2i clamps negatives and NaN to zero')
                    laws[placed]['constants']['%s src%d->dst%d' % (name, src_w, dst_w)] += 1
        if not placed and got and plan and row['arity'] in (1, 2):
            # THE DECLARED DESTINATION WIDTH, APPLIED AS A LAW RATHER THAN AS NEW CANDIDATES.
            # Third time today that a narrow destination was the missing term: op11668's `sub` at
            # a 16-bit second operand, two csel forms returning 0x5A5A from a 0x5A5A5A5A mask, and
            # now op16819 `asr` and op17045 `shr` - both declaring GPR16 for the destination AND
            # both sources, both exact at 10 of 10 once the output is truncated to 16 bits and 9
            # of 10 without.
            #
            # MASKING REDUCES DISCRIMINATION, so uniqueness is RE-CHECKED under the mask. Two
            # candidates differing only above bit 15 become indistinguishable once the top half is
            # discarded, and a law that accepted the first match would manufacture determinations
            # out of exactly that collision. If more than one candidate survives the mask this
            # records nothing.
            try:
                classes = g17auth.operand_classes(row['op'])
                dsts, _srcs = g17auth.register_operands(row['op'])
                narrow = 'GPR16' in (classes[dsts[0]] or '')
            except Exception:
                narrow = False
            if narrow and (row.get('outputs_equal_an_input_column_at_the_destination_width')
                           or row.get('outputs_constant_at_the_destination_width')):
                # THE LAW'S OWN MASK IS WHAT MADE THESE FIT. Six records of three csel opcodes
                # returned their second operand truncated to the declared 16-bit destination and
                # were explained as `bitwise_A`, which is that truncation spelled as a function.
                # A law that narrows the comparison has to narrow its degeneracy gates with it.
                placed = ('narrow destination, DEGENERATE EVIDENCE: at the declared 16-bit '
                          'destination the output is an input column or a constant, so the mask '
                          'this law applies is what makes the fit hold')
                laws[placed]['constants']['op%d at 16 bits' % row['op']] += 1
            elif narrow:
                # ACROSS EVERY LIBRARY OF THIS ARITY, NOT ONE OF THEM. `_kind_of` derives the
                # interpretation from a record's SURVIVORS, so for a no-fit record it returns None
                # and the lookup came back empty - the law silently explained nothing while
                # looking implemented. That is circular by construction: the records this law
                # exists for are exactly the ones with no survivor to read a kind from.
                #
                # Searching all three and requiring a single survivor ACROSS them is also the
                # stricter test, so it needs no help from the scheduling-class predictor and
                # cannot inherit that predictor's calibration error.
                library = {}
                for kind, table in (LIBRARY.get(row['arity']) or {}).items():
                    for cand, fn in table.items():
                        library['%s/%s' % (kind, cand)] = fn
                survivors = []
                for cand, fn in library.items():
                    try:
                        want = [fn(*case) for case in plan['cases']]
                    except Exception:
                        continue
                    if all(w is not None and (int(w) & 0xFFFF) == (g & 0xFFFF)
                           for w, g in zip(want, got)):
                        survivors.append(cand)
                # THE NAME'S OWN CANDIDATE WAS OFFERED AND ELIMINATED, WHICH REFUTES THE FIT.
                # A unique survivor is not an explanation when the operation the table NAMES was
                # in the library and did not survive: op3801 `floor` kept `fhalf` while `ffloor`
                # and `hfloor` were eliminated, and op3785 `rint` and op3833 `trunc` did the same
                # thing. Three differently-named operations cannot all be a multiply by one half,
                # so what the masked comparison found is not the instruction's semantics.
                #
                # This asks only about candidates the library ACTUALLY OFFERS. `rint` has no
                # candidate of its own, so for that opcode the correspondence table can say
                # nothing either way - identical failures across subjects measure the question,
                # not the subjects.
                predicted_by_name = {c for c, means in NAME_MEANS.items()
                                     if means[0] == name.split('.')[0]}
                offered = {c.split('/', 1)[1] for c in library}
                survived = {c.split('/', 1)[1] for c in survivors}
                if (len(survivors) == 1 and (predicted_by_name & offered)
                        and not (predicted_by_name & survived)):
                    placed = ('narrow destination, CONTRADICTS THE TABLE NAME: the candidate the '
                              'name predicts was offered and eliminated while something else '
                              'survived the mask, so this is evidence AGAINST the name rather '
                              'than an explanation of the opcode')
                    laws[placed]['constants']['%s named %s, fitted %s' % (
                        'op%d' % row['op'], name, survivors[0])] += 1
                elif len(survivors) == 1:
                    placed = ('narrow destination law: the output is a library function '
                              'TRUNCATED to the declared 16-bit destination, and the fit is still '
                              'unique after masking')
                    laws[placed]['constants']['%s -> 16 bits%s' % (
                        survivors[0],
                        '' if predicted_by_name & survived
                        else ' (no candidate corresponds to the table name %r, so the name '
                             'neither agrees nor disagrees)' % name)] += 1
                elif len(survivors) > 1:
                    placed = ('narrow destination, NOT DETERMINED: several candidates agree once '
                              'the output is masked to the declared 16-bit destination, so the '
                              'mask destroyed the discrimination')
                    laws[placed]['constants']['%d survivors' % len(survivors)] += 1
        if not placed and 'csel' in name and row['arity'] == 2 and got and plan:
            # THE CONDITIONAL-SELECT LAW, FOR ONE OPERAND LAYOUT AND SAID SO.
            #
            #   dst = trunc_to_dst_width(mask) if cond_k(src, HI) else trunc_to_dst_width(imm_f)
            #
            # with cond at operand 2, HI at 5, mask at 6 and imm_f at 8. Another lane determined
            # op11456 this way (dst = mask if src <u HI else imm_f, imm_f NOT always zero - their
            # earlier "else 0" held only because every compiler-emitted witness carried imm_f=0),
            # and applying it to the whole layout covers four opcodes rather than one.
            #
            # THE SCOPE IS THE POINT. The csel family is 46 opcodes in 28 DISTINCT operand
            # layouts, with sources at [3,5], [3,6], [3,7], [3,5,7], [3,5,8] and more. Roles do
            # not sit at fixed indices across them, so this law is keyed to the layout and claims
            # nothing about the other 27. My first attempt indexed "the Nth immediate" instead of
            # the operand number and held for 4 of 17 records - the same positional assumption the
            # peer lane warned about after hitting it three times in a day.
            #
            # The destination WIDTH is load-bearing and was the last piece: two records returned
            # 0x5A5A where the mask was 0x5A5A5A5A, and those opcodes declare a GPR16 destination.
            # With the width applied the law is 7 of 7; without it, 5 of 7.
            try:
                classes = g17auth.operand_classes(row['op'])
                dsts, srcs = g17auth.register_operands(row['op'])
            except Exception:
                classes, dsts, srcs = None, None, None
            e = encoded.get((row['stem'], row['id']))
            if classes and dsts and tuple(srcs or ()) == (3, 6) and e:
                try:
                    vs = list(next(decode_instruction(bytes.fromhex(e), 0)).values)
                except Exception:
                    vs = []
                if len(vs) >= 9 and len(plan['cases']) == len(got):
                    imm_f = vs[8][1]
                    wide = 0xFFFF if 'GPR16' in (classes[dsts[0]] or '') else M32
                    # THE DEGENERACY GATE, AND IT RETIRED MOST OF THIS LAW'S OWN RECORDS.
                    # "output is the mask or imm_f" is trivially true when imm_f is ZERO and the
                    # outputs are mostly zero, and equally trivial when only ONE branch is ever
                    # taken - then the else value is never observed and half the law is untested.
                    # Of the ten records where the structure held at all, five had imm_f = 0 and
                    # three exercised one branch; ONE survived. Naming what a degenerate pass
                    # looks like before trusting a pass is the discipline that caught it.
                    matches = all((g == (c[1] & wide)) or (g == (imm_f & wide))
                                  for c, g in zip(plan['cases'], got))
                    both = sum(1 for c, g in zip(plan['cases'], got)
                               if g == (c[1] & wide) and g == (imm_f & wide))
                    took_mask = sum(1 for c, g in zip(plan['cases'], got)
                                    if g == (c[1] & wide)) - both
                    took_else = sum(1 for c, g in zip(plan['cases'], got)
                                    if g == (imm_f & wide)) - both
                    if matches and took_mask > 0 and took_else > 0 and (imm_f & wide) != 0:
                        placed = ('csel select law (layout dst[0] srcs[3,6]): output is the mask '
                                  'or imm_f, truncated to the destination width - BOTH branches '
                                  'exercised and imm_f nonzero')
                        laws[placed]['constants']['cond=%s HI=%s imm_f=%s'
                                                  % (vs[2][1], vs[5][1], imm_f)] += 1
                    elif matches:
                        placed = ('csel select structure, DEGENERATE EVIDENCE: consistent with '
                                  'the select law but imm_f is zero or only one branch is taken, '
                                  'so the law is not tested by this record')
                        laws[placed]['constants'][
                            'imm_f=%s mask_branch=%d else_branch=%d'
                            % (imm_f, took_mask, took_else)] += 1
        if not placed and name.startswith('addr16') and got:
            # THE RANGE-PREDICATE LAW, determined on op612 and op621 today (24 of 24 values):
            # bit j of the destination, j in 0..3, is set iff lo <= src + j < lo + w, where lo and
            # w are the last two operands. The record's own EMITTED bytes are decoded for lo and w
            # rather than assumed, because both are operands a record inherits unless it writes
            # them - which is exactly why these came back as no-fit.
            lo_w = _addr16_window(row, plan, encoded)
            if lo_w and plan and len(plan['cases']) == len(got):
                lo, w = lo_w
                want = [sum(1 << j for j in range(4) if lo <= c[0] + j < lo + w)
                        for c in plan['cases']]
                if want == [g & 0xF for g in got] and len(set(want)) > 1:
                    placed = 'addr16 range predicate: bit j set iff lo <= src+j < lo+w'
                    laws[placed]['constants']['lo=%d w=%d' % (lo, w)] += 1
        if placed:
            laws[placed]['records'] += 1
            laws[placed]['opcodes']['op%d' % row['op']] += 1
            laws[placed]['explains'].append([row['stem'], row['id']])
        else:
            unexplained.append(row)
    return dict(
        # `tested` IS PUBLISHED PER LAW so a consumer does not have to re-derive the predicate.
        # The map needs it to tell a determination apart from a degenerate match, and a second
        # copy of `_law_is_tested` over there would be a control sharing its predicate with what
        # it controls - the aggregate `records_matched_but_not_tested` below cannot answer it per
        # form, which is the unit a denominator counts.
        laws={k: dict(records=v['records'], opcodes=sorted(v['opcodes']),
                      constants=dict(v['constants']), explains=v['explains'],
                      tested=_law_is_tested(k),
                      **_discrimination_fields(v.get('discrimination'), v['records']))
              for k, v in sorted(laws.items())},
        # TWO TOTALS, BECAUSE A DEGENERATE OR UNCORROBORATED MATCH IS NOT AN EXPLANATION.
        # Laws whose name says WEAKER or DEGENERATE are counted apart: their records are
        # consistent with the law and do not test it - five csel records matched "output is the
        # mask or imm_f" only because imm_f was zero or one branch was never taken, which is the
        # law's own trivial case. Reporting one total would have let those five read as coverage.
        # `NOT DETERMINED` BELONGS WITH `WEAKER` AND `DEGENERATE`, AND IT WAS NOT. The split
        # below excluded the two words I had thought of and let a third through, so the eight
        # records of the narrow-destination law that says in its own name that several candidates
        # survive the mask were counted as EXPLAINED while `accounted_for` - three fields down -
        # excluded them by name. One artifact, two incompatible answers to "is this record
        # explained", and the permissive one was the headline. Adding the carrier-ambiguous arm of
        # the immediate-operand law would have made it eleven.
        #
        # The predicate is shared now so a fourth such word cannot disagree with itself again.
        records_explained=sum(v['records'] for k, v in laws.items() if _law_is_tested(k)),
        records_matched_but_not_tested=sum(v['records'] for k, v in laws.items()
                                           if not _law_is_tested(k)),
        records_remaining=len(unexplained),
        # OPCODES, NOT RECORDS, IS THE COVERAGE FIGURE - and the two differ by a factor of five.
        # 101 records sounds like coverage and is not: op10279 contributes 50 of them because it
        # is the mandatory control in every batch, and op612/op621 contribute 26 because they were
        # probed repeatedly. Counting records measures work DONE; counting opcodes measures the
        # ISA surface EXPLAINED, and the claim being made is about the surface.
        opcodes_explained=len({o for k, v in laws.items()
                               if _law_is_tested(k) for o in v['opcodes']}),
        opcodes_in_the_population=len({'op%d' % r['op'] for r in rows
                                       if r['verdict'] == 'no candidate fits'
                                       and not r['outputs_constant_across_cases']
                                       and not r['outputs_equal_an_input_column']}),
        # THE COMBINED FIGURE, BECAUSE I CONFLATED THE TWO AND REPORTED A NUMBER THAT WAS
        # NEITHER. A family law explains a record the LIBRARY CANNOT FIT; a census determination is
        # the library fitting a record uniquely. They are different populations that overlap, and
        # quoting "coverage went 35 to 43" by adding eight new determinations to the law count
        # produced a figure that appears in no measurement. What a reader wants is the union: how
        # many opcodes of the no-fit population are accounted for by EITHER route.
        accounted_for=len({'op%d' % r['op'] for r in rows
                           if r['verdict'] == 'unique'
                           and not r['outputs_constant_across_cases']
                           and not r['outputs_equal_an_input_column']}
                          | {o for k, v in laws.items()
                             if _law_is_tested(k) for o in v['opcodes']}),
        accounted_for_means=('opcodes with EITHER a tested family law or a unique non-degenerate '
                             'library fit. The union, because the two routes are different '
                             'populations: opcodes_explained counts only the law route and would '
                             'understate, while adding the two counts double-counts their '
                             'overlap'),
        why_two_units=('records_explained counts RECORDS and opcodes_explained counts OPCODES. '
                       'The second is the coverage claim; the first is dominated by opcodes '
                       'measured many times - the control op10279 alone accounts for 50 records '
                       'across every batch that carried it. Do not quote the record count as '
                       'coverage'),
        remaining_by_name=dict(collections.Counter(
            (universe.get('op%d' % r['op']) or {}).get('name') or '(unnamed)'
            for r in unexplained).most_common(20)),
        means=('a family law is predicted from the table NAME plus the declared operand classes '
               'and then checked against retained values - never fitted per opcode. A record it '
               'explains is not an unknown instruction: it is a known operation whose immediate '
               'the candidate library cannot parameterise, and the constant is reported here '
               'rather than absorbed into a candidate'))


def schedclass_predicts_interpretation(rows, contract):
    """Does an opcode's SCHEDULING CLASS predict which interpretation its function lives in?

    Root's standing instruction is to exhaust operand classes and structural metadata before
    dispatching silicon, and the scheduling class is the one field on every opcode this census had
    never used. Two questions, and they have different answers.

    IT DOES NOT PREDICT THE FUNCTION. Of 16 classes with two or more determined members (15 once
    the degenerate fits are excluded, 2 of them single-function, on 2026-09-23), 10 hold
    members with DIFFERENT functions - sched 5 alone holds twelve. And the six that appeared to
    agree were all `trunc16` on the cross-lane opcodes retracted on 2026-09-18 as degenerate: the
    uniform-lane probe gave every one of them the same answer, so their agreement was an artifact
    of the defect and not evidence for the predictor. Excluding them is not tuning the result; it
    is refusing to count a known artifact as support.

    IT DOES PREDICT THE INTERPRETATION: 14 of 15 testable classes on 2026-09-23 (this read "11 of
    11" and went stale; the published figure is derived). This is not a tautology: the census
    tries integer, float and half against every record independently, so members of a class
    arriving at functions from the SAME library is an empirical regularity. But the raw ratio
    overstates it - 67 of the 99 determined opcodes are integer, so a class of two agrees by chance
    about half the time, and the null computed beside it (4.4 of 15, exact, for opcodes drawn
    at random into classes of the same sizes) is the denominator the 14 should be read against. And sched 144,
    which this docstring calls float, holds four float and five half members: it is the one class
    that disagrees.

    WHAT IT IS WORTH. It narrows from three interpretations to one and leaves the function open -
    eight `interpretation-ambiguous` records resolve to integer and each still has seven surviving
    shift and max candidates. And it names the input design error that cost the f32 rounding family
    its margin: sched 144 is float, my case set was sixteen half bit patterns which read as tiny
    positive denormals in f32, and truncation and flooring and rounding all give zero there. The
    class would have said so before the dispatch.

    CALIBRATION IS PUBLISHED BECAUSE THE POPULATION IS SMALL: 15 classes, most with two to four
    determined members, and no class without a determination can be tested at all.
    """
    kind_of = {}
    for arity, kinds in LIBRARY.items():
        for kind, lib in kinds.items():
            for name in lib:
                kind_of.setdefault(name, set()).add(kind)
    # opcodes whose promoted evidence was retracted as degenerate; their agreement is an artifact
    degenerate = {14040, 14301, 16860, 16868, 13853, 13855, 13873, 13875, 13881, 13883,
                  16838, 16840}
    per_class = collections.defaultdict(lambda: dict(kinds=set(), functions=set(), opcodes=set()))
    for row in rows:
        if row['verdict'] != 'unique':
            continue
        if row['outputs_constant_across_cases'] or row['outputs_equal_an_input_column']:
            continue
        if row['op'] in degenerate:
            continue
        names = {f for v in row['survivors'].values() for f in v}
        if len(names) != 1:
            continue
        sched = (contract.get(row['op']) or {}).get('schedclass')
        bucket = per_class[sched]
        bucket['functions'].add(sorted(names)[0])
        bucket['opcodes'].add(row['op'])
        bucket['kinds'] |= kind_of.get(sorted(names)[0], set())
    testable = {k: v for k, v in per_class.items() if len(v['opcodes']) > 1}
    # THE NULL BESIDE THE SCORE. Two-thirds of the determined opcodes are integer, so classes of
    # two would agree often by chance. The expectation below is exact, not sampled: for each
    # testable class of n members, the chance that n opcodes drawn without replacement from the
    # whole determined population share one interpretation, summed over classes.
    _op_kind = {}
    for row in rows:
        if (row['verdict'] != 'unique' or row['outputs_constant_across_cases']
                or row['outputs_equal_an_input_column'] or row['op'] in degenerate):
            continue
        _names = {f for v in row['survivors'].values() for f in v}
        if len(_names) == 1:
            _op_kind.setdefault(row['op'], set()).update(kind_of.get(sorted(_names)[0], set()))
    _pool = collections.Counter(frozenset(k) for k in _op_kind.values())
    _total = sum(_pool.values())
    _null = sum(sum(math.comb(c, len(v['opcodes'])) for kinds, c in _pool.items()
                    if len(kinds) == 1) / math.comb(_total, len(v['opcodes']))
                for v in testable.values()) if _total else 0.0
    _one_kind = sum(1 for v in testable.values() if len(v['kinds']) == 1)
    _one_function = sum(1 for v in testable.values() if len(v['functions']) == 1)
    return dict(
        interpretation_by_class={str(k): sorted(v['kinds'])[0]
                                 for k, v in sorted(per_class.items())
                                 if len(v['kinds']) == 1 and k is not None},
        calibration=dict(
            classes_with_two_or_more_determined_members=len(testable),
            of_those_one_interpretation=_one_kind,
            of_those_one_function=_one_function,
            one_interpretation_expected_by_chance=round(_null, 1),
            degenerate_opcodes_excluded=sorted(degenerate),
            why_excluded=('their promoted fits were retracted as probe artefacts - a uniform-lane '
                          'probe returned the input for every cross-lane opcode, so they all '
                          'fitted trunc16 and would have supplied six of the six classes that '
                          'appear to predict the FUNCTION')),
        # THIS SENTENCE WAS A LITERAL, "11 of 11 ... 6 of 16", and it outlived the calibration
        # three fields above it, which had moved to 14 of 15 and 2 of 15. Derived now.
        means=('the scheduling class predicts the INTERPRETATION and not the function: %d of %d '
               'testable classes agree on interpretation, against %.1f expected by chance - '
               'two-thirds of the determined opcodes are integer, so a small class agrees often '
               'whatever the class means; %d of %d agree on '
               'function once the degenerate cross-lane fits are excluded. Use it to choose '
               'which library a probe should be designed against, never to propagate a '
               'determination from one opcode to another'
               % (_one_kind, len(testable), _null, _one_function, len(testable))))


def fit_margins(rows):
    """For each unique fit: how many cases separate it from its NEAREST rival.

    "The unique survivor of 26 candidates" sounds decisive and can rest on ONE input. Measured
    2026-09-18 across every unique fit in this census: 31 of them have a nearest rival that differs
    on a single case. op3786 is the clearest - its fit `ftrunc` differs from `ffloor` on exactly one
    of sixteen inputs, because the other fifteen are half bit patterns that read as tiny positive
    denormals in f32, where truncation and flooring and rounding all give zero.

    THIS IS NOT THE DEGENERACY DEFECT AND THE DIFFERENCE MATTERS. A degenerate probe discriminates
    NOTHING - a uniform-lane reduction returns its input whatever the semantics, and a zero
    else-value makes a select law trivially true. A margin of one DOES discriminate; it just does
    so on a single observation, so one flaky value flips the fit to its rival and the "26
    candidates eliminated" figure is carried by one input rather than spread across the set.

    So this is published rather than gated: the count of competitors and the MARGIN are different
    facts and a consumer needs both. `min_competing_candidates` says how much was on the table;
    the margin says how much of the answer came from any one case.
    """
    out = {}
    for row in rows:
        if row['verdict'] != 'unique':
            continue
        if row['outputs_constant_across_cases'] or row['outputs_equal_an_input_column']:
            continue
        names = {f for v in row['survivors'].values() for f in v}
        if len(names) != 1:
            continue
        fit = sorted(names)[0]
        library = (LIBRARY.get(row['arity']) or {}).get(_kind_of(row)) or {}
        plan = _plans_by_stem_and_id().get((row['stem'], row['id']))
        if fit not in library or not plan:
            continue
        chosen, nearest = library[fit], None
        for name, other in library.items():
            if name == fit:
                continue
            differs = 0
            for case in plan['cases']:
                try:
                    if other(*case) != chosen(*case):
                        differs += 1
                except Exception:
                    differs += 1
            if nearest is None or differs < nearest[1]:
                nearest = (name, differs)
        if nearest:
            key = '%d/%s' % (row['op'], (row['widths'] or ['?'])[0])
            prior = out.get(key)
            # BOTH ENDS, BECAUSE THEY ANSWER DIFFERENT QUESTIONS AND I HAD ONLY THE PESSIMISTIC
            # ONE. Keeping the minimum across a form's records answers "how weak is the weakest
            # evidence" - useful, but it means a better case set can NEVER show an improvement,
            # and the whole point of the 31-case margin batch was to improve margins. The maximum
            # answers "how strong is the best evidence", which is the question for deciding
            # whether to trust a determination: one record with a good margin suffices, and a
            # weaker record of the same form does not undo it.
            row_out = dict(function=fit, nearest_rival=nearest[0],
                           margin_worst=nearest[1], margin_best=nearest[1],
                           cases=len(plan['cases']), record=row['id'], stem=row['stem'])
            if prior is None:
                out[key] = row_out
            else:
                if nearest[1] < prior['margin_worst']:
                    prior['margin_worst'] = nearest[1]
                if nearest[1] > prior['margin_best']:
                    prior.update(margin_best=nearest[1], nearest_rival=nearest[0],
                                 record=row['id'], stem=row['stem'], cases=len(plan['cases']))
    return out


def _contract():
    import g17isamap
    return g17isamap.sources()['contract']


def _universe():
    return json.loads((ROOT/'isa'/'g17-universe.json').read_text())


def _receipts_by_id():
    """runs and cb_status per (stem, id), for the hardware-execution axis."""
    out = {}
    for path in sorted(ISA.glob('g17-execution-*-results.json')):
        stem = path.name[:-len('-results.json')]
        try:
            rows = json.loads(path.read_text())
        except Exception:
            continue
        for row in (rows if isinstance(rows, list) else rows.get('results') or []):
            if isinstance(row, dict) and row.get('id'):
                out[(stem, row['id'])] = dict(runs=row.get('runs'),
                                              cb_status=row.get('cb_status'),
                                              finished=row.get('finished'))
    return out


def _lane_variation(stem):
    """Did this batch's probe give the lanes DIFFERENT values?

    This harness gives every lane the same value in every batch but one, so for a cross-lane
    opcode it cannot observe the operation at all - a reduction returns its input and a shuffle
    returns its neighbour's identical value. Ten determinations were retracted on 2026-09-18 for
    exactly that. The one batch built for the purpose is named in
    docs/archive/g17-isa-crosslane-batch2-predictions.md and its records carry per-lane values.
    """
    return stem in ('g17-execution-crosslane', 'g17-execution-crosslane2')


def three_evidence_dimensions(rows, universe):
    """Per determined form: hardware semantics, operand-class agreement, Apple-name agreement.

    Root's rule, 2026-09-18: a form carrying all three is much stronger than one merely named by
    the table, and the three must be recorded SEPARATELY rather than collapsed into a verdict.
    That is the same discipline the three denominators already follow - a single number hides which
    instrument produced it.

    Dimension 2 is the one root asked to promote from cross-check to hypothesis source: the
    operand widths this candidate was MEASURED to read, against the widths Apple's operand classes
    DECLARE. op11668 is the case that earned the rule - its fitted `sub_b16` reads (32, 16) and its
    declared classes are (GPR32, GPR16), and those two facts were established independently.

    Dimension 3 scores against NAME_MEANS, which is declared. An unlisted function is
    `unknown_correspondence`, never a guess: accidental agreement is the thing this dimension
    exists to rule out, so it must not be manufactured by a substring test.
    """
    widths = {}
    for arity, kinds in LIBRARY.items():
        for kind in kinds:
            if LIBRARY[arity][kind]:
                widths[(arity, kind)] = candidate_operand_widths(arity, kind)
    # HOISTED. These were called per form inside the loop, which made the census hang - each call
    # re-reads every results file and re-solves every margin. Computing them once is the same
    # answer at a fraction of the cost; a helper being cheap to call is not the same as being
    # cheap to call n times.
    _receipt_cache = _receipts_by_id()
    _margin_cache = fit_margins(rows)
    out = {}
    for row in rows:
        if row['verdict'] != 'unique':
            continue
        if row['outputs_constant_across_cases'] or row['outputs_equal_an_input_column']:
            continue
        fns = {f for v in row['survivors'].values() for f in v}
        if len(fns) != 1:
            continue
        fn = sorted(fns)[0]
        kind = _kind_of(row)
        sig = (widths.get((row['arity'], kind)) or {}).get(fn)
        try:
            classes = g17auth.operand_classes(row['op'])
            dsts, srcs = g17auth.register_operands(row['op'])
            declared_src = [16 if 'GPR16' in (classes[i] or '') else 32 for i in srcs]
            declared_dst = 16 if 'GPR16' in (classes[dsts[0]] or '') else 32
        except Exception:
            declared_src, declared_dst = None, None
        structural = None
        if sig and declared_src is not None and len(declared_src) == row['arity']:
            structural = (sig['operand_widths'] == declared_src
                          and sig['destination_width'] == declared_dst)
        table_name = (universe.get('op%d' % row['op']) or {}).get('name') or ''
        means = NAME_MEANS.get(fn)
        if not means:
            name_agrees = 'unknown_correspondence'
        elif not table_name:
            name_agrees = 'the table names this opcode nothing'
        else:
            token, _w = means
            name_agrees = bool(table_name == token or table_name.startswith(token + '.')
                               or table_name.split('.')[0] == token.split('.')[0])
        key = '%d/%s' % (row['op'], (row['widths'] or ['?'])[0])
        # ROOT'S FIVE AXES, 2026-09-18, kept apart so that "stronger fit" can never be read as
        # "stronger architectural proof" - which is exactly what happened when ten cross-lane
        # forms held their predictions 16 of 16 on a probe that could not see a cross-lane
        # operation.
        receipt = _receipt_cache.get((row['stem'], row['id'])) or {}
        margins = _margin_cache.get(key) or {}
        crosslane_name = (table_name or '').startswith(('simd.', 'quad.'))
        out[key] = dict(
            hardware_semantics=fn,
            semantic_fit_confidence=dict(
                margin_best=margins.get('margin_best'),
                margin_worst=margins.get('margin_worst'),
                nearest_rival=margins.get('nearest_rival'),
                cases=margins.get('cases'),
                means=('cases separating this fit from its nearest library rival. 1 means the '
                       'determination hangs on a single observation')),
            cross_lane_observability=(
                'NOT OBSERVED: the name is cross-lane and this probe gave every lane the same '
                'value, so the operation was not exercised' if crosslane_name
                and not _lane_variation(row['stem'])
                else 'observed: the probe carries per-lane variation' if crosslane_name
                else 'not applicable: this is not a cross-lane operation'),
            hardware_execution_evidence=dict(
                runs=receipt.get('runs'), cb_status=receipt.get('cb_status'),
                finished=receipt.get('finished'),
                means='the record ran on hardware; cb_status 0 and 3 runs is this harness\'s bar'),
            operand_class_agreement=structural,
            measured_widths=(sig or {}).get('operand_widths'),
            measured_destination_width=(sig or {}).get('destination_width'),
            declared_widths=declared_src, declared_destination_width=declared_dst,
            apple_name=table_name or None, apple_name_agreement=name_agrees,
            dimensions_held=sum(1 for x in (True, structural, name_agrees) if x is True),
            axes_note=('FIVE axes, reported apart: semantic fit (the margin), cross-lane '
                       'observability, hardware execution, operand-class agreement and '
                       'Apple-name agreement. dimensions_held counts only the last three for '
                       'continuity with the earlier field; it is NOT a score out of five and a '
                       'form with dimensions_held 3 can still be unobservable'))
    return out


def _record_agrees_at_its_destination_width(row, name, plans, values):
    """Does `name` reproduce this record's observed values, masked to its destination width?

    A RECORD THAT AGREES WITH A HYPOTHESIS CANNOT REFUTE IT, and the refutation test could not
    see that. Row-level survivors are computed at FULL width, so for a form whose declared
    destination is 16 bits every record where the function's 32-bit result OVERFLOWS reads as
    "no candidate fits" - and the refutation filter then counted that as evidence against the
    function. It is not: the hardware cannot return the bits the comparison demanded.

    Measured when this was added, over the 25 no-fit records whose form has a non-degenerate
    unique fit: 20 genuinely disagree and 5 agree once masked. The three forms those five block -
    410/10 `bitwise_4`, 9989/10 `msb`, 14423/14 `shl` - each ALSO carry a full-width unique fit
    on another record, so removing the false refutation promotes on a full-width measurement and
    not on a masked one. The test discriminates rather than rescuing everything narrow: op9990's
    `msb` records still disagree while its sibling op9989's do not.

    This does NOT widen the row-level fit, which stays at full width. It only stops a consistent
    record from vetoing a form.
    """
    if (row.get('destination_width') or 32) >= 32:
        return False
    plan = plans.get((row['stem'], row['id']))
    got = values.get((row['stem'], row['id']))
    if not plan or not got or not plan.get('cases'):
        return False
    table = (LIBRARY.get(row['arity']) or {})
    fn = next((t[name] for t in table.values() if name in t), None)
    if fn is None:
        return False
    mask = (1 << row['destination_width']) - 1
    try:
        predicted = [int(fn(*case)) & mask for case in plan['cases']]
    except Exception:
        return False
    observed = [int(v) & mask for v in got]
    return len(predicted) == len(observed) and predicted == observed


def no_fit_rows_accounted(rows, laws, transcendentals, csel=None):
    """Every "no candidate fits" row, and WHICH INSTRUMENT explains it - or that none does.

    "No candidate fits" reads as a gap and is usually a statement about the wrong instrument. Of
    the 564 rows that meet the case bar and are not degenerate, 255 are explained by a family law
    (the immediate-operand family alone accounts for 164, addr16 for 23, the narrow destination
    for 28 across its three qualifiers, funnel for 15, conversion for 11) and 25 belong to opcodes
    the ULP transcendental instrument covers - which exists precisely because an exact-match
    library can never promote an f32 exp2, and mixing the two instruments is refused elsewhere in
    this file.

    That leaves 284 genuinely unaccounted, and they are grouped by APPLE'S OWN NAME so the
    residual is a work list rather than a number: csel 67, fadd 46, unnamed 28, fmul 26, simd 15,
    fcmp 13, add 12, mul 12, quad 8, floor 6. The pattern worth reading is that `fadd`, `fmul`,
    `add` and `mul` ARE in the library at their arity, so those rows are not a missing
    interpretation - something about the operand configuration differs, which is where the
    immediate-operand law came from. `simd` and `quad` are cross-lane and need per-lane readback
    this harness cannot do.

    WHAT THIS IS NOT: a claim that the 284 are unknowable, and not a coverage number. It is the
    subset of no-fit rows that no instrument in this repository currently addresses, which is the
    only honest basis for choosing what to build next.
    """
    explained = {}
    for name, law in (laws or {}).items():
        for stem, rid in law.get('explains') or []:
            explained[(stem, rid)] = name
    ulp_opcodes = set()
    for key in (transcendentals or {}):
        try:
            ulp_opcodes.add(int(str(key).split('/')[0]))
        except Exception:
            continue
    # THE CSEL CENSUS IS A THIRD INSTRUMENT AND THIS FIELD DID NOT KNOW ABOUT IT. A row counted
    # as unaccounted whenever no FAMILY LAW explained it, which left 67 rows named `csel` in the
    # residual - and 57 of those are records the csel predicate census has already read. Its
    # determinations are the fourteen forms printed as `D3 candidate, NOT counted`, held out of
    # D3 pending root's decision on a sixth column: HELD OUT OF A DENOMINATOR IS NOT THE SAME AS
    # UNEXAMINED, and calling them unaccounted overstated the residual by exactly the work that
    # instrument did. Same shape as the ULP transcendental instrument this field already consults.
    csel_records = set()
    for entry in ((csel or {}).get('forms') or {}).values():
        for record in entry.get('records') or []:
            if isinstance(record, (list, tuple)) and len(record) == 2:
                csel_records.add((record[0], record[1]))
        for record in entry.get('refuted_by') or []:
            if isinstance(record, dict):
                csel_records.add((record.get('stem'), record.get('id')))
    names = _apple_names()
    by_law = collections.Counter()
    by_token = collections.Counter()
    ulp = 0
    csel_seen = 0
    total = 0
    for row in rows:
        if row['verdict'] != 'no candidate fits':
            continue
        if row.get('distinct_cases', row['cases']) < MIN_CASES:
            continue
        if (row['outputs_equal_an_input_column'] or row['outputs_constant_across_cases']
                or row['outputs_equal_an_input_column_at_the_destination_width']
                or row['outputs_constant_at_the_destination_width']):
            continue
        total += 1
        law = explained.get((row['stem'], row['id']))
        if law:
            by_law[law] += 1
        elif row['op'] in ulp_opcodes:
            ulp += 1
        elif (row['stem'], row['id']) in csel_records:
            csel_seen += 1
        else:
            by_token[str(names.get(row['op'], '(no Apple name)')).split('.')[0]] += 1
    return dict(
        rows_considered=total,
        explained_by_a_family_law=sum(by_law.values()),
        by_law=dict(by_law.most_common()),
        covered_by_the_ulp_transcendental_instrument=ulp,
        read_by_the_csel_predicate_census=csel_seen,
        unaccounted=sum(by_token.values()),
        unaccounted_by_apple_name=dict(by_token.most_common()),
        means=('no-fit rows that meet the case bar and are not degenerate, split by which '
               'instrument explains them. "No candidate fits" is usually a statement about the '
               'wrong instrument rather than about the instruction: a family law explains most of '
               'these, and the ULP transcendental instrument owns the ones an exact-match library '
               'can never promote. The remainder is grouped by APPLE\'S OWN NAME so it reads as a '
               'work list. Note that fadd, fmul, add and mul ARE in the library at their arity, so '
               'those rows are not a missing interpretation - the operand configuration differs. '
               'This is not a coverage number and not a claim that the remainder is unknowable. '
               '`read_by_the_csel_predicate_census` counts rows a THIRD instrument has already '
               'read: its determinations are the fourteen forms published as a candidate column '
               'and held out of D3 pending a decision, and held out of a denominator is not the '
               'same as unexamined'))


def _apple_names():
    """opcode -> Apple's name, from the contract. Empty if the contract is absent."""
    out = {}
    path = ISA / 'g17-contract.jsonl'
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get('opcode') is not None and row.get('name'):
            out[int(row['opcode'])] = row['name']
    return out


def census():
    plans, results = load_pairs()
    rows, summary = [], collections.Counter()
    for stem in sorted(set(plans) & set(results)):
        for rid, res in sorted(results[stem].items()):
            plan = plans[stem].get(rid) or {}
            cases, values = plan.get('cases'), res.get('values')
            if res.get('status') != 'ok' or not cases or not values:
                continue
            # A RECORD WITH NO OPCODE IS NOT AN ISOLATED RECORD of any form - a per-lane program of
            # several forms (g17lanereceipt's butterflies, the runtime loops) - and has nothing for
            # this census to fit. It used to reach g17auth with op None and crash the --check.
            if not isinstance(res.get('op'), int):
                summary['no opcode (a multi-form program)'] += 1
                continue
            if len(cases) != len(values) or not all(isinstance(c, list) for c in cases):
                continue
            arities = {len(c) for c in cases}
            if len(arities) != 1:
                summary['mixed arity'] += 1
                continue
            arity = arities.pop()
            if arity not in LIBRARY:
                summary['arity %d not offered' % arity] += 1
                continue
            op = res.get('op')
            found = survivors(cases, values, arity)
            unique = {k: v[0] for k, v in found.items() if len(v) == 1}
            # TWO DEGENERACIES, AND MISSING THE SECOND FALSELY PROMOTED FIVE FORMS. The first is
            # a record whose outputs are just an operand column: the surviving function is the
            # identity, an answer but a weak one, and it is how an operand-ignoring instruction
            # looks. The second is a record whose outputs are CONSTANT across every case, and it
            # is worse, because an ABSORBING function fits it perfectly. The sweep file's cases
            # are raw words that read as tiny positive denormals, so `fceil` returns 1.0 for all
            # four and "exactly one candidate survives" is satisfied by a function that has
            # collapsed the input rather than computed with it. Five forms - op1007/12, op16849/10,
            # op3979/10, op3981/10, op904/12, every one of them all-1.0f - were promoted on that
            # basis before this check existed. A constant output cannot distinguish the semantics
            # from any function that maps this input set to one value.
            degenerate = any([int(c[i]) & M32 for c in cases] == [int(v) & M32 for v in values]
                             for i in range(arity))
            constant = len({int(v) & M32 for v in values}) == 1
            # BOTH DEGENERACY TESTS ARE WIDTH-BLIND, AND A NARROW DESTINATION IS WHERE THE
            # IDENTITY HIDES FROM THEM. `outputs_equal_an_input_column` compares whole words, so
            # a GPR16 destination returning the low half of its input reads as a genuine
            # measurement: 0x0ABCDEF1 in, 0x0000DEF1 out, and the full-width comparison calls
            # those different. Three csel opcodes fit `bitwise_A` - which IS "return the operand"
            # - on exactly that basis, and my own narrow-destination law then counted six such
            # records as explained, because it applies a 16-bit mask WITHOUT re-deriving the
            # degeneracy gates under the same mask.
            #
            # Worse, three promoted forms rest on it. 14040/10, 14301/10 and 590/8 are determined
            # as `trunc16` at declared 16-bit destinations, and trunc16 IS the identity under a
            # 16-bit mask - so the fit cannot separate truncation from a move, a sign-extension,
            # or anything else agreeing on the low half. A mask that makes the claim true is not
            # evidence for it.
            try:
                _classes = g17auth.operand_classes(op)
                _dsts, _ = g17auth.register_operands(op)
                dst_mask = 0xFFFF if 'GPR16' in (_classes[_dsts[0]] or '') else M32
            except Exception:
                dst_mask = M32
            narrow_degenerate = dst_mask != M32 and any(
                [int(c[i]) & dst_mask for c in cases] == [int(v) & dst_mask for v in values]
                for i in range(arity))
            narrow_constant = (dst_mask != M32
                               and len({int(v) & dst_mask for v in values}) == 1)
            verdict = ('unique' if len(unique) == 1 and
                       all(len(v) <= 1 for v in found.values()) else
                       'interpretation-ambiguous' if len(unique) > 1 else
                       'ambiguous' if any(len(v) > 1 for v in found.values()) else
                       'no candidate fits')
            summary[verdict] += 1
            if degenerate:
                summary['outputs equal an input column'] += 1
            if constant:
                summary['outputs constant across cases'] += 1
            if narrow_degenerate and not degenerate:
                summary['outputs equal an input column AT THE DECLARED DESTINATION WIDTH'] += 1
            if narrow_constant and not constant:
                summary['outputs constant AT THE DECLARED DESTINATION WIDTH'] += 1
            rows.append(dict(stem=stem, id=rid, op=op, widths=widths_of(res, op), arity=arity,
                             cases=len(cases),
                             # A REPEATED CASE IS NOT A SECOND MEASUREMENT, and nothing said so.
                             # `cases` counted the length of the list, so a plan that presents the
                             # same inputs twice reports twice the evidence it carries. Found in a
                             # batch I authored on 2026-09-18: my generator's index arithmetic
                             # cycled with period four, so nineteen records ran eight cases of
                             # which four were distinct, and every returned vector had period
                             # four. Both counts are published, and the BAR below uses the
                             # distinct one - a duplicate can no longer buy a row past MIN_CASES.
                             #
                             # A REPEAT IS NOT ALWAYS A MISTAKE, and the field says which rather
                             # than condemning it. Six pre-existing rows repeat a single input
                             # many times - `op621.same-value`, `w.b2b4`, `bx.00` - and their ids
                             # say they mean to: they hold the inputs fixed to vary something
                             # else. All six read `no candidate fits` and no promoted form rests
                             # on one, so tightening the bar retracts nothing; what was wrong was
                             # only that 22 identical cases were presented as 22 cases.
                             distinct_cases=len({tuple(c) for c in cases}),
                             runs=res.get('runs'),
                             had_expect=bool(plan.get('expect')), survivors=found,
                             verdict=verdict, outputs_equal_an_input_column=degenerate,
                             outputs_constant_across_cases=constant,
                             destination_width=16 if dst_mask == 0xFFFF else 32,
                             outputs_equal_an_input_column_at_the_destination_width=(
                                 narrow_degenerate),
                             outputs_constant_at_the_destination_width=narrow_constant,
                             constant_value=('0x%08X' % (int(values[0]) & M32)
                                             if constant and values else None),
                             unwritten_immediates=unwritten_immediates(op, plan),
                             stated_immediates=_stated_configuration(op, plan),
                             note=str(plan.get('note') or '')[:160]))
    forms = {}
    for row in rows:
        if (row['verdict'] != 'unique' or row['outputs_equal_an_input_column']
                or row['outputs_constant_across_cases']
                or row['outputs_equal_an_input_column_at_the_destination_width']
                or row['outputs_constant_at_the_destination_width']):
            continue
        name = list({v[0] for v in row['survivors'].values() if len(v) == 1})[0]
        for width in row['widths']:
            forms.setdefault('%d/%d' % (row['op'], width), []).append(
                dict(function=name, stem=row['stem'], id=row['id'], cases=row['cases'],
                     runs=row['runs'], had_expect=row['had_expect'],
                     competitors=COMPETITORS.get((row['arity'], _kind_of(row)), 0),
                     # carried so a consumer can assert on them rather than trust this filter
                     outputs_equal_an_input_column=row['outputs_equal_an_input_column'],
                     outputs_constant_across_cases=row['outputs_constant_across_cases'],
                     stated_immediates=row['stated_immediates']))

    # THE BAR IS DEFINED HERE, ONCE, so a consumer cannot quietly relax it. "One candidate
    # survived" means nothing without knowing how many were offered: a unique survivor of three
    # eliminated two, and of twenty-eight eliminated twenty-seven. MIN_COMPETITORS is why the
    # arity-3 fits (three candidates each) are NOT promotable even though they are unique, and the
    # arity-1 float fits (eight) are.
    promotable, rejected, conflicts = {}, collections.Counter(), {}
    blocked_by_complement = {}
    # forms whose opcode is half of a complement pair that returned identical values
    _complement_blocked = {}
    _degen = probe_degeneracy(rows, _universe(), _values_by_id())
    _row_by = {(r['stem'], r['id']): r for r in rows}
    # KEYED ON THE RECORD, NOT THE OPCODE. Blocking every form of an opcode that appears in any
    # complement pair in any batch is the over-reporting I had already warned about once: eleven
    # opcodes were touched by a degenerate probe somewhere while only six were degenerate in their
    # own promotion. Coarsely applied here it blocked op16807 `sar` and op17014/op17015 `shr` -
    # the 32-BIT forms measured to differ from each other at margin 4 - because some OTHER record
    # of those opcodes agreed with a complement elsewhere.
    #
    # A promotion is blocked only when the RECORD that supports it is itself one half of a pair
    # that returned identical values. That is the claim the gate is about.
    # opcode -> the record pair that told it apart from its complement, so a lifted block shows
    _separated_ops = {}
    for pair in _degen['separated_at_the_same_inputs']:
        for side, other in (('a', 'b'), ('b', 'a')):
            _separated_ops.setdefault(pair[side]['op'], '%s/%s returned %s while %s returned %s'
                                      % (pair['stem'], pair[side]['id'],
                                         ['0x%X' % v for v in pair[side]['values']][:4],
                                         pair[other]['name'],
                                         ['0x%X' % v for v in pair[other]['values']][:4]))
    # A LANE-VARYING RECORD ANSWERS THE IDENTITY CAUSE TOO, not only the complement one. The
    # cross-lane forms' remaining cause was "the output is an input column", which is a fact about
    # their UNIFORM-LANE records: with every lane holding the same value a reduction returns that
    # value, so of course the output is the input. A lane-varying record of the same opcode
    # returning 0x3C1F from a base of 0x3C00 contradicts it directly, and leaving the cause
    # standing made the status keep naming a probe that had already been built and run.
    # ANY LATER RECORD OF THE SAME FORM ANSWERS IT, not only a lane-varying one. The first
    # version of this keyed on `lane_varying` because the cross-lane forms were the case in hand,
    # and that was too narrow by exactly one class: op9754 `csel` is identity-looking in every
    # retained record, and what contradicts it is not lane variation but a record that MOVED THE
    # CONDITION BIT and got 0x3C800000 where the input column would have given 0xFFFFFFFF. Keying
    # on the mechanism instead of on the observation left that form reading "blocked on explicit
    # source-mode authoring" after the authoring had been done.
    #
    # SCOPED TO THE FORM AND NOT THE OPCODE. A 32-bit record cannot answer a 16-bit form's cause:
    # the identity claim is about what a destination of that width returns, and this file has
    # already inflated three denominators by applying an opcode-level fact per form.
    _lane_answered = {}
    _plans_once = _plans_by_stem_and_id()
    _values_once = _values_by_id()
    for row in rows:
        if (row['outputs_equal_an_input_column']
                or row['outputs_equal_an_input_column_at_the_destination_width']):
            continue
        # THE CONSTANT TEST IS VACUOUS ON A SINGLE-CASE RECORD and applying it here silently
        # disqualified every answerer I had just built. A lane-varying record carries ONE case, so
        # "outputs constant across cases" is true of it by construction - the same wrong-axis
        # mistake the cross-lane law exists to avoid, re-made one function away, and it took the
        # answered count from twelve back to nine while looking like a tightening.
        if row['cases'] > 1 and (row['outputs_constant_across_cases']
                                 or row['outputs_constant_at_the_destination_width']):
            continue
        plan = _plans_once.get((row['stem'], row['id'])) or {}
        how = ('varies its lanes and does not'
               if plan.get('lane_varying') else
               'states operand %s and does not' % ','.join(
                   str(k) for k, _v in (row.get('stated_immediates') or []))
               if row.get('stated_immediates') else 'does not')
        _lane_answered.setdefault((row['op'], row['destination_width']),
                                  '%s/%s %s' % (row['stem'], row['id'], how))
    _degenerate_records = set()
    for pair in _degen['pairs']:
        for side in ('a', 'b'):
            _degenerate_records.add((pair['stem'], pair[side]['id']))
    _complement_twin = {}
    for pair in _degen['pairs']:
        for side, other in (('a', 'b'), ('b', 'a')):
            _complement_twin[(pair['stem'], pair[side]['id'])] = pair[other]['name']
    for key, fits in sorted(forms.items()):
        strong = [f for f in fits if f['cases'] >= MIN_CASES and (f['runs'] or 0) >= MIN_RUNS
                  and f['competitors'] >= MIN_COMPETITORS]
        if not strong:
            rejected['below the bar'] += 1
            continue
        # ONE OPERAND CONFIGURATION AT A TIME, the rule the refutation below already applies. A
        # record that writes an immediate the base form inherits is a DIFFERENT instruction, and
        # its function is a result about that operand: op16805's operand 4 = 3 and = 5
        # (g17-execution-confound3) return asr3 and asr5, which is the shift distance measured,
        # not a disagreement with sar1 at the base form's own value. Counting them together made
        # 16805/12 "records disagree" and unpromotable the moment the fits were regenerated. The
        # base configuration (no stated immediate) speaks for the form when present; otherwise
        # the records must share one configuration, as before.
        configs = {json.dumps(f.get('stated_immediates') or []) for f in strong}
        if '[]' in configs:
            strong = [f for f in strong if not f.get('stated_immediates')]
        names = {f['function'] for f in strong}
        if len(names) > 1:
            rejected['records disagree on the function'] += 1
            continue
        # A RECORD THAT REFUTES THE FUNCTION MUST BE ABLE TO VETO IT, AND IT COULD NOT.
        # `forms` above is built only from rows with a UNIQUE survivor, so a record where the
        # function does NOT survive contributes nothing and the disagreement is invisible. op16838
        # and op16840 were promotable as trunc16 on a four-case record while a sixteen-case record
        # of the same form returned the canonical quiet half NaN at 0x00007FFF, which truncation
        # cannot produce - a later, stronger, refuting measurement outranked by an earlier weaker
        # one because the filter only ever saw confirmations.
        name = sorted(names)[0]
        contradicting = [
            dict(stem=r['stem'], id=r['id'], cases=r['cases'], verdict=r['verdict'],
                 survivors={k: v for k, v in r['survivors'].items() if v})
            for r in rows
            if '%d/%d' % (r['op'], (r['widths'] or [0])[0]) == key
            and not r['outputs_constant_across_cases']
            and not r['outputs_equal_an_input_column']
            # a degenerate record cannot refute a fit either, for the same reason it cannot
            # support one: it did not measure the function
            and not r['outputs_equal_an_input_column_at_the_destination_width']
            and not r['outputs_constant_at_the_destination_width']
            and r['cases'] >= MIN_CASES
            # ONLY THE SAME OPERAND CONFIGURATION MAY REFUTE. This file's own scope sentence says
            # a fit holds "at the operand and modifier configuration each record was dispatched
            # with", and a record that deliberately writes a different immediate is a different
            # configuration measuring a different function. c1004.imm16 moved op1004's operand 4
            # from 128 to 16 and returned something else entirely - which is a RESULT about that
            # operand, not evidence against h2f at the operand's own value, and it vetoed the base
            # form until this line existed.
            and r.get('stated_immediates') == next(
                (f.get('stated_immediates') for f in strong), [])
            and name not in {f for v in r['survivors'].values() for f in v}
            # AND IT MUST ACTUALLY DISAGREE. `survivors` is computed at full width, so a record
            # of a 16-bit-destination form whose result overflows lists nothing and looked like a
            # refutation of the very function it reproduces once masked to the width the hardware
            # writes. See `_record_agrees_at_its_destination_width`.
            and not _record_agrees_at_its_destination_width(
                r, name, _plans_once, _values_once)]
        if contradicting:
            rejected['a record of the same form refutes the function'] += 1
            conflicts[key] = dict(function=name, refuted_by=contradicting,
                                  supported_by=[dict(stem=f['stem'], id=f['id'],
                                                     cases=f['cases']) for f in strong],
                                  means=('one record of this form fits %s and another of at least '
                                         'MIN_CASES cases does not admit it at all. The form is '
                                         'NOT promoted and the two records are named, because the '
                                         'disagreement is the finding - usually the wider record '
                                         'reaching a special value the narrower one never asked '
                                         'about' % name))
            continue
        # THE COMPLEMENT GATE, root's rule of 2026-09-18: if two semantically distinct
        # candidates produce identical outputs over all current observations, NEITHER is
        # determined unless another independent observable distinguishes them. Until now the
        # complement detector only REPORTED; a table name could still pull an ambiguous
        # measurement over the line, which is how op16819 `asr` came to fit a 16-bit LOGICAL
        # shift at 17 of 17 while op17045 `shr` returned byte-identical values.
        supporting = {(f['stem'], f['id']) for f in strong}
        degenerate_support = sorted(supporting & _degenerate_records)
        if degenerate_support and not (supporting - _degenerate_records):
            rejected['a complementary opcode returns identical values'] += 1
            blocked_by_complement[key] = dict(
                function=name,
                twin=_complement_twin.get(degenerate_support[0]),
                degenerate_records=[list(r) for r in degenerate_support],
                # THE BLOCKER IS PER CAUSE, because a wrong blocker sends the next person to
                # the wrong experiment. A cross-lane form needs a probe with per-lane variation;
                # a narrow shift form needs the disputed source/lifetime field authored
                # explicitly. Naming one blocker for both would have pointed the lane-variation
                # cases at an encoder they do not need.
                status=['encoding structurally known',
                        'candidate semantics known from table and family',
                        'current execution probe DEGENERATE',
                        ('blocked on a probe carrying per-lane variation'
                         if (_universe().get('op%d' % int(key.split('/')[0])) or {}
                             ).get('name', '').startswith(('simd.', 'quad.'))
                         else 'blocked on explicit source-mode authoring')],
                means=('an opcode whose name cannot describe the same function returned the same '
                       'values on every observation, so this measurement cannot choose between '
                       'them. Not unknown - the encoding and the candidate family are known and '
                       'the probe is the thing that failed'))
            continue
        promotable[key] = dict(function=name, records=strong)
    # A FORM THAT NEVER REACHES THE GATE CANNOT BE REPORTED BY IT, and widening the degeneracy
    # filters upstream is exactly what stops it reaching. The four forms the complement gate was
    # blocking - quad.fmax/fmin.f16 and simd.fmax/fmin.f16 - all declare 16-bit destinations and
    # all return an input column at that width, so the new filter removes them from `forms`
    # before the loop above runs. The gate then had nothing to block and the artifact lost the
    # four-line status root asked for: a form that went from "blocked, and here is the
    # experiment that would unblock it" to absent.
    #
    # So the disqualification report is built HERE, from the rows, and names every cause it finds
    # rather than the first one a control-flow path happened to test. A form disqualified for two
    # independent reasons now says both.
    disqualified = {}
    for row in rows:
        if row['verdict'] != 'unique':
            continue
        unique_names = {v[0] for v in row['survivors'].values() if len(v) == 1}
        if len(unique_names) != 1:
            continue
        competitors = COMPETITORS.get((row['arity'], _kind_of(row)), 0)
        if (row['cases'] < MIN_CASES or (row['runs'] or 0) < MIN_RUNS
                or competitors < MIN_COMPETITORS):
            continue
        # A CAUSE CAN BE ANSWERED WITHOUT THE OLD RECORDS CHANGING, and nothing here could say
        # so. The lane-varying probe makes op16860 `simd.fmax.f16` return 0x3C1F while op16868
        # `simd.fmin.f16` returns 0x3C00 at the same inputs - the observation this whole
        # cross-lane class was blocked on - but the original uniform-lane records still pair, and
        # the new records fit no candidate because the library holds no cross-lane function. So
        # without this the forms would keep reading "blocked on a probe carrying per-lane
        # variation" after that probe had been built and run.
        causes, answered = [], []
        if (row['stem'], row['id']) in _degenerate_records:
            cause = ('a complementary opcode returned identical values (%s)'
                     % _complement_twin.get((row['stem'], row['id'])))
            if row['op'] in _separated_ops:
                answered.append(cause + ' - ANSWERED: %s' % _separated_ops[row['op']])
            else:
                causes.append(cause)
        identity_cause = ('the output is an input column at full width'
                          if row['outputs_equal_an_input_column'] else
                          'the output is an input column at the declared %d-bit destination'
                          % row['destination_width']
                          if row['outputs_equal_an_input_column_at_the_destination_width']
                          else None)
        _answer = _lane_answered.get((row['op'], row['destination_width']))
        if identity_cause and _answer:
            answered.append(identity_cause + ' - ANSWERED: %s' % _answer)
        elif identity_cause:
            causes.append(identity_cause)
        if row['outputs_constant_across_cases']:
            causes.append('the output is constant across every case')
        elif row['outputs_constant_at_the_destination_width']:
            causes.append('the output is constant at the declared %d-bit destination'
                          % row['destination_width'])
        if not causes and not answered:
            continue
        opcode_name = (_universe().get('op%d' % row['op']) or {}).get('name') or ''
        for width in row['widths']:
            key = '%d/%d' % (row['op'], width)
            if key in promotable:
                continue
            entry = disqualified.setdefault(key, dict(
                function=sorted(unique_names)[0], opcode_name=opcode_name,
                causes=[], causes_answered_by_a_later_observation=[], records=[],
                # THE BLOCKER IS PER CAUSE, because a wrong blocker sends the next person to the
                # wrong experiment. A cross-lane form needs a probe carrying per-lane variation;
                # a narrow form needs the disputed source-mode field authored explicitly; a form
                # whose output is its own input needs inputs the operation can actually change.
                status=['encoding structurally known',
                        'candidate semantics known from table and family',
                        'current execution probe DEGENERATE',
                        ('blocked on a probe carrying per-lane variation'
                         if opcode_name.startswith(('simd.', 'quad.'))
                         else 'blocked on explicit source-mode authoring')]))
            entry['records'].append([row['stem'], row['id']])
            for cause in causes:
                if cause not in entry['causes']:
                    entry['causes'].append(cause)
            for cause in answered:
                if cause not in entry['causes_answered_by_a_later_observation']:
                    entry['causes_answered_by_a_later_observation'].append(cause)
    for entry in disqualified.values():
        if entry['causes_answered_by_a_later_observation'] and not entry['causes']:
            # every cause answered: the probe that was missing exists now
            # THE FOURTH LINE MUST NOT ASSERT SOMETHING FALSE ABOUT THE FORM. It read "blocked
            # on the candidate library, which holds no CROSS-LANE function" for every fully
            # answered form - true of the simd and quad reductions it was written for, and false
            # of op9754 `csel`, which is not a cross-lane instruction at all. A status line is a
            # claim.
            _cross = entry['opcode_name'].startswith(('simd.', 'quad.'))
            entry['status'] = [
                'encoding structurally known',
                'candidate semantics known from table and family',
                'a later probe DISTINGUISHES this form from the degenerate reading',
                'not blocked on a probe any more - blocked on the candidate library, which holds '
                + ('no cross-lane function' if _cross else
                   'no function of the operands this form actually reads')
                + ' for the census to eliminate against']
        elif entry['causes_answered_by_a_later_observation']:
            entry['status'][3] = (entry['status'][3]
                                  + '; the complement cause is answered but another remains')
    blocked_by_complement = {k: v for k, v in disqualified.items()
                             if any('complementary opcode' in c for c in v['causes'])}

    # WHAT THE CHOSEN INPUTS CANNOT SEPARATE, aggregated. An ambiguous record is not a failure of
    # the library, it is a statement that these four cases do not distinguish the survivors - and
    # the pairs that survive TOGETHER most often are precisely what a future probe should be
    # designed to split. Reported because "record what could not be reached" is the point.
    inseparable = collections.Counter()
    for row in rows:
        for names in row['survivors'].values():
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    inseparable[' / '.join(sorted((names[i], names[j])))] += 1
    dupes = duplicate_candidates()
    doc = dict(candidate_library={str(k): {kind: sorted(lib) for kind, lib in v.items()}
                                   for k, v in LIBRARY.items()},
                library_duplicate_groups={str(k): v for k, v in dupes.items()},
                library_duplicates_at_a_declared_width=(
                    duplicate_candidates_at_a_declared_width()),
                library_dedup_probe=dict(seed=DEDUP_SEED, samples=400,
                                         values=['0x%08X' % v for v in DEDUP_PROBE]),
                inseparable_pairs=dict(sorted(inseparable.most_common(40))),
                # A TRUNCATED LIST READ AS AN ABSENCE, and it nearly cost two of root's three
                # named pairs. `bitwise_A / umin` and `bitwise_C / rotl` were absent from the 40
                # published rows and I read that as "separated" - they are still co-surviving at
                # 19 and 21 records and had simply fallen BELOW the cap as other pairs grew, since
                # the smallest published count is 49. This is the same shape as the 10-row cap
                # that deleted three withdrawn peer rows earlier: a cap is not a filter a reader
                # can see unless the omission is counted.
                cross_lane_forms_this_harness_cannot_reach=(
                    cross_lane_forms_this_harness_cannot_reach(_universe())),
                constant_output_causes=constant_output_causes(rows, _universe()),
                the_input_that_would_split_it=the_input_that_would_split_it(rows),
                named_pairs_per_form=named_pairs_per_form(rows),
                inseparable_pairs_cap=dict(
                    published=min(40, len(inseparable)), distinct_pairs=len(inseparable),
                    omitted=max(0, len(inseparable) - 40),
                    smallest_published=(min(dict(inseparable.most_common(40)).values())
                                        if inseparable else 0),
                    named_pairs_checked_in_full={
                        pair: inseparable.get(pair, 0) for pair in (
                            'abs_s32 / identity', 'bitwise_A / umin', 'bitwise_C / rotl')},
                    means=('the published list is the TOP 40 of `distinct_pairs`, so a pair '
                           'missing from it may be separated OR may merely rank below '
                           '`smallest_published`. Anything below the cap is invisible, which is '
                           'why the pairs root named are looked up IN FULL here regardless of '
                           'rank - a reader asking about a specific pair must not have to infer '
                           'its absence from a ranking')),
                bar=dict(min_cases=MIN_CASES, min_runs=MIN_RUNS,
                         min_competing_candidates=MIN_COMPETITORS,
                         competitors_per_arity_and_interpretation={
                             '%d/%s' % k: v for k, v in sorted(COMPETITORS.items())},
                         why=('a unique survivor of three candidates eliminated two, and of '
                              'twenty-eight eliminated twenty-seven; the count of competitors is '
                              'part of the claim, so it is carried per record')),
                forms_promotable=promotable,
                forms_blocked_by_a_complementary_opcode=dict(sorted(
                    blocked_by_complement.items())),
                forms_a_degenerate_probe_disqualified=dict(sorted(disqualified.items())),
                forms_a_degenerate_probe_disqualified_means=(
                    'forms where a record MEETING THE BAR fitted exactly one candidate and the '
                    'measurement is degenerate anyway, with every cause named rather than the '
                    'first one a code path tested. Not unknowns: the encoding and the candidate '
                    'family are known and the PROBE is what failed, so each carries the blocker '
                    'that would lift it. Built from the rows rather than from the promotion loop, '
                    'because a form the degeneracy filters remove never reaches that loop and '
                    'stopped being reported at all when they were tightened'),
                forms_where_records_refute_the_fit=dict(sorted(conflicts.items())),
                forms_rejected_by_the_bar=dict(sorted(rejected.items())),
                scope=('uniqueness holds ONLY within the library above, at the operand and '
                       'modifier configuration each record was dispatched with; a function of '
                       'another arity, reading the operands differently, or depending on a '
                       'modifier is not excluded'),
                schedclass_predicts_interpretation=schedclass_predicts_interpretation(
                    rows, _contract()),
                fit_margins=fit_margins(rows),
                fit_margin_distribution=dict(sorted(collections.Counter(
                    v['margin_best'] for v in fit_margins(rows).values()).items())),
                fit_margin_distribution_worst=dict(sorted(collections.Counter(
                    v['margin_worst'] for v in fit_margins(rows).values()).items())),
                fit_margin_means=(
                    'how many cases separate each unique fit from its NEAREST rival. A '
                    'margin of 1 means the determination hangs on ONE input - real '
                    'evidence, unlike a degenerate probe which discriminates nothing, but '
                    'far weaker than "unique survivor of 26 candidates" sounds. Read it '
                    'beside the competitor count, never instead of it'),
                family_laws=family_laws(rows, _universe()),
                records_that_add_no_information=records_that_add_no_information(rows),
                csel_boolean_comparisons=csel_boolean_comparisons(rows, _universe()),
                csel_boolean_comparisons_means=(
                    'conditional-select opcodes whose output is a two-valued BOOLEAN, with the '
                    'predicate eliminated against it. The branch-pattern model was the wrong model '
                    'and the else-values said so - they came back as 0 and 1, not a large '
                    'inherited constant - so these are comparison instructions producing a flag '
                    'and the earlier a/b/K classification was an artifact of an input set that '
                    'contains 0 and 1 as operand values. Each source is read AT ITS DECLARED '
                    'WIDTH, without which op11492 explains nothing; the true-value is recognised '
                    'rather than assumed, since two opcodes here compute the same predicate and '
                    'encode the answer differently'),
                csel_predicate_elimination=csel_predicate_elimination(rows, _universe()),
                csel_predicate_elimination_means=(
                    'which PREDICATE each conditional-select opcode tests, by elimination over '
                    'the branch its records took. The comparison in this family is carried by the '
                    'OPCODE, so determining one means naming a predicate rather than a function - '
                    'and a record exercising both branches has already measured that predicate at '
                    'every input it carries. Says nothing about what the two branches ARE, only '
                    'when each is taken; the else-value is an operand these records do not write '
                    'and is identified by exclusion. A tie is published as a tie, a form whose '
                    'records give different answers is published as DISAGREEING rather than '
                    'resolved by preference, and `true_at_fewest_cases` is the margin: a '
                    'predicate pinned at ONE input means no other library predicate is true at '
                    'exactly that position, which is elimination but thin, and reads identically '
                    'to one pinned at five unless the number is beside it'),
                the_condition_is_selected_by_the_opcode=the_condition_is_selected_by_the_opcode(
                    rows, _universe()),
                the_condition_is_selected_by_the_opcode_means=(
                    'csel-family opcodes of the SAME declared shape, dispatched on the SAME input '
                    'list in the SAME batch, grouped with how many distinct answers they gave. '
                    'Operand 2 has at most ONE recovered bit on every one of the 232 csel, clamp '
                    'and fselect opcodes in the table, which reads as a decode-side gap until '
                    'these groups are looked at: six opcodes returning five different vectors '
                    'means the comparison is enumerated into the OPCODE and there is no multi-bit '
                    'condition field to recover. Produced by GROUPING RETAINED VALUES, with no '
                    'dispatch. Opcodes that agree with another inside a group are named, not '
                    'counted: identical vectors either mean the same condition or inputs that do '
                    'not separate two conditions, and this instrument cannot tell which'),
                transcendental_accuracy=transcendental_accuracy(rows, _universe()),
                peer_reported_conventions=_peer_reported_conventions(),
                transcendental_accuracy_means=(
                    'distance in units in the last place between each approximated form and the '
                    'correctly-rounded value of the operation its own name states, with the '
                    'OVERFLOW cases counted apart - at the boundary the largest finite value and '
                    'infinity are ADJACENT encodings, so a saturating opcode scores one ULP '
                    'against an IEEE reference and would read as a rounding difference. FEEDS NO '
                    'DENOMINATOR: D3 comes from exact-match elimination, and a form agreeing to '
                    'one ULP has not satisfied that instrument. No tolerance is applied anywhere '
                    'in this field; the distance is reported and the reader draws the line'),
                probe_degeneracy=probe_degeneracy(rows, _universe(), _values_by_id()),
                three_evidence_dimensions=three_evidence_dimensions(rows, _universe()),
                three_evidence_dimensions_means=(
                    'per determined form, the three dimensions kept APART: hardware semantics '
                    '(the surviving candidate), operand-class agreement (the widths the candidate '
                    'was MEASURED to read against the widths Apple DECLARES), and Apple-name '
                    'agreement (scored against a declared correspondence table, never a substring '
                    'test - an unlisted function reads unknown_correspondence rather than a '
                    'guess). A form holding all three is much stronger than one merely named by '
                    'the table; collapsing them into one verdict would hide which instrument '
                    'produced it, the same reason the three denominators are never summed'),
                shapes_the_library_cannot_express=shapes_the_library_cannot_express(rows),
                shapes_means=(
                    'no-fit records grouped by DECLARED operand widths, with whether any '
                    'candidate of that arity reads those widths. Measured 2026-09-18: of 259 '
                    'non-degenerate no-fit records, only SEVEN sit in a shape no candidate can '
                    'read - so operand width is nearly exhausted as an explanation and the '
                    'remaining holes are elsewhere. A shape being expressible does NOT mean the '
                    'opcode is understood; it means the failure to fit is not about widths'),
                asked_outside_the_domain=dict(sorted(asked_outside_the_domain(rows).items())),
                asked_outside_the_domain_by_destination=_by_destination(
                    asked_outside_the_domain(rows)),
                asked_outside_the_domain_means=(
                    'CANDIDATE SET, not findings: constant-output records whose constant is '
                    'absorbing (all-zero or all-ones) and at least one non-register operand was '
                    'never stated, so it holds the authoring witness\'s value. This is op612\'s '
                    'signature - a range predicate probed entirely outside its window, recorded '
                    'as an unreachable, then determined in one batch once the window was written. '
                    'Each entry is worth one dispatch with an unwritten immediate moved; a '
                    'constant output, a dead instrument and a correct answer outside the domain '
                    'all produce the same vector, and moving the immediate is what separates '
                    'them. Operand 1 is excluded from `unwritten_immediates` because every opcode '
                    'here has it free, and lifetime carriers because a record cannot state one. '
                    'READ THE BY-DESTINATION SPLIT BEFORE PROBING ANY OF IT: a record writing '
                    'FLAGR can return a robust zero because the flag file is not visible to '
                    'the GPR store this harness reads back with, which is a competing cause '
                    'and not this one, so those entries are not the cheap dispatches the '
                    'others are. A THIRD COMPETING CAUSE, AND ITS FIRST VERSION WAS WRONG. '
                    'I recorded here, hours ago, that the TensorOps lane had measured an '
                    'accelerator result a following instruction reads correctly while the store '
                    'of it is wrong - a store-visibility hazard. THAT FINDING IS RETRACTED by the '
                    'lane that made it (sections 114-115): a spacing sweep of 0 to 32 intervening '
                    'instructions gave a bit-identical error at every value, which rules out '
                    'instruction timing, and the real variable was their DISPATCH CALL\'s '
                    'completion-scope argument. Too small a value silently under-covers the '
                    'output buffer and every byte past it reads back wrong or zero with no error '
                    'returned; they measured the threshold exactly. So there is no store-'
                    'visibility hazard to inherit, and the confound worth naming is the other '
                    'shape: A DISPATCH PARAMETER MASQUERADING AS AN INSTRUCTION-LEVEL EFFECT. '
                    'Every value in this census is read back from that output buffer, so a record '
                    'reading past the dispatch\'s scope would return fill or stale data and '
                    'present as a dead instruction - which is what this population looks like. '
                    'MEASURED, not assumed: the highest word any record here reads is 68 and the '
                    'canary sits at 6, against a lower bound of 1024 words, so nothing in this '
                    'artifact is exposed. spike/accel/re/oracle.py asserts it per record now, '
                    'because that margin held by luck rather than by construction'),
                summary=dict(sorted(summary.items())), records=len(rows),
                forms_uniquely_fitted=dict(sorted(forms.items())), rows=rows)
    # attached AFTER the document is built, so the table is checked against what was actually
    # published rather than against what I remember publishing
    doc['the_control_gate'] = the_control_gate(
        rows, promotable, doc['family_laws']['laws'])
    # ATTACHED AFTER THE DOCUMENT IS BUILT, like the other audits, so it reads the finished
    # family-law and transcendental sections rather than rebuilding them.
    doc['no_fit_rows_accounted'] = no_fit_rows_accounted(
        rows, doc['family_laws']['laws'], doc.get('transcendental_accuracy') or {},
        doc.get('csel_boolean_comparisons') or {})
    doc['instrument_failure_modes'] = instrument_failure_modes(doc)
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true', help='write isa/g17-execution-fits.json')
    ap.add_argument('--op', type=int, help='show one opcode with its surviving candidates')
    ap.add_argument('--check', action='store_true',
                    help='compare the written artifact against a fresh census')
    ap.add_argument('--check-committed', action='store_true',
                    help='compare HEAD\'s artifact against a fresh census')
    args = ap.parse_args()
    doc = census()
    if args.op:
        for row in doc['rows']:
            if row['op'] == args.op:
                print(json.dumps(row, indent=1))
        return 0
    path = ISA / 'g17-execution-fits.json'
    fresh = json.dumps(doc, indent=1, sort_keys=True) + '\n'
    if args.write:
        path.write_text(fresh)
        print('wrote %s' % path.relative_to(ROOT))
    # THIS ARTIFACT HAD NO FRESHNESS CHECK AT ALL, AND IT IS THE ONE THE GUARDS READ.
    # `test_g17askedoutsidethedomain` opens isa/g17-execution-fits.json from disk and asserts
    # against its fields - forty-odd guards, every one of them about a stored snapshot that
    # nothing proved was what this code produces. The map layer had `--check` for its own
    # artifacts and this layer had nothing, so a census change that moved a number the tests
    # assert would pass until someone happened to regenerate. Two modes rather than one, for the
    # reason `--check` alone was not enough next door: the working tree and the commit are
    # different questions and only the commit is what merging publishes.
    if args.check or args.check_committed:
        if args.check:
            if not path.exists():
                raise SystemExit('%s is absent' % path.relative_to(ROOT))
            if path.read_text() != fresh:
                raise SystemExit('%s differs from a fresh census - run --write'
                                 % path.relative_to(ROOT))
            print('regenerable: %s matches a fresh census' % path.relative_to(ROOT))
        if args.check_committed:
            import subprocess
            rel = path.relative_to(ROOT).as_posix()
            proc = subprocess.run(['git', '-C', str(ROOT), 'show', 'HEAD:' + rel],
                                  capture_output=True)
            if proc.returncode != 0:
                print('not asked of HEAD (untracked there, or not a checkout): %s' % rel)
            elif proc.stdout.decode() != fresh:
                raise SystemExit('the COMMITTED %s differs from a fresh census, so the guards '
                                 'that read it are asserting against a stale snapshot - commit '
                                 'the regeneration' % rel)
            else:
                print('committed: %s at HEAD matches a fresh census' % rel)
        return 0
    print('records considered      %d' % doc['records'])
    for key, count in doc['summary'].items():
        print('  %-34s %d' % (key, count))
    print('forms with ONE surviving candidate, outputs neither an input column nor constant: %d'
          % len(doc['forms_uniquely_fitted']))
    print('  of those, meeting the bar (%d cases, %d runs, %d competitors): %d'
          % (doc['bar']['min_cases'], doc['bar']['min_runs'],
             doc['bar']['min_competing_candidates'], len(doc['forms_promotable'])))
    # THE DISCRIMINATING DENOMINATOR BESIDE THE RAW ONE. "N promotable" reads as N forms each
    # separated from its nearest rival; margin 1 means one input separates it, so one flaky
    # value flips the fit. Both ends, because a form whose BEST record has margin 1 hangs on one
    # case outright, while worst-margin 1 only says some record of it is thin.
    _margins = doc.get('fit_margins') or {}
    for end in ('best', 'worst'):
        _dist = collections.Counter(
            (_margins.get(k) or {}).get('margin_' + end) for k in doc['forms_promotable'])
        print('    margin (%s record) 1: %d   2-3: %d   4+: %d   none: %d'
              % (end, _dist.get(1, 0), _dist.get(2, 0) + _dist.get(3, 0),
                 sum(c for m, c in _dist.items() if m is not None and m >= 4),
                 _dist.get(None, 0)))
    for key, count in doc['forms_rejected_by_the_bar'].items():
        print('  rejected, %-32s %d' % (key, count))
    return 0


if __name__ == '__main__':
    sys.exit(main())
