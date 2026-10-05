"""Bit-exact reference model of the M5 Neural Accelerator MMA (op5106 family) and of matmul2d built on it.

Measured on H17s (docs/g17-tensorops-accelerator-recon.md, sections 4, 5, 6, 12). Everything here is what the
hardware was observed to do; nothing is assumed beyond the measured domain (f16/bf16/f32-truncated/int8
operands, fp32 or int32 accumulation, 16x16x16 tiles, ascending K chunks).

    mma16(A, B, C)              one 16x16x16 issue: exact products, adjacent-pair sums, pairs interleaved by 8,
                                C first, four quads sequentially, RNE32 at every step
    matmul_ref(A, B, C, K)      matmul2d half/bfloat -> float: mma16 over ascending 16-wide K chunks through
                                the rounded fp32 accumulator (mode multiply: C = 0; multiply_accumulate: C given)
    to_f32_operand(x)           the accelerator's fp32 operand: 10 significand bits truncated toward zero,
                                fp32 exponent range, subnormals flushed
    to_half_rne / to_bf16_rne   destination conversions (RNE from the fp32 accumulator)
Self-test: python3 tensorops_model.py  (replays the retained stimulus records against the model).
"""
import json
from fractions import Fraction as Fr
from pathlib import Path
import numpy as np

def rne32(v):
    """round an exact rational to fp32, round-to-nearest-even (normal and subnormal range)"""
    v = Fr(v)
    if v == 0: return Fr(0)
    s = -1 if v < 0 else 1; a = abs(v)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fr(2) ** e > a: e -= 1
    e = max(e, -126)
    q = Fr(2) ** (e - 23); m = a / q; fl = m.numerator // m.denominator; fr = m - fl
    n = fl + (1 if fr > Fr(1, 2) else (fl % 2 if fr == Fr(1, 2) else 0))
    return s * n * q

def mma16(a_row, b_col, c, integer=False):
    """D element from 16 exact products (a_row[k] * b_col[k]) and accumulator c. Values are Fractions or ints."""
    prods = [Fr(a_row[k]) * Fr(b_col[k]) for k in range(16)]
    if integer:
        return ((int(sum(prods)) + int(c) + 2 ** 31) % 2 ** 32) - 2 ** 31          # i32 wraps
    R = rne32
    p = [R(prods[2 * i] + prods[2 * i + 1]) for i in range(8)]
    q = [R(p[j] + p[j + 4]) for j in range(4)]
    acc = Fr(c)
    for j in range(4): acc = R(acc + q[j])
    return acc

def matmul_ref(A, B, C=None, integer=False):
    """A: M x K, B: K x N (exact values as Fractions/floats), C: M x N or None. K multiple of 16. Returns Fractions."""
    M, K = len(A), len(A[0]); N = len(B[0])
    out = [[None] * N for _ in range(M)]
    for r in range(M):
        for c in range(N):
            acc = Fr(C[r][c]) if C is not None else Fr(0)
            for s in range(0, K, 16):
                acc = mma16([Fr(A[r][k]) for k in range(s, s + 16)], [Fr(B[k][c]) for k in range(s, s + 16)], acc, integer)
            out[r][c] = acc
    return out

def to_f32_operand(x):
    """what the accelerator reads from an fp32 operand: the low 13 bits of the bit pattern cleared - 10 explicit
    mantissa bits kept, truncation toward zero, full exponent range; subnormals are treated the same way (their
    bits above mantissa bit 12 survive), NOT flushed - recon section 33 corrects the earlier reading"""
    u = int(np.array([np.float32(x)]).view(np.uint32)[0]) & 0xffffe000
    return Fr(float(np.array([u], np.uint32).view(np.float32)[0]))

def _round_to(v, mant_bits, max_norm, tie_even=True):
    if v == 0: return Fr(0)
    s = -1 if v < 0 else 1; a = abs(v); e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fr(2) ** e > a: e -= 1
    q = Fr(2) ** (e - mant_bits); m = a / q; fl = m.numerator // m.denominator; fr = m - fl
    n = fl + (1 if fr > Fr(1, 2) else (fl % 2 if fr == Fr(1, 2) else 0))
    r = s * n * q
    return float('inf') * s if abs(r) > max_norm else r
def to_half_rne(v):
    v = Fr(v)
    if abs(v) < Fr(1, 2 ** 24): return Fr(0) if abs(v) <= Fr(1, 2 ** 25) else Fr(1, 2 ** 24) * (1 if v > 0 else -1)
    if abs(v) < Fr(1, 2 ** 14): q = Fr(1, 2 ** 24); m = abs(v) / q; fl = m.numerator // m.denominator; fr = m - fl; n = fl + (1 if fr > Fr(1, 2) else (fl % 2 if fr == Fr(1, 2) else 0)); return (1 if v > 0 else -1) * n * q
    return _round_to(v, 10, Fr(65504) + Fr(16) - Fr(1, 2 ** 10))
def to_bf16_rne(v): return _round_to(Fr(v), 7, Fr(2) ** 128)

if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    eps = Fr(1, 2 ** 24); ok = 0; tot = 0
    # arith stimuli on the single MMA (element (0,0), B column of ones)
    from arith import STIMULI
    a = json.loads((here / 'arith-new_f16f16_nn.json').read_text())
    for n, (arow, bcol, c) in STIMULI.items():
        tot += 1; ok += float(mma16(arow, bcol, c)) == a['stimuli'][n]['result']
    # chained kernels
    for name, k in (('new_f16f16_nn_x2', 2), ('new_f16f16_nn_x3', 3)):
        d = json.loads((here / ('chain-%s.json' % name)).read_text())
        _extra = {'chain_tiny_on_one': ([1] + [Fr(1, 2 ** 24)] * 15, [1] * 16, 0), 'chain_halfulp': ([2 ** 12] + [Fr(1, 2 ** 12)] * 15, [1] * 16, 0),
                  'chain_c_tiny': ([Fr(1, 2 ** 24)] * 16, [1] * 16, 1), 'chain_1p15eps_x': ([1] + [Fr(15, 2 ** 24)] + [0] * 14, [1] * 16, 0)}
        for sname, (arow, bcol, c) in {**STIMULI, **_extra}.items():
            acc = Fr(c)
            for _ in range(k): acc = mma16(arow, bcol, acc)
            tot += 1; ok += float(acc) == d['stimuli'][sname]['measured']
    # C placement
    cp = json.loads((here / 'cplacement-new_f16f16_nn.json').read_text())
    for n, t in cp['tests'].items():
        vals = {'c1_q0_3eps_q1_3eps': {0: 3 * eps, 2: 3 * eps}, 'c1_q0_3eps_q2_3eps': {0: 3 * eps, 4: 3 * eps}, 'c1_q3_3eps_q2_3eps': {6: 3 * eps, 4: 3 * eps}, 'c1_q0_3eps_q1_3eps_q2_3eps': {0: 3 * eps, 2: 3 * eps, 4: 3 * eps}}[n]
        arow = [vals.get(k, Fr(0)) for k in range(16)]; tot += 1; ok += float(mma16(arow, [1] * 16, 1)) == t['measured']
    # fp32 operand truncation records
    for name in ('new_f32f32_nn', 'new_f16f32_nn'):
        p = json.loads((here / ('precision-%s.json' % name)).read_text())
        for side, sw in p['sweeps'].items():
            for m, r in sw['rows'].items():
                v = to_f32_operand(r['input']); tot += 1; ok += float(v) == r['output']
    print('model replays %d / %d retained measurements' % (ok, tot))
