#!/usr/bin/env python3
"""The split-fp32 GEMM (Set A item 10): emission shape, refusals, and what the split buys (compile only).

The accuracy claim is checked in software against the measured arithmetic: the MMA model is
section 136's (exact products, RNE per stage, C first) with the fp32 operand truncation of section
33, applied to the split operands exactly as the emitted ALU sequence computes them.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agxforge.g17 import model, tlower

f32 = np.float32


def trunc(x):
    return (np.asarray(x, np.float32).view(np.uint32) & np.uint32(0xFFFFE000)).view(np.float32)


def split(x):
    """The emitted sequence, operation for operation, each rounded to fp32."""
    x = np.asarray(x, np.float32)
    c = (x * f32(8193.0)).astype(np.float32)
    d = (c + (x * f32(-1.0)).astype(np.float32)).astype(np.float32)
    hi = (c + (d * f32(-1.0)).astype(np.float32)).astype(np.float32)
    lo = (x + (hi * f32(-1.0)).astype(np.float32)).astype(np.float32)
    return hi, lo


def mma(a, b, c):
    """One 16x16x16 issue: products exact, pairs, quads, C first (section 136), operands truncated."""
    a, b = trunc(a).astype(np.float64), trunc(b).astype(np.float64)
    out = np.empty((16, 16), np.float32)
    for r in range(16):
        for col in range(16):
            p = [f32(a[r, 2 * i] * b[2 * i, col] + a[r, 2 * i + 1] * b[2 * i + 1, col]) for i in range(8)]
            q = [f32(p[j] + p[j + 4]) for j in range(4)]
            acc = q[0] if c is None else f32(c[r, col])
            for v in (q[1:] if c is None else q):
                acc = f32(acc + v)
            out[r, col] = acc
    return out


def split_gemm(A, B):
    acc = None
    for k in range(0, A.shape[1], 16):
        ah, al = split(A[:, k:k + 16]); bh, bl = split(B[k:k + 16])
        for a, b in ((ah, bh), (ah, bl), (al, bh)):
            acc = mma(a, b, acc)
    return acc


def plain_gemm(A, B):
    acc = None
    for k in range(0, A.shape[1], 16):
        acc = mma(A[:, k:k + 16], B[k:k + 16], acc)
    return acc


class Emission(unittest.TestCase):
    def test_seven_alu_operations_per_element_and_three_mmas_per_step(self):
        ops = [i.opcode.id for i in model.decode(
            tlower.lower(16, 16, 32, 32, 16, 16, a_type="float", b_type="float", split_fp32=True)[0], 0) if i.opcode]
        steps, fragments = 2, 2
        self.assertEqual(ops.count(3290), steps * fragments * 8 * 4)
        self.assertEqual(ops.count(998), steps * (1 + fragments * 8 * 3))
        mmas = [o for o in ops if 5098 <= o <= 5107]
        self.assertEqual(mmas, [5099] + [5098] * (3 * steps - 1))

    def test_other_types_and_transposes_refuse(self):
        for kw in (dict(a_type="half", b_type="half"), dict(a_type="float", b_type="float", transA=True)):
            with self.assertRaises(ValueError):
                tlower.lower(16, 16, 32, 32, 16, 16, split_fp32=True, **kw)


class WhatTheSplitBuys(unittest.TestCase):
    def test_hi_survives_the_truncation_and_lo_is_the_remainder(self):
        x = np.random.default_rng(3).uniform(-4, 4, 4096).astype(np.float32)
        hi, lo = split(x)
        self.assertTrue(np.array_equal(trunc(hi), hi))
        self.assertTrue(np.array_equal((hi + lo).astype(np.float32), x))

    def test_error_falls_from_2e_minus_11_to_about_2e_minus_20(self):
        rng = np.random.default_rng(5)
        A = rng.uniform(-1, 1, (16, 32)).astype(np.float32); B = rng.uniform(-1, 1, (32, 16)).astype(np.float32)
        exact = A.astype(np.float64) @ B.astype(np.float64)
        scale = np.abs(A).astype(np.float64) @ np.abs(B).astype(np.float64)
        plain = np.max(np.abs(plain_gemm(A, B) - exact) / scale)
        split_err = np.max(np.abs(split_gemm(A, B) - exact) / scale)
        self.assertGreater(plain, 2.0 ** -14)
        self.assertLess(split_err, 2.0 ** -19)


if __name__ == "__main__":
    unittest.main()
