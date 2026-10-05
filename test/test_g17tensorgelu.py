#!/usr/bin/env python3
"""The GELU register epilogue (Set A goal item 6): the compiler's memory-stage GELU, x*sigmoid(1.702x)
(tensorreduce.emit_row_gelu), emitted on the accumulators in registers from cc's own instruction
forms. Default bodies are byte-identical to main; the reference is the instruction sequence."""
import collections
import hashlib
import itertools
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import g17tensorcommonruntime as R  # noqa: E402
from agxforge.g17 import cc, epienc, ir, model, tlower  # noqa: E402

GELU_OPS = [3290, 3290, 1272, 998, 3658, 3290]      # fmul, fmul, exp2, fadd, recip, fmul per slot


def memory_stage_program():
    a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function("g", [a, b, c]); builder = ir.Builder(fn, fn.block("entry"))
    builder.tensor_matmul(a, b, c, M=16, N=16, K=64)
    builder.tensor_row_gelu(c, row=0, M=16, N=16, K=64)
    builder.ret()
    return cc.compile_function(fn)


class CcsOwnForms(unittest.TestCase):
    def test_recip_and_fadd_are_the_bytes_cc_emits_for_the_memory_stage(self):
        emitted = collections.defaultdict(set)
        for i in model.decode(memory_stage_program().code, 0):
            if i.opcode and i.opcode.id in (3658, 998):
                emitted[i.opcode.id].add(i.raw.hex())
        # cc's recip R11 <- R10 and fadd R10 <- R11 + R48 (source released, constant kept)
        self.assertIn(epienc.recip(11, 10).hex(), emitted[3658])
        self.assertIn(epienc.fadd(10, 11, 48).hex(), emitted[998])
        self.assertEqual(epienc.recip(11, 10).hex(), "b70004b82a20a10a3000")
        self.assertEqual(epienc.fadd(10, 11, 48).hex(), "8102049a0220a00a64041200")

    def test_six_instructions_per_accumulator_slot(self):
        for M, N in ((16, 16), (32, 32)):
            body, _ = tlower.lower(M, N, 64, 64, N, N, epilogue=(("gelu",),))
            plain, _ = tlower.lower(M, N, 64, 64, N, N)
            ops = collections.Counter(i.opcode.id for i in model.decode(body, 0) if i.opcode)
            base = collections.Counter(i.opcode.id for i in model.decode(plain, 0) if i.opcode)
            slots = M * N // 32
            self.assertEqual(ops[1272] - base[1272], slots)
            self.assertEqual(ops[3658] - base[3658], slots)
            self.assertEqual(ops[3290] - base[3290], 3 * slots)
            self.assertEqual(ops[11842] - base[11842], 3)        # the three constants, once


class DefaultBodiesUnchanged(unittest.TestCase):
    def test_default_bodies_are_mains(self):
        # 120 configurations hashed with origin/main's tlower (bc6fde3b) and with this branch's
        h = hashlib.sha256(); n = 0
        for (M, N, K), a, ep, kw in itertools.product(
                [(16, 16, 32), (32, 32, 64), (48, 32, 64), (64, 64, 64), (16, 48, 96)], ["half", "bfloat"],
                [(), (("relu",),), (("exp2",),), (("scale", 0x3fb8aa3b), ("relu",))], [{}, dict(grid=2), dict(sg=2)]):
            try:
                body, _ = tlower.lower(M * (2 if kw else 1), N, K, K, N, N, a_type=a, b_type=a, epilogue=ep, **kw)
                h.update(body)
            except ValueError as e:
                h.update(str(e).encode())
            n += 1
        self.assertEqual((n, h.hexdigest()[:16]), (120, "1e2f75cdfe173328"))


class TheReference(unittest.TestCase):
    x = np.concatenate([np.linspace(-20, 20, 401), [0.0, -0.0, 1e-30, -1e-30, 88.0, -88.0]]).astype("<f4")

    def test_the_model_is_the_memory_stages_arithmetic(self):
        # the transformer-layer reference's GELU lines (the released memory stage's reference), with
        # its float32 np.exp2 replaced by the exact value rounded once: numpy's float32 exp2 is itself
        # not correctly rounded, which is a property of numpy, not of either program
        scaled = np.asarray(self.x * np.float32(1.702), dtype="<f4")
        with np.errstate(over="ignore"):
            e = np.exp2(np.asarray(scaled * np.float32(-1.4426950408889634), dtype="<f4").astype(np.float64)).astype("<f4")
        mem = np.asarray(self.x * np.asarray(np.float32(1.0) / np.asarray(e + np.float32(1.0), dtype="<f4"), dtype="<f4"), dtype="<f4")
        self.assertEqual(int(R._ulp_distance(R.gelu_model(self.x), mem).max()), 0)

    def test_the_envelope_covers_each_perturbation_and_is_not_zero(self):
        ref = R.gelu_model(self.x).astype(np.float64)
        env = np.zeros_like(ref)
        for de, dr in itertools.product((-1, 0, 1), repeat=2):
            env = np.maximum(env, np.abs(R.gelu_model(self.x, de, dr).astype(np.float64) - ref))
        # the control: a one-ulp recip or exp2 does move some outputs, so the bound is not a free pass
        self.assertGreater(int((env > 0).sum()), 100)
        # and it stays small against the values: at most a few ulps of the output
        big = np.abs(ref) > 1e-3
        self.assertLess(float((env[big] / np.abs(ref[big])).max()), 4 * 2.0 ** -23)


class Refusals(unittest.TestCase):
    def test_spec_rules(self):
        for spec in (dict(M=32, N=32, K=64, epilogue=["gelu", "relu"]),
                     dict(M=32, N=32, K=64, epilogue=["mx32", "gelu"]),
                     dict(M=32, N=32, K=64, epilogue=["gelu"], gelu_stage="memory"),
                     dict(M=16, N=16, K=64, epilogue=["relu", "gelu"], gelu_stage="memory")):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                R.generic_spec(spec)
        R.generic_spec(dict(M=16, N=16, K=64, epilogue=["gelu"], gelu_stage="memory"))      # the control

    def test_the_memory_arm_builds_the_memory_stage_and_the_register_arm_does_not(self):
        reg = R.build_generic_program(R.generic_spec(dict(M=16, N=16, K=64, epilogue=["gelu"]))).code
        mem = R.build_generic_program(R.generic_spec(dict(M=16, N=16, K=64, epilogue=["gelu"], gelu_stage="memory"))).code
        self.assertNotEqual(reg, mem)
        r = collections.Counter(i.opcode.id for i in model.decode(reg, 0) if i.opcode)
        m = collections.Counter(i.opcode.id for i in model.decode(mem, 0) if i.opcode)
        self.assertEqual((r[1272], r[3658]), (8, 8))          # one per accumulator slot: 256 values / 32 lanes
        self.assertEqual(m[1272], m[3658])                     # the memory stage: row by row over the stored tile
        self.assertGreater(m[1272], 8)


class TheModelFlushesSubnormals(unittest.TestCase):
    def test_a_subnormal_recip_becomes_a_signed_zero(self):
        # x near -52: 1/(1 + 2^t) is about 3.6e-39, subnormal; the GPU returned -0 there (57 of 65,536
        # elements of gelu_M512N128K256_g8), as section 138 part 5 measured for fmul and fadd
        import g17tensorcommonruntime as R
        y = R.gelu_model(np.array([-52.0, -51.6, -10.0, 3.0], np.float32))
        self.assertEqual(y[:2].tolist(), [0.0, 0.0])
        self.assertTrue(np.signbit(y[0]) and np.signbit(y[1]))
        self.assertNotEqual(float(y[2]), 0.0)           # the control: a normal result is kept


if __name__ == "__main__":
    unittest.main()
