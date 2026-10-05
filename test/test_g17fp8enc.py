#!/usr/bin/env python3
"""op17642/10 (the fp8 unpack) against Apple's own bytes, and the fp8 GEMM body's shape (compile only).

The sixteen instances are Apple's compilation of an fp8 x fp8 AIR MMA
(`multiply_accumulate.f.f.v8f32.v8f8{e4m3fn,e5m2}...`), eight per format. They are pinned here
as bytes, so the test needs no toolchain.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agxforge.g17 import fp8enc, model, tlower

# (dest, source half, wait slot or None) in Apple's order; the bytes differ per format only in bytes 8-9
APPLE = [(12, "R8L", 7), (13, "R8H", None), (14, "R9L", None), (15, "R9H", None),
         (8, "R10L", 0), (9, "R10H", None), (10, "R11L", None), (11, "R11H", None)]
BYTES = {"e4m3": ["af0000982500ac121002", "b70100982500ac121002"],
         "e5m2": ["af0000982500ac125000", "b70100982500ac125000"]}


class Unpack(unittest.TestCase):
    def test_apples_first_two_of_each_format_are_reproduced(self):
        for fmt, want in BYTES.items():
            for (dest, src, slot), hexb in zip(APPLE, want):
                self.assertEqual(fp8enc.unpack(dest, src, fmt, wait_slot=slot).hex(), hexb, (fmt, dest, src))

    def test_every_instance_decodes_as_asked(self):
        names = model.registers()
        for fmt in ("e4m3", "e5m2"):
            for dest, src, slot in APPLE + [(60, "R41H", 3), (0, "R0L", None)]:
                d = list(model.decode(fp8enc.unpack(dest, src, fmt, wait_slot=slot), 0))[0]
                self.assertEqual([names.get(v) for k, v in d.values if k == "reg"], ["R%d" % dest, src])
                self.assertEqual(d.values[2][1], fp8enc.FORMAT_CODE[fmt])

    def test_a_register_the_field_cannot_name_refuses(self):
        with self.assertRaises(ValueError):
            fp8enc.unpack(200, "R8L")


class Fp8Body(unittest.TestCase):
    def test_four_unpacks_per_fragment_and_a_bf16_mma(self):
        body = tlower.lower(32, 32, 64, 64, 32, 32, a_type="fp8e4m3", b_type="fp8e5m2")[0]
        ins = [i for i in model.decode(body, 0) if i.opcode]
        ops = [i.opcode.id for i in ins]
        self.assertEqual(ops.count(17642), 4 * (2 + 2) * 4)       # 4 K steps x (2 A + 2 B fragments) x 4
        mmas = [i for i in ins if i.opcode.id in (5106, 5107)]
        self.assertEqual(len(mmas), 16)
        self.assertTrue(all(i.values[5][1] == 3 and i.values[8][1] == 3 for i in mmas))   # bf16 type codes

    def test_unmeasured_pairings_refuse(self):
        for a, b, kw in (("fp8e4m3", "half", {}), ("fp8e4m3", "fp8e5m2", {"transA": True})):
            with self.assertRaises(ValueError):
                tlower.lower(32, 32, 64, 64, 32, 32, a_type=a, b_type=b, **kw)


if __name__ == "__main__":
    unittest.main()
