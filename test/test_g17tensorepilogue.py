#!/usr/bin/env python3
"""The register epilogue on a tensor body: what it emits, and where it refuses (compile only)."""
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agxforge.g17 import epienc, model, tlower

HALF = struct.unpack("<I", struct.pack("<f", 0.5))[0]


def opcodes(body):
    return [i.opcode.id for i in model.decode(body, 0) if i.opcode]


class Encoders(unittest.TestCase):
    """Each encoder reproduces cc's own bytes for the same operation, both lifetime variants."""

    def test_cc_instances_are_reproduced_byte_for_byte(self):
        self.assertEqual(epienc.fmul(20, 17, 16).hex(), "210205100280a0321c0810000210")
        self.assertEqual(epienc.fmul(17, 18, 16, keep_b=False).hex(), "310005300280a0221c0820000210")
        self.assertEqual(epienc.fmax(16, 20, 19).hex(), "2200074b2380a022b40991542844")
        self.assertEqual(epienc.fmax(18, 17, 19, keep_b=False).hex(), "2202070b2380a02ab409a1642244")
        self.assertEqual(epienc.movimm(16, 0x3F000000).hex(), "0c80423e60200008")
        self.assertEqual(epienc.movimm(16, 0).hex(), "0c80420060200000")

    def test_a_register_the_decoder_would_not_read_back_is_refused(self):
        with self.assertRaises(Exception):
            epienc.fmul(200, 1, 2)


class Emission(unittest.TestCase):
    def test_each_step_is_eight_instructions_per_tile(self):
        base = opcodes(tlower.lower(32, 32, 64, 64, 32, 32)[0])
        body = opcodes(tlower.lower(32, 32, 64, 64, 32, 32,
                                    epilogue=(("bias", 1, 0), ("scale", HALF), ("relu",)))[0])
        tiles = 4
        self.assertEqual(body.count(3290) - base.count(3290), 8 * tiles)
        self.assertEqual(body.count(9700) - base.count(9700), 8 * tiles)
        self.assertEqual(body.count(998) - base.count(998), 8 * tiles)
        self.assertEqual(body.count(11842) - base.count(11842), 2)   # scale and zero, once per body

    def test_no_epilogue_is_the_old_body(self):
        self.assertEqual(tlower.lower(32, 32, 64, 64, 32, 32)[0],
                         tlower.lower(32, 32, 64, 64, 32, 32, epilogue=())[0])

    def test_partial_tiles_and_integer_accumulators_refuse(self):
        for kwargs in (dict(M=17, N=32), dict(M=32, N=19)):
            with self.assertRaises(ValueError):
                tlower.lower(kwargs["M"], kwargs["N"], 64, 64, kwargs["N"], kwargs["N"], epilogue=(("relu",),))
        with self.assertRaises(ValueError):
            tlower.lower(32, 32, 64, 64, 32, 32, a_type="int8", b_type="int8", epilogue=(("relu",),))

    def test_unknown_steps_refuse(self):
        with self.assertRaises(ValueError):
            tlower.lower(32, 32, 64, 64, 32, 32, epilogue=(("tanh",),))   # gelu is a step since MM 25.109


if __name__ == "__main__":
    unittest.main()
