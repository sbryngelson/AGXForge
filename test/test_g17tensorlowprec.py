#!/usr/bin/env python3
"""Set A item 9: fp8 quantize-out (op13618 packs, op17202 word stores) and software microscaling
(post-MMA fp32 block scaling) in tlower, their references, and the byte identity of every default
body (compile only; machine model 25.105)."""
import hashlib
import os
import struct
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from agxforge.g17 import epienc, model, tlower  # noqa: E402

HALF = struct.unpack("<I", struct.pack("<f", 0.5))[0]
MX = ("mx", 0, 32 * 64, 1, 64 * 32)            # tables right after a 32x64 A and a 64x32 B (one byte each)


def opcodes(body):
    return [i.opcode.id for i in model.decode(body, 0) if i.opcode]


def default_configs():
    """The configurations whose bodies origin/main (cbc8e19b) produced, refusals included."""
    for (M, N, K) in ((16, 16, 16), (32, 32, 64), (48, 32, 64), (64, 32, 64), (32, 48, 96), (17, 19, 16), (64, 64, 128)):
        for a, b in (("half", "half"), ("bfloat", "bfloat"), ("fp8e4m3", "fp8e5m2"), ("float", "half")):
            for ep in ((), (("scale", HALF),), (("bias", 1, 0), ("scale", HALF), ("relu",))):
                for grid in (1, 2):
                    yield (M, N, K, K, N, N), dict(a_type=a, b_type=b, epilogue=ep, grid=grid)
    for acc in (False, True):
        yield (32, 32, 64, 64, 32, 32), dict(accumulate=acc)
    yield (32, 32, 64, 64, 32, 32), dict(a_type="int8", b_type="int8")
    yield (32, 32, 32, 32, 32, 32), dict(a_type="float", b_type="float", split_fp32=True)


def digest(configs):
    h = hashlib.sha256()
    n = 0
    for args, kwargs in configs:
        try:
            body = tlower.lower(*args, **kwargs)[0]
        except ValueError as e:
            body = b"REFUSED:" + str(e).encode()
        h.update(len(body).to_bytes(4, "little") + body)
        n += 1
    return n, h.hexdigest()


class Encoders(unittest.TestCase):
    """Each encoder reproduces Apple's own instruction from the OS compiler's lowering of
    air.convert to fp8 and two word stores (tools/tensorops-model/fp8_store_witness.py)."""

    def test_apple_pack_and_store_are_reproduced_byte_for_byte(self):
        self.assertEqual(epienc.fp8pack(4, "H", 6, 7, "e4m3fn").hex(), "2700006b2500ad1290836000")
        self.assertEqual(epienc.fp8pack(1, "H", 2, 3, "e4m3fn").hex(), "3700002b2500ad0290816000")
        self.assertEqual(epienc.fp8pack(4, "H", 6, 7, "e5m2").hex(), "2700006b2500ad1290832040")
        self.assertEqual(epienc.fp8pack(1, "H", 2, 3, "e5m2").hex(), "3700002b2500ad0290812040")
        self.assertEqual(epienc.store_word(4, 2, 1, True, True).hex(), "47040304210e1040")
        self.assertEqual(epienc.store_word(1, 0, 1, True, True).hex(), "17040300210e1040")

    def test_the_formats_are_different_bytes(self):
        # the control for the format templates: were they the same, an e5m2 program would be e4m3
        self.assertNotEqual(epienc.fp8pack(9, "L", 10, 11, "e4m3fn"), epienc.fp8pack(9, "L", 10, 11, "e5m2"))

    def test_apple_low_half_differs_only_by_its_wait_bit(self):
        # Apple's R4L pack waits on its load's scoreboard slot 7 (word 1<<31, byte 0 bit 3); ours waits on nothing
        ours, apple = epienc.fp8pack(4, "L", 4, 5, "e4m3fn"), bytes.fromhex("2f00004a2500ad1290826000")
        self.assertEqual(bytes(x ^ y for x, y in zip(ours, apple)), bytes([8] + [0] * 11))

    def test_unreadable_registers_refuse(self):
        with self.assertRaises(Exception):
            epienc.fp8pack(200, "L", 1, 2, "e4m3fn")
        with self.assertRaises(KeyError):
            epienc.fp8pack(4, "L", 1, 2, "e4m3")


class DefaultBodies(unittest.TestCase):
    # computed from origin/main cbc8e19b's own tlower (git archive, same configurations).
    # AUDIT (c5b157635, M6's fuzzer): the digest also hashes REFUSAL TEXT, and the one capacity refusal among these
    # configurations ("no register plan for 2x2 tiles in 126 registers") gained the "refused: " prefix. All 127
    # compiled bodies are byte-identical; with the old wording substituted back the digest is main's 17dc0e03... again,
    # so no bytes changed and no hardware receipt is needed.
    MAIN = (172, "a499ed31d99407727346ffc865e0902589c1d12f725a44daa20b0dfbf661d8ee")

    def test_every_default_body_is_byte_identical_to_main(self):
        self.assertEqual(digest(default_configs()), self.MAIN)

    def test_the_digest_sees_the_new_steps(self):
        # the failing control: adding the fp8 step to one configuration must move the digest
        cfgs = list(default_configs())
        args, kw = cfgs[30]
        cfgs[30] = (args, dict(kw, epilogue=tuple(kw["epilogue"]) + (("fp8", "e4m3fn"),)))
        self.assertNotEqual(digest(cfgs), self.MAIN)

    def test_documented_generic_programs_keep_their_hashes(self):
        # docs/archive/g17-tensor-generic.md: both ran bit-exact on hardware with these bytes
        import g17tensorcommonruntime as R
        for spec, want in ((dict(M=64, N=32, K=64, threadgroups=2),
                            "7157eefc3cc7c692d91731ced55b3d3f7e876579cb9127b4f70db7faa83f57e4"),
                           (dict(M=48, N=32, K=64, a="bfloat", b="bfloat", epilogue=["scale:0x3f400000", "relu"]),
                            "da97deebbf4103e5101af84fc3f85095fc9c892f00be16ce86784f7e38112ea2")):
            self.assertEqual(hashlib.sha256(R.build_generic_program(spec).code).hexdigest(), want, spec)


class Fp8Out(unittest.TestCase):
    def test_four_packs_and_two_word_stores_per_tile_replace_the_tensor_store(self):
        base = opcodes(tlower.lower(32, 32, 64, 64, 32, 32)[0])
        body = opcodes(tlower.lower(32, 32, 64, 64, 32, 32, epilogue=(("fp8", "e5m2"),))[0])
        self.assertEqual(body.count(13618), 16)
        self.assertEqual(body.count(17202), 8)
        self.assertEqual(body.count(17257), 0)
        self.assertEqual(base.count(17257), 8)

    def test_the_stores_address_bytes_row_major(self):
        # word offsets (C_OFF + (16 mi + 8 p) ldc + 16 ni) / 4 at ldc 32: 0, 64, 4, 68, 128, 192, 132, 196
        body = tlower.lower(32, 32, 64, 64, 32, 32, epilogue=(("fp8", "e4m3fn"),))[0]
        imms = [[v for k, v in i.values if k == "imm"][-1] for i in model.decode(body, 0)
                if i.opcode and i.opcode.id == 11842]
        self.assertEqual(imms, [64, 4, 68, 128, 192, 132, 196])

    def test_misplaced_or_unstorable_fp8_steps_refuse(self):
        for kwargs in (dict(epilogue=(("fp8", "e4m3fn"), ("relu",))), dict(epilogue=(("fp8", "e4m3"),)),
                       dict(epilogue=(("fp8", "e4m3fn"),), keep=True),
                       dict(epilogue=(("fp8", "e4m3fn"),), offsets=(0, 0, 2))):
            with self.assertRaises(ValueError, msg=kwargs):
                tlower.lower(32, 32, 64, 64, 32, 32, **kwargs)


class Microscaling(unittest.TestCase):
    def test_each_block_scales_every_tile_once(self):
        body = opcodes(tlower.lower(32, 32, 64, 64, 32, 32, a_type="fp8e4m3", b_type="fp8e5m2", epilogue=(MX,))[0])
        tiles, blocks = 4, 2
        self.assertEqual(body.count(5107), tiles * blocks)            # every block starts with the no-C form
        self.assertEqual(body.count(5106), tiles * blocks)
        self.assertEqual(body.count(3290), tiles * 8 * 2 * blocks)    # T * SA, then * SB
        self.assertEqual(body.count(998), tiles * (blocks - 1) * 8 + tiles * blocks)   # D += T, plus one wait per tile-block

    def test_refusals(self):
        for args, kwargs in (((32, 32, 48, 48, 32, 32), dict(epilogue=(MX,))),                          # K not whole blocks
                             ((32, 32, 64, 64, 32, 32), dict(a_type="int8", b_type="int8", epilogue=(MX,))),
                             ((32, 32, 64, 64, 32, 32), dict(a_type="float", b_type="float", epilogue=(MX,))),
                             ((32, 32, 64, 64, 32, 32), dict(accumulate=True, epilogue=(MX,))),
                             ((32, 32, 64, 64, 32, 32), dict(epilogue=(("relu",), MX)))):               # not first
            with self.assertRaises(ValueError, msg=kwargs):
                tlower.lower(*args, **kwargs)

    def test_the_parallel_branches_are_refused_together(self):
        # the K loop, register-fed B and the column reduction were built on other branches and never
        # composed with the scale step or the byte store
        for kw in (dict(epilogue=(MX,), kloop=True), dict(epilogue=(MX,), reduce=("col", "sum")),
                   dict(epilogue=(("fp8", "e4m3fn"),), reduce=("col", "sum"))):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                tlower.lower(32, 32, 64, 64, 32, 32, **kw)


class References(unittest.TestCase):
    def test_fp8_quantize_is_rne_without_saturation(self):
        import g17tensorcommonruntime as R
        x = np.array([464, 464.01, -500, 448, 61439, 61440, -7e4, np.nan, -np.nan, -0.0, 2 ** -9, 1.5 * 2 ** -10], np.float32)
        # a NEGATIVE overflow keeps its sign (-500 and -7e4 -> 0xff): measured on hardware, 471 of 471
        # negative overflows (machine model 25.105.3); only NaN inputs become 0x7f
        self.assertEqual(R.fp8_quantize(x, "fp8e4m3").tolist(),
                         [0x7e, 0x7f, 0xff, 0x7e, 0x7f, 0x7f, 0xff, 0x7f, 0x7f, 0x80, 0x01, 0x01])
        self.assertEqual(R.fp8_quantize(x, "fp8e5m2").tolist(),
                         [0x5f, 0x5f, 0xe0, 0x5f, 0x7b, 0x7c, 0xfc, 0x7e, 0x7e, 0x80, 0x18, 0x16])

    def test_a_saturating_pack_would_be_told_apart(self):
        # the control: a saturating model gives the largest finite code where the measured pack gives NaN
        import g17tensorcommonruntime as R
        import ml_dtypes
        x = np.array([500.0, -500.0], np.float32)
        sat = np.clip(x, -448, 448).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
        self.assertFalse(np.array_equal(sat, R.fp8_quantize(x, "fp8e4m3")))

    def test_mx_reference_reads_the_block_axis(self):
        import g17tensorcommonruntime as R
        rng = np.random.default_rng(3)
        a = rng.uniform(-2, 2, (16, 64)).astype(np.float16).astype(np.float32)
        b = rng.uniform(-2, 2, (64, 16)).astype(np.float16).astype(np.float32)
        _, sa = R.mx_scale_table(rng, 2 * 16)
        _, sb = R.mx_scale_table(rng, 2 * 16)
        sa, sb = sa.reshape(2, 16), sb.reshape(2, 16)
        right, shifted = R.mx_post_reference(a, b, sa, sb), R.mx_post_reference(a, b, sa, sb, shift=1)
        self.assertGreater(int((right.view("<u4") != shifted.view("<u4")).sum()), 200)
        # unit scales: the block chain plus one rounding per block, never the one-chain GEMM exactly
        ones = np.ones((2, 16), np.float32)
        self.assertTrue(np.allclose(R.mx_post_reference(a, b, ones, ones), a @ b, rtol=1e-5, atol=1e-5))

    def test_mx_reference_refuses_the_flush_range(self):
        import g17tensorcommonruntime as R
        a = np.full((16, 32), 1.0, np.float32); b = np.full((32, 16), 1.0, np.float32)
        tiny = np.full((1, 16), 2.0 ** -120, np.float32)
        with self.assertRaises(ValueError):
            R.mx_post_reference(a, b, tiny, tiny)


if __name__ == "__main__":
    unittest.main()
