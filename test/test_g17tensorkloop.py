#!/usr/bin/env python3
"""tlower's runtime K loop (performance items 2 and 5): the default body is unchanged, the loop body
is under the 16 KiB loop-body cliff, its latch is this repository's executed counted-loop shape, and
the static check refuses a body whose counter is written by anything but its own increment."""
import hashlib
import itertools
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agxforge.g17 import asm, model, tlower  # noqa: E402

# 81 default bodies hashed on origin/main before the loop existed (4cb3da1f)
DEFAULT_SWEEP_SHA256 = "e46e69b271627be7f9be77c3dd0c7a4c69941de789a803a3e12de59bb0b779a9"
CLIFF = 16 * 1024


def _ins(body):
    out, off = [], 0
    for i in model.decode(body, 0):
        out.append((off, i)); off += len(i.raw)
    return out


class Default(unittest.TestCase):
    def test_the_default_body_is_byte_identical_to_main(self):
        h = hashlib.sha256()
        for M, N, K in itertools.product((16, 32, 64), (16, 32, 128), (32, 64, 256)):
            for a, b in (('half', 'half'), ('float', 'half'), ('int8', 'int8')):
                try:
                    body = tlower.lower(M, N, K, K, N, N, a_type=a, b_type=b)[0]
                except Exception as e:
                    body = ('ERR ' + type(e).__name__ + str(e)).encode()
                h.update(body)
        self.assertEqual(h.hexdigest(), DEFAULT_SWEEP_SHA256)


class Loop(unittest.TestCase):
    def test_the_loop_body_fits_under_the_cliff_where_unrolling_does_not(self):
        unrolled = tlower.lower(64, 128, 256, 256, 128, 128)[0]
        looped = tlower.lower(64, 128, 256, 256, 128, 128, kloop=True)[0]
        self.assertGreater(len(unrolled), CLIFF)
        self.assertLess(len(looped), CLIFF)
        self.assertLess(len(tlower.lower(64, 128, 1024, 1024, 128, 128, kloop=True)[0]), CLIFF)

    def test_the_latch_is_the_executed_counted_loop_shape(self):
        body = tlower.lower(32, 32, 64, 64, 32, 32, kloop=True)[0]
        ins = _ins(body)
        ops = [i.opcode.id for _o, i in ins if i.opcode]
        j = ops.index(458)
        self.assertEqual(ops[j - 4:j + 2], [10279, 577, 10369, 582, 458, 577])
        at, br = next((o, i) for o, i in ins if i.opcode and i.opcode.id == 458)
        target = at + asm.decode_branch10(br.raw)
        self.assertIn(target, [o for o, _i in ins])                  # an instruction boundary
        first = next(i for o, i in ins if o == target)
        self.assertIn(first.opcode.id, (12674, 12675, 17016, 12656))  # the loop's first load
        cmp = next(i for _o, i in ins if i.opcode and i.opcode.id == 10369)
        self.assertEqual([v for _k, v in cmp.values][5], 3)          # K 64: slices 1..3 in the loop

    def test_k_above_256_compiles_only_as_a_loop(self):
        body = tlower.lower(32, 32, 4096, 4096, 32, 32, kloop=True)[0]
        cmp = next(i for i in model.decode(body, 0) if i.opcode and i.opcode.id == 10369)
        self.assertEqual([v for _k, v in cmp.values][5], 255)

    def test_a_kloop_request_bypasses_the_witness_stream(self):
        # 32x32xK half has a cached Apple witness, whose own loop uses op579; a kloop request must
        # reach tlower, not silently dispatch Apple's program
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import g17tensorcommonruntime as R
        s = R.generic_spec(dict(M=32, N=32, K=64, kloop=True))
        ins = [i for i in model.decode(R.build_generic_program(s).code, 0) if i.opcode]
        self.assertEqual(sum(i.opcode.id == 579 for i in ins), 0)
        cmp = [i for i in ins if i.opcode.id == 10369]
        self.assertEqual([[v for _k, v in i.values][5] for i in cmp], [3])

    def test_refusals(self):
        # fp8 operands left this list with production row P5 (machine model 25.128): the loop admits
        # them after hardware (evidence set g17-tensor-p5-v1, arm kloop_fp8_K512), tested in test_g17tensormxcodes
        for kw in (dict(K=24), dict(K=16), dict(transA=True),
                   dict(a_type='float', b_type='float', split_fp32=True), dict(K=16 * 258),
                   dict(reduce=('col', 'sum')), dict(b_regs=((0,),))):
            K = kw.pop('K', 64)
            with self.subTest(**kw, K=K), self.assertRaises(ValueError):
                tlower.lower(32, 32, K, K, 32, 32, kloop=True, **kw)
        # the control for the two composed-branch refusals: the reduction lowers without the loop
        self.assertTrue(tlower.lower(32, 32, 64, 64, 32, 32, reduce=('col', 'sum')))

    def test_the_termination_check_fires_on_a_second_writer_of_the_counter(self):
        from agxforge.g17 import ledgerenc
        inc = ledgerenc.encode(10279, {0: 'R5', 1: 0, 3: 'R5', 4: 0, 2: 1})
        clobber = ledgerenc.encode(10279, {0: 'R5', 1: 0, 3: 'R6', 4: 0, 2: 7})
        tlower._check_counted_loop(bytes(inc), 5, 3)                  # the control: one writer passes
        with self.assertRaises(RuntimeError):
            tlower._check_counted_loop(bytes(inc) + bytes(clobber), 5, 3)
        with self.assertRaises(RuntimeError):
            tlower._check_counted_loop(bytes(inc), 5, 256)


def _loop_body(code):
    ins = _ins(code)
    at, br = next((o, i) for o, i in ins if i.opcode and i.opcode.id == 458)
    target = at + asm.decode_branch10(br.raw)
    return [i for o, i in ins if target <= o < at]


class Unroll(unittest.TestCase):
    """kloop_unroll=2 (MM 25.124.6): the body is two slices, every load before the first MMA. On
    hardware (the g17-proj-timing-v1 receipt, warm clock, 66 bit-exact units per arm) the M16 N128 K2048
    projection at 8 column threadgroups went 43.2 -> 26.9 us against Apple's 25.9; the same two slices
    with each slice's MMA right after its own loads (the control) took 42.5."""

    # the receipted programs: (grid_n, kloop_unroll) -> sha256[:16] of gemm_generic's code
    RECEIPTED = {(2, 1): "408ce4ec8f264ba0", (4, 1): "5d96350a803101ee", (8, 1): "5ff0d886e2658112",
                 (2, 2): "375b14d494adda41", (4, 2): "f505eb45d275c605", (8, 2): "738c975ece9e95cc"}

    def test_the_receipted_programs_reproduce(self):
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import g17tensorcommonruntime as R
        for (c, u), want in sorted(self.RECEIPTED.items()):
            spec = dict(M=16, N=128, K=2048, grid_n=c, kloop=True, **({"kloop_unroll": u} if u > 1 else {}))
            code = bytes(R.build_generic_program(R.generic_spec(spec)).code)
            with self.subTest(grid_n=c, kloop_unroll=u):
                self.assertEqual(hashlib.sha256(code).hexdigest()[:16], want)

    def test_every_load_is_issued_before_the_first_mma(self):
        for M, N, K in ((16, 16, 2048), (16, 64, 2048), (32, 32, 512), (32, 32, 48)):
            body = _loop_body(tlower.lower(M, N, K, K, N, N, kloop=True, kloop_unroll=2)[0])
            ops = [i.opcode.id for i in body]
            mmas = [j for j, op in enumerate(ops) if op in (5106, 5107)]
            loads = [j for j, op in enumerate(ops) if op in (12674, 12675)]
            with self.subTest(M=M, N=N, K=K):
                self.assertEqual(len(mmas), 2 * (M // 16) * (N // 16))
                self.assertLess(max(loads), min(mmas))
                # slice 0's MMAs wait on load slot 1, slice 1's on slot 2 (the wait mask, operand 1 bits 24-30)
                slots = [(i.values[1][1] >> 24) & 0x7F for i in body if i.opcode.id in (5106, 5107)]
                self.assertEqual(slots, [2] * (len(slots) // 2) + [4] * (len(slots) // 2))

    def test_the_trip_count_covers_k_after_the_peel(self):
        # K 2048: 128 slices, 2 peeled, 63 trips of 2; K 48: 3 slices, 1 peeled, 1 trip of 2
        for K, trips in ((2048, 63), (48, 1), (64, 1), (80, 2)):
            code = tlower.lower(32, 32, K, K, 32, 32, kloop=True, kloop_unroll=2)[0]
            cmp = next(i for i in model.decode(code, 0) if i.opcode and i.opcode.id == 10369)
            with self.subTest(K=K):
                self.assertEqual([v for _k, v in cmp.values][5], trips)

    def test_refusals(self):
        for kw in (dict(kloop=False), dict(kloop_unroll=5), dict(K=32), dict(a_type='fp8e4m3', b_type='fp8e5m2'),
                   dict(N=128)):          # N 128 needs two register groups with two buffer sets
            kw = dict(dict(kloop=True, kloop_unroll=2), **kw)
            K, N = kw.pop('K', 512), kw.pop('N', 32)
            with self.subTest(**kw, K=K, N=N), self.assertRaises(ValueError):
                tlower.lower(16, N, K, K, N, N, **kw)
        self.assertTrue(tlower.lower(16, 128, 512, 512, 128, 128, kloop=True))   # the control: unroll 1 lowers it

    def test_three_and_four_slices_per_trip(self):
        # MM 25.144.1: 3 or 4 slices per trip (MLX's steel loop runs 4), bit-exact on hardware; each slice's loads land
        # in their own scoreboard slot 1 + kd, so the body holds loads in slots 1..U
        for u in (3, 4):
            code = tlower.lower(16, 32, 512, 512, 32, 32, kloop=True, kloop_unroll=u)[0]
            slots = {(v >> 20) & 7 for i in model.decode(code, 0) if i.opcode and i.opcode.id == 12674
                     for n, (k, v) in enumerate(i.values) if n == 1}
            with self.subTest(u=u):
                self.assertTrue(set(range(1, u + 1)) <= slots, slots)

    def test_the_loop_carried_check(self):
        from agxforge.g17 import ledgerenc
        keep = bytes(ledgerenc.encode(10279, {0: 'R6', 1: 0, 3: 'R5', 4: 0, 2: 1}))
        release = bytes(ledgerenc.encode(10279, {0: 'R6', 1: 0, 3: 'R5', 4: 16, 2: 1}))
        rewrite = bytes(ledgerenc.encode(10279, {0: 'R5', 1: 0, 3: 'R5', 4: 16, 2: 1}))   # Apple's own counter form
        self.assertEqual(tlower._check_loop_carried(keep), {5})
        self.assertEqual(tlower._check_loop_carried(rewrite), {5})
        self.assertEqual(tlower._check_loop_carried(release + rewrite), {5})   # released, then written again
        with self.assertRaises(RuntimeError):
            tlower._check_loop_carried(release)
        with self.assertRaises(RuntimeError):
            tlower._check_loop_carried(keep + release)


if __name__ == "__main__":
    unittest.main()
