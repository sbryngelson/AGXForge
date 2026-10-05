"""64-K partial/fold loop: numerical grouping and emitted-loop invariants."""
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agxforge.g17 import cc, ir, model, tensor, tlower


class ChunkLoop(unittest.TestCase):
    def test_default_bodies_are_unchanged(self):
        cases = [({}, 32, '41e0132ae42a217f85e7a8e75d7789570427878dcef54076ef3a95b1e950c3b4'),
                 ({'kloop': True}, 32, 'ff56130b74697a684893bc993a2c22e94356cf3b6fa6eee9d136b0003588260e'),
                 ({'kloop': True, 'grid_n': 48}, 1536, '15acad27c0085f485b243d97fad7aa020ef601d126cce7303cdf0691e54b830a')]
        for kw, n, expected in cases:
            body, _ = tlower.lower(32, n, 384, 384, n, n, **kw)
            self.assertEqual(hashlib.sha256(body).hexdigest(), expected)

    def test_application_shapes_have_fresh_partials_and_one_bounded_loop(self):
        for k, n, grid in ((384, 1536, 48), (1536, 384, 12)):
            lowered = tensor.emit_gemm(32, n, k, grid_n=grid, kloop=True, kloop_chunk=64)
            plan = lowered.plan
            self.assertEqual(plan['groups'], [[[0, 0], [0, 1], [1, 0], [1, 1]]])
            self.assertLessEqual(plan['registers'], 126)
            self.assertEqual(plan['accumulator_groups'], 8)
            instructions = list(model.decode(lowered.body, 0))
            self.assertTrue(all(i.opcode is not None for i in instructions))
            opcodes = [i.opcode.id for i in instructions]
            # One peeled block plus one four-slice runtime block, four output tiles.
            self.assertEqual(opcodes.count(5107), 8)
            self.assertEqual(opcodes.count(5106), 24)
            self.assertEqual(opcodes.count(458), 1)
            self.assertEqual(opcodes.count(684), 1)
            folded = [o for o in plan['ops'] if o['what'].startswith('chunk fold D')]
            self.assertEqual(len(folded), 64)
            self.assertTrue(all(o['op'] == 998 for o in folded))
            self.assertIn('cmp cnt < %d' % (k // 64 - 1), [o['what'] for o in plan['ops']])
            self.assertIn('advance kA += 64, kB += 64*ldb', [o['what'] for o in plan['ops']])

    def test_unmeasured_chunk_domains_are_refused(self):
        for changed in ({'kloop': False}, {'kloop_chunk': 32}, {'K': 192 + 16},
                        {'a_type': 'int8', 'b_type': 'int8'}, {'kloop_unroll': 2},
                        {'transA': True}, {'split_k': 2}):
            kw = dict(kloop=True, kloop_chunk=64)
            kw.update(changed)
            k = kw.pop('K', 384)
            with self.assertRaisesRegex(ValueError, 'refused: kloop_chunk'):
                tensor.emit_gemm(32, 32, k, **kw)

    def test_ordinary_ir_preserves_chunk_fold(self):
        buffers = [ir.Buffer('a', 1, elem=ir.F16), ir.Buffer('b', 2, elem=ir.F16),
                   ir.Buffer('out', 3, elem=ir.F32)]
        fn = ir.Function('chunk_fold', buffers)
        builder = ir.Builder(fn, fn.block('entry'))
        builder.tensor_matmul(*buffers, M=32, N=32, K=128, kloop=True, kloop_chunk=64)
        builder.ret()
        compiled = cc.compile_function(fn)
        expected = tensor.emit_gemm(32, 32, 128, kloop=True, kloop_chunk=64).body
        self.assertEqual(compiled.code, expected)


if __name__ == '__main__':
    unittest.main()
