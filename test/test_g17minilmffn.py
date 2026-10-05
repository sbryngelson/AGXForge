import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import g17ir as ir
import g17minilmffn as ffn
import g17minilmquerycheck as check


class FeedForwardTest(unittest.TestCase):
    def test_rectangular_reductions_include_last_input_and_distinct_rows(self):
        # Two CPU grid rows discriminate the output stride without a full GPU
        # campaign. Both actual reduction widths and all output columns remain.
        for ni, no in ((384, 1536), (1536, 384)):
            with self.subTest(input_width=ni):
                x = np.zeros((2, ni), np.float32)
                x[:, -1] = [2, -3]
                w = np.zeros((no, ni), np.float32)
                w[:, -1] = np.arange(no, dtype=np.float32) + 1
                bias = np.full(no, .25, np.float32)
                actual = check.evaluate(ffn.dense_ir(ni, no), x, w, bias)
                np.testing.assert_array_equal(actual, ffn.dense_reference(x, w, bias))

    def test_wrong_rectangular_output_stride_is_detected(self):
        f = ffn.dense_ir(384, 1536)
        base = next(op for block in f.blocks for op in block.ops
                    if op.dest is not None and op.dest.name == 'output_base')
        base.args[1] = ir.Imm(384)
        x = np.zeros((2, 384), np.float32)
        w = np.zeros((1536, 384), np.float32)
        with self.assertRaisesRegex(ValueError, 'every output exactly once'):
            check.evaluate(f, x, w, np.zeros(1536, np.float32))

    def test_gelu_definition_and_missing_ir_operation(self):
        # Exact erf GELU has g(x)-g(-x)=x and tends to x/0 at either tail.
        x = np.array([0., .5, 1., 3., 10.])
        np.testing.assert_allclose(ffn.gelu_reference(x) - ffn.gelu_reference(-x), x,
                                   rtol=0, atol=1e-15)
        self.assertEqual(ffn.gelu_reference(np.array([-10., 10.])).tolist(), [-0., 10.])
        # `erf` WAS refused as an unknown defining operation; it is now an IR-level operation
        # (g17ir.EXPANDED) expanded before selection into executed forms - tools/g17gelu.py
        # states the bound and the domain. The application's GELU compiles UNCHANGED, and the
        # forms it compiles to are the executed float set with no fma (every executed fma
        # consumed a load; one over ALU results is an unmeasured modifier).
        f = ffn.gelu_ir()
        ir.verify(f)
        self.assertEqual([o.kind for b in f.blocks for o in b.ops if o.kind == 'erf'], ['erf'])
        import g17cc
        p = g17cc.compile_function(f)
        self.assertNotIn(2190, {op for op, _l in p.abi()['forms']})
        self.assertTrue({(998, 12), (3290, 14), (1272, 10), (3658, 10), (9700, 14)} <= set(p.abi()['forms']))


if __name__ == '__main__':
    unittest.main()
