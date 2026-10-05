from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import g17tensordelivery as T


class TensorApplicationDelivery(unittest.TestCase):
    def test_reference_separates_operand_and_layout_errors(self):
        a, b, ref = (T.arrays()[k] for k in ('a', 'b', 'reference'))
        self.assertFalse(np.array_equal(ref, a.astype(np.float64) @ a.astype(np.float64).T))
        self.assertFalse(np.array_equal(ref, ref.T))
        self.assertFalse(np.array_equal(ref, a.astype(np.float64) @ b.ravel().reshape(32, 64).T.astype(np.float64)))
        self.assertEqual(ref.shape, (32, 32))
        self.assertTrue(np.array_equal(ref, ref.astype(np.float32)))

    def test_actual_ir_names_three_buffers_and_requested_operation(self):
        fn = T.program()
        op = next(o for block in fn.blocks for o in block.ops if o.kind == 'tensor_matmul')
        self.assertEqual(op.args, fn.buffers)
        self.assertEqual([b.slot for b in fn.buffers], [1, 2, 3])
        self.assertEqual([op.attrs[k] for k in ('M', 'N', 'K')], [32, 32, 64])

    def test_refused_compile_preserves_inputs_without_dispatchable_bytes(self):
        import g17cc
        with tempfile.TemporaryDirectory() as tmp, patch.object(g17cc, 'compile_function', side_effect=KeyError('tensor.seq')):
            dst = Path(tmp) / 'delivery'
            report = T.deliver(dst)
            self.assertEqual(report['status'], 'compiler_refused')
            self.assertFalse(report['dispatch_eligible'])
            self.assertFalse((dst / 'program.bin').exists())
            self.assertTrue((dst / 'reference.npy').is_file())
            with self.assertRaises(FileExistsError):T.deliver(dst)


if __name__ == '__main__':
    unittest.main()
