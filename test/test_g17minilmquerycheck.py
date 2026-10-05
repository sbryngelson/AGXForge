import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import g17ir as ir
import g17minilmquery as query
import g17minilmquerycheck as check


class QueryIRCheckTest(unittest.TestCase):
    def setUp(self):
        # The final reduction element is the only nonzero product. A shortened
        # loop can look plausible on ordinary inputs but must fail this control.
        self.x = np.zeros((1, 384), np.float32)
        self.x[0, -1] = 2
        self.w = np.zeros((384, 384), np.float32)
        self.w[:, -1] = np.arange(384, dtype=np.float32) + 1
        self.b = np.full(384, 0.25, np.float32)

    def test_full_reduction_and_last_column(self):
        result = check.evaluate(query.query_projection_ir(1), self.x, self.w, self.b)
        expected = 2 * (np.arange(384) + 1) + 0.25
        np.testing.assert_array_equal(result[0], expected)

    def test_shortened_reduction_is_wrong(self):
        function = query.query_projection_ir(1)
        comparison = next(op for block in function.blocks for op in block.ops if op.kind == "cmp")
        comparison.args[1] = ir.Imm(383)
        result = check.evaluate(function, self.x, self.w, self.b)
        report = check.compare(result, query.reference(self.x, self.w, self.b))
        self.assertEqual(report["failures"], 384)

    def test_readonly_store_is_refused(self):
        function = query.query_projection_ir(1)
        store = next(op for block in function.blocks for op in block.ops if op.kind == "store_at")
        store.args[0] = function.buffers[0]
        with self.assertRaisesRegex(ValueError, "read-only"):
            check.evaluate(function, self.x, self.w, self.b)

    def test_wrong_grid_axis_causes_address_refusal(self):
        function = query.query_projection_ir(1)
        row = next(op for block in function.blocks for op in block.ops
                   if op.kind == "builtin" and op.attrs["axis"] == "y")
        row.attrs["axis"] = "x"
        with self.assertRaisesRegex(ValueError, "outside its declared buffer"):
            check.evaluate(function, self.x, self.w, self.b)


if __name__ == "__main__":
    unittest.main()
