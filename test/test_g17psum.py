#!/usr/bin/env python3
"""The batched decode's split-K partial sum (tools/g17psum.py, MM 25.172; swiglu MM 25.185): all five modes bit-exact on g17emu at 16 and
8 rows (the partial stride 16 N), with the control that a reference summing in another order differs. CPU only."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402


class PartialSum(unittest.TestCase):
    def test_every_mode_is_bit_exact(self):
        import g17psum as PS
        for rows in (16, 8):
            with self.subTest(rows=rows):
                self.assertEqual(PS.check(N=256, sk=4, rows=rows), {m: 0 for m in PS.MODES})

    def test_the_order_is_observed(self):
        """The control: summing the partials in descending order differs from the stated ascending order."""
        import g17psum as PS
        lay = PS.layout(256, 4, "sum", rows=16)
        rng = np.random.default_rng(1)
        parts = (rng.standard_normal((4, 16, 256)) * np.array([1e4, 1, 1e-4, 1])[:, None, None]).astype(np.float32)
        down = ((parts[3] + parts[2]).astype(np.float32) + parts[1]).astype(np.float32) + parts[0]
        self.assertGreater(int((PS.reference(lay, parts).view("<u4") != down.astype(np.float32).reshape(-1).view("<u4")).sum()), 0)

    def test_refusals(self):
        import g17psum as PS
        for kw in (dict(mode="max"), dict(sk=0), dict(N=100), dict(pstride=10)):
            args = dict(N=256, sk=2, mode="sum", rows=16)
            args.update(kw)
            with self.subTest(**{k: str(v) for k, v in kw.items()}), self.assertRaises(ValueError):
                PS.layout(args["N"], args["sk"], args["mode"], rows=args["rows"], pstride=args.get("pstride"))


if __name__ == "__main__":
    unittest.main()
