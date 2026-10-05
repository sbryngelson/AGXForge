#!/usr/bin/env python3
"""tlower refuses a body that reads a loaded register before waiting on its slot (tensorview.hazards).
The check accepts every body tlower emits and fires on a real body whose MMA waits are removed."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agxforge.g17 import mmaenc, model, tensorview as TV, tlower  # noqa: E402


class LoadWaitsAreChecked(unittest.TestCase):
    def test_emitted_bodies_are_clean(self):
        for M, kw in ((32, dict()), (32, dict(kloop=True)), (32, dict(epilogue=(("gelu",),))), (64, dict(grid=4)),
                      (32, dict(sg=2))):
            with self.subTest(M=M, **{k: str(v) for k, v in kw.items()}):
                body, _ = tlower.lower(M, 32, 64, 64, 32, 32, **kw)
                self.assertEqual(TV.hazards(TV.view(bytes(body))), [])

    def test_the_check_fires_when_an_mma_stops_waiting(self):
        body, _ = tlower.lower(32, 32, 64, 64, 32, 32)
        # every MMA's wait mask cleared (clearing only the first can leave its slot covered by another
        # instruction's wait, which is not a hazard)
        out, cleared = bytearray(), 0
        for ins in model.decode(bytes(body), 0):
            raw = bytes(ins.raw)
            if ins.opcode and ins.opcode.id in (5106, 5107):
                vals = {k: v[1] for k, v in enumerate(ins.values)}
                if vals[1] & 0x7F000000:
                    vals[1] &= ~0x7F000000
                    raw = bytes(mmaenc.encode(ins.opcode.id, vals))
                    cleared += 1
            out += raw
        self.assertGreater(cleared, 0)
        self.assertNotEqual(TV.hazards(TV.view(bytes(out))), [])


if __name__ == "__main__":
    unittest.main()
