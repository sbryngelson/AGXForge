#!/usr/bin/env python3
"""One-dispatch attention past two rows (Set A goal item 6, machine model 25.110): the
allocator's compact read_sr retry, the clobber check that guards any narrower tensor reservation,
and the attention programs it admits. Offline: nothing here dispatches."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import g17tensorcommonruntime as R  # noqa: E402
from agxforge.g17 import cc, tensorlife  # noqa: E402

# The dispatched two-row program (fusion v2, machine model 25.102.3, passed on hardware) and the spec it
# came from; the other row counts differ only in `between`.
BASE = {"M": 32, "N": 16, "K": 64, "a": "half", "b": "half", "simdgroups": 1, "threadgroups": 1,
        "epilogue": [], "split_fp32": False, "stages": [[16, 16, "float"]], "guard": "sg0",
        "accumulate": False, "saturate": False, "imageblock": None, "between": "softmax:2",
        "independent": 0, "independent_wrong_a": False, "ib_op1": None}
PROGRAMS = {2: "51c90d46e4cffc86", 4: "7c83c0e50aa65f6d", 8: "0f4692daf152b135", 16: "33525e6aeeb0c65a"}


def program(rows):
    return bytes(R.build_generic_program(R.generic_spec(dict(BASE, between="softmax:%d" % rows))).code)


class AttentionPastTwoRows(unittest.TestCase):
    def test_the_dispatched_two_row_program_is_unchanged(self):
        # the unchanged-behaviour control: the retry runs only after a refusal
        self.assertEqual(hashlib.sha256(program(2)).hexdigest()[:16], PROGRAMS[2])

    def test_four_eight_and_sixteen_rows_compile_to_their_preregistered_bytes(self):
        for rows in (4, 8, 16):
            with self.subTest(rows=rows):
                code = program(rows)
                self.assertEqual(hashlib.sha256(code).hexdigest()[:16], PROGRAMS[rows])
                self.assertEqual(tensorlife.released_reads(code), [])

    def test_the_first_attempt_alone_still_refuses_four_rows(self):
        # the retry is what admits them; without it the first attempt's refusal stands
        saved = cc.Alloc.run

        def first_attempt_only(self, insts):
            occupied = set()
            for m in insts:
                occupied |= set(m.fields.get("_occupies", ()))
            self.regs = [r for r in self.regs if r not in occupied]
            self.wide = [r for r in self.wide if r not in occupied]
            return self._run(insts)
        cc.Alloc.run = first_attempt_only
        try:
            with self.assertRaises(cc.Unsupported):
                program(4)
        finally:
            cc.Alloc.run = saved

    def test_rows_beyond_the_measured_tile_are_refused_by_the_emitter(self):
        for spec in (dict(BASE, between="softmax:32"),
                     dict(BASE, N=32, stages=[[32, 32, "float"]], between="softmax:16")):
            with self.subTest(between=spec["between"], N=spec["N"]), self.assertRaises(Exception):
                R.build_generic_program(R.generic_spec(spec))

    def test_the_control_misses_by_far_more_than_the_bound(self):
        with tempfile.TemporaryDirectory() as td:
            b = Path(td) / "a16"
            R.author_generic(b, dict(BASE, between="softmax:16"))
            s = R.generic_spec(json.loads((b / "generic.json").read_text()))
            bound = R.generic_abs_bound(b, s)
            M, W = bound.shape
            pos = np.asarray(R.generic_reference(b, s), np.float64).reshape(-1)[:M * W].reshape(M, W)
            neg = np.asarray(R.generic_reference(b, R.generic_spec(dict(s, between=None))),
                             np.float64).reshape(-1)[:M * W].reshape(M, W)
            gap = np.abs(pos - neg)
            self.assertGreater(gap[:16, :16].min(), 5 * bound[:16, :16].max())
            self.assertEqual(int((gap[16:] != 0).sum()), 0)


class TheClobberCheck(unittest.TestCase):
    def _stream(self, named):
        define = cc.MInst("movimm.8", 8, {"imm": 1, "_defs": [40]})
        body = cc.MInst("tensor.wholekernel", 10, {"bytes": b"\0" * 10, "_defs": [], "_uses": []},
                        note="synthetic body")
        use = cc.MInst("alu.12", 12, {"_uses": [40], "_defs": [41]})
        return [define, body, use], (lambda m: set(named))

    def test_it_fires_on_a_value_live_across_a_body_in_a_register_the_body_names(self):
        insts, regs_of = self._stream({40})
        with self.assertRaises(cc.Unsupported):
            cc._check_tensor_body_clobbers(insts, regs_of)

    def test_and_passes_when_the_body_names_other_registers(self):
        insts, regs_of = self._stream({41, 42})
        cc._check_tensor_body_clobbers(insts, regs_of)

    def test_and_passes_when_nothing_lives_across(self):
        insts, regs_of = self._stream({40})
        insts = [insts[0], insts[2], insts[1]]      # the value dies before the body
        cc._check_tensor_body_clobbers(insts, regs_of)


if __name__ == "__main__":
    unittest.main()
