#!/usr/bin/env python3
"""P10 focused relocation tests: an A-, B- or C-buffer byte offset folds correctly into the tensor
load/store addressing.

Each offset class is GPU-measured and admitted by cc's routes (A and C: machine model 25.102.2, the
g17-tensor-fusion-v1 receipt with its `neg_indep2_wrong_a` control where a zeroed offset makes every
body read row 0 and the reference rejects it; B: the measured even-transport stream 2..47104, cc's
`_measured_tensor_stream_offset`, 25.102.4). Those numerical proofs live in the git-ignored evidence
tree; these tests need no evidence -- they are the committed, GPU-free guard that the relocation
ADDRESSING itself (`tlower.based`) stays correct across codegen changes, for every buffer:

- an offset is CONSUMED (the failing control: an ignored offset would leave the bytes identical to
  offset 0, which is exactly the `independent_wrong_a` bug the GPU control catches);
- while it fits the instruction's displacement field it rides that displacement, adding no base
  register;
- beyond the field it crosses to a base register (the displacement/base-register transition);
- a residual folded into the base must be element-aligned, or it is refused.

The fold is a function of the BYTE offset and the element size only, so it is shape- and
route-independent -- which is why cc applies the one measured B range across the six-body, chain and
group routes (25.102.4). Element sizes here: A and B are half (2 bytes), C is fp32 (4 bytes). The
buffer-level alignment rules (A whole-row, C fp32-aligned, B even 2..47104) are cc admission rules
checked in cc; these tests check the addressing fold beneath them, relative and structural (not pinned
hashes), so only a relocation-addressing regression reds them.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agxforge.g17 import tlower, model  # noqa: E402

K = 64
DISP_MAX = 32767             # tlower.based folds an offset <= this into the displacement (tlower.py)
ADD = 10282                  # the integer add tlower.based emits when it builds a base register


def body(offsets):
    code, _ = tlower.lower(16, 32, K, K, 32, 32, a_type="half", b_type="half", offsets=offsets)
    return code


def address_adds(code):
    return sum(1 for i in model.decode(code, 0) if i.opcode and i.opcode.id == ADD)


class _Relocation:
    """Shared checks; a subclass sets INDEX (0=A, 1=B, 2=C), SMALL (a valid in-field offset) and BIG
    (a valid offset just past DISP_MAX). Not a TestCase itself, so it is not collected alone."""
    INDEX = SMALL = BIG = None

    def _off(self, value):
        triple = [0, 0, 0]
        triple[self.INDEX] = value
        return tuple(triple)

    def test_the_offset_is_consumed(self):
        # The control that FAILS if the offset is dropped: offset 0 and a real offset must differ.
        self.assertNotEqual(body(self._off(self.SMALL)), body(self._off(0)))

    def test_an_offset_within_the_field_rides_the_displacement(self):
        # <= DISP_MAX: no base register is built, so no extra addressing add over offset 0.
        self.assertLessEqual(self.SMALL, DISP_MAX)
        self.assertEqual(address_adds(body(self._off(self.SMALL))),
                         address_adds(body(self._off(0))))

    def test_an_offset_beyond_the_field_builds_a_base_register(self):
        # > DISP_MAX: the displacement/base-register transition, so more addressing adds appear.
        self.assertGreater(self.BIG, DISP_MAX)
        self.assertGreater(address_adds(body(self._off(self.BIG))),
                           address_adds(body(self._off(self.SMALL))))

    def test_a_residual_beyond_the_field_must_be_element_aligned(self):
        # BIG + 1 byte is not a whole element (2 for half, 4 for fp32); folded past the field it is
        # refused. BIG is already > DISP_MAX, so BIG + 1 takes the base-register fold that checks it.
        with self.assertRaises(ValueError):
            body(self._off(self.BIG + 1))


class AOffsetRelocation(_Relocation, unittest.TestCase):
    INDEX, SMALL, BIG = 0, 2 * K, 2 * K * 257          # half A, whole rows (2K); big whole-row > DISP


class BOffsetRelocation(_Relocation, unittest.TestCase):
    INDEX, SMALL, BIG = 1, 256, 2 * (DISP_MAX // 2 + 1)   # half B, even; big even > DISP


class COffsetRelocation(_Relocation, unittest.TestCase):
    INDEX, SMALL, BIG = 2, 4, 4 * (DISP_MAX // 4 + 1)     # fp32 C, 4-aligned; big 4-aligned > DISP


if __name__ == "__main__":
    unittest.main()
