"""g17auth.encode writes a slot the way Apple's decoder reads it at the authored length.

The authoring table's slot maps were fitted at one length each, and a slot's layout depends on the
encoded length (isa/g17-slot-truth-by-length.json, tools/g17slottruth.py). Written the way cc writes
registers, op11994/16 named r105 for every register from 16 up, and op3291/3294/3307 at four bytes
wrapped r64 and above by writing a bit past byte 4. g17auth.encode now reads each slot back through
the decoder-measured layout for its length, rewrites it from that layout on a mismatch, and refuses
a value the layout cannot hold. Where the map was already right, no byte changes.
"""
import json, os, sys, unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from agxforge.g17 import auth as A, cc
import g17packedcheck as D

END = bytes.fromhex("0e000000")


def emit(op, j, r):
    b = A.encode(op, {j: A.field_value(op, j, r)}, template=cc.CLEARED_WITNESS.get(op))
    return b[:A.length(op)]


def named(op, j, r):
    t = D.decode(emit(op, j, r) + END)[0][3][j]
    return int(t[4:]) if t.startswith("reg:") else None


def slots():
    truth = json.load(open(os.path.join(ROOT, "isa", "g17-slot-truth-by-length.json")))["truth"]
    for key in sorted(truth):
        op, spec = key.split()
        yield int(op), int(spec.split(":")[0])


class TheDecoderNamesTheRegisterAsked(unittest.TestCase):

    def test_op11994_above_r16(self):
        base = named(11994, 5, 0)
        for r in (15, 16, 17, 31, 64, 127):
            with self.subTest(r=r):
                self.assertEqual(named(11994, 5, r), base + r)

    def test_four_byte_slots_refuse_what_they_cannot_hold(self):
        for op, j in ((3291, 4), (3294, 2), (3307, 0)):
            with self.subTest(op=op):
                base = named(op, j, 0)
                self.assertEqual(named(op, j, 63), base + 63)
                with self.assertRaises(ValueError):
                    emit(op, j, 64)


class NothingChangesWhereTheMapWasRight(unittest.TestCase):

    def test_bytes_are_unchanged_unless_the_old_bytes_were_wrong(self):
        changed = []
        for op, j in slots():
            try:
                A.length(op); A.register_operands(op)
            except Exception:
                continue
            for r in range(64):
                with patch.object(A, "_slot_at_length", lambda *a: None):
                    try:
                        old = emit(op, j, r)
                    except ValueError:
                        continue
                try:
                    new = emit(op, j, r)
                except ValueError:
                    continue
                if new != old:
                    changed.append((op, j, r))
        self.assertTrue(all(op == 11994 and r >= 16 for op, j, r in changed), changed[:5])
        # the control: the one known-wrong slot must show up, or this compared nothing
        self.assertEqual(sorted(r for op, j, r in changed), list(range(16, 64)))


if __name__ == "__main__":
    unittest.main()
