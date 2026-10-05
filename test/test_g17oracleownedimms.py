"""A record cannot state an operand the compiler owns, and asking must refuse rather than lie.

This exists because the silent version nearly published a refutation of another lane's finding.
A record asked for op612's operand 3 = 0, the emitted program came back BYTE-IDENTICAL to the
record that asked for nothing, it ran, and its two mismatches read exactly like hardware
disagreeing with their sentence. The operand is the source's LIFETIME carrier (32 keeps, 16
releases) and the liveness pass writes it into the finished bytes after the encoder has already
honoured the record - deliberately, so an authored program cannot emit a stale lifetime.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
import g17auth
import g17oracle as M

BASE = dict(op=612, author="fieldmap", witness_required=False, cases=[[13]])


class AnOwnedOperandCannotBeStatedByARecord(unittest.TestCase):
    def test_op612_operand_3_is_the_lifetime_carrier_of_its_register_source(self):
        """The premise, asserted against the field map rather than taken from the refusal.

        If this moves, the guard below is guarding the wrong operand and would start refusing a
        record that builds correctly - which is the failure mode that made the first version of it
        useless.
        """
        self.assertEqual(g17auth.register_operands(612), ([0], [2]))
        self.assertEqual(g17auth.lifetime_operand(612, 2), 3)

    def test_a_record_naming_the_lifetime_operand_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            M.program(dict(BASE, id="t.owned", imms={"3": 0}))
        message = str(caught.exception)
        self.assertIn("operand(s) [3]", message)
        self.assertIn("liveness", message)
        # The refusal has to say what to do instead, or it just moves the dead end.
        self.assertIn("bytes", message)

    def test_the_operands_the_compiler_does_not_own_still_build(self):
        """THE ARM WHERE EVERY REFUSAL IS A FALSE POSITIVE BY CONSTRUCTION.

        op612's operands 4 and 5 are its lower bound and its window width, and the batch in
        isa/g17-execution-op612range.json wrote both and read the moved edge back off the
        hardware. The first version of this guard refused them anyway, because lifetime_operand
        answers for any index it is handed and reports a carrier for 3, 4 and 5 alike; only the
        REGISTER sources have lifetimes, and op612 has exactly one.
        """
        for imms in ({"4": 8}, {"5": 8}, {"4": 8, "5": 32}):
            P, check = M.program(dict(BASE, id="t.free", imms=imms))
            self.assertTrue(check["ok"], check)
        # and the immediate must actually reach the bytes, which is the whole point
        a = M.program(dict(BASE, id="t.a", imms={"5": 8}))[1]["encoded"][0]
        b = M.program(dict(BASE, id="t.b", imms={"5": 16}))[1]["encoded"][0]
        self.assertNotEqual(a, b, "the window width did not change the encoding, so a record "
                                  "stating it is measuring the default program")

    def test_a_record_stating_nothing_is_not_refused(self):
        P, check = M.program(dict(BASE, id="t.plain"))
        self.assertTrue(check["ok"], check)


if __name__ == "__main__":
    unittest.main()
