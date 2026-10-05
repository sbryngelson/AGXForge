"""A whole-program oracle record that names an opcode must contain exactly that form, once.

The runner copies a plan's `op` and `length` into its result, and the compiler inventory's
isolated index counts a finished ok record by those two fields alone. A program record naming a
form it does not contain would be counted as that form's isolated evidence.
"""
import contextlib
import io
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import g17oracle


def build(rec):
    with contextlib.redirect_stdout(io.StringIO()):
        return g17oracle.program(rec)[1]


class AProgramRecordMustContainTheFormItNames(unittest.TestCase):

    def test_the_named_form_is_accepted_and_its_bytes_reported(self):
        chk = build(dict(id="t", program="store_word_8", op=17235, length=8))
        self.assertEqual(chk["want"], 17235)
        self.assertEqual(len(chk["encoded"]), 1)
        self.assertEqual(len(bytes.fromhex(chk["encoded"][0])), 8)

    def test_another_length_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            build(dict(id="t", program="store_word_8", op=17235, length=14))
        self.assertIn("contains it as [8]", str(cm.exception))

    def test_an_absent_opcode_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            build(dict(id="t", program="store_word_8", op=17199, length=8))
        self.assertIn("no instance", str(cm.exception))

    def test_a_missing_length_is_refused(self):
        with self.assertRaises(ValueError):
            build(dict(id="t", program="store_word_8", op=17235))

    def test_a_program_record_naming_nothing_is_unchanged(self):
        chk = build(dict(id="t", program="store_word_8"))
        self.assertIsNone(chk["want"])
        self.assertEqual(chk["encoded"], [])

    def test_each_one_store_program_holds_its_store_exactly_once(self):
        for name, op, ln in (("store_word_8", 17235, 8), ("store_word_10", 17235, 10),
                             ("store_word_14", 17235, 14), ("store_half_8", 17199, 8),
                             ("store_half_10", 17199, 10), ("store_half_14", 17199, 14)):
            with self.subTest(name=name):
                self.assertEqual(build(dict(id=name, program=name, op=op, length=ln))["want"], op)

    def test_the_canary_store_is_scaffold_not_a_second_instance(self):
        """The wrapper's canary is a default store, op17244 at 8 bytes. A probe OF op17244/8 holds
        one instance as written, and must not be refused for the one the scaffold appends."""
        chk = build(dict(id="t", program="range2_8", op=17244, length=8))
        self.assertEqual(chk["want"], 17244)

    def test_a_program_that_writes_two_instances_itself_is_refused(self):
        """store_word_8's canary adds an op17244/8; the program itself writes none, so naming
        op17244 must refuse as 'no instance' - the count is of the program, not the wrapper."""
        with self.assertRaises(ValueError) as cm:
            build(dict(id="t", program="store_word_8", op=17244, length=8))
        self.assertIn("no instance", str(cm.exception))

    def test_a_declared_instance_count_must_match_exactly(self):
        """halfvec2 writes two op590 packing moves; naming op590 needs `instances: 2`, and 1 or 3
        refuses - the count is a claim about the program, checked like the length."""
        self.assertEqual(build(dict(id="t", program="halfvec2", op=590, length=4, instances=2))["want"], 590)
        for n in (1, 3):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    build(dict(id="t", program="halfvec2", op=590, length=4, instances=n))


if __name__ == "__main__":
    unittest.main()
