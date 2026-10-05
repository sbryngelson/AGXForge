"""A one-element store whose value is read again must KEEP its source, or refuse.

Measured 2026-09-22: op17235 and op17199 at 8, 10 and 14 bytes, with the source lifetime 16
(release) the liveness pass used to write unconditionally, made `store v; w = v + 1` give w = 1 -
the released source reads 0 (isa/g17-execution-storereread-results.json). Written 0 or 32 at
14 bytes, the value survived (isa/g17-execution-storelifetime-results.json).
"""
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import g17cc
import g17storeprobes as L
from agxforge.g17 import ref as g17ref


def lifetime(name):
    """The store under test's operand 1, as Apple's decoder reads it."""
    code = g17cc.compile_function(getattr(L, name)()).code
    for at, ln, op in g17ref.walk(code, 0):
        if op in (17235, 17199):
            with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
                fh.write(code[at:at + ln]); fh.flush()
                out = subprocess.run([g17ref.binary(), fh.name, "0", str(ln), "--pc", "0"],
                                     capture_output=True, text=True).stdout
            return int(re.findall(r"imm:(\d+)", out.splitlines()[0])[0])
    raise AssertionError("no element store in %s" % name)


class AStoredValueReadAgainIsKept(unittest.TestCase):

    def test_the_8_and_14_byte_forms_keep_it(self):
        for name in ("reread_word_8", "reread_word_14", "reread_half_8", "reread_half_14"):
            with self.subTest(name=name):
                self.assertEqual(lifetime(name) & 0x30, 0x20, "keep is 32 in the lifetime bits")

    def test_the_10_byte_forms_refuse_rather_than_release(self):
        """Their lifetime does not survive Apple's decoder on the template, so they cannot keep."""
        for name in ("reread_word_10", "reread_half_10"):
            with self.subTest(name=name):
                with self.assertRaises(g17cc.Unsupported) as cm:
                    g17cc.compile_function(getattr(L, name)())
                self.assertIn("cannot be read again", str(cm.exception))

    def test_a_value_stored_last_still_releases(self):
        """The control: a dead value's store is unchanged - 16, with the 10-byte wait bit intact."""
        for name in ("store_word_8", "store_word_14", "store_half_8", "store_half_14"):
            with self.subTest(name=name):
                self.assertEqual(lifetime(name), 16)
        for name in ("store_word_10", "store_half_10"):
            with self.subTest(name=name):
                self.assertEqual(lifetime(name), 0x80000010)


if __name__ == "__main__":
    unittest.main()
