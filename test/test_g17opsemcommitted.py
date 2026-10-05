import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import g17normcheck as N


class ExecutedEvidenceIsCommitted(unittest.TestCase):
    def test_the_default_source_is_in_the_repository(self):
        self.assertTrue(os.path.abspath(N.OPSEM).startswith(ROOT))
        self.assertGreaterEqual(len(N.executed_opcodes()), 500)

    def test_a_missing_source_refuses_rather_than_returning_nothing(self):
        with self.assertRaises(OSError):
            N.executed_opcodes(os.path.join(ROOT, "isa", "no-such-opsem.json"))


if __name__ == "__main__":
    unittest.main()
