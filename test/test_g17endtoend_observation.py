"""CPU controls for the legacy execution instrument; these never dispatch."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import g17endtoend as E


class Observation(unittest.TestCase):
    def compare(self, mutate=lambda side, record: None, returncode=0):
        def child(argv, **kwargs):
            record = {"status": 0, "C": E._rangestore_py(None)}
            mutate(argv[-3], record)
            Path(argv[-1]).write_text(json.dumps(record))
            return subprocess.CompletedProcess(argv, returncode, b"", b"child failed")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(E, "SCRATCH", directory), patch.object(E.subprocess, "run", child):
                return E.one("rangestore", "unused.py")

    def test_readback_reaches_every_store_and_trailing_neighbours(self):
        self.assertEqual(E.output_count("rangestore", None), 92)
        self.assertEqual(E.dispatch_threads("rangestore"), 1)
        self.assertTrue(self.compare()[0])
        for slot in (70, 72, 80, 81, 82, 91):
            def corrupt(side, record):
                if side == "g17":
                    record["C"][slot] ^= 1
            with self.subTest(slot=slot):
                self.assertFalse(self.compare(corrupt)[0])

    def test_old_native_truncation_cannot_pass(self):
        def truncate(side, record):
            if side == "g17":
                record["C"] = record["C"][:32]
        self.assertFalse(self.compare(truncate)[0])

    def test_matching_numbers_do_not_override_failed_dispatch(self):
        for bad_status in (1, None, False):
            with self.subTest(status=bad_status):
                self.assertFalse(self.compare(lambda side, r: r.update(status=bad_status))[0])
        self.assertFalse(self.compare(returncode=1)[0])


if __name__ == "__main__":
    unittest.main()
