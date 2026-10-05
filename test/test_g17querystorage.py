"""Exercise the query allocation using actual C guard and readonly checks."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_g17layernormstorage import HARNESS

ROOT = Path(__file__).resolve().parents[1]


class QueryStorage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        source = root / "query-storage.c"
        # Share the mutations and C checks, but initialize the real query sizes.
        source.write_text(HARNESS.replace("g17LayerNormStorageInit", "g17QueryProjectionStorageInit"))
        cls.binary = root / "query-storage"
        subprocess.run(["clang", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=address,undefined", "-I", str(ROOT / "tools"),
                        str(source), "-o", str(cls.binary)],
                       check=True, timeout=30, capture_output=True)

    def test_query_allocations_guards_and_readonly_payloads(self):
        for rows in (1, 32, 128):
            with self.subTest(rows=rows):
                result = subprocess.run([str(self.binary), str(rows), "384"],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["payload_bytes"],
                                 [rows * 384 * 4, 384 * 384 * 4, 384 * 4, rows * 384 * 4])
                self.assertEqual(report["guard_cases"], 1024)
                self.assertEqual(report["readonly_cases"], 9)
                self.assertEqual(report["outputs_checked"], rows * 384)
                self.assertFalse(report["gpu_dispatched"])

    def test_invalid_shapes(self):
        for rows, width in ((0, 384), (129, 384), (32, 383), (32, 385)):
            result = subprocess.run([str(self.binary), str(rows), str(width)],
                                    capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 3)


if __name__ == "__main__":
    unittest.main()
