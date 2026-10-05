import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import g17wordcontrol as control
import g17halfruntime
import g17halfcheck
import g17packedcheck


class WordControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.bundle = Path(cls.temp.name) / "control"
        cls.report = control.prepare(cls.bundle)

    def test_delivered_program_writes_exactly_one_bounded_word(self):
        self.assertEqual(self.report["write_byte_range"], [0, 4])
        self.assertEqual(self.report["allocation_bytes"], 130)
        self.assertEqual(self.report["word_stores"], 1)
        self.assertEqual([c["word_bits"] for c in self.report["cases"]],
                         [0x3f800000, 0xc0000000, 0x3f800000])
        self.assertEqual(self.report["preserved_guard_byte_range"], [4, 130])

    def test_normal_runtime_rejects_control_before_gpu(self):
        with self.assertRaisesRegex(ValueError, "FP32 device memory"):
            g17halfruntime.verify_bundle(self.bundle)
        for flag, limit in (("--half-validation-approved", "1"),
                            ("--half-word-control-approved", "2")):
            result = subprocess.run([str(ROOT / "tools/g17scanworker"), str(self.bundle),
                                     "missing", limit, flag], capture_output=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn(b"create_pipeline", result.stderr)

    def test_cpu_control_refuses_wider_grid_and_address_changes(self):
        _, _, program = control.build()
        ins = g17packedcheck.decode(program.code)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            g17halfcheck.simulate(ins, np.ones((2, 1), np.float16), np.ones(1, np.float16),
                                 rows=2, row_ids=[0, 1], store_bytes=4)
        next(t for _, _, op, t in ins if op == 17229)[7] = "imm:4"
        with self.assertRaisesRegex(ValueError, "displacement"):
            g17halfcheck.simulate(ins, np.ones((1, 1), np.float16), np.ones(1, np.float16),
                                 rows=1, row_ids=[0], store_bytes=4)

    def test_failed_offline_stage_records_failure_and_never_starts_process(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in g17halfruntime.FILES:
                (directory / name).write_bytes((self.bundle / name).read_bytes())
            with patch.object(control, "verify", side_effect=ValueError("bad code")), \
                 patch.object(control.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "bad code"):
                    control.execute(directory)
                run.assert_not_called()
            self.assertEqual(json.loads((directory / "execution.json").read_text())["status"], "failed")
            with self.assertRaises(FileExistsError):
                control.execute(directory)

    def test_control_frame_parser_rejects_incomplete_or_oversized_reply(self):
        header = json.dumps({"bytes": 4}).encode()
        good = struct.pack("<I", len(header)) + header + bytes(4)
        self.assertEqual(control.frames(good)[0][1], bytes(4))
        for bad in (good[:-1], b"\0", struct.pack("<I", 4097), struct.pack("<I", 0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                control.frames(bad)


if __name__ == "__main__":
    unittest.main()
