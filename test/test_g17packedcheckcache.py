from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import g17packedcheck as packedcheck


class PackedDecoderCacheTests(unittest.TestCase):
    def setUp(self):
        packedcheck._decode_cached.cache_clear()

    def tearDown(self):
        packedcheck._decode_cached.cache_clear()

    def test_identical_streams_share_one_native_launch_and_return_fresh_lists(self):
        calls = []

        def fake_run(argv, **kwargs):
            with open(argv[1], "rb") as stream:
                calls.append((stream.read(), tuple(argv[2:])))
            return SimpleNamespace(stdout="0 4 684 imm:0\n")

        with patch.object(packedcheck, "g17ref") as ref, \
             patch.object(packedcheck.subprocess, "run", side_effect=fake_run):
            ref.binary.return_value = "/decoder"
            first = packedcheck.decode(b"\x01\x02\x03\x04")
            second = packedcheck.decode(bytearray(b"\x01\x02\x03\x04"))
            self.assertEqual(first, second)
            self.assertIsNot(first, second)
            first.clear()
            self.assertEqual(len(second), 1)
            self.assertEqual(calls, [(b"\x01\x02\x03\x04", ("0", "4", "--expr"))])

            packedcheck.decode(b"\x01\x02\x03\x05")
            self.assertEqual(len(calls), 2)

    def test_failed_decode_is_retried_instead_of_cached(self):
        calls = []

        def fake_run(argv, **kwargs):
            calls.append(tuple(argv))
            return SimpleNamespace(stdout="0 2 684 imm:0\n")

        with patch.object(packedcheck, "g17ref") as ref, \
             patch.object(packedcheck.subprocess, "run", side_effect=fake_run):
            ref.binary.return_value = "/decoder"
            for _ in range(2):
                with self.assertRaises(ValueError):
                    packedcheck.decode(b"\x09\x08\x07\x06")
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
