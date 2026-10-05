from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agxforge.g17 import model


class DecoderOutputCacheTests(unittest.TestCase):
    def setUp(self):
        model._decode_stdout.cache_clear()

    def tearDown(self):
        model._decode_stdout.cache_clear()

    def test_identical_decode_requests_share_one_native_launch(self):
        calls = []

        def fake_run(argv, **kwargs):
            with open(argv[1], "rb") as stream:
                calls.append((bytes(stream.read()), tuple(argv[2:])))
            return SimpleNamespace(stdout="0 4 423 reg:105\n")

        with patch.object(model, "opcodes", return_value={423: object()}), \
             patch("agxforge.g17.ref.binary", return_value="/decoder"), \
             patch.object(model.subprocess, "run", side_effect=fake_run):
            first = list(model.decode(b"\x01\x02\x03\x04"))
            second = list(model.decode(bytearray(b"\x01\x02\x03\x04")))
            self.assertEqual(len(first), 1)
            self.assertEqual(first[0].raw, second[0].raw)
            self.assertEqual(len(calls), 1)
            # The key is the complete stream plus the decode offset; a different offset
            # must still reach the decoder.
            list(model.decode(b"\x01\x02\x03\x04", start=1))
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
