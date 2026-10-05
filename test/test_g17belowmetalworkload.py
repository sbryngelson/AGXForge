"""Persistent-runtime transport and refusal controls; no GPU dispatch."""
import hashlib
import json
import os
import sys
from pathlib import Path
import unittest
from unittest import mock
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import g17belowmetalworkload as W


class RuntimeControls(unittest.TestCase):
    def test_partial_writes_preserve_whole_request(self):
        class Partial:
            def __init__(self): self.data = bytearray()
            def write(self, data):
                n = min(3, len(data)); self.data.extend(data[:n]); return n
        p = Partial(); W.write_all(p, b'0123456789')
        self.assertEqual(p.data, b'0123456789')

    def test_zero_write_is_an_error(self):
        class Closed:
            def write(self, data): return 0
        with self.assertRaises(BrokenPipeError): W.write_all(Closed(), b'x')

    def test_short_native_reply_cannot_be_a_pass(self):
        r, w = os.pipe()
        with os.fdopen(r, 'rb', buffering=0) as stream:
            os.write(w, b'123'); os.close(w)
            with self.assertRaisesRegex(RuntimeError, 'closed before reply'):
                W.read_exact(stream, 4)

    def test_idle_reply_has_a_deadline(self):
        r, w = os.pipe()
        try:
            with os.fdopen(r, 'rb', buffering=0) as stream:
                with self.assertRaises(TimeoutError): W.read_exact(stream, 1, seconds=0.01)
        finally: os.close(w)

    def test_unsupported_numerics_refuse_before_conversion(self):
        for value in (0.5, 5, float('nan'), float('inf'), 65536):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'refused:'):
                W.control_operand(np.full((32, 64), value), (32, 64))

    def test_wrong_shape_refuses(self):
        with self.assertRaisesRegex(ValueError, 'refused:'):
            W.control_operand(np.zeros((16, 64)), (32, 64))

    def test_exact_integer_control_accepts_both_signs(self):
        a = np.arange(2048).reshape(32, 64) % 9 - 4
        self.assertTrue(np.array_equal(W.control_operand(a, (32, 64)).astype(int), a))

    def test_resigning_inventory_cannot_admit_an_altered_launch(self):
        # An attacker or buggy generator can rewrite its own JSON hashes. The
        # measured launch reconstruction must still reject changed native pages.
        code = bytes(1276)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(W, 'PROGRAM_SHA', W.sha(code)), mock.patch.object(W, 'compile_control', return_value=code):
            bundle = Path(directory) / 'bundle'; W.prepare(bundle)
            W.verify_bundle(bundle)
            pages = bytearray((bundle / 'pages.bin').read_bytes()); pages[0x3c0] ^= 1
            (bundle / 'pages.bin').write_bytes(pages)
            manifest = json.loads((bundle / 'manifest.json').read_text())
            manifest['files']['pages.bin']['sha256'] = hashlib.sha256(pages).hexdigest()
            (bundle / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'launch surface outside measured class'):
                W.verify_bundle(bundle)


if __name__ == '__main__': unittest.main()
