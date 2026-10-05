import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import g17checkpoint as C
from agxforge.g17 import inferencegraph as I


class Checkpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name).resolve() / 'weights'
        self.header = b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
        self.payload = len(self.header).to_bytes(8, 'little') + self.header + b'\x00\x00\x80?'
        self.manifest = dict(repository='test/model', revision='a'*40, checkpoint_bytes=len(self.payload),
                             expected_checkpoint_sha256=hashlib.sha256(self.payload).hexdigest())

    def fetch(self, payload):
        return C.fetch(self.path, self.manifest, self.header,
                       opener=lambda *a, **k: io.BytesIO(payload))

    def test_fetch_verify_and_reuse_without_network(self):
        self.assertTrue(self.fetch(self.payload)['weight_payload_verified'])
        self.assertEqual(self.path.read_bytes(), self.payload)
        C.fetch(self.path, self.manifest, self.header, opener=lambda *a, **k: self.fail('network'))
        self.assertEqual(self.path.stat().st_nlink, 1)

    def test_corrupt_payload_not_published(self):
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.fetch(self.payload[:-1] + b'!')
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_short_payload_not_published(self):
        with self.assertRaisesRegex(ValueError, 'size'):
            self.fetch(self.payload[:-1])
        self.assertFalse(self.path.exists())

    def test_oversized_payload_not_published(self):
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            self.fetch(self.payload + b'!')
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_wrong_header_even_with_matching_payload_hash(self):
        self.path.write_bytes(self.payload)
        with self.assertRaisesRegex(ValueError, 'header'):
            C.verify(self.path, self.manifest, b'?' * len(self.header))

    def test_existing_corrupt_file_preserved(self):
        self.path.write_bytes(b'original')
        with self.assertRaises(ValueError):
            self.fetch(self.payload)
        self.assertEqual(self.path.read_bytes(), b'original')

    def test_tensor_raw_chunks_and_bounds(self):
        self.fetch(self.payload)
        parameter = I.checkpoint_header(self.header, file_bytes=len(self.payload))['x']
        chunks = list(C.tensor_chunks(self.path, self.manifest, self.header, parameter, chunk_bytes=3))
        self.assertEqual(chunks, [self.payload[-4:-1], self.payload[-1:]])
        with self.assertRaisesRegex(ValueError, 'chunk size'):
            list(C.tensor_chunks(self.path, self.manifest, self.header, parameter, chunk_bytes=0))

    def test_download_deadline_cleans_partial(self):
        with self.assertRaises(TimeoutError):
            C.fetch(self.path, self.manifest, self.header,
                    opener=lambda *a, **k: io.BytesIO(self.payload), deadline_seconds=-1)
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_concurrent_creator_is_not_overwritten(self):
        def opener(*args, **kwargs):
            self.path.write_bytes(b'other owner')
            return io.BytesIO(self.payload)
        with self.assertRaises(FileExistsError):
            C.fetch(self.path, self.manifest, self.header, opener=opener)
        self.assertEqual(self.path.read_bytes(), b'other owner')
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_symlink_refused(self):
        target = self.path.with_name('target')
        target.write_bytes(self.payload)
        self.path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.fetch(self.payload)
        self.assertEqual(target.read_bytes(), self.payload)

    def test_reader_named_chunks_limit_and_closed_state(self):
        self.fetch(self.payload)
        with C.CheckpointReader(self.path, self.manifest, self.header) as reader:
            self.assertEqual(b''.join(reader.chunks('x', chunk_bytes=3)), self.payload[-4:])
            with self.assertRaisesRegex(ValueError, 'bounded conversion'):
                reader.small_tensor('x', limit=3)
            with self.assertRaises(KeyError):
                list(reader.chunks('absent'))
        with self.assertRaisesRegex(ValueError, 'closed'):
            reader.check()

    def test_reader_rejects_mutation_after_verification(self):
        self.fetch(self.payload)
        reader = C.CheckpointReader(self.path, self.manifest, self.header)
        with self.assertRaisesRegex(ValueError, 'changed'):
            with reader:
                with self.path.open('r+b') as output:
                    output.seek(-1, 2)
                    output.write(b'!')
                reader.small_tensor('x')
        self.assertIsNone(reader.stream)


if __name__ == '__main__':
    unittest.main()
