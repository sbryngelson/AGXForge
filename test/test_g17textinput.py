import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import g17textinput as T


class TokenizerIdentity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.model = dict(repository='test/model', revision='a'*40)
        files = {}
        for name in ('tokenizer.json', 'tokenizer_config.json'):
            raw = b'{}'
            (self.path / name).write_bytes(raw)
            files[name] = dict(bytes=2, sha256=hashlib.sha256(raw).hexdigest())
        self.identity = dict(format='g17-pinned-tokenizer-v1', **self.model, files=files)
        self.write_identity()

    def write_identity(self):
        (self.path / 'identity.json').write_text(json.dumps(self.identity))

    def test_valid_identity(self):
        self.assertEqual(T.validate_tokenizer(self.path, self.model), self.identity)

    def test_wrong_model_revision(self):
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            T.validate_tokenizer(self.path, dict(self.model, revision='b'*40))

    def test_changed_tokenizer(self):
        (self.path / 'tokenizer.json').write_bytes(b'[]')
        with self.assertRaisesRegex(ValueError, 'changed'):
            T.validate_tokenizer(self.path, self.model)

    def test_unpinned_local_config_refused(self):
        (self.path / 'config.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'unpinned'):
            T.validate_tokenizer(self.path, self.model)

    def test_symlink_file_refused(self):
        (self.path / 'tokenizer.json').unlink()
        (self.path / 'tokenizer.json').symlink_to(self.path / 'tokenizer_config.json')
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            T.validate_tokenizer(self.path, self.model)

    def test_traversal_filename_refused(self):
        self.identity['files']['../outside'] = dict(bytes=2, sha256='0'*64)
        self.write_identity()
        with self.assertRaises(ValueError):
            T.validate_tokenizer(self.path, self.model)


if __name__ == '__main__':
    unittest.main()
