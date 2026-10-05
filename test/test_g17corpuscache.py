"""A probe name must not make a different source appear successfully compiled."""
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import g17corpus as C


class CorpusCache(unittest.TestCase):
    def test_changed_or_unattributed_source_cannot_reuse_cached_success(self):
        old='kernel void k() {}\n';new='kernel void k() { /* changed */ }\n'
        with tempfile.TemporaryDirectory() as tmp,patch.object(C,'CACHE',tmp),\
             patch.object(C.subprocess,'run',side_effect=AssertionError('no compiler')),\
             patch.object(C.ctypes,'CDLL',side_effect=AssertionError('no Metal')):
            d=Path(tmp)/'probe';obj=d/'out/object/0-0';obj.parent.mkdir(parents=True)
            obj.write_bytes(b'retained native evidence')
            source=d/'s.metal';source.write_text(old)
            self.assertEqual(C.build('probe',old),'cached')
            self.assertTrue(C.build('probe',new).startswith('CACHE REFUSED: requested Metal source differs'))
            self.assertEqual(source.read_text(),old)
            self.assertEqual(obj.read_bytes(),b'retained native evidence')
            source.unlink()
            self.assertEqual(C.build('probe',old),'CACHE REFUSED: retained Metal source is unavailable')
            self.assertEqual(obj.read_bytes(),b'retained native evidence')

class ClassCacheIdentity(unittest.TestCase):
    def test_cache_implementation_is_an_input_to_its_own_stamp(self):
        from agxforge.g17 import cache
        implementation = Path(cache.__file__).resolve().relative_to(Path(cache.ROOT).resolve()).as_posix()
        self.assertIn(implementation, cache.KEY_SOURCES)

    def test_same_size_source_edit_with_preserved_mtime_changes_stamp(self):
        import os
        from agxforge.g17 import cache
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "reader.py"
            source.write_bytes(b"answer = 1\n")
            stat = source.stat()
            with patch.object(cache, 'ROOT', tmp), patch.object(cache, 'KEY_SOURCES', ('reader.py',)), \
                 patch.object(cache, '_corpus_dir', return_value=tmp):
                before = cache.stamp()
                source.write_bytes(b"answer = 2\n")
                os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                self.assertNotEqual(cache.stamp(), before)

if __name__=='__main__':unittest.main()
