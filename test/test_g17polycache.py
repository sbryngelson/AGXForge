"""The solver cache must be keyed on what its answers actually depend on.

`assembler._poly_stamp()` decides whether cached bit solves are still valid. It named
("g17as.py", "g17encode.py") and joined them against a directory: before the migration that was
tools/ and both resolved; afterwards the anchor was the checkout root and NEITHER did, so each was
stamped "missing" and the cache stopped noticing when its inputs changed. Root found it.

The module-constant sweep in test_g17librarycompat could not catch this one, because these paths
are assembled inside a function from a tuple of basenames rather than bound at module level. That
is the reason this file exists: the property to test is not "the constants resolve" but "the stamp
changes when the thing it stands for changes".
"""
import os
import shutil
import sys
import unittest
import tempfile
import json
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import agxforge.g17.assembler as A


class TheStampNamesTheRealImplementations(unittest.TestCase):
    def test_the_sources_exist_and_are_the_package_modules(self):
        self.assertEqual([Path(p).name for p in A._POLY_SOURCES], ["assembler.py", "encode.py"])
        for path in A._POLY_SOURCES:
            with self.subTest(path):
                self.assertTrue(os.path.exists(path), path)
                self.assertEqual(Path(path).parent, ROOT / "agxforge" / "g17")

    def test_it_does_not_name_the_compatibility_shims(self):
        """Solver semantics live in the implementation; a shim holds none, so hashing one would
        make the stamp blind to every change that matters."""
        for path in A._POLY_SOURCES:
            with self.subTest(path):
                self.assertNotIn("tools", Path(path).parts)

    def test_a_stamp_is_produced_at_all(self):
        self.assertIsInstance(A._poly_stamp(), str)


class _Edited:
    """Change a source's CONTENT while preserving its size and mtime, then restore it."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.original = open(self.path, "rb").read()
        self.stat = os.stat(self.path)
        body = bytearray(self.original)
        i = body.find(b"\n# ")
        assert i > 0, "no comment line to perturb"
        body[i + 2:i + 3] = b"#" if body[i + 2:i + 3] != b"#" else b" "
        assert len(body) == len(self.original)
        open(self.path, "wb").write(bytes(body))
        os.utime(self.path, ns=(self.stat.st_atime_ns, self.stat.st_mtime_ns))
        return self

    def __exit__(self, *exc):
        open(self.path, "wb").write(self.original)
        os.utime(self.path, ns=(self.stat.st_atime_ns, self.stat.st_mtime_ns))
        return False


class IsolatedCacheCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        sources = []
        for source in A._POLY_SOURCES:
            target = self.directory / Path(source).name
            shutil.copy2(source, target)
            sources.append(str(target))
        maps = self.directory / 'maps.jsonl'
        shutil.copy2(A.MAPS, maps)
        self.disk = self.directory / 'cache.json'
        for name, value in {
            '_POLY_SOURCES': tuple(sources), 'MAPS': str(maps),
            '_POLY_DIR': str(self.directory), '_POLY_DISK': str(self.disk),
            '_POLY_CACHE': {}, '_POLY_LOADED': [False], '_POLY_IDENTITY': [None],
        }.items():
            replacement = patch.object(A, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)


class TheStampFollowsContentNotStat(IsolatedCacheCase):
    def test_an_edit_that_preserves_size_and_mtime_still_invalidates(self):
        before = A._poly_stamp()
        for source in A._POLY_SOURCES:
            with self.subTest(Path(source).name):
                with _Edited(source) as e:
                    self.assertEqual(os.stat(source).st_size, e.stat.st_size)
                    self.assertEqual(os.stat(source).st_mtime_ns, e.stat.st_mtime_ns)
                    self.assertNotEqual(A._poly_stamp(), before)
                self.assertEqual(A._poly_stamp(), before)


class CacheResultsKeepTheirSourceIdentity(IsolatedCacheCase):
    def test_loaded_results_are_not_relabelled_after_a_source_change(self):
        old = A._poly_stamp()
        self.disk.write_text(json.dumps({'stamp': old,
                                         'entries': {'old-result': [7, None]}}))
        A._poly_disk_load()
        self.assertIn('old-result', A._POLY_CACHE)
        with _Edited(A._POLY_SOURCES[0]):
            new = A._poly_stamp()
            self.assertNotEqual(new, old)
            A.poly_cache_save()
            saved = json.loads(self.disk.read_text())
            if saved['stamp'] == new:
                self.assertNotIn('old-result', saved['entries'],
                                 'old results were published under a new source identity')


class TheWarmPathStillPersists(IsolatedCacheCase):
    """The control for the refusal above: refusing to rebrand must not refuse everything.

    Binding entries to their source identity risks the opposite defect - a cache that never writes,
    which costs every run a full solve and would look like a speed regression rather than a bug.
    """

    def test_a_normal_load_then_save_writes_the_entries(self):
        stamp = A._poly_stamp()
        self.disk.write_text(json.dumps({'stamp': stamp, 'entries': {'loaded': [3, None]}}))
        A._poly_disk_load()
        A._POLY_CACHE['solved-now'] = (9, None)
        A.poly_cache_save()
        saved = json.loads(self.disk.read_text())
        self.assertEqual(saved['stamp'], stamp)
        self.assertIn('loaded', saved['entries'], "a loaded entry was dropped with no drift")
        self.assertIn('solved-now', saved['entries'], "a fresh solve was not persisted")

    def test_a_first_save_with_no_prior_load_still_writes(self):
        """Nothing has established an identity yet, so there is nothing to conflict with."""
        A._POLY_CACHE['fresh'] = (5, None)
        A.poly_cache_save()
        saved = json.loads(self.disk.read_text())
        self.assertEqual(saved['stamp'], A._poly_stamp())
        self.assertIn('fresh', saved['entries'])

    def test_the_identity_is_recorded_even_when_the_disk_holds_nothing(self):
        """A load that finds no usable file still fixes what this process's solves belong to."""
        A._poly_disk_load()
        self.assertEqual(A._POLY_IDENTITY[0], A._poly_stamp())


class AMissingSourceDisablesTheCache(IsolatedCacheCase):
    def test_a_missing_source_yields_no_stamp(self):
        Path(A._POLY_SOURCES[1]).unlink()
        self.assertIsNone(A._poly_stamp())

    def test_the_disk_cache_neither_loads_nor_saves_without_a_stamp(self):
        # A null-stamped cache is precisely what a failed-open loader could accept.
        self.disk.write_text(json.dumps({'stamp': None,
                                         'entries': {'stale': [123, None]}}))
        original = self.disk.read_bytes()
        Path(A._POLY_SOURCES[1]).unlink()
        self.assertIsNone(A._poly_stamp())
        A._poly_disk_load()
        self.assertEqual(A._POLY_CACHE, {}, 'an unkeyed disk entry was loaded')
        A._POLY_CACHE['new'] = (456, None)
        A.poly_cache_save()
        self.assertEqual(self.disk.read_bytes(), original, 'unkeyed entries were saved')
        self.assertEqual(sorted(p.name for p in self.directory.iterdir()),
                         ['assembler.py', 'cache.json', 'maps.jsonl'])


if __name__ == "__main__":
    unittest.main()
