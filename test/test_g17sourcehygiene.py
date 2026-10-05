#!/usr/bin/env python3
r"""Two whole-tree guards for defects that make OTHER guards stop running.

Both of these were live in this repository on 2026-09-20 and neither was caught by a test:

  A test class defined BELOW `if __name__ == "__main__":` is never created when the file is run
  directly, because main() runs first.  `NobodyRespellsTheRegisterPattern` sat there and reported a
  silent pass to anyone not using discover -- a ratchet that does not run is the defect it exists to
  prevent, one level up.  The tell was a test COUNT that differed between two invocations of the
  same commit (11 direct, 14 under discover).

  An invalid escape (`"\d"`) is a SyntaxWarning today and becomes a SyntaxError in a future Python,
  which would stop the file importing at all.

Each guard here is paired with a control that constructs the defect and requires the detector to
report it, because a scan that silently finds nothing satisfies a "no hits" assertion vacuously.
"""
import ast
import pathlib
import unittest
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _defs_after_main(source):
    """Names defined at module level AFTER an `if __name__ == "__main__"` block."""
    tree = ast.parse(source)
    main_line = None
    for node in tree.body:
        if isinstance(node, ast.If) and ast.unparse(node.test).replace('"', "'") == "__name__ == '__main__'":
            main_line = node.lineno
    if main_line is None:
        return []
    return [n.name for n in tree.body
            if n.lineno > main_line and isinstance(n, (ast.ClassDef, ast.FunctionDef))]


def _syntax_warnings(source, filename):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile(source, filename, "exec")
        return [str(w.message) for w in caught if issubclass(w.category, SyntaxWarning)]


def _python_files():
    return [p for p in sorted(ROOT.rglob("*.py")) if ".git" not in p.parts]


class NothingIsDefinedBelowTheMainGuard(unittest.TestCase):
    def test_no_test_file_defines_a_class_or_function_after_main(self):
        offenders = {}
        scanned = 0
        for path in sorted((ROOT / "test").glob("*.py")):
            scanned += 1
            names = _defs_after_main(path.read_text(errors="replace"))
            if names:
                offenders[path.relative_to(ROOT).as_posix()] = names
        self.assertGreater(scanned, 100, "the scan must actually reach the test tree")
        self.assertEqual(offenders, {},
                         "these are never defined when the file is run directly: %r" % (offenders,))

    def test_the_detector_reports_a_definition_placed_below_the_guard(self):
        # The control: without this, the assertion above passes just as well on a broken scan.
        below = 'import unittest\nif __name__ == "__main__":\n    unittest.main()\n\n\nclass Late:\n    pass\n'
        above = 'import unittest\n\n\nclass Early:\n    pass\n\n\nif __name__ == "__main__":\n    unittest.main()\n'
        self.assertEqual(_defs_after_main(below), ["Late"])
        self.assertEqual(_defs_after_main(above), [])


class NoSourceFileCarriesASyntaxWarning(unittest.TestCase):
    def test_every_python_file_compiles_without_a_syntax_warning(self):
        offenders = {}
        files = _python_files()
        for path in files:
            try:
                found = _syntax_warnings(path.read_text(errors="replace"), str(path))
            except SyntaxError as exc:
                offenders[path.relative_to(ROOT).as_posix()] = "SyntaxError: %s" % exc
                continue
            if found:
                offenders[path.relative_to(ROOT).as_posix()] = found
        self.assertGreater(len(files), 500, "the scan must actually reach the source tree")
        self.assertEqual(offenders, {},
                         "an invalid escape is a SyntaxError in a future Python: %r" % (offenders,))

    def test_the_detector_reports_an_invalid_escape(self):
        self.assertTrue(_syntax_warnings('x = "\\d"', "<control>"))
        self.assertTrue(_syntax_warnings('def f():\n    """quotes \\d"""\n', "<control-docstring>"))
        self.assertFalse(_syntax_warnings('x = r"\\d"', "<control-raw>"))


class NoTestMutatesATrackedFile(unittest.TestCase):
    r"""A test that rewrites a committed file is safe serially and a race in parallel.

    `test_g17ledgeraudit` proved its freshness check could fail by APPENDING to
    docs/g17-tensorops-machine-model.md and restoring it. A dozen modules read that document.
    Run concurrently, two instances each save an "original" and the later restore writes the
    earlier one's mutated copy back -- measured, five times out of five, with this repository's
    machine-model document left modified on disk carrying the test's own fixture sentence.

    Four modules had the shape, found by this scan rather than by the one that bit: ledgeraudit,
    platform, questions and vendorforms. The repair is the same in each -- the tool takes a path
    (`--doc`, `--record`, `--out`, `--index`), the test copies the file to a temporary directory
    and points the tool at the copy. The runner-side hold-out that preceded this was a net, not a
    fix: it made the race impossible for the modules someone had remembered to list.

    So the list is not maintained by hand. This asks the AST.
    """

    def _offenders(self):
        import ast
        out = {}
        for path in sorted((ROOT / "test").glob("test_g17*.py")):
            source = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            consts = {n.targets[0].id: (ast.get_source_segment(source, n.value) or "")
                      for n in ast.walk(tree)
                      if isinstance(n, ast.Assign) and len(n.targets) == 1
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id.isupper()}
            bad = set()
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "open" and node.args):
                    target = node.args[0]
                    mode = ""
                    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                        mode = node.args[1].value
                    for kw in node.keywords or []:
                        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                            mode = kw.value.value
                    if (isinstance(target, ast.Name) and target.id in consts
                            and isinstance(mode, str) and set("wa") & set(mode)):
                        bad.add(target.id)
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("remove", "unlink") and node.args):
                    target = node.args[0]
                    if isinstance(target, ast.Name) and target.id in consts:
                        bad.add(target.id)
            # a constant pointing into the checkout, not into a temporary directory
            tracked = sorted(b for b in bad
                             if ("ROOT" in consts[b] or "os.path.join" in consts[b])
                             and "tempfile" not in consts[b] and "/tmp" not in consts[b])
            if tracked:
                out[path.name] = tracked
        return out

    def test_no_test_module_writes_a_repo_rooted_path(self):
        self.assertEqual(self._offenders(), {},
                         "these modules write or remove a file inside the checkout. Serial that "
                         "is safe; in parallel it races every reader and the loser leaves the "
                         "tracked file modified on disk. Give the tool a path argument and point "
                         "it at a copy, as test_g17ledgeraudit now does with --doc/--record.")

    def test_the_scan_can_actually_find_one(self):
        """A sweep that reports nothing is worthless until it has been shown to report something.

        The offender is synthesized, so this cannot pass by the population happening to be clean.
        """
        import ast, tempfile, textwrap
        with tempfile.TemporaryDirectory() as d:
            planted = ROOT / "test" / "test_g17__planted_offender.py"
            try:
                planted.write_text(textwrap.dedent('''
                    import os
                    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    DOC = os.path.join(ROOT, "docs", "some-tracked-file.md")
                    def touch():
                        with open(DOC, "w") as fh:
                            fh.write("x")
                '''), encoding="utf-8")
                found = self._offenders()
                self.assertIn("test_g17__planted_offender.py", found,
                              "the scan did not see a module that plainly writes a tracked path")
                self.assertEqual(found["test_g17__planted_offender.py"], ["DOC"])
            finally:
                planted.unlink(missing_ok=True)
        self.assertEqual(self._offenders(), {}, "the planted offender was not removed")


if __name__ == "__main__":
    unittest.main()
