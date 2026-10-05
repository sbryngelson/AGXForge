"""No test file may define a class or function BELOW its `if __name__` guard.

THE DEFECT THIS CATCHES, measured before it was fixed. These files grew by appending, and each
append landed under an existing guard block, so `python3 test/<file>.py` stopped at the first guard
and never defined the classes beneath it - while `unittest discover` imports the module and sees
them all. Both print "Ran N ... OK", which is why it survived:

    test_g17abi                 discover 34   direct  8      26 cases invisible
    test_g17ffnchainrequirements discover 20   direct  6      14 invisible
    test_g17textureclass        discover 56   direct 37      19 invisible
    test_g17capcompiler         discover 24   direct 19       5 invisible

integration's acceptance commands use the direct form, so accepting on one of those files was
accepting a quarter of it. The linker found it first, on their own file, and their checker's first
version failed on its own docstring by counting occurrences of the guard TEXT.

SO THIS USES `ast`, not text. A docstring mentioning `if __name__` is not a guard, and a checker
that cannot tell a quoted string from a statement is the same defect one level up - which is the
lesson from an audit that found the comment forbidding the thing it was looking for."""
import ast, glob, os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _guard_and_definitions(path):
    """(line of the first __main__ guard, lines of top-level defs/classes after it), from the AST."""
    tree = ast.parse(open(path, errors="ignore").read())
    guard = None
    for node in tree.body:
        if isinstance(node, ast.If):
            t = ast.dump(node.test)
            if "__name__" in t and "__main__" in t:
                guard = node.lineno
                break
    if guard is None:
        return None, []
    below = [n.lineno for n in tree.body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
             and n.lineno > guard]
    return guard, below


class EveryTestFileIsFullyVisibleToADirectRun(unittest.TestCase):
    def test_nothing_is_defined_below_the_main_guard(self):
        bad = []
        for path in sorted(glob.glob(os.path.join(ROOT, "test", "test_g17*.py"))):
            guard, below = _guard_and_definitions(path)
            if below:
                bad.append("%s: guard at line %d with %d definition(s) below it"
                           % (os.path.basename(path), guard, len(below)))
        self.assertEqual(bad, [], "a direct `python3 test/<file>.py` would not define these: "
                                  + "; ".join(bad))

    def test_no_file_has_more_than_one_guard(self):
        many = []
        for path in sorted(glob.glob(os.path.join(ROOT, "test", "test_g17*.py"))):
            tree = ast.parse(open(path, errors="ignore").read())
            n = sum(1 for node in tree.body if isinstance(node, ast.If)
                    and "__name__" in ast.dump(node.test) and "__main__" in ast.dump(node.test))
            if n > 1:
                many.append("%s has %d" % (os.path.basename(path), n))
        self.assertEqual(many, [], "; ".join(many))

    def test_the_checker_reads_the_ast_and_not_the_text(self):
        """A docstring that mentions the guard is not a guard. This file's own docstring contains
        `if __name__` four times over; a text-counting version would fail on itself, which is how
        the linker's first version failed."""
        text = open(os.path.abspath(__file__), errors="ignore").read()
        self.assertGreater(text.count("__name__"), 3, "the docstring really does mention it")
        guard, below = _guard_and_definitions(os.path.abspath(__file__))
        self.assertEqual(below, [], "and this file is itself clean by its own rule")

    def test_it_would_fire_on_a_planted_file(self):
        """THE CONTROL. Without it, an empty finding list is equally consistent with a walk that
        cannot find anything."""
        import tempfile
        src = ("import unittest\n\n\nclass A(unittest.TestCase):\n    def test_a(self): pass\n\n\n"
               "if __name__ == '__main__':\n    unittest.main()\n\n\n"
               "class B(unittest.TestCase):\n    def test_b(self): pass\n")
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(src); path = fh.name
        try:
            guard, below = _guard_and_definitions(path)
            self.assertIsNotNone(guard)
            self.assertEqual(len(below), 1, "class B is below the guard and must be reported")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
