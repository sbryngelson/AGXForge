"""The inherited-bit audit never reports "nothing inherited" for a table it could not read (agxforge/g17/cc.py
_inherited_bits; ledger g17-the-inherited-bit-audit-reads-an-unreadable-table-as-none).

It used to `except Exception: return {}`, so g17regress's `p.inherited == {}` passed VACUOUSLY on an unreadable
isa/g17-form-constants.toml. (With the default template source the emitter reads the same table first and fails
loudly; with another source - the harvested registry - the audit is the only reader, and it answered {}.) Each test
below compiles a program with the real table, then breaks the table on purpose and requires the audit of that
program's layout to FAIL; the control audits the same layout with the real table."""
import os, sys, tempfile, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]


def _program():
    from agxforge.g17 import ir
    f = ir.Function("t", [ir.Buffer("a", 1), ir.Buffer("c", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    b.store_at(f.buffers[1], t, b.add(b.load(f.buffers[0], t, name="x"), ir.Imm(1), name="y"))
    b.ret()
    return f


class AnUnreadableTableIsNotClean(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agxforge.g17 import cc
        cls.layout = cc.compile_function(_program()).layout

    def setUp(self):
        from agxforge.g17 import const
        self.const, self.load, self.inventory = const, const.load, const.inventory

    def tearDown(self):
        self.const.load, self.const.inventory = self.load, self.inventory

    def test_the_real_table_audits(self):
        """The control: with the committed table the audit runs and returns a dict (here: nothing inherited)."""
        from agxforge.g17 import cc
        self.assertIsInstance(cc._inherited_bits(self.layout), dict)

    def test_a_corrupt_table_fails_the_audit(self):
        from agxforge.g17 import cc
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write("[[form]\nthis is not toml = = =\n")
            bad = f.name
        try:
            self.const.load = lambda path=bad: self.load(path)
            with self.assertRaisesRegex(cc.InheritedBitsUnreadable, "form-constants table"):
                cc._inherited_bits(self.layout)
        finally:
            os.unlink(bad)

    def test_a_missing_table_fails_the_audit(self):
        from agxforge.g17 import cc
        missing = os.path.join(tempfile.gettempdir(), "no-such-g17-form-constants.toml")
        self.const.load = lambda path=missing: self.load(path)
        with self.assertRaisesRegex(cc.InheritedBitsUnreadable, "FileNotFoundError"):
            cc._inherited_bits(self.layout)

    def test_an_unreadable_inventory_fails_the_audit(self):
        """One level down: an unreadable inventory silently switched off the UNCOVERED check."""
        from agxforge.g17 import cc

        def broken():
            raise OSError("inventory unreadable (deliberately)")
        self.const.inventory = broken
        with self.assertRaisesRegex(cc.InheritedBitsUnreadable, "form inventory"):
            cc._inherited_bits(self.layout)


if __name__ == "__main__":
    unittest.main()
