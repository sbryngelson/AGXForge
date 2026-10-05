"""Author-side consumption of ABI v5's execution requirement."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import g17authorobj as A

TEXT = bytes.fromhex("0e000000") + bytes.fromhex("0600") * 30 + b"\x00" * 16
BINDINGS = [(1, 0, False, "half"), (2, 2, False, "half"), (3, 4, True, "float")]


def abi(**over):
    base = dict(abi_version=5, entry=64, arch_flag=True, has_stores=True, writes_buffer=True,
                writes_texture=False, uses_threadgroup=False, system_registers=[160],
                register_count=2)
    base.update(over)
    return base


class TheVersionIsAccepted(unittest.TestCase):
    def test_v5_is_known(self):
        self.assertIn(5, A.ABI_VERSIONS)

    def test_an_unknown_version_is_still_refused_by_name(self):
        # THE VERSION IS DERIVED, NOT WRITTEN DOWN. This test named 6 as "unknown", and 6 became
        # known when the texture contract landed - so the test failed for the one reason that is
        # not a defect. What it means is "a version outside the known set", and that is now what
        # it asks for, one past the highest.
        unknown = max(A.ABI_VERSIONS) + 1
        self.assertNotIn(unknown, A.ABI_VERSIONS)
        with self.assertRaisesRegex(ValueError, "not one this side knows"):
            A.author(TEXT, 64, BINDINGS, abi(abi_version=unknown))

    def test_v6_is_known_and_demands_its_block(self):
        self.assertIn(6, A.ABI_VERSIONS)
        with self.assertRaisesRegex(A.Missing, "resources block"):
            A.author(TEXT, 64, BINDINGS, abi(abi_version=6))


class TheRequirementIsRecordedNotEncoded(unittest.TestCase):
    """Closed on the compiler owner's written statement that it is a launch fact."""

    def ledger(self, **over):
        led = {}
        try:
            A._metadata([(1, 0, False), (2, 2, False), (3, 4, True)],
                        abi(execution=dict(simd_width=32, tensor=True), **over), led)
        except Exception:
            pass
        return led

    def test_it_reaches_the_field_ledger(self):
        entry = self.ledger().get("execution requirement", "")
        self.assertIn("LAUNCH BOUNDARY", entry)
        self.assertIn("simd_width", entry)

    def test_the_ledger_says_the_basis_is_a_statement_not_a_measurement(self):
        # The distinction is the whole point: a reader six weeks from now must not take this
        # for a measured fact about the section.
        entry = self.ledger().get("execution requirement", "")
        self.assertIn("statement", entry)
        self.assertIn("not on a measurement", entry)

    def test_a_contract_without_the_field_records_nothing(self):
        led = {}
        try:
            A._metadata([(1, 0, False), (2, 2, False), (3, 4, True)], abi(), led)
        except Exception:
            pass
        self.assertNotIn("execution requirement", led)

    def test_a_typed_execution_object_is_recorded_the_same_way(self):
        import g17abi
        led = {}
        try:
            A._metadata([(1, 0, False), (2, 2, False), (3, 4, True)],
                        abi(execution=g17abi.ExecutionABI(simd_width=32, tensor=True)), led)
        except Exception:
            pass
        self.assertIn("LAUNCH BOUNDARY", led.get("execution requirement", ""))

    def test_no_class_claims_to_encode_it(self):
        self.assertEqual(A.MEASURED_EXECUTION_CLASSES, ())


if __name__ == "__main__":
    unittest.main()
