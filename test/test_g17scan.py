"""CPU-only acceptance tests; no Metal imports or GPU dispatch."""
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import g17scan as scan
import g17scanlink as link


class ReferenceTests(unittest.TestCase):
    def test_contract_preserves_real_input_layout(self):
        contract = scan.Scan().contract()
        self.assertEqual([b["bytes"] for b in contract["bindings"]],
                         [384_000_000, 768, 1_000_000])
        self.assertEqual([b["readonly"] for b in contract["bindings"]], [True, True, False])

    def test_reference_handles_exact_products_and_cancellation(self):
        a = np.array([[1, 2, 3], [1, -1, 0], [0, 0, 0]], dtype=np.float16)
        x = np.array([2, 2, 4], dtype=np.float16)
        exact, _ = scan.reference(a, x)
        np.testing.assert_array_equal(exact, [18, 0, 0])
        self.assertTrue(scan.check_output(a, x, exact.astype(np.float16))["ok"])
        bad = exact.astype(np.float16)
        bad[1] = 1
        self.assertEqual(scan.check_output(a, x, bad)["failed_rows"], [1])

    def test_real_shape_sequential_fp32_fits_reference_bound(self):
        rng = np.random.default_rng(7)
        a = (rng.standard_normal((33, 384)) * 0.05).astype(np.float16)
        x = (rng.standard_normal(384) * 0.05).astype(np.float16)
        result = np.zeros(33, np.float32)
        for k in range(384):
            result = np.float32(result + a[:, k].astype(np.float32) * np.float32(x[k]))
        self.assertTrue(scan.check_output(a, x, result.astype(np.float16))["ok"])
        # A wrong-buffer-like perturbation must fail, even near cancellation.
        self.assertFalse(scan.check_output(a, x, (result + 0.1).astype(np.float16))["ok"])

    def test_output_shape_dtype_and_nonfinite_are_not_accepted(self):
        a, x = np.ones((2, 3), np.float16), np.ones(3, np.float16)
        for out in (np.ones(2, np.float32), np.ones(1, np.float16),
                    np.array([np.nan, 3], np.float16)):
            self.assertFalse(scan.check_output(a, x, out)["ok"])
        with self.assertRaises(ValueError):
            scan.reference(a.astype(np.float32), x)
        with self.assertRaises(ValueError):
            scan.reference(a, x * np.float16(np.inf))

    def test_fixtures_are_reproducible(self):
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            aa = scan.write_fixtures(Path(one), 384)
            bb = scan.write_fixtures(Path(two), 384)
            self.assertEqual(aa, bb)
            for case in aa:
                with np.load(Path(one) / case["path"]) as arrays:
                    exact, error = scan.reference(arrays["index"], arrays["embedding"])
                    np.testing.assert_array_equal(exact, arrays["exact"])
                    np.testing.assert_array_equal(error, arrays["tolerance"])

    def test_precision_and_address_domain_are_enforced(self):
        for kwargs in ({"rows": 0}, {"columns": 0}, {"rows": 2**32, "columns": 2},
                       {"input_dtype": "float32"}):
            with self.assertRaises(ValueError):
                scan.Scan(**kwargs)


class ObjectAdapterTests(unittest.TestCase):
    @staticmethod
    def kernel():
        import g17link as L
        return L.Kernel(bytes.fromhex("0e000000"),
                        [L.Binding(1, readonly=True), L.Binding(2, readonly=False)],
                        prologue=L.PROLOGUE_WORD + L.FILLER * 30)

    @staticmethod
    def fixture_abi():
        # Known synthetic authoring fixture ONLY. These numbers never supply
        # missing fields in the scan path or its readiness report.
        import g17authorobj as A
        return A._abi()

    def test_complete_fixture_links_and_checks_delivered_bytes(self):
        import g17verify as V
        image = link.link(self.kernel(), self.fixture_abi(), binding_offsets=[0, 2])
        self.assertFalse(V.verify(image.archive, image.library))
        self.assertEqual(link.object_from_archive(image.archive), image.object)
        self.assertEqual(link.binding_records(V._objsect(image.object, "__GPU_METADATA")),
                         [(1, 0, False), (2, 2, True)])
        with self.assertRaisesRegex(ValueError, "differs from compiler contract"):
            link.verify_contract(image.archive, image.library, [(2, 2, True), (1, 0, False)])

    def test_abi_omission_and_resource_errors_refuse_before_authoring(self):
        import g17authorobj as A
        with self.assertRaises(A.Missing):
            link.link(self.kernel(), {}, binding_offsets=[0, 2])
        for offsets in ([0], [0, 0], [0, 3], [-2, 0]):
            with self.assertRaises(ValueError):
                link.link(self.kernel(), self.fixture_abi(), binding_offsets=offsets)
        k = self.kernel()
        k.prologue = None
        with self.assertRaises(A.Missing):
            link.link(k, self.fixture_abi(), binding_offsets=[0, 2])
        k = self.kernel()
        k.bindings[0].element_type = "half"
        typed = link.link(k, self.fixture_abi(), binding_offsets=[0, 2])
        self.assertIn("element types", typed.field_ledger)
        k.bindings[0].element_type = "unknown_resource_type"
        with self.assertRaisesRegex(ValueError, "element type"):
            link.link(k, self.fixture_abi(), binding_offsets=[0, 2])

    def test_binding_metadata_is_part_of_archive_identity(self):
        k = self.kernel()
        first = link.link(k, self.fixture_abi(), binding_offsets=[0, 2])
        k.bindings[1].index = 3
        second = link.link(k, self.fixture_abi(), binding_offsets=[0, 2])
        self.assertNotEqual(first.library, second.library)
        self.assertNotEqual(first.sha256, second.sha256)

    def test_mutated_archive_cannot_pass(self):
        image = link.link(self.kernel(), self.fixture_abi(), binding_offsets=[0, 2])
        damaged = bytearray(image.archive)
        struct.pack_into(">I", damaged, 16, len(damaged) + 100)
        with self.assertRaises((ValueError, struct.error)):
            link.verify_contract(bytes(damaged), image.library, [(1, 0, False), (2, 2, True)])


class ProbeTests(unittest.TestCase):
    def test_guard_is_proven_to_block_real_dependency_attempts(self):
        r = scan.run_probe("guard_control", 10)
        self.assertEqual(r["status"], "passed", r)
        self.assertEqual(r["caught"], ["subprocess", "shader_cache"])

    def test_half_load_probe_is_not_confounded_by_store_hazard(self):
        r = scan.run_probe("half_load", 10)
        self.assertNotEqual(r["status"], "error", r)
        self.assertIn("half", r)
        # Half lowering now lands a distinct memory form. Selection alone does
        # not establish the addressing or hardware semantics of the new form.
        self.assertFalse(r["identical"], r)
        self.assertEqual(r["half"][0]["fields"]["half"], 1)
        self.assertEqual(r["word"][0]["fields"]["half"], 0)
        self.assertEqual(r["status"], "needs_review")

    def test_no_loop_override_or_shared_solver_cache_leaks(self):
        old = os.environ.get("G17_ALLOW_LOOP")
        os.environ["G17_ALLOW_LOOP"] = "1"
        try:
            r = scan.run_probe("abi_handoff", 10)
            self.assertEqual(os.environ["G17_ALLOW_LOOP"], "1")
            self.assertEqual(r["status"], "blocked", r)
            self.assertIn("ld_md_slots", r["missing_required_keys"])
            self.assertIsNone(r["environment"]["loop_override"])
            self.assertNotIn("/.cache/agxforge/", r["environment"]["solver_cache"])
        finally:
            if old is None:
                os.environ.pop("G17_ALLOW_LOOP", None)
            else:
                os.environ["G17_ALLOW_LOOP"] = old

    def test_actual_target_retains_half_operations(self):
        r = scan.run_probe("target", 10, scan.Scan(rows=33, columns=384))
        self.assertEqual(r["kind"], "target")
        self.assertEqual(r["status"], "blocked", r)
        for op in ("load_f16", "f16_to_f32", "f32_to_f16_rte", "store_f16"):
            self.assertIn(op, r["requested_ir"])
        self.assertIn("stride = const 384", r["requested_ir"])


if __name__ == "__main__":
    unittest.main()
