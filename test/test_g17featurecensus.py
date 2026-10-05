import copy
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import g17featurecensus as F


class FeatureCensusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = F.load()
        cls.facts = F._facts()

    def rows(self):
        return copy.deepcopy(self.doc["rows"])

    def test_the_written_census_checks(self):
        F.check()

    def test_fp64_is_refused_by_the_language(self):
        v = self.doc["verdict"]["conv.fp64"]
        self.assertEqual(v["verdict"], "refused_by_the_language")
        self.assertIn("'double' is not supported in Metal", v["compiler_errors"])

    def test_a_refusal_that_builds_refuses_the_census(self):
        rows = self.rows()
        rows["conv.fp64"] = dict(rows["baseline"], role="refusal", capability="compiler.conv.fp64", signature=[])
        with self.assertRaises(F.Refused):
            F.classify(rows, self.facts)

    def test_a_stale_signature_refuses(self):
        rows = self.rows()
        rows["mem.sampler"]["forms"] = [f for f in rows["mem.sampler"]["forms"] if f != "14661/22"]
        with self.assertRaises(F.Refused):
            F.classify(rows, self.facts)

    def test_cross_lane_records_are_not_counted_as_semantics(self):
        v = self.doc["verdict"]["simd.reductions"]
        self.assertEqual(len(v["cross_lane_with_uninformative_isolated_record"]), 5)
        self.assertEqual(v["with_isolated_record"], 1)   # the fadd, not a simd opcode

    def test_a_vertex_slot_difference_would_be_reported(self):
        rows = self.rows()
        rows["graphics.stages.vertex"]["metadata"]["per_kernel_slots"]["99"] = [0, 4, 1]
        twin = F.classify(rows, self.facts)["graphics.stages.vertex"]["against_its_compute_twin"]
        self.assertEqual(twin["per_kernel_slots_only_in_vertex"], ["99"])
        self.assertEqual(self.doc["verdict"]["graphics.stages.vertex"]["against_its_compute_twin"]["per_kernel_slots_only_in_vertex"], [])

    def test_a_different_vertex_form_sequence_is_reported(self):
        rows = self.rows()
        self.assertTrue(self.doc["verdict"]["graphics.stages.vertex"]["against_its_compute_twin"]["same_form_sequence"])
        rows["graphics.stages.vertex"]["forms"] = rows["graphics.stages.vertex"]["forms"][:-1] + ["586/4", "684/4"]
        self.assertFalse(F.classify(rows, self.facts)["graphics.stages.vertex"]["against_its_compute_twin"]["same_form_sequence"])

    def test_apple_emits_slot_44_for_a_tensor_kernel(self):
        v = self.doc["verdict"]["tensor.matmul.apple"]
        self.assertTrue(v["apple_compiler_emits_slot_44"])
        self.assertTrue({32, 33, 44} <= set(v["per_kernel_slots"]))
        rows = self.rows()
        rows["tensor.matmul.apple"]["metadata"]["per_kernel_slots"].pop("44")
        self.assertFalse(F.classify(rows, self.facts)["tensor.matmul.apple"]["apple_compiler_emits_slot_44"])

    def test_moved_tables_refuse_and_refresh_repairs(self):
        # Another lane landing an isolated record must fail --check with the repair named, and
        # --refresh must restore it without a compile.
        import shutil, tempfile
        from unittest.mock import patch
        names, isolated, vendor = self.facts
        grown = (names, isolated | {14661}, vendor)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "census.json")
            shutil.copy(F.DEST, path)
            with patch.object(F, "_facts", lambda root=F.ROOT: grown):
                with self.assertRaisesRegex(F.Refused, "--refresh"):
                    F.check(path)
                F.refresh(path)
                doc = F.check(path)
            self.assertTrue(doc["verdict"]["mem.sampler"]["signature"]["14661/22"]["isolated_record"])

    def test_the_rasterizing_stages_are_retained_and_compared(self):
        st = self.doc["verdict"]["graphics.stages.raster"]["stages"]
        self.assertEqual(set(st), {"__vertex", "__fragment"})
        self.assertEqual(st["__vertex"]["slots_added_over_buffer_only_vertex"], [38, 42])
        self.assertEqual(st["__fragment"]["slots_added_over_buffer_only_vertex"], [19])
        self.assertIn("__GPU_METADATA_2,__fragment", st["__fragment"]["sections_not_described"])
        objs = self.doc["rows"]["graphics.stages.raster"]["objects"].values()
        self.assertTrue(all(o["metadata_hex"] for o in objs))
        rows = self.rows()
        next(iter(rows["graphics.stages.raster"]["objects"].values()))["per_kernel_slots"].append(99)
        got = F.classify(rows, self.facts)["graphics.stages.raster"]["stages"]
        self.assertTrue(any(99 in s["slots_added_over_buffer_only_vertex"] for s in got.values()))

    def test_a_stale_verdict_refuses(self):
        doc = copy.deepcopy(self.doc)
        doc["verdict"]["mem.sampler"]["named"] += 1
        path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "featcensus-stale-%d.json" % os.getpid())
        with open(path, "w") as fh:
            json.dump(doc, fh)
        self.addCleanup(os.remove, path)
        with self.assertRaises(F.Refused):
            F.check(path)


if __name__ == "__main__":
    unittest.main()
