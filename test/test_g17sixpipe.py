"""Item 1: the six-buffer image creates a compute pipeline; the crashed baseline reproduces
the -11. Asserts the committed receipt (a hardware create_pipelines run)."""
import json, os, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = json.load(open(os.path.join(ROOT, "isa", "g17-six-buffer-pipeline-receipt.json")))


class SixBufferPipeline(unittest.TestCase):
    def test_the_corrected_image_creates_a_pipeline(self):
        self.assertTrue(REC["corrected"]["created"])
        self.assertEqual(REC["corrected"]["returncode"], 0)
        self.assertEqual(REC["corrected"]["function"], "six")

    def test_the_crashed_baseline_reproduces_the_minus_11(self):
        self.assertFalse(REC["crashed"]["created"])
        self.assertTrue(REC["crashed"]["segfault"])
        self.assertIn(REC["crashed"]["returncode"], (-11, 139))

    def test_the_pair_is_discriminating(self):
        # only the pointer bytes differ; the same loader passes one and segfaults on the other,
        # so the test can fail -- a control that could not fail proves nothing
        self.assertNotEqual(REC["corrected"]["arc_sha256"], REC["crashed"]["arc_sha256"])
        self.assertEqual(REC["phase"], "create_pipelines")


if __name__ == "__main__":
    unittest.main()
