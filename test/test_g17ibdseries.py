"""The imageblock addressing D-series located the cc coordinate-lifetime bug. Asserts the
committed receipt (a GPU run's evidence): D1/D3 read the neighbour, D2 reads its own element,
and D4 (the coordinate lifetime released) collapses every lane to element 0."""
import json, os, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = json.load(open(os.path.join(ROOT, "isa", "g17-imageblock-dseries-receipt.json")))
A = REC["arms"]
NEIGHBOUR = [0xA5000000 | ((t + 1) % 32) for t in range(32)]
OWN = [0xA5000000 | t for t in range(32)]


class ImageblockDSeries(unittest.TestCase):
    def test_all_arms_agree_across_runs(self):
        for name, a in A.items():
            self.assertTrue(a["agree"], name)
            self.assertTrue(a["matches_expected"], name)

    def test_the_neighbour_read_is_member_a_and_the_y_width_is_not_the_bug(self):
        self.assertEqual(A["apple_unmodified"]["buf0"], NEIGHBOUR)
        self.assertEqual(A["D1_member_a_only"]["buf0"], NEIGHBOUR)      # member b removed, still neighbour
        self.assertEqual(A["D3_y_by_four_byte_read"]["buf0"], NEIGHBOUR)  # 4-byte y read reaches R3H

    def test_the_coordinate_operand_selects_own_vs_neighbour(self):
        self.assertEqual(A["D2_read_own_element"]["buf0"], OWN)

    def test_D4_collapses_to_element_zero_the_bug_signature(self):
        # releasing the write's coordinate lifetime makes every lane read element 0
        self.assertEqual(set(A["D4_write_releases_coordinate"]["buf0"]), {0xA5000000})
        # and it is a real discriminator: D2 (own element) is not a collapse
        self.assertGreater(len(set(A["D2_read_own_element"]["buf0"])), 1)

    def test_the_conclusion_names_the_lifetime_and_the_fix(self):
        self.assertIn("LIFETIME", REC["conclusion"])
        self.assertIn("cc_fix_validated", REC)
        self.assertIn("one bit", REC["cc_fix_validated"])


if __name__ == "__main__":
    unittest.main()
