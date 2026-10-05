"""The fragment __GPU_METADATA_2 reading (isa/g17-fragment-metadata2.json), held to its own variants."""
import json, os, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = json.load(open(os.path.join(ROOT, "isa", "g17-fragment-metadata2.json")))
V = {k: v for k, v in D["variants"].items() if v.get("status") == "ok"}


def ents(k):
    return V[k]["interpolants"]["entries"]


class FragmentMetadata2(unittest.TestCase):
    def test_enough_variants_compiled(self):
        self.assertGreaterEqual(len(V), 12)

    def test_discard_alone_clears_bit_zero_of_the_root_field(self):
        for k, v in V.items():
            self.assertEqual(v["fields"]["1"], 754 if k == "discard" else 755, k)

    def test_no_varying_read_has_an_empty_interpolant_vector(self):
        self.assertEqual(ents("no_varying_read"), [])

    def test_perspective_adds_a_leading_one_over_w_record_and_flat_does_not(self):
        self.assertEqual(ents("base.one_varying")[0], {"2": 1})
        for k in ("flat", "no_perspective"):
            self.assertNotIn({"2": 1}, ents(k), k)

    def test_component_count_and_cumulative_start(self):
        self.assertEqual([e.get("2") for e in ents("eight_varyings")[1:]], [4, 4])
        self.assertEqual([e.get("3") for e in ents("eight_varyings")[1:]], [1, 5])
        self.assertEqual([e.get("3") for e in ents("two_varyings")[1:]], [1, 2])

    def test_the_interpolation_mode_is_not_in_this_section(self):
        # A reading that claimed the mode would need these to differ; they do not.
        self.assertEqual(ents("flat"), ents("no_perspective"))
        self.assertEqual(ents("centroid"), ents("base.one_varying"))


if __name__ == "__main__":
    unittest.main()
