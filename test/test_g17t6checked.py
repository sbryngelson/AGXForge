"""Guards for the T6 receipt-attribution result.

The claim is that checked values exist and cannot be attributed, so the tests care about the two
ways that could be wrong: a scan that found no checked values at all, and a recovery that silently
attributed them to the wrong thing.
"""
import json, os, unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RECORD = os.path.join(ROOT, "tools", "tensorops-model", "t6-checked-forms.json")


def _r():
    with open(RECORD, encoding="utf-8") as fh:
        return json.load(fh)


class TheScanFoundCheckedValues(unittest.TestCase):

    def test_receipts_were_read(self):
        r = _r()
        self.assertGreater(r["files"], 50)
        self.assertGreater(r["records"], 500)

    def test_checked_values_exist(self):
        """If none carried a match, 'the checks were run' is unsupported."""
        self.assertGreater(_r()["checked"], 100)


class DecodingRecoversWhatTheFieldCannot(unittest.TestCase):

    def test_decoding_reaches_far_more_forms_than_the_length_field(self):
        r = _r()
        self.assertGreater(r["forms_from_decode"], 10 * max(1, r["forms_from_length"]))

    def test_most_records_had_bytes_to_decode(self):
        r = _r()
        self.assertGreater(r["recovered"], 0.8 * r["checked"])

    def test_every_recovered_form_names_a_real_width(self):
        for form in _r()["forms"]:
            op, w = form.split("/")
            with self.subTest(form=form):
                self.assertTrue(op.isdigit() and w.isdigit())
                self.assertIn(int(w), (2, 4, 6, 8, 10, 12, 14, 16))


class TheUnintersectableClaimIsNotOverstated(unittest.TestCase):
    """The artifact publishes a count with no population; the section must not claim an overlap."""

    def test_the_record_keeps_the_coverage_totals_it_cannot_intersect(self):
        t = _r()["coverage_totals"]
        self.assertEqual(t["dispatched"], 466)
        self.assertEqual(t["executed_unchecked"], 425)

    def test_the_recovered_form_count_is_not_presented_as_a_subset(self):
        r = _r()
        self.assertNotIn("of_the_425", r)
        self.assertLess(r["forms_from_decode"], r["coverage_totals"]["executed_unchecked"])


if __name__ == "__main__":
    unittest.main()
