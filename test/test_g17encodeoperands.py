"""encode_operands re-authors Apple's own instances of a form, not only the witness it was fit on.

Each case is an Apple instance of a D2 form that is NOT the form's witness, and that failed before
the fix named on it. Asked for the instance's printed operands (and its free-bit values), the
encoder must return Apple's bytes exactly. Held-out over a seeded sample of 50 D2 forms, 553
instances: 156 byte-exact before these fixes, 407 after.
"""
import os, sys, unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from agxforge.g17 import encode as E


def author(form, hexbytes, tokens, hints=None):
    op, ln = map(int, form.split("/"))
    got, un = E.encode_operands(op, ln, tokens, hints=hints)
    return got, un, bytes.fromhex(hexbytes)


class HeldOutInstancesReauthor(unittest.TestCase):

    def test_a_map_probed_at_another_length_is_not_laid_over_the_form(self):
        """op13575's authoring map was probed on a 16-byte instance; its bit 0.5 is an immediate
        there and a register bit in the 4-byte form, and it hid the form's own bit."""
        got, un, want = author("13575/4", "2b053c07",
                               ["reg:107", "imm:32", "reg:107", "imm:16", "reg:108", "imm:16"])
        self.assertEqual((got, un), (want, []))

    def test_the_wrong_length_map_fails_the_same_instance(self):
        """The control: hand the search the opcode-wide map and the same request fails."""
        real = E.extended_fields
        with patch.object(E, "extended_fields", lambda op, length_hint=None, own_form_only=False: real(op, length_hint)):
            got, un, want = author("13575/4", "2b053c07",
                                   ["reg:107", "imm:32", "reg:107", "imm:16", "reg:108", "imm:16"])
        self.assertNotEqual(got, want)
        self.assertTrue(un)

    def test_an_address_expression_is_solved_sub_field_by_sub_field(self):
        got, un, want = author("3142/8", "49000d8341208002",
                               ["expr:bin(op0,const(8),4)", "imm:0", "imm:16777216", "reg:425",
                                "imm:2064", "reg:106", "imm:32"])
        self.assertEqual((got, un), (want, []))

    def test_a_bit_the_closed_form_model_left_out_is_still_searched(self):
        """op2191/8 bit 5.0 takes operand 6 from register 427 to a different class, so it has no
        power-of-two weight; register 294 is unreachable without it."""
        got, un, want = author("2191/8", "790b1e07811b0000",
                               ["reg:112", "imm:0", "reg:110", "imm:16", "reg:108", "imm:16",
                                "reg:294", "imm:16"])
        self.assertEqual((got, un), (want, []))

    def test_free_bits_are_in_place_before_the_search(self):
        """op11487/10's free bits 4.3 and 4.4 move immediate 7 (16 -> 22), and bit 4.1, recorded
        against operand 8, also moves it - so the search needs the hints first and a second sweep. Bit 8.1 is a hint too since
        the second corpus showed it varying (it had been constant 1 over 309 instances)."""
        got, un, want = author("11487/10", "0a0229809a0107028a01",
                               ["reg:281", "imm:32", "imm:13", "reg:426", "imm:16", "imm:0",
                                "reg:281", "imm:22", "reg:281", "imm:20"],
                               # (8, 1): adopted 2026-09-24 as FREE (constant 1 -> varies, cleared
                               # on hardware by tools/g17adoptcheck.py), so the encoder emits 0 unless
                               # asked; reproducing Apple's instance byte-exact now needs the hint.
                               hints={(2, 2): 0, (4, 3): 1, (4, 4): 1, (8, 3): 1, (8, 1): 1})
        self.assertEqual((got, un), (want, []))


    def test_the_requested_operand_kind_sets_the_mode_bit(self):
        """op798/12's witness carries a REGISTER in operand 4; this instance carries an address
        expression there, which only mode bit 10.3 can give - no operand field reaches it."""
        self.assertEqual(E.spec_for(798, 12)["10.3"]["switches"], [[4, "reg", "expr"]])
        got, un, want = author("798/12", "3080463b2b20a002c10d8800",
                               ["reg:282", "imm:1073741856", "reg:439", "imm:32",
                                "expr:bin(op0,const(55),2)", "imm:0", "reg:426", "imm:16"])
        self.assertEqual((got, un), (want, []))


class TheCompilerPathIsUnchanged(unittest.TestCase):
    """The closed-form encoder and the per-form bases still read the opcode-wide map: dropping it
    there changes compiled bytes that were validated on hardware, which needs a GPU batch first."""

    def test_default_fields_still_carry_the_opcode_wide_map(self):
        ext, _, _, _ = E.extended_fields(13575, 4)
        self.assertTrue(any(b >= 4 for v in ext.values() for _, b, _, _ in v))

    def test_the_search_fields_stay_inside_the_form(self):
        ext, _, _, _ = E.extended_fields(13575, 4, own_form_only=True)
        self.assertFalse(any(b >= 4 for v in ext.values() for _, b, _, _ in v))


if __name__ == "__main__":
    unittest.main()
