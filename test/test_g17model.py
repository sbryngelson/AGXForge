from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import g17model


class G17ModelMetadataTests(unittest.TestCase):
    def test_operand_descriptors_start_after_implicit_fields(self):
        """The parser must not treat flags/uses/defs as encoded operands."""
        table = g17model.opcodes()
        self.assertGreater(len(table), 17000)
        for opcode in table.values():
            self.assertEqual(len(opcode.operands), opcode.nops)
            self.assertTrue(all(len(operand) == 3 for operand in opcode.operands))
        # A multi-operand row catches the old p[5:] parser directly: it used
        # to try parsing the flags word and the uses/defs tokens as operands.
        self.assertEqual(table[17257].nops, 10)
        self.assertEqual(len(table[17257].operands), 10)

    def test_decode_accepts_hex_operand_values_without_changing_decimal(self):
        fake_decoder = "00000000 4 123 reg:0x10 expr:7"
        with mock.patch.object(g17model, "_decode_stdout", return_value=fake_decoder), \
             mock.patch.object(g17model, "opcodes", return_value={123: object()}):
            values = next(g17model.decode(b"\0" * 4)).values
        self.assertEqual(values, [("reg", 16), ("expr", 7)])


if __name__ == "__main__":
    unittest.main()
