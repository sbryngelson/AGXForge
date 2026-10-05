"""Focused checks for the measured six-byte requantization field handoff."""

import os
import re
import subprocess
import tempfile
import unittest

from agxforge.g17 import requantenc


class RequantEncoders(unittest.TestCase):
    def test_retained_templates_round_trip_the_measured_registers(self):
        cases = (
            (11375, requantenc.SIGNED_SELECT_TEMPLATE, {0: 107, 3: 107, 7: 107},
             requantenc.encode_11375),
            (11364, requantenc.SIGNED_CLAMP_TEMPLATE, {0: 108, 3: 107, 6: 107},
             requantenc.encode_11364),
            (10369, requantenc.SIGNED_CLAMP_LOW_TEMPLATE, {0: 74, 3: 106},
             requantenc.encode_10369),
            (10369, requantenc.UNSIGNED_CLAMP_LOW_TEMPLATE, {0: 74, 3: 105},
             requantenc.encode_10369),
            (11364, requantenc.UNSIGNED_CLAMP_HIGH_TEMPLATE, {0: 107, 3: 107, 6: 107},
             requantenc.encode_11364),
            (11364, requantenc.UNSIGNED_CLAMP_FINAL_TEMPLATE, {0: 107, 3: 107, 6: 107},
             requantenc.encode_11364),
        )
        for opcode, template, regs, encode in cases:
            with self.subTest(opcode=opcode, template=template.hex()):
                raw = encode(regs, template=template)
                self.assertEqual(raw, template)
                self.assertEqual(requantenc.decode_measured_registers(opcode, raw), regs)

    def test_register_fields_can_change_without_touching_template_owned_bits(self):
        raw = requantenc.encode_11364({0: 105, 3: 119, 6: 119})
        self.assertEqual(requantenc.decode_measured_registers(11364, raw),
                         {0: 105, 3: 119, 6: 119})
        template = requantenc.SIGNED_CLAMP_TEMPLATE
        # The fixed immediate/expr bytes are owned by the retained template.  The field map
        # changes only the register carriers for this check.
        reg_positions = {(by, bit) for _operand, (_base, bits) in requantenc._REG_FIELDS[11364].items()
                         for _j, by, bit, _inv in bits}
        for i, (before, after) in enumerate(zip(template, raw)):
            if all((i, bit) not in reg_positions for bit in range(8)):
                self.assertEqual(before, after, "template-owned byte %d changed" % i)

    def test_measured_register_class_boundaries_refuse(self):
        with self.assertRaisesRegex(ValueError, "outside operand 0's measured class"):
            requantenc.encode_11364({0: 233})
        with self.assertRaisesRegex(ValueError, "outside operand 0's measured class"):
            requantenc.encode_10369({0: 82})
        with self.assertRaisesRegex(ValueError, "exactly six bytes"):
            requantenc.encode_11375({}, template=b"\0" * 14)
        with self.assertRaisesRegex(ValueError, "overlap with conflicting values"):
            requantenc.encode_11364({3: 112, 6: 119})

    def test_fmul6_tied_scale_template_round_trips(self):
        for reg in (0, 1, 27, 127):
            with self.subTest(reg=reg):
                raw = requantenc.encode_fmul6(requantenc.FMUL6_TEMPLATE, [reg], [reg])
                decoded = requantenc.decode_fmul6(raw)
                self.assertEqual(decoded["register"], reg + 105)
                self.assertEqual(decoded["dest"], reg + 105)
        with self.assertRaisesRegex(ValueError, "tied"):
            requantenc.encode_fmul6(requantenc.FMUL6_TEMPLATE, [1], [2])
        with self.assertRaisesRegex(ValueError, "template"):
            requantenc.encode_fmul6(b"\0" * 6, [0], [0])

    @staticmethod
    def _decoder(path, raw):
        with tempfile.NamedTemporaryFile() as f:
            f.write(raw)
            f.flush()
            text = subprocess.check_output([path, f.name, "0", str(len(raw))], text=True)
        return text.strip()

    @unittest.skipUnless(os.path.exists(os.path.join(os.path.dirname(__file__), "../tools/agx3dis")),
                         "Apple decoder helper is not built in this checkout")
    def test_apple_decoder_reads_changed_registers(self):
        path = os.path.join(os.path.dirname(__file__), "../tools/agx3dis")
        cases = (
            (requantenc.encode_11375({0: 105, 3: 119, 7: 119}), 11375,
             ("reg:105", "reg:119")),
            (requantenc.encode_11364({0: 105, 3: 119, 6: 119}), 11364,
             ("reg:105", "reg:119")),
            (requantenc.encode_10369({0: 74, 3: 112}), 10369,
             ("reg:74", "reg:112")),
        )
        for raw, opcode, regs in cases:
            with self.subTest(opcode=opcode):
                text = self._decoder(path, raw)
                self.assertIn(" %d " % opcode, text)
                for reg in regs:
                    self.assertIn(reg, text)


if __name__ == "__main__":
    unittest.main()
