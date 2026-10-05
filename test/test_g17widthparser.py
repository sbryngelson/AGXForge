"""Guards for the operand-token parser and the cache that can hide its state.

This defect published `no_width: 72` for weeks with the reason recorded as the bare class name
"ValueError". Three things had to be true at once for that to survive: the parser read hex as
base 10, nothing exercised the parser directly, and the cache that stores the result is keyed on
its INPUT rather than on the code that produces it - so a fix would have gone on serving the
broken answer and a regression would go on serving the good one.

The population matters here. The breaking token shape, `unknown:0x00`, appears for exactly 72 of
the 6,001 repair-walk witnesses and for NONE of the 717 Apple ones - so a guard that exercised
the Apple population would be a control that cannot fail.
"""
import hashlib
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

import g17isamap as MAP
from agxforge.g17 import model as M

CONTRACT = os.path.join(ROOT, "isa", "g17-contract.jsonl")
# Four of the 72 opcodes whose repair-walk witness decodes through the hex arm, so the guard
# below costs four subprocess calls rather than six thousand.
HEX_ARM_OPCODES = (2820, 2824, 3076, 13516)


def _contract():
    rows = {}
    with open(CONTRACT) as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["opcode"]] = row
    return rows


class TheOperandTokenParser(unittest.TestCase):

    def test_hex_and_decimal_both_parse(self):
        self.assertEqual(0, M.operand_value("0x00"))
        self.assertEqual(31, M.operand_value("0x1f"))
        self.assertEqual(42, M.operand_value("0X2A"))
        self.assertEqual(-16, M.operand_value("-0x10"))
        self.assertEqual(16, M.operand_value("16"))
        self.assertEqual(-3, M.operand_value("-3"))
        self.assertEqual(0, M.operand_value("-0"))

    def test_a_decimal_token_is_unchanged_by_the_fix(self):
        """The fix's own claim: nothing that decoded before decodes differently now."""
        for token in ("0", "1", "7", "16", "32", "105", "255", "-1", "-105"):
            self.assertEqual(int(token), M.operand_value(token), token)

    def test_an_unparseable_token_still_RAISES(self):
        """Loudly, rather than being skipped. A dropped token leaves the operand list short and
        the caller reads a value belonging to another field."""
        for bad in ("ff", "", "0xzz", "1.5", "reg", "0x"):
            with self.assertRaises(ValueError, msg=bad):
                M.operand_value(bad)


class TheHexArmHasALivePopulation(unittest.TestCase):

    def test_the_breaking_token_shape_really_occurs(self):
        """If no witness produced a hex token the parser guard above would be arguing with
        nobody, and the 72 lost widths would have had some other cause."""
        rows = _contract()
        seen = []
        for opcode in HEX_ARM_OPCODES:
            witness = (rows[opcode].get("encoding") or {}).get("witness")
            self.assertTrue(witness, "op%d has no repair-walk witness" % opcode)
            text = M._decode_stdout(bytes.fromhex(witness), 0)
            tokens = [tok for line in text.splitlines() for tok in line.split()
                      if ":" in tok and "0x" in tok.split(":", 1)[1].lower()]
            seen.append((opcode, tokens))
        for opcode, tokens in seen:
            self.assertTrue(tokens, "op%d no longer emits a hex token" % opcode)

    def test_those_opcodes_now_decode_to_a_width(self):
        """The outcome the 72 lost: a decoded instruction with a length."""
        rows = _contract()
        for opcode in HEX_ARM_OPCODES:
            witness = (rows[opcode].get("encoding") or {}).get("witness")
            instructions = list(M.decode(bytes.fromhex(witness), 0))
            self.assertTrue(instructions, "op%d decodes to nothing" % opcode)
            self.assertGreater(len(instructions[0].raw), 0, opcode)


class TheRepairWidthCacheCannotHideTheDecoder(unittest.TestCase):

    def test_the_key_moves_when_the_DECODER_moves(self):
        """The property the old key did not have. Same witnesses, different decoder, same key
        meant a parser fix kept serving the broken widths."""
        rows = _contract()
        apple = {o for o, r in rows.items()
                 if isinstance((r.get("encoding") or {}).get("apple_witness"), str)}
        a = MAP.repair_cache_key(rows, apple, decoder=b"def decode(): pass")
        b = MAP.repair_cache_key(rows, apple, decoder=b"def decode(): pass  # changed")
        self.assertNotEqual(a, b, "the cache key ignores the decoder's code")

    def test_the_key_still_moves_when_the_WITNESSES_move(self):
        """And the property it did have is not lost by adding the decoder to it."""
        rows = _contract()
        apple = {o for o, r in rows.items()
                 if isinstance((r.get("encoding") or {}).get("apple_witness"), str)}
        base = MAP.repair_cache_key(rows, apple, decoder=b"same")
        victim = next(o for o in sorted(rows) if o not in apple
                      and (rows[o].get("encoding") or {}).get("witness"))
        moved = json.loads(json.dumps(rows[victim]))
        moved["encoding"]["witness"] = "00" * 16
        other = dict(rows)
        other[victim] = moved        # an int key, so this cannot go through **kwargs
        self.assertNotEqual(base, MAP.repair_cache_key(other, apple, decoder=b"same"))

    def test_the_live_key_reads_the_real_decoder(self):
        rows = _contract()
        apple = {o for o, r in rows.items()
                 if isinstance((r.get("encoding") or {}).get("apple_witness"), str)}
        with open(os.path.join(ROOT, "agxforge", "g17", "model.py"), "rb") as handle:
            live = handle.read()
        self.assertEqual(MAP.repair_cache_key(rows, apple, decoder=live),
                         MAP.repair_cache_key(rows, apple))


if __name__ == "__main__":
    unittest.main(verbosity=0)
