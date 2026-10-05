import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import g17dis as D


class Class4AndCLength(unittest.TestCase):
    """The rule mined from Apple's decoder, the instruction that desynced three constant programs,
    and the form a NOP probe had called four bytes."""

    def test_the_table(self):
        pad = bytes(12)
        for b0 in (0x24, 0x1c, 0x04, 0x3c):
            self.assertEqual(D.length(bytes([b0, 0x00, 0x21]) + pad, 0), 2)     # b1.7 clear
            self.assertEqual(D.length(bytes([b0, 0x80, 0x00]) + pad, 0), 4)
            self.assertEqual(D.length(bytes([b0, 0x80, 0x01]) + pad, 0), 8)
            self.assertEqual(D.length(bytes([b0, 0x80, 0x02]) + pad, 0), 8)
            self.assertEqual(D.length(bytes([b0, 0x80, 0x03]) + pad, 0), 10)

    def test_the_read_sr_that_desynced_the_constant_programs(self):
        self.assertEqual(D.length(bytes.fromhex("248021104701a0821c8a08270700"), 0), 8)

    def test_byte2_01_is_eight_bytes_as_apple_reads_it(self):
        self.assertEqual(D.length(bytes.fromhex("24800100000000000000"), 0), 8)

    def test_the_rule_against_every_probe_corpus_instance(self):
        bad = 0
        for line in open(os.path.join(ROOT, "isa", "g17-corpus-programs.jsonl")):
            r = json.loads(line)
            t = bytes.fromhex(r["text"])
            for pc, ln, _op in r["spans"]:
                if pc + 3 <= len(t) and t[pc] & 7 == 4 and D._class4_len(t, pc) != ln:
                    bad += 1
        self.assertEqual(bad, 0)


if __name__ == "__main__":
    unittest.main()
