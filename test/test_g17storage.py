"""Compile the actual host buffer code without Metal and inject missing/wide writes."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from g17native import StorageLayout

HARNESS = r'''
#include "g17scanstorage.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  if (argc != 4) return 2;
  G17Storage s;
  if (!g17StorageInit(&s, strtoul(argv[1], NULL, 10), strtoul(argv[2], NULL, 10),
                      strtoul(argv[3], NULL, 10) != 0)) return 3;
  // Full-target layout arithmetic requires no matrix allocation.
  printf("{\"matrix\":%zu,\"query\":%zu,\"reply\":%zu,\"output\":%zu}\n",
         s.matrixBytes, s.queryBytes, s.replyBytes, s.outputBytes);
  unsigned char *out = malloc(s.outputBytes);
  if (!out) return 4;
  g17StorageReset(&s, out);
  if (!g17StorageCheck(&s, out)) return 5; // No row was written.
  if (s.half) {
    uint16_t one = 0x3c00;
    for (size_t i=0; i<s.rows; ++i) memcpy(out+i*2, &one, 2);
  } else {
    uint32_t one = 0x3f800000, marker = 0x5a17c0de;
    for (size_t i=0; i<s.rows; ++i) {
      memcpy(out+i*4, &one, 4);
      memcpy(out+(s.rows+i)*4, &marker, 4);
    }
  }
  if (g17StorageCheck(&s, out)) return 6;
  if (s.half) {
    // Full-word host control clobbers the adjacent unwritten halfword.
    // This tests the instrument, not GPU half-store behavior.
    uint32_t wide = 0x00003c00;
    memcpy(out+(s.rows-1)*2, &wide, 4);
  } else {
    out[s.replyBytes] ^= 1;
  }
  const char *error = g17StorageCheck(&s, out);
  if (!error || strcmp(error, "boundary_guard")) return 7;
  g17StorageReset(&s, out); // A previous query cannot supply completion.
  if (!g17StorageCheck(&s, out)) return 8;
  free(out);
  return 0;
}
'''


class HostStorage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        source = root / "storage.c"
        source.write_text(HARNESS)
        cls.binary = root / "storage"
        subprocess.run(["clang", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=address,undefined", "-I", str(ROOT / "tools"),
                        str(source), "-o", str(cls.binary)], check=True, timeout=30,
                       capture_output=True)

    def test_guard_instrument_and_c_python_layouts_agree(self):
        for rows, columns in ((1, 1), (3, 7), (32, 384), (33, 384), (500000, 384)):
            for dtype in ("float16", "float32"):
                with self.subTest(rows=rows, columns=columns, dtype=dtype):
                    result = subprocess.run([str(self.binary), str(rows), str(columns),
                                             str(int(dtype == "float16"))],
                                            check=True, capture_output=True, text=True, timeout=5)
                    got = json.loads(result.stdout)
                    layout = StorageLayout(rows, columns, dtype)
                    self.assertEqual(got, dict(matrix=layout.matrix_bytes, query=layout.query_bytes,
                                               reply=layout.reply_bytes, output=layout.output_bytes))
        full = StorageLayout(500000, 384, "float16")
        self.assertEqual(full.matrix_bytes, 384000000)
        self.assertEqual(full.query_bytes, 768)
        self.assertEqual(full.reply_bytes, 1000000)
        self.assertEqual(full.output_bytes, 1000128)

    def test_invalid_storage_contracts_fail(self):
        for shape in ((0, 384), (500001, 384), (33, 385)):
            with self.assertRaises(ValueError):
                StorageLayout(*shape, "float16")
            result = subprocess.run([str(self.binary), str(shape[0]), str(shape[1]), "1"],
                                    capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 3)
        with self.assertRaises(ValueError):
            StorageLayout(33, 384, "float64")
