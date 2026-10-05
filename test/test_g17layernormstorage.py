"""Exercise actual four-buffer host memory code under ASan/UBSan, without Metal."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

HARNESS = r'''
#include "g17scanstorage.h"
#include <stdio.h>
#include <stdlib.h>
#define CHECK(condition) do { if (!(condition)) { \
  fprintf(stderr, "memory check failed at line %d\n", __LINE__); return 1; } } while (0)

static void complete(const G17LayerNormStorage *s, void *output) {
  for (size_t k=0; k<s->payloadBytes[3]/4; ++k) {
    const float value = (float)k + 0.25f;
    memcpy((uint8_t *)output+G17_LN_GUARD+k*4, &value, 4);
  }
}

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  G17LayerNormStorage s;
  if (!g17LayerNormStorageInit(&s, strtoul(argv[1], NULL, 10), strtoul(argv[2], NULL, 10)))
    return 3;
  void *buffers[4];
  const void *view[4], *inputs[3];
  for (size_t i=0; i<4; ++i) {
    buffers[i] = malloc(s.allocationBytes[i]);
    CHECK(buffers[i]);
    view[i] = buffers[i];
  }
  for (size_t i=0; i<3; ++i) {
    float *input = malloc(s.payloadBytes[i]);
    CHECK(input);
    for (size_t k=0; k<s.payloadBytes[i]/4; ++k) input[k] = (float)(k%19) - (float)i;
    inputs[i] = input;
  }
  size_t failed = 99;
  CHECK(!g17LayerNormPrepare(&s, buffers, inputs));
  CHECK(!strcmp(g17LayerNormCheck(&s, view, inputs, &failed), "nonfinite_output"));
  CHECK(failed == 3);
  complete(&s, buffers[3]);
  CHECK(!g17LayerNormCheck(&s, view, inputs, &failed));

  // Flip every byte of both guards around every buffer independently. This
  // includes the bytes immediately adjoining the first and last float.
  size_t guardCases = 0;
  for (size_t i=0; i<4; ++i) {
    uint8_t *bytes = buffers[i];
    for (size_t side=0; side<2; ++side) for (size_t k=0; k<G17_LN_GUARD; ++k) {
      size_t at = side ? G17_LN_GUARD+s.payloadBytes[i]+k : k;
      bytes[at] ^= 1;
      CHECK(!strcmp(g17LayerNormCheck(&s, view, inputs, &failed), "boundary_guard"));
      CHECK(failed == i);
      bytes[at] ^= 1;
      CHECK(!g17LayerNormCheck(&s, view, inputs, &failed));
      ++guardCases;
    }
  }

  // A misranked store can hit a valid finite input without touching a guard.
  // Check the full readonly payload against its independently held snapshot.
  size_t readonlyCases = 0;
  for (size_t i=0; i<3; ++i) {
    uint8_t *payload = (uint8_t *)buffers[i]+G17_LN_GUARD;
    const size_t positions[] = {0, s.payloadBytes[i]/2, s.payloadBytes[i]-1};
    for (size_t k=0; k<3; ++k) {
      payload[positions[k]] ^= 1;
      CHECK(!strcmp(g17LayerNormCheck(&s, view, inputs, &failed), "readonly_input_changed"));
      CHECK(failed == i);
      payload[positions[k]] ^= 1;
      ++readonlyCases;
    }
  }

  // Writing all but the final output must fail, even beyond the first `rows`
  // floats that the former scalar transport considered the complete reply.
  uint8_t *out = (uint8_t *)buffers[3]+G17_LN_GUARD;
  const uint32_t nan = 0x7fc01234, infinity = 0x7f800000;
  memcpy(out+s.payloadBytes[3]-4, &nan, 4);
  CHECK(!strcmp(g17LayerNormCheck(&s, view, inputs, &failed), "nonfinite_output"));
  memcpy(out+s.payloadBytes[3]-4, &infinity, 4);
  CHECK(!strcmp(g17LayerNormCheck(&s, view, inputs, &failed), "nonfinite_output"));
  complete(&s, buffers[3]);

  // An invalid parameter prevents all buffer writes, including earlier source
  // buffers. Validation cannot leave partially replaced input behind.
  void *before[4];
  for (size_t i=0; i<4; ++i) {
    before[i] = malloc(s.allocationBytes[i]);
    CHECK(before[i]);
    memcpy(before[i], buffers[i], s.allocationBytes[i]);
  }
  uint32_t saved;
  memcpy(&saved, inputs[2], 4);
  memcpy((void *)inputs[2], &nan, 4);
  CHECK(!strcmp(g17LayerNormPrepare(&s, buffers, inputs), "nonfinite_input"));
  for (size_t i=0; i<4; ++i) CHECK(!memcmp(before[i], buffers[i], s.allocationBytes[i]));
  memcpy((void *)inputs[2], &saved, 4);

  // Parameters persist across queries; reset only output and replace source.
  // Previous finite output must not supply completion for the next query.
  float *changed = (float *)inputs[0];
  for (size_t k=0; k<s.payloadBytes[0]/4; ++k) changed[k] = -changed[k] - 0.5f;
  CHECK(!strcmp(g17LayerNormCheck(&s, view, inputs, &failed), "readonly_input_changed"));
  CHECK(failed == 0); // The old upload cannot stand in for the changed request.
  CHECK(!g17LayerNormBeginQuery(&s, buffers, inputs[0]));
  CHECK(!strcmp(g17LayerNormCheck(&s, view, inputs, &failed), "nonfinite_output"));
  CHECK(!memcmp(before[1], buffers[1], s.allocationBytes[1]));
  CHECK(!memcmp(before[2], buffers[2], s.allocationBytes[2]));
  complete(&s, buffers[3]);
  CHECK(!g17LayerNormCheck(&s, view, inputs, &failed));
  memcpy(&saved, inputs[0], 4);
  memcpy((void *)inputs[0], &infinity, 4);
  CHECK(!strcmp(g17LayerNormBeginQuery(&s, buffers, inputs[0]), "nonfinite_input"));
  memcpy((void *)inputs[0], &saved, 4);
  CHECK(!g17LayerNormCheck(&s, view, inputs, &failed)); // Refusal did not reset output.
  printf("{\"guard_cases\":%zu,\"readonly_cases\":%zu,\"outputs_checked\":%zu,"
         "\"payload_bytes\":[%zu,%zu,%zu,%zu],\"gpu_dispatched\":false}\n",
         guardCases, readonlyCases, s.rows*s.columns, s.payloadBytes[0], s.payloadBytes[1],
         s.payloadBytes[2], s.payloadBytes[3]);
  for (size_t i=0; i<4; ++i) { free(before[i]); free(buffers[i]); }
  for (size_t i=0; i<3; ++i) free((void *)inputs[i]);
  return 0;
}
'''


class LayerNormStorage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        source = root/"layernorm-storage.c"
        source.write_text(HARNESS)
        cls.binary = root/"layernorm-storage"
        subprocess.run(["clang", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-fsanitize=address,undefined", "-I", str(ROOT/"tools"), str(source),
            "-o", str(cls.binary)], check=True, timeout=30, capture_output=True)

    def test_missing_outputs_guard_corruption_and_readonly_writes_are_detected(self):
        for rows, columns in ((1,1), (1,4), (3,7), (32,384), (128,384)):
            with self.subTest(rows=rows, columns=columns):
                p = subprocess.run([str(self.binary), str(rows), str(columns)],
                    capture_output=True, text=True, timeout=5)
                self.assertEqual(p.returncode, 0, p.stderr)
                report = json.loads(p.stdout)
                self.assertEqual(report["guard_cases"], 1024)
                self.assertEqual(report["readonly_cases"], 9)
                self.assertEqual(report["outputs_checked"], rows*columns)
                self.assertEqual(report["payload_bytes"], [rows*columns*4, columns*4,
                                                          columns*4, rows*columns*4])
                self.assertFalse(report["gpu_dispatched"])

    def test_invalid_shapes_refuse_without_allocating(self):
        for rows, columns in ((0,384), (129,384), (1,0), (32,385)):
            p = subprocess.run([str(self.binary), str(rows), str(columns)],
                capture_output=True, timeout=5)
            self.assertEqual(p.returncode, 3)


if __name__ == "__main__":
    unittest.main()
