// Host buffer mechanics shared by scalar execution and offline half preparation.
// This header neither validates an image nor authorizes a Metal operation.
#ifndef G17_SCAN_STORAGE_H
#define G17_SCAN_STORAGE_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
  size_t rows, columns, elementBytes, matrixBytes, queryBytes, inputBytes;
  size_t replyBytes, outputBytes;
  bool half;
} G17Storage;

static inline bool g17StorageInit(G17Storage *s, size_t rows, size_t columns, bool half) {
  if (!rows || rows > 500000 || !columns || columns > 384) return false;
  size_t width = half ? 2 : 4;
  *s = (G17Storage){.rows=rows, .columns=columns, .elementBytes=width,
    .matrixBytes=rows*columns*width, .queryBytes=columns*width,
    .inputBytes=(rows+1)*columns*width, .replyBytes=rows*(half ? 2 : 8),
    .outputBytes=rows*(half ? 2 : 8)+128, .half=half};
  return true;
}

static inline bool g17StorageFinite(const G17Storage *s, const void *data, size_t count) {
  const uint8_t *bytes = data;
  for (size_t i=0; i<count; ++i) {
    if (s->half) {
      uint16_t bits;
      memcpy(&bits, bytes+i*2, 2);
      if ((bits & 0x7c00) == 0x7c00) return false;
    } else {
      uint32_t bits;
      memcpy(&bits, bytes+i*4, 4);
      if ((bits & 0x7f800000) == 0x7f800000) return false;
    }
  }
  return true;
}

static inline void g17StorageReset(const G17Storage *s, void *output) {
  uint8_t *bytes = output;
  if (s->half) {
    // A missing score remains NaN. The first guard begins at the very next
    // halfword, including the upper half of the final word for an odd grid.
    const uint16_t sentinel = 0x7e01, guard = 0xa55a;
    for (size_t i=0; i<s->rows; ++i) memcpy(bytes+i*2, &sentinel, 2);
    for (size_t i=s->replyBytes; i<s->outputBytes; i+=2) memcpy(bytes+i, &guard, 2);
  } else {
    const uint32_t guard = 0xdeadbeef;
    for (size_t i=0; i<s->outputBytes; i+=4) memcpy(bytes+i, &guard, 4);
  }
}

// NULL means every host-side check passed. Numerical correctness is separate.
static inline const char *g17StorageCheck(const G17Storage *s, const void *output) {
  const uint8_t *bytes = output;
  if (!g17StorageFinite(s, output, s->rows)) return "nonfinite_score";
  if (!s->half) {
    for (size_t i=s->rows*4; i<s->replyBytes; i+=4) {
      uint32_t bits;
      memcpy(&bits, bytes+i, 4);
      if (bits != 0x5a17c0de) return "completion_marker";
    }
  }
  for (size_t i=s->replyBytes; i<s->outputBytes; i+=s->elementBytes) {
    if (s->half) {
      uint16_t bits;
      memcpy(&bits, bytes+i, 2);
      if (bits != 0xa55a) return "boundary_guard";
    } else {
      uint32_t bits;
      memcpy(&bits, bytes+i, 4);
      if (bits != 0xdeadbeef) return "boundary_guard";
    }
  }
  return NULL;
}

// Four FP32 application buffers: source, gamma, beta, output. These are host
// allocations, not compiler resource ranks. Metal binding indices come from ABI.
enum { G17_LN_BUFFERS = 4, G17_LN_INPUTS = 3, G17_LN_GUARD = 128 };
typedef struct {
  size_t rows, columns, payloadBytes[G17_LN_BUFFERS], allocationBytes[G17_LN_BUFFERS];
} G17LayerNormStorage;

static inline bool g17LayerNormStorageInit(G17LayerNormStorage *s, size_t rows, size_t columns) {
  if (!s || !rows || rows > 128 || !columns || columns > 384) return false;
  *s = (G17LayerNormStorage){.rows=rows, .columns=columns,
    .payloadBytes={rows*columns*4, columns*4, columns*4, rows*columns*4}};
  for (size_t i=0; i<G17_LN_BUFFERS; ++i)
    s->allocationBytes[i] = s->payloadBytes[i] + 2*G17_LN_GUARD;
  return true;
}

// Query projection uses the same four-allocation guard/readonly protocol with
// different payloads: source, square query weight, bias, output. This describes
// host storage only; it neither admits an image nor supplies its launch metadata.
static inline bool g17QueryProjectionStorageInit(G17LayerNormStorage *s,
                                                 size_t rows, size_t columns) {
  if (!s || !rows || rows > 128 || columns != 384) return false;
  *s = (G17LayerNormStorage){.rows=rows, .columns=columns,
    .payloadBytes={rows*columns*4, columns*columns*4, columns*4, rows*columns*4}};
  for (size_t i=0; i<G17_LN_BUFFERS; ++i)
    s->allocationBytes[i] = s->payloadBytes[i] + 2*G17_LN_GUARD;
  return true;
}

static inline void g17LayerNormResetOutput(const G17LayerNormStorage *s, void *allocation) {
  uint8_t *bytes = allocation;
  const uint32_t sentinel = 0x7fc01234;
  memset(bytes, 0xa5 ^ 3, s->allocationBytes[3]);
  for (size_t i=0; i<s->payloadBytes[3]; i+=4)
    memcpy(bytes+G17_LN_GUARD+i, &sentinel, 4);
}

// Allocations must have the sizes above. Inputs are separate owned snapshots;
// retaining them permits detecting a store into any read-only payload.
// Validate all inputs before changing any allocation.
static inline const char *g17LayerNormPrepare(const G17LayerNormStorage *s,
    void *const allocations[G17_LN_BUFFERS], const void *const inputs[G17_LN_INPUTS]) {
  const G17Storage f32 = {.half=false};
  for (size_t i=0; i<G17_LN_BUFFERS; ++i)
    if (!allocations[i]) return "missing_buffer";
  for (size_t i=0; i<G17_LN_INPUTS; ++i)
    if (!inputs[i] || !g17StorageFinite(&f32, inputs[i], s->payloadBytes[i]/4))
      return "nonfinite_input";
  for (size_t i=0; i<G17_LN_INPUTS; ++i) {
    memset(allocations[i], 0xa5 ^ (int)i, s->allocationBytes[i]);
    memcpy((uint8_t *)allocations[i]+G17_LN_GUARD, inputs[i], s->payloadBytes[i]);
  }
  g17LayerNormResetOutput(s, allocations[3]);
  return NULL;
}

// Begin another query only after checking the previous one. Parameters retain
// their allocations and payloads. Source is an independently owned snapshot.
static inline const char *g17LayerNormBeginQuery(const G17LayerNormStorage *s,
    void *const allocations[G17_LN_BUFFERS], const void *source) {
  const G17Storage f32 = {.half=false};
  if (!source || !g17StorageFinite(&f32, source, s->payloadBytes[0]/4))
    return "nonfinite_input";
  if (!allocations[0] || !allocations[3]) return "missing_buffer";
  memcpy((uint8_t *)allocations[0]+G17_LN_GUARD, source, s->payloadBytes[0]);
  g17LayerNormResetOutput(s, allocations[3]);
  return NULL;
}

// Check all four allocations and all output elements after command completion.
// NULL means memory/completion checks passed; application math is checked apart.
static inline const char *g17LayerNormCheck(const G17LayerNormStorage *s,
    const void *const allocations[G17_LN_BUFFERS], const void *const inputs[G17_LN_INPUTS],
    size_t *failedBuffer) {
  for (size_t i=0; i<G17_LN_BUFFERS; ++i) {
    if (failedBuffer) *failedBuffer = i;
    const uint8_t *bytes = allocations[i];
    if (!bytes) return "missing_buffer";
    const uint8_t guard = (uint8_t)(0xa5 ^ i);
    for (size_t k=0; k<G17_LN_GUARD; ++k)
      if (bytes[k] != guard || bytes[G17_LN_GUARD+s->payloadBytes[i]+k] != guard)
        return "boundary_guard";
    if (i<G17_LN_INPUTS && (!inputs[i] ||
        memcmp(bytes+G17_LN_GUARD, inputs[i], s->payloadBytes[i])))
      return "readonly_input_changed";
  }
  const G17Storage f32 = {.half=false};
  if (!g17StorageFinite(&f32, (const uint8_t *)allocations[3]+G17_LN_GUARD,
                         s->payloadBytes[3]/4)) return "nonfinite_output";
  return NULL;
}
#endif
