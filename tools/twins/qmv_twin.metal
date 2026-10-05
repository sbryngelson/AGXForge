// The Apple-compiled twin of g17qmv.build_qmv2's cooperative split-K q4 matrix-vector kernel (the decode
// graph's qkv, wo_res1, ffn (fused SwiGLU), w2_res2 and lm head): the same threads, the same work partition and
// the same fp32 operation order as g17qmv.qmv2_ksplit_reference / qmv2_reference_fast, written in Metal and
// compiled by xcrun metal. Variant macros (-D):
//   KQ, NOUT, SGS, SOFF, BOFF, RESOFF   shapes and byte offsets (W at 0 in buffer 2)
//   CHAINS      position chains (pool_masks a16 epi_fma) - else the plain fma chain with acc + s t, acc + b sx
//   EPI         0 y fp32 | 1 add16 (h = y + x16) | 2 add32_to16 (x16 = fp16(y + h)) | 3 SwiGLU act32
//   FFN         the SwiGLU's F (gate rows [0, F), up rows [F, 2F))
// Buffers as the graph binds them: 0 the written region, 1 x fp32 [K], 2 the weights (W u32 | S bf16 | B bf16).
#include <metal_stdlib>
using namespace metal;

#ifndef EPI
#define EPI 0
#endif
#if EPI == 3
#define NR 2
#else
#define NR 1
#endif

constant constexpr int K = KQ;
constant constexpr int WORDS = K / 8;
constant constexpr int GPR = K / 64;
constant constexpr int E = 16;                       // wpt 2 words x 8 fields
constant constexpr int TRIPS = K / (32 * E);
constant constexpr int TS = TRIPS / SGS;

static inline float bf(ushort h) { return as_type<float>(uint(h) << 16); }

static inline float row_partial(device const uint *W, device const ushort *S, device const ushort *B,
                                device const float *x, uint row, uint sg, uint lane) {
  float acc = 0.0f;
  for (int k = int(sg) * TS; k < int(sg + 1) * TS; ++k) {
    const int wi = int(row) * WORDS + k * 64 + int(lane) * 2;     // coalesced: trip k lane l reads words 64 k + 2 l
    const uint w0 = W[wi], w1 = W[wi + 1];
    const int e0 = k * 512 + int(lane) * 16;
    float xs[16];
    for (int e = 0; e < 16; ++e) xs[e] = x[e0 + e];
    float sx = xs[0];
    for (int e = 1; e < 16; ++e) sx = sx + xs[e];
    float t;
#if CHAINS
    // one chain per field position c = (e % 8) % 4 over the 16-bit masked field q 16^c and the raw x, then
    // t = t0, t = fma(t_c, 2^-4c, t)
    float tp[4];
    for (int e = 0; e < 16; ++e) {
      const uint w = e < 8 ? w0 : w1;
      const int j = e % 8, c = j % 4;
      const float qs = float(((w >> (16 * (j / 4))) & 0xFFFFu) & (0xFu << (4 * c)));
      if (e < 4) tp[c] = qs * xs[e];
      else tp[c] = fma(qs, xs[e], tp[c]);
    }
    t = tp[0];
    t = fma(tp[1], 0.0625f, t);
    t = fma(tp[2], 0.00390625f, t);
    t = fma(tp[3], 0.000244140625f, t);
#else
    t = float(w0 & 0xFu) * xs[0];
    for (int e = 1; e < 16; ++e) {
      const uint w = e < 8 ? w0 : w1;
      t = fma(float((w >> (4 * (e % 8))) & 0xFu), xs[e], t);
    }
#endif
    const int gi = e0 / 64;
    const float s = bf(S[int(row) * GPR + gi]), b = bf(B[int(row) * GPR + gi]);
#if CHAINS
    acc = fma(s, t, acc);
    acc = fma(b, sx, acc);
#else
    acc = acc + s * t;
    acc = acc + b * sx;
#endif
  }
  // the measured butterfly (masks 1, 8, 2, 4, 16)
  acc = acc + simd_shuffle_xor(acc, 1);
  acc = acc + simd_shuffle_xor(acc, 8);
  acc = acc + simd_shuffle_xor(acc, 2);
  acc = acc + simd_shuffle_xor(acc, 4);
  acc = acc + simd_shuffle_xor(acc, 16);
  return acc;
}

static inline float silu(float v) {
  // g17decodestep.silu: t = v (-1/ln 2), t = 2^t, t = t + 1, t = 1 / t, y = v t
  float t = v * -1.4426950408889634f;
  t = exp2(t);
  t = t + 1.0f;
  t = 1.0f / t;
  return v * t;
}

kernel void qmv_twin(device uchar *b0 [[buffer(0)]], device const float *x [[buffer(1)]],
                     device const uchar *b2 [[buffer(2)]],
                     uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  device const uint *W = (device const uint *)b2;
  device const ushort *S = (device const ushort *)(b2 + SOFF);
  device const ushort *B = (device const ushort *)(b2 + BOFF);
  threadgroup float part[SGS][NR];
  for (int r = 0; r < NR; ++r) {
#if EPI == 3
    const uint row = r == 0 ? tg : FFN + tg;
#else
    const uint row = tg;
#endif
    const float p = row_partial(W, S, B, x, row, sg, lane);
    if (lane == 0) part[sg][r] = p;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0) return;
  float y[NR];
  for (int r = 0; r < NR; ++r) {
    float v = part[0][r];
    for (int s = 1; s < SGS; ++s) v = v + part[s][r];          // ((p0 + p1) + p2) + ...
    y[r] = v;
  }
  device float *out = (device float *)b0;
#if EPI == 0
  out[tg] = y[0];
#elif EPI == 1
  device const half *x16 = (device const half *)(b0 + RESOFF);
  out[tg] = y[0] + float(x16[tg]);
#elif EPI == 2
  device half *x16 = (device half *)(b0 + RESOFF);
  x16[tg] = half(y[0] + out[tg]);
#else
  out[tg] = float(half(silu(y[0]) * y[1]));
#endif
}
