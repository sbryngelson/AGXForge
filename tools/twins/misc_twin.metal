// Apple-compiled twins of the decode graph's small kernels, same threads and order as ours:
//   norm_twin     g17decodeops.build_rmsnorm_wide (out32, rs_seed): one 1,024-thread threadgroup, d = 2,048
//   argmax1_twin  g17gen.build_pass1 (batched): threadgroup t reduces logits [t C, t C + C) to one (value, index) pair
//   gen_twin      g17gen.build_gen (batched): pass 2 over the G pairs, the token log, q0, the embedding row copy
// Macros: N_IN_HALF, N_X, N_OUT, N_G (byte offsets); A_C, A_PL, A_G, A_CAP, A_D, A_RX16, A_GEN, A_LOG.
#include <metal_stdlib>
using namespace metal;

static inline float bsum(float v) {
  v = v + simd_shuffle_xor(v, 1);
  v = v + simd_shuffle_xor(v, 8);
  v = v + simd_shuffle_xor(v, 2);
  v = v + simd_shuffle_xor(v, 4);
  v = v + simd_shuffle_xor(v, 16);
  return v;
}

#ifdef N_X
kernel void norm_twin(device uchar *b0 [[buffer(0)]], device const uchar *b1 [[buffer(1)]],
                      device const uchar *b2 [[buffer(2)]],
                      uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint DD = 2048, TPG = 1024;
  threadgroup float part[32];
#if N_IN_HALF
  device const half *x = (device const half *)(b1 + N_X);
#else
  device const float *x = (device const float *)(b1 + N_X);
#endif
  device const half *g = (device const half *)(b2 + N_G);
  device float *out = (device float *)(b0 + N_OUT);
  const float v0 = float(x[t]), v1 = float(x[t + TPG]);
  float acc = v0 * v0;
  acc = acc + v1 * v1;
  const float s = bsum(acc);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float tot = bsum(part[lane]);
  const float mean = tot * (1.0f / float(DD));
  const float r = fast::rsqrt(mean + 0x1.4f8b58p-17f);      // our kernel's hardware rsqrt seed (op3850)
  for (uint e = t; e < DD; e += TPG) {
    const float y = (float(x[e]) * r) * float(g[e]);
    out[e] = float(half(y));
  }
}
#endif

#ifdef A_C
static inline void combine(thread float &v, thread float &i, float v2, float i2) {
  const float m = fmax(v, v2);
  const bool ea = v == m, eb = v2 == m;
  const float lo = fmin(i, i2);
  const float pick = ea ? i : i2;
  i = (ea && eb) ? lo : pick;
  v = m;
}

static inline void lane_reduce(thread float &v, thread float &i) {
  const ushort masks[5] = {1, 8, 2, 4, 16};
  for (int k = 0; k < 5; ++k) {
    const float vs = simd_shuffle_xor(v, masks[k]), is = simd_shuffle_xor(i, masks[k]);
    combine(v, i, vs, is);
  }
}

kernel void argmax1_twin(device const float *logits [[buffer(1)]], device float *pairs [[buffer(3)]],
                         uint t [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint base = t * A_C + lane;
  float x[A_PL];
  for (int k = 0; k < A_PL; ++k) x[k] = logits[base + 32 * k];
  float v = x[0], i = float(base);
  for (int k = 1; k < A_PL; ++k) combine(v, i, x[k], float(base + 32 * k));
  lane_reduce(v, i);
  pairs[2 * t] = v;
  pairs[2 * t + 1] = i;
}

kernel void gen_twin(device const float *pairs [[buffer(1)]], device const uint *emb [[buffer(2)]],
                     device uint *R [[buffer(3)]], uint lane [[thread_index_in_simdgroup]]) {
  float v = 0.0f, i = 0.0f;
  for (int k = 0; k < (A_G + 31) / 32; ++k) {
    uint p = lane + 32 * k;
    if (32 * (k + 1) > A_G) p = p > A_G - 1 ? 0 : p;
    const float pv = pairs[2 * p], pi = pairs[2 * p + 1];
    if (k == 0) { v = pv; i = pi; } else combine(v, i, pv, pi);
  }
  lane_reduce(v, i);
  const uint targ = uint(i);
  const uint q0 = R[A_GEN / 4];
  uint nxt = q0 + 1;
  nxt = nxt > A_CAP - 1 ? A_CAP - 1 : nxt;
  const uint li = nxt + A_LOG / 4;
  const uint forced = R[li];
  const uint tok = forced == 0xFFFFFFFFu ? targ : forced;
  // every lane has read the state before lane 0 writes it
  simdgroup_barrier(mem_flags::mem_device);
  if (lane == 0) { R[li] = tok; R[A_GEN / 4] = nxt; }
  for (uint w = lane; w < A_D / 2; w += 32) R[A_RX16 / 4 + w] = emb[tok * (A_D / 2) + w];
}
#endif
