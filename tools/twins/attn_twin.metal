// The Apple-compiled twin of g17attn.build_attn_split_rope's delivered decode attention (wide, rope_tables,
// fused_merge, bfly_merge, attn32, kvvec, hw_exp2): one 1,024-thread threadgroup per q head, simdgroup s walks keys
// s, s + 32, ...; RoPE and the cache append in-kernel; the 32 slice partials meet in threadgroup memory and every
// simdgroup merges four output dims by lane butterflies. Same threads, same partition, same fp32 order as
// g17attn.attn_reference (exp2 is Metal's, as the hardware exp2 is ours). Macros: A_H, A_KVH, A_CAP, A_KOFF, A_VOFF, A_OUTAT, A_COST, A_SINT.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint D = 128;
constant constexpr uint S = 32;
constant constexpr uint PW = 4 + D;

static inline float bsum(float v) {
  v = v + simd_shuffle_xor(v, 1);
  v = v + simd_shuffle_xor(v, 8);
  v = v + simd_shuffle_xor(v, 2);
  v = v + simd_shuffle_xor(v, 4);
  v = v + simd_shuffle_xor(v, 16);
  return v;
}

static inline float bmax(float v) {
  v = fmax(v, simd_shuffle_xor(v, 1));
  v = fmax(v, simd_shuffle_xor(v, 8));
  v = fmax(v, simd_shuffle_xor(v, 2));
  v = fmax(v, simd_shuffle_xor(v, 4));
  v = fmax(v, simd_shuffle_xor(v, 16));
  return v;
}

kernel void attn_twin(device uchar *b0 [[buffer(0)]], device const float *qkv [[buffer(1)]],
                      device const uchar *b2 [[buffer(2)]],
                      uint g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                      uint s [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[S * PW];
  const uint q0raw = *(device const uint *)b2;
  const uint q0 = q0raw > A_CAP - 1 ? A_CAP - 1 : q0raw;
  device const float *cosT = (device const float *)(b2 + A_COST);
  device const float *sinT = (device const float *)(b2 + A_SINT);
  const uint h = g, kvh = h / (A_H / A_KVH), d0 = lane * 4;
  const uint dm = 4 * (lane & 15) + q0 * (D / 2);
  const float sgn = float(lane >> 4) * 2.0f + -1.0f;
  float cs[4], sn[4];
  for (int i = 0; i < 4; ++i) { cs[i] = cosT[dm + i]; sn[i] = sinT[dm + i]; }

  // the append first: k rotated and v, as fp16, at row q0 of this KV head
  device half *Kc = (device half *)(b0 + A_KOFF) + kvh * A_CAP * D + d0;
  device half *Vc = (device half *)(b0 + A_VOFF) + kvh * A_CAP * D + d0;
  for (int i = 0; i < 4; ++i) {
    const float kin = qkv[A_H * D + kvh * D + d0 + i];
    const float vin = qkv[A_H * D + A_KVH * D + kvh * D + d0 + i];
    const float pt = simd_shuffle_xor(kin, 16);
    const float kr = kin * cs[i] + (pt * sn[i]) * sgn;
    Kc[q0 * D + i] = half(kr);
    Vc[q0 * D + i] = half(vin);
  }
  float qv[4];
  for (int i = 0; i < 4; ++i) {
    const float qin = qkv[h * D + d0 + i];
    const float pt = simd_shuffle_xor(qin, 16);
    const float qr = qin * cs[i] + (pt * sn[i]) * sgn;
    qv[i] = float(half(qr * 0x1.0527dcp-3f));       // log2(e) / sqrt(128), rounded to fp16
  }

  uint trips = (q0 + S - s) >> 5;
  if (trips == 0) trips = 1;
  float m = -FLT_MAX, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint t = 0; t < trips; ++t) {
    const uint j = s + S * t;
    const uint jr = j > q0 ? q0 : j;
    const half4 k = *(device const half4 *)(Kc + jr * D);
    const half4 v = *(device const half4 *)(Vc + jr * D);
    float pd = qv[0] * float(k[0]);
    pd = fma(qv[1], float(k[1]), pd);
    pd = fma(qv[2], float(k[2]), pd);
    pd = fma(qv[3], float(k[3]), pd);
    float sc = bsum(pd);
    sc = j > q0 ? m : sc;
    const float mn = fmax(m, sc);
    const float nmn = mn * -1.0f;
    const float al = exp2(m + nmn);
    const float p = j > q0 ? 0.0f : exp2(sc + nmn);
    l = l * al + p;
    for (int i = 0; i < 4; ++i) o[i] = fma(p, float(v[i]), o[i] * al);
    m = mn;
  }
  if (lane == 0) { part[s * PW] = m; part[s * PW + 1] = l; }
  for (int i = 0; i < 4; ++i) part[s * PW + 4 + d0 + i] = o[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // the butterfly merge: simdgroup s merges dims 4 s .. 4 s + 3, lane l holds slice l
  const float m_ = part[lane * PW], l_ = part[lane * PW + 1];
  const float M = bmax(m_);
  float w = exp2(m_ + M * -1.0f);
  w = lane > q0 ? 0.0f : w;
  const float L = bsum(l_ * w);
  const float rL = 1.0f / L;
  device float *out = (device float *)(b0 + A_OUTAT) + h * D + 4 * s;
  for (int i = 0; i < 4; ++i) {
    const float y = bsum(part[lane * PW + 4 + 4 * s + i] * w) * rL;
    if (lane == 0) out[i] = float(half(y));
  }
}
