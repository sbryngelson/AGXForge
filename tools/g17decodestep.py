#!/usr/bin/env python3
"""One transformer decoder-layer DECODE STEP (one new token): the host reference and the dispatch plan.

The milestone (MM 25.132) is this step run through the repository's compiler (cc + tlower) on the GPU,
bit-exact against a host reference, then timed against MLX. This module is the REFERENCE and the PLAN; it
imports no GPU code (tools/g17decodestep_gpu.py dispatches, tools/g17decodestep_mlx.py is the baseline).

The step, in order (`STAGES`):

    attn_norm    h1 = RMSNorm(x) * g1                        -> half
    qkv_proj     [q | k | v] = h1 Wqkv                       -> fp32
    rope_append  RoPE on q and k at position kv_len; q scaled by log2(e)/sqrt(head_dim); the new k, v
               appended to the cache at row kv_len           -> half
    attention    per head, the base-2 online softmax over kv_len + 1 keys in 16-key blocks (the P7/P9
               attention class's arithmetic), O normalised    -> fp32 O, then half
    o_proj       h = x + attn Wo (the residual is the GEMM's accumulate add)             -> fp32
    ffn_norm     h2 = RMSNorm(h) * g2                        -> half
    ffn_gate_up  gate = h2 Wgate, up = h2 Wup                  -> fp32
    ffn_swiglu   a = silu(gate) * up                           -> half
    ffn_down     out = h + a Wdown                             -> fp32, then half

`reference()` mirrors the GPU's numerics stage by stage: 16-bit storage, fp32 accumulation, and the
rounding points the tensor path uses. It REUSES the measured models rather than restating them:
  - every GEMM is the section-136 MMA (docs/g17-tensorops-machine-model.md chapter 4): per 16-wide
    issue the products, adjacent-pair sums, interleave-by-8 sums, then C FIRST and the four sums in
    order, RNE32 at every step; issues compose in ascending K through the rounded fp32 accumulator; the
    first issue is the no-C form; an ACCUMULATE input is added once after the chain (section 29).
    `gemm` is a vectorised transcription of g17tensorcommonruntime._gemm_mma, and the tests check it
    equals `_gemm_mma` bit for bit;
  - fp32 A operands (the attention's P) are truncated to 19 bits (chapter 5, `_truncate_fp32`);
  - every fp32 ALU op rounds to nearest even and flushes subnormal inputs and results to a signed zero
    (recon 138 part 5); in the attention exp2 and recip are the exact value rounded once
    (`_StreamPoint`); rsqrt and SiLU's recip are rounded once and the GPU makes them so (an exact
    midpoint test on the hardware seed, MM 25.136); SiLU's exp2 is `exp2_soft` (MM 25.136);
  - the attention's row statistics are `_StreamPoint` with `_stream_rowmax` / `_stream_rowsum`, in the
    order `_kv_step_trace` walks them; the causal mask value is tensorreduce.CAUSAL_MASK_VALUE;
  - SiLU is the GELU register step's instruction sequence without its 1.702 multiply (gelu_model),
    with exp2_soft for its exp2 (MM 25.136);
  - the 16-bit narrowing is op1016's RNE (numpy's astype(float16) is RNE).
Rounding points that no built class fixes yet are CHOSEN here and stated, so the class that is built
must match them (not the other way round): the RMSNorm lane order (lane l owns elements l, l + 32, ...,
a sequential fp32 chain per lane, then the measured row butterfly (1, 8) and column butterfly
(2, 4, 16)), rsqrt as exact-rounded-once (op3850 is within one ulp and not correctly rounded, MM
25.136, so the GPU corrects its seed with an exact midpoint test), the RoPE arithmetic
(two fmuls and one fadd per output, no fused multiply-add), and q's softmax scale folded into RoPE.

`ideal()` is the same step in float64 from the same stored inputs, with no intermediate rounding and the
natural-base softmax: the error reference, not a pass rule.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import g17tensorcommonruntime as TCR     # noqa: E402  (numpy only at import; no GPU code)

F32, F64 = np.float32, np.float64
LOG2E = 1.4426950408889634
KEY_BLOCK = 16                             # keys per score tile (runtime.ATTENTION_BLOCK)


@dataclasses.dataclass(frozen=True)
class LayerSpec:
    """One decoder layer's decode step. `storage` is the 16-bit type of every stored tensor (weights,
    activations, cache); accumulation is fp32 throughout. `k_chunk` 0 is the single-chain class (one
    MMA chain over the whole K); k_chunk > 0 is split-K on every GEMM with slices of that length, under
    gemm_reference's contract (single-chain partials, ascending fp32 left fold, the residual C last). The
    host-orchestrated pipeline needs it: a straight-line gemm_generic body admits K <= 256 (the K loop is
    a loop, and wide_n excludes it)."""
    d_model: int = 2048
    n_heads: int = 16
    head_dim: int = 128
    ffn_dim: int = 8192
    kv_len: int = 256
    storage: str = "half"                  # "half" or "bfloat"
    norm_eps: float = 1.0e-5
    rope_base: float = 10000.0
    k_chunk: int = 0
    # k_route True: each projection at projection_route's split_k, the GPU pipeline's route (G 8 for the
    # 2048-wide qkv, o_proj and ffn_down at the milestone, G 1 for gate/up), instead of k_chunk on every
    # GEMM. It is the reference tools/g17decodestep_gpu.py checks the whole step against when k_chunk is 0.
    k_route: bool = False
    # GQA (MM 25.138.2): n_kv_heads KV heads, each shared by n_heads / n_kv_heads query heads; 0 is n_heads (MHA).
    # The qkv projection is then N = d_model + 2 n_kv_heads head_dim, and the RoPE append writes the cache of
    # every QUERY head from its KV head, so the attention sees the MHA layout unchanged
    n_kv_heads: int = 0

    @property
    def kv_heads(self):
        return self.n_kv_heads or self.n_heads

    @property
    def qkv_width(self):
        return self.d_model + 2 * self.kv_heads * self.head_dim

    def __post_init__(self):
        if self.storage not in ("half", "bfloat"):
            raise ValueError("storage is half or bfloat")
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError("n_heads * head_dim must equal d_model")
        for name in ("d_model", "head_dim", "ffn_dim"):
            if getattr(self, name) % 16:
                raise ValueError("%s must be a multiple of 16 (the MMA issue)" % name)
        if self.head_dim % 2:
            raise ValueError("RoPE needs an even head_dim")
        if self.kv_len < 0:
            raise ValueError("kv_len must be non-negative")
        if self.k_chunk and (self.k_chunk < 0 or self.k_chunk % 16 or self.d_model % self.k_chunk
                             or self.ffn_dim % self.k_chunk):
            raise ValueError("k_chunk must be a multiple of 16 dividing d_model and ffn_dim")

    @property
    def n_keys(self):
        return self.kv_len + 1

    @property
    def key_blocks(self):
        return -(-self.n_keys // KEY_BLOCK)

    def as_dict(self):
        return dataclasses.asdict(self)


MILESTONE = LayerSpec()
# a layer every stage of which the host-orchestrated pipeline (tools/g17decodestep_gpu.py) runs on
# today's straight-line classes: the attention class admits QK head 64, value 16 (a V slice per
# dispatch) and 8 key blocks (so kv_len <= 127); a straight-line GEMM admits K <= 256 (k_chunk 256)
TODAY = LayerSpec(d_model=512, n_heads=8, head_dim=64, ffn_dim=1024, kv_len=127, k_chunk=256)
TODAY_WIDE = LayerSpec(d_model=1024, n_heads=16, head_dim=64, ffn_dim=4096, kv_len=127, k_chunk=256)
TINY = LayerSpec(d_model=128, n_heads=2, head_dim=64, ffn_dim=256, kv_len=20, k_chunk=64)
SPECS = {"milestone": MILESTONE, "today": TODAY, "today_wide": TODAY_WIDE, "tiny": TINY}


# ---------------------------------------------------------------------------------------------------
# element arithmetic (the ALU model: RNE fp32, subnormal inputs and results flushed to signed zero)

_TINY32 = F32(2.0 ** -126)


def ftz(v):
    v = np.array(v, dtype=F32, copy=True)
    tiny = (v != 0) & (np.abs(v) < _TINY32)
    v[tiny] = np.copysign(F32(0.0), v[tiny])
    return v


def fadd(x, y):
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(ftz(x) + ftz(y))


def fmul(x, y):
    with np.errstate(over="ignore", invalid="ignore"):
        return ftz(ftz(x) * ftz(y))


def exp2(x):
    with np.errstate(over="ignore"):
        return ftz(np.exp2(ftz(x).astype(F64)).astype(F32))


def _recip64(x):
    with np.errstate(divide="ignore", over="ignore"):
        return ftz((1.0 / ftz(x).astype(F64)).astype(F32))


def _rsqrt64(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return ftz((1.0 / np.sqrt(ftz(x).astype(F64))).astype(F32))


# THE EXACT MIDPOINT TEST (MM 25.136). For a positive normal candidate a, the midpoint between a and its
# successor is (2 Ma + 1) 2^(Ea - 151) (Ma the 24-bit significand, Ea the biased exponent, also across a
# binade edge). 1/sqrt(x) is below it exactly when Mx (2 Ma + 1)^2 > 2^(452 - Ex - 2 Ea), and 1/g exactly
# when Mg (2 Ma + 1) > 2^(301 - Eg - Ea). The products are formed in 15-bit limbs, every intermediate
# below 2^31, keeping only the bits from 2^60 (2^30) up: the comparison with a power of two at least
# that large needs no more. Equality is impossible (2 Ma + 1 is odd and at least 2^24), so the test never
# ties. tools/g17decodeops.py emits the same limbs in the GPU program, instruction for instruction.
_M15 = 0x7FFF


def _fields(bits):
    bits = np.asarray(bits, np.int64)
    return (bits & 0x7FFFFF) | 0x800000, (bits >> 23) & 0xFF


def mid_below_rsqrt(bx, ba):
    """True where 1/sqrt(x) < mid(a, a+) (x and a as float32 bit patterns)."""
    Mx, Ex = _fields(bx)
    Ma, Ea = _fields(ba)
    q = 2 * Ma + 1
    q0, q1 = q & _M15, q >> 15
    t0 = q0 * q0
    s0, c = t0 & _M15, t0 >> 15
    t1 = ((q0 * q1) << 1) + c
    s1, c = t1 & _M15, t1 >> 15
    t2 = q1 * q1 + c
    s2, s3 = t2 & _M15, t2 >> 15
    m0, m1 = Mx & _M15, Mx >> 15
    k = (s0 * m0) >> 15
    k = (s1 * m0 + s0 * m1 + k) >> 15
    k = (s2 * m0 + s1 * m1 + k) >> 15
    k = (s3 * m0 + s2 * m1 + k) >> 15
    top = s3 * m1 + k                                     # floor(Mx q^2 / 2^60)
    sh = 392 - Ex - 2 * Ea                                # S - 60
    if np.any((sh < 0) | (sh > 30)):
        raise ValueError("rsqrt midpoint test: the candidate is not within a binade of 1/sqrt(x)")
    return top >= (np.int64(1) << sh)


def mid_below_recip(bg, ba):
    """True where 1/g < mid(a, a+)."""
    Mg, Eg = _fields(bg)
    Ma, Ea = _fields(ba)
    q = 2 * Ma + 1
    q0, q1 = q & _M15, q >> 15
    m0, m1 = Mg & _M15, Mg >> 15
    k = (q0 * m0) >> 15
    k = (q1 * m0 + q0 * m1 + k) >> 15
    top = q1 * m1 + k                                     # floor(Mg q / 2^30)
    sh = 271 - Eg - Ea                                    # S - 30
    if np.any((sh < 0) | (sh > 30)):
        raise ValueError("recip midpoint test: the candidate is not within a binade of 1/g")
    return top >= (np.int64(1) << sh)


def round_once(below, x, seed):
    """The correctly rounded value from any seed within one ulp of it: y0 - 1 where f lies below
    mid(y0 - 1, y0), y0 + 1 where it lies above mid(y0, y0 + 1), else y0 (the GPU program's rule)."""
    bx = np.asarray(x, F32).view(np.uint32)
    by = np.asarray(seed, F32).view(np.uint32).astype(np.int64)
    down = below(bx, by - 1)
    up = ~below(bx, by)
    return np.where(down, by - 1, np.where(up, by + 1, by)).astype(np.uint32).view(F32)


def recip(x):
    """1/x rounded once (correctly rounded), for a positive normal x whose reciprocal is normal: the
    float64 value, within one ulp, put through the exact midpoint test, so no double rounding
    survives. On the GPU, op3658's seed through the same test (MM 25.136: op3658 is within one ulp
    and not correctly rounded)."""
    x = np.asarray(x, F32)
    return round_once(mid_below_recip, x, _recip64(x))


def rsqrt(x):
    """1/sqrt(x) rounded once (correctly rounded), for a positive normal x: as `recip`. On the GPU,
    op3850's seed through the same test (MM 25.136 measured op3850: within one ulp, not correctly
    rounded)."""
    x = np.asarray(x, F32)
    return round_once(mid_below_rsqrt, x, _rsqrt64(x))


_SEED = None


def rsqrt_seed(x):
    """op3850 itself (cc `b.rsqrt`, the hardware rsqrt seed) for a positive normal x, bit for bit (MM 25.141.16).
    It is faithful (within 0.82 ulp) and equals `rsqrt` on 95.7 percent of inputs. Measured as an exact function:
    2^24 inputs per band, and seed(x 4^k) == seed(x) 2^-k on every input at k = -8 and k = +5. So the seed is
    `rsqrt` except at the 725,821 (parity, mantissa) keys in isa/g17-rsqrt-seed.npz, where it takes the stored
    [1, 4) value scaled by the same power of two."""
    global _SEED
    if _SEED is None:
        import os
        z = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "isa", "g17-rsqrt-seed.npz"))
        _SEED = (z["index"].astype(np.int64), z["seed"].astype(np.int64))
    x = np.asarray(x, F32)
    out = rsqrt(x).view(np.uint32).astype(np.int64)
    bits = x.view(np.uint32).astype(np.int64)
    e = (bits >> 23) & 0xFF
    if np.any((e == 0) | (e == 0xFF) | (bits >> 31 != 0)):
        raise ValueError("rsqrt_seed: positive normal inputs only")
    par = (e - 127) & 1
    key = (par << 23) | (bits & 0x7FFFFF)
    k = (e - 127 - par) >> 1                          # x = x0 4^k, x0 in [1, 4)
    pos = np.searchsorted(_SEED[0], key)
    pos = np.minimum(pos, len(_SEED[0]) - 1)
    hit = _SEED[0][pos] == key
    out = np.where(hit, _SEED[1][pos] - (k << 23), out)
    return out.astype(np.uint32).view(F32).reshape(x.shape)


# SILU'S EXP2 (MM 25.136). op1272 is within one ulp both ways and not correctly rounded, and no cheap
# exact test decides 2^t against a midpoint, so no reference that rounds exp2 once can be matched bit
# for bit (the GELU register step passes by an enclosure for this reason, 25.109). exp2_soft is a fixed
# sequence of RNE fp32 and integer steps the GPU reproduces by construction.
EXP2_MAGIC = F32(12582912.0)                 # 1.5 * 2^23: t + MAGIC holds rint(t) in its low bits
# [-125, 125]: p 2^n stays normal and finite, and SiLU's 1/(1 + 2^t) stays normal, so the exact
# midpoint test (which needs a normal result) applies wherever SiLU calls recip
EXP2_LO, EXP2_HI = F32(-125.0), F32(125.0)
EXP2_COEF = tuple(F32(math.log(2.0) ** k / math.factorial(k)) for k in range(8))


def exp2_soft(t):
    """t clamped to [-125, 125]; s = t + 1.5 * 2^23; n = s - 1.5 * 2^23 (rint(t), exact); f = t - n
    (exact, |f| <= 1/2); p = the degree-7 Taylor polynomial of 2^f in Horner form (fmul then fadd,
    RNE); then n added to p's exponent field as an integer. Within one ulp of 2^t."""
    t = np.asarray(t, F32)
    t = np.minimum(np.maximum(t, EXP2_LO), EXP2_HI)
    s = fadd(t, EXP2_MAGIC)
    nf = fadd(s, -EXP2_MAGIC)
    f = fadd(t, -nf)
    p = np.full(f.shape, EXP2_COEF[7], F32)
    for c in EXP2_COEF[6::-1]:
        p = fadd(fmul(p, f), c)
    n = s.view(np.uint32).astype(np.int64) - 0x4B400000
    return ((p.view(np.uint32).astype(np.int64) + (n << 23)) & 0xFFFFFFFF).astype(np.uint32).view(F32)


def narrow(x, storage="half"):
    """fp32 -> the 16-bit storage type, RNE (op1016 for half), returned as the fp32 value it holds."""
    x = np.asarray(x, dtype=F32)
    if storage == "half":
        with np.errstate(over="ignore"):
            return x.astype(np.float16).astype(F32)
    u = x.view(np.uint32).astype(np.uint64)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16
    out = rounded.astype(np.uint32).view(F32)
    return np.where(np.isnan(x), x, out).astype(F32)


def to_bits16(x, storage="half"):
    """The stored 16-bit patterns of values already representable in `storage`."""
    x = np.asarray(x, dtype=F32)
    if storage == "half":
        return x.astype(np.float16).view(np.uint16)
    return (x.view(np.uint32) >> 16).astype(np.uint16)


def truncate19(x):
    """The measured fp32-A operand quantisation (chapter 5): the low 13 bits cleared."""
    u = np.asarray(x, dtype=F32).view(np.uint32) & np.uint32(0xFFFFE000)
    return u.view(F32)


# ---------------------------------------------------------------------------------------------------
# the MMA (chapter 4), vectorised

def gemm(a, b, c=None, *, truncate_a=False, truncate_b=False):
    """D = A B (+ C once, after the chain), A (M x K), B (K x N), fp32 out: _gemm_mma vectorised.

    Each 16-wide issue: products rounded to fp32 (exact for 16-bit operands), p_i = RNE32(p_2i +
    p_2i+1), q_j = RNE32(p_j + p_j+4), acc = C (the previous issue's D; the first issue is the no-C
    form, acc = q_0) then acc = RNE32(acc + q_j). A float64 sum of two fp32 values rounded to fp32 is
    the correctly rounded fp32 sum, which is what `_rne32` of a Python float computes."""
    a = np.asarray(a, dtype=F32)
    b = np.asarray(b, dtype=F32)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("gemm: A is M x K and B is K x N")
    M, K = a.shape
    N = b.shape[1]
    if K % 16:
        raise ValueError("gemm: K must be a multiple of 16 (the MMA issue)")
    if truncate_a:
        a = truncate19(a)
    if truncate_b:
        b = truncate19(b)
    a64, b64 = a.astype(F64), b.astype(F64)
    r = lambda v: v.astype(F32).astype(F64)
    acc = None
    with np.errstate(over="ignore", invalid="ignore"):
        for s in range(0, K, 16):
            prod = r(a64[:, s:s + 16, None] * b64[None, s:s + 16, :])        # M x 16 x N
            p = r(prod[:, 0::2] + prod[:, 1::2])                                # p_0..p_7
            q = r(p[:, 0:4] + p[:, 4:8])                                        # q_0..q_3
            if acc is None:
                acc, rest = q[:, 0], range(1, 4)
            else:
                rest = range(4)
            for j in rest:
                acc = r(acc + q[:, j])
        if c is not None:
            acc = r(acc + np.asarray(c, dtype=F32).astype(F64).reshape(M, N))
    return acc.astype(F32)


def gemm_reference(A, B, C=None, split_k=1, *, truncate_a=False, truncate_b=False):
    """THE SPLIT-K CONTRACT (Set C's split-K kernel implements exactly this; MM 25.132):
      - K is partitioned into G = split_k contiguous slices; slice t owns [t K/G, (t+1) K/G). A K that
        is not a multiple of 16 G is refused (every slice is whole 16-wide MMA issues);
      - each partial p_t is the single-chain MMA over slice t alone (`gemm`, _gemm_mma's order within
        the slice: no-C first issue, C-first issues after it, ascending K);
      - the partials are reduced by an fp32 LEFT fold in ascending t: acc = ((p_0 + p_1) + p_2) + ...,
        each add RNE;
      - an accumulating op adds C LAST, out = RNE32(acc + C); a non-accumulating op (C None) returns acc.
    G = 1 is `gemm(A, B, C)` bit for bit. Different G are DIFFERENT VALUES, not a reordering to be
    checked bitwise: split_k_tolerance states the change (as the KV split's merge row does, 25.131)."""
    A = np.asarray(A, dtype=F32)
    B = np.asarray(B, dtype=F32)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError("gemm_reference: A is M x K and B is K x N")
    K = A.shape[1]
    G = int(split_k)
    if G < 1 or K % (16 * G):
        raise ValueError("refused: split_k %r does not split K %d into whole 16-wide issues (K %% (16 G) != 0)" % (split_k, K))
    ks = K // G
    acc = gemm(A[:, :ks], B[:ks], truncate_a=truncate_a, truncate_b=truncate_b)
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(1, G):
            acc = (acc + gemm(A[:, t * ks:(t + 1) * ks], B[t * ks:(t + 1) * ks],
                              truncate_a=truncate_a, truncate_b=truncate_b)).astype(F32)
        if C is not None:
            acc = (acc + np.asarray(C, dtype=F32).reshape(acc.shape)).astype(F32)
    return acc


def split_k_tolerance(A, B, C=None, split_k=2, *, truncate_a=False, truncate_b=False):
    """The value change split-K makes: max |gemm_reference(split_k=G) - gemm_reference(split_k=1)| on
    these operands (float64 difference of the fp32 results). A stated change, like the KV split's merge
    (25.131), not a bound that asserts bit-identity across G."""
    a = gemm_reference(A, B, C, split_k, truncate_a=truncate_a, truncate_b=truncate_b).astype(np.float64)
    b = gemm_reference(A, B, C, 1, truncate_a=truncate_a, truncate_b=truncate_b).astype(np.float64)
    with np.errstate(invalid="ignore"):
        return float(np.nanmax(np.abs(a - b))) if a.size else 0.0


def gemv(x, w, c=None, k_chunk=0):
    """One row (the GPU pads it to a tile; the padding rows never reach row 0). k_chunk > 0 is split-K
    with slices of k_chunk (gemm_reference, split_k = K / k_chunk); LayerSpec.k_chunk."""
    x = np.asarray(x, dtype=F32).reshape(1, -1)
    K = x.shape[1]
    G = 1 if not k_chunk or k_chunk >= K else K // k_chunk
    if G > 1 and K % k_chunk:
        raise ValueError("gemv: k_chunk must divide K")
    return gemm_reference(x, w, None if c is None else np.asarray(c).reshape(1, -1), split_k=G)[0]


# ---------------------------------------------------------------------------------------------------
# inputs

def make_inputs(spec: LayerSpec, seed: int = 20260924):
    """Seeded weights, token and KV cache, every tensor already a value of the storage type (fp32
    arrays holding 16-bit values). The RoPE tables are fp32 host constants (the angle in float64,
    cos/sin rounded once). Scales keep every stage well inside the half range."""
    rng = np.random.default_rng(seed)
    st = spec.storage
    d, H, hd, f, n = spec.d_model, spec.n_heads, spec.head_dim, spec.ffn_dim, spec.kv_len
    def w(rows, cols):
        return narrow(rng.uniform(-1.0, 1.0, size=(rows, cols)) * math.sqrt(3.0 / rows), st)
    inp = dict(
        x=narrow(rng.standard_normal(d), st),
        g1=narrow(1.0 + 0.1 * rng.standard_normal(d), st),
        wqkv=w(d, 3 * d),
        wo=w(d, d),
        g2=narrow(1.0 + 0.1 * rng.standard_normal(d), st),
        wgate=w(d, f),
        wup=w(d, f),
        wdown=w(f, d),
        k_cache=narrow(rng.standard_normal((H, n, hd)), st),
        v_cache=narrow(rng.standard_normal((H, n, hd)), st),
    )
    inv_freq = spec.rope_base ** (-np.arange(0, hd, 2, dtype=F64) / hd)
    angle = spec.kv_len * inv_freq
    inp["rope_cos"] = np.cos(angle).astype(F32)
    inp["rope_sin"] = np.sin(angle).astype(F32)
    return inp


# ---------------------------------------------------------------------------------------------------
# stages: each takes and returns a dict of arrays, so each GPU dispatch can be checked on its own

def rmsnorm(v, g, spec):
    """y = (v * rsqrt(mean(v^2) + eps)) * g, narrowed. Lane l owns elements l, l + 32, ... (a
    sequential fp32 chain, ascending), then the measured row butterfly (lane ^ 1, lane ^ 8) and column
    butterfly (lane ^ 2, ^ 4, ^ 16) of tensorreduce: every lane ends with the same sum."""
    from agxforge.g17 import tensorreduce as TR
    v = np.asarray(v, dtype=F32)
    d = v.size
    if d % 32:
        raise ValueError("rmsnorm: the row must fill 32 lanes")
    sq = fmul(v, v).reshape(d // 32, 32)
    local = sq[0].copy()
    for i in range(1, d // 32):
        local = fadd(local, sq[i])
    lanes = TR.butterfly([float(x) for x in local], TR.ROW_BUTTERFLY_MASKS, "sum")
    lanes = TR.butterfly(list(lanes), TR.COLUMN_BUTTERFLY_MASKS, "sum")
    if len(set(lanes)) != 1:
        raise AssertionError("rmsnorm: the butterfly left lanes disagreeing")
    ss = F32(lanes[0])
    mean = fmul(ss, F32(1.0 / d))
    r = rsqrt(fadd(mean, F32(spec.norm_eps)))
    return narrow(fmul(fmul(v, r), np.asarray(g, dtype=F32)), spec.storage)


def stage_attn_norm(spec, x, g1):
    return dict(h1=rmsnorm(x, g1, spec))


def proj_chunk(spec, n_block, K):
    """The k_chunk of one projection: spec.k_chunk, or under k_route the slice length of
    projection_route's split_k for a block of n_block columns (0 = the single chain)."""
    if not spec.k_route:
        return spec.k_chunk
    G = projection_route(n_block, K, spec.k_chunk)[1]
    return 0 if G == 1 else K // G


def stage_qkv_proj(spec, h1, wqkv):
    return dict(qkv32=gemv(h1, wqkv, k_chunk=proj_chunk(spec, spec.d_model, spec.d_model)))


def split_qkv(spec, qkv32):
    """q (H x hd), and k and v per QUERY head: under GQA each KV head's row repeated for its query heads."""
    H, hd, KV, d = spec.n_heads, spec.head_dim, spec.kv_heads, spec.d_model
    q = qkv32[:d].reshape(H, hd)
    k = qkv32[d:d + KV * hd].reshape(KV, hd)
    v = qkv32[d + KV * hd:d + 2 * KV * hd].reshape(KV, hd)
    if KV != H:
        k, v = np.repeat(k, H // KV, axis=0), np.repeat(v, H // KV, axis=0)
    return q, k, v


def rope_rotate(t, cos, sin):
    """Rotate-half RoPE (pairs i and i + head_dim/2), per output two fmuls and one fadd."""
    h = t.shape[-1] // 2
    x1, x2 = t[..., :h], t[..., h:]
    o1 = fadd(fmul(x1, cos), -fmul(x2, sin))
    o2 = fadd(fmul(x2, cos), fmul(x1, sin))
    return np.concatenate([o1, o2], axis=-1)


def q_scale(spec):
    """The softmax scale 1/sqrt(head_dim), in base 2 (the attention class exponentiates with exp2)."""
    return F32(LOG2E / math.sqrt(spec.head_dim))


def stage_rope_append(spec, qkv32, rope_cos, rope_sin, k_cache, v_cache):
    q, k, v = split_qkv(spec, qkv32)
    q_rot = rope_rotate(q, rope_cos, rope_sin)
    k_rot = rope_rotate(k, rope_cos, rope_sin)
    q16 = narrow(fmul(q_rot, q_scale(spec)), spec.storage)
    k16, v16 = narrow(k_rot, spec.storage), narrow(v, spec.storage)
    k_all = np.concatenate([k_cache, k16[:, None, :]], axis=1)
    v_all = np.concatenate([v_cache, v16[:, None, :]], axis=1)
    return dict(q16=q16, k_new=k16, v_new=v16, k_all=k_all, v_all=v_all)


def attention_head(q, k_all, v_all, q0):
    """One head, one query row at position q0: the P7/P9 class's online softmax (the `_kv_step_trace`
    arithmetic at row 0) over ceil(keys / 16) blocks, keys past the end zero-padded and masked. The
    value width is any multiple of 16: PV's columns are independent, so a 16-wide slice of V gives the
    matching slice of O bit for bit."""
    ar = TCR._StreamPoint()
    kp, vp, nb = _padded_kv(k_all, v_all, -(-np.asarray(k_all).shape[0] // KEY_BLOCK))
    o, _m, l = attention_blocks(q, kp, vp, q0, range(nb))
    return fmul(o, F32(ar.recip(l)))


def _padded_kv(k_all, v_all, nb):
    """K and V zero-padded to nb blocks of 16 keys (fp32)."""
    n, hd = np.asarray(k_all).shape
    vw = np.asarray(v_all).shape[1]
    kp = np.zeros((nb * KEY_BLOCK, hd), F32); kp[:n] = k_all
    vp = np.zeros((nb * KEY_BLOCK, vw), F32); vp[:n] = v_all
    return kp, vp, nb


def attention_blocks(q, kp, vp, q0, blocks):
    """The online softmax of one query row at position q0 over the key blocks `blocks` (ascending, absolute
    indices into the padded kp / vp), UN-NORMALISED: (O, m, l), O scaled to the running max m. The first
    block takes the first-block stage, every later one the rescaling stage, in attention_head's arithmetic.
    A block list this row sees no key of (every key past q0, or no block) is the stream's initial state:
    O = +0, m = -FLT_MAX, l = 0 (the KV split's empty slice, MM 25.135.5)."""
    from agxforge.g17 import tensorreduce as TR
    ar = TCR._StreamPoint()
    blocks = list(blocks)
    vw = vp.shape[1]
    if not blocks or KEY_BLOCK * blocks[0] > q0:
        return np.zeros(vw, F32), float(TR.FP32_NEG_MAX), 0.0
    q = np.asarray(q, dtype=F32).reshape(1, kp.shape[1])
    neg_one, masked = ar.val(-1.0), ar.val(TR.CAUSAL_MASK_VALUE)
    o = m = l = None
    for i, j in enumerate(blocks):
        s = gemm(q, kp[16 * j:16 * j + 16].T.copy())[0]
        t = TR.causal_threshold(q0, 0, KEY_BLOCK * j)
        vals = [masked if col > t else float(s[col]) for col in range(16)]
        if i == 0:
            m = TCR._stream_rowmax(ar, vals)
            neg = ar.fmul(m, neg_one)
            p = [ar.exp2(ar.fadd(v, neg)) for v in vals]
            l = TCR._stream_rowsum(ar, p)
            c = None
        else:
            m_old, l_old = m, l
            m = ar.fmax(m_old, TCR._stream_rowmax(ar, vals))
            neg = ar.fmul(m, neg_one)
            alpha = ar.exp2(ar.fadd(m_old, neg))
            p = [ar.exp2(ar.fadd(v, neg)) for v in vals]
            l = ar.fadd(ar.fmul(alpha, l_old), TCR._stream_rowsum(ar, p))
            c = fmul(o, F32(alpha)).reshape(1, vw)
        o = gemm(np.asarray(p, F32).reshape(1, 16), vp[16 * j:16 * j + 16], c, truncate_a=True)[0]
    return o, m, l


# THE S-WAY KV SPLIT (MM 25.135.5): opt-in and VALUE-CHANGING, as P9's allow_value_change. The key blocks are
# padded to S equal slices (runtime.kv_split_slices); each slice runs attention_blocks into its own
# un-normalised partial, then the merge combines them; the split-aware reference the GPU merge is checked
# against. KV_MERGE_MODELS: "merge" (the contract), and the claims each control makes about the same output:
# "no_rescale" (f_s = 1), "drop_last_slice" (the merge over slices 0 .. S-2).
KV_MERGE_MODELS = ("merge", "no_rescale", "drop_last_slice")


def kv_merge_trace(ar, ms, ls, os_, model="merge", f=None):
    """The merge of S partial states on one row under arithmetic `ar` (TCR._StreamPoint or _StreamInterval):
    ms, ls the per-slice m_s and l_s, os_ the per-slice O words (lists of ar values). M = the fmax fold in
    ascending s; f_s = exp2(fadd(m_s, fmul(M, -1))); l = the left fold of fmul(f_s, l_s) with fadd, O
    likewise, word by word. `f` overrides the f_s (the one-ulp search over the hardware exp2). Returns
    (M, l, O, f)."""
    S = len(ms)
    if model == "drop_last_slice":
        S -= 1
    elif model not in ("merge", "no_rescale"):
        raise ValueError("kv merge model %r" % (model,))
    big = ms[0]
    for s in range(1, S):
        big = ar.fmax(big, ms[s])
    if f is None:
        if model == "no_rescale":
            f = [ar.val(1.0)] * S
        else:
            neg = ar.fmul(big, ar.val(-1.0))
            f = [ar.exp2(ar.fadd(ms[s], neg)) for s in range(S)]
    l = ar.fmul(f[0], ls[0])
    o = [ar.fmul(f[0], x) for x in os_[0]]
    for s in range(1, S):
        l = ar.fadd(l, ar.fmul(f[s], ls[s]))
        o = [ar.fadd(a, ar.fmul(f[s], x)) for a, x in zip(o, os_[s])]
    return big, l, o, f


def attention_head_partials(q, k_all, v_all, q0, S):
    """The S slices' un-normalised partials [(O, m, l)] of one head and query row: the keys padded to S equal
    slices of whole blocks (runtime.kv_split_slices), each slice's online softmax from its own start."""
    from agxforge.g17 import runtime as R
    nb = -(-np.asarray(k_all).shape[0] // KEY_BLOCK)
    padded, slices = R.kv_split_slices(nb, S)
    kp, vp, _nb = _padded_kv(k_all, v_all, padded)
    return [attention_blocks(q, kp, vp, q0, blocks) for blocks in slices]


def attention_head_split(q, k_all, v_all, q0, S, *, allow_value_change=False, model="merge"):
    """attention_head through an S-way KV split: the per-slice partials, the merge (kv_merge_trace), then the
    normalisation O * recip(l). It CHANGES THE VALUES against attention_head (a different association), so
    it runs only with allow_value_change=True. Returns (O normalised, M, l)."""
    if allow_value_change is not True:
        raise ValueError("refused: an S-way KV split changes the values; opt in with allow_value_change=True")
    ar = TCR._StreamPoint()
    parts = attention_head_partials(q, k_all, v_all, q0, S)
    big, l, o, _f = kv_merge_trace(ar, [float(p[1]) for p in parts], [float(p[2]) for p in parts],
                                   [[float(x) for x in p[0]] for p in parts], model)
    return fmul(np.asarray(o, F32), F32(ar.recip(l))), F32(big), F32(l)


def stage_attention(spec, q16, k_all, v_all):
    o32 = np.stack([attention_head(q16[h], k_all[h], v_all[h], spec.kv_len) for h in range(spec.n_heads)])
    return dict(o32=o32, attn=narrow(o32, spec.storage).reshape(-1))


def stage_o_proj(spec, attn, wo, x):
    return dict(h=gemv(attn, wo, x, k_chunk=proj_chunk(spec, spec.d_model, spec.d_model)))


def stage_ffn_norm(spec, h, g2):
    return dict(h2=rmsnorm(h, g2, spec))


def stage_ffn_gate_up(spec, h2, wgate, wup):
    kc = proj_chunk(spec, spec.ffn_dim, spec.d_model)
    return dict(gate32=gemv(h2, wgate, k_chunk=kc), up32=gemv(h2, wup, k_chunk=kc))


def silu(x):
    """The GELU register step (tlower 'gelu', gelu_model) without its 1.702 multiply: t = x*(-1/ln 2),
    t = 2**t, t = t + 1, t = 1/t, y = x*t. CHANGED (MM 25.136): 2**t is exp2_soft, not the exact value
    rounded once, because op1272 is not correctly rounded and nothing cheap corrects it; 1/t stays
    rounded once (the GPU corrects op3658 exactly)."""
    t = fmul(x, TCR.GELU_NEG_INV_LN2)
    t = exp2_soft(t)
    t = fadd(t, F32(1.0))
    t = recip(t)
    return fmul(x, t)


def stage_ffn_swiglu(spec, gate32, up32):
    return dict(act=narrow(fmul(silu(gate32), up32), spec.storage))


def stage_ffn_down(spec, act, wdown, h):
    """out = h + act Wdown (the residual is the accumulate add), then narrowed."""
    out32 = gemv(act, wdown, h, k_chunk=proj_chunk(spec, spec.d_model, spec.ffn_dim))
    return dict(out32=out32, out=narrow(out32, spec.storage))


# (name, function, input names, output names), in execution order
STAGES = (
    ("attn_norm", stage_attn_norm, ("x", "g1"), ("h1",)),
    ("qkv_proj", stage_qkv_proj, ("h1", "wqkv"), ("qkv32",)),
    ("rope_append", stage_rope_append, ("qkv32", "rope_cos", "rope_sin", "k_cache", "v_cache"),
     ("q16", "k_new", "v_new", "k_all", "v_all")),
    ("attention", stage_attention, ("q16", "k_all", "v_all"), ("o32", "attn")),
    ("o_proj", stage_o_proj, ("attn", "wo", "x"), ("h",)),
    ("ffn_norm", stage_ffn_norm, ("h", "g2"), ("h2",)),
    ("ffn_gate_up", stage_ffn_gate_up, ("h2", "wgate", "wup"), ("gate32", "up32")),
    ("ffn_swiglu", stage_ffn_swiglu, ("gate32", "up32"), ("act",)),
    ("ffn_down", stage_ffn_down, ("act", "wdown", "h"), ("out32", "out")),
)
STAGE_NAMES = tuple(s[0] for s in STAGES)


def run_stage(spec, name, env):
    """Run one stage on the named arrays of `env`; returns its outputs."""
    for sname, fn, ins, outs in STAGES:
        if sname == name:
            got = fn(spec, **{k: env[k] for k in ins})
            if tuple(got) != outs:
                raise AssertionError("stage %s returned %s, declared %s" % (name, tuple(got), outs))
            return got
    raise KeyError(name)


def reference(spec: LayerSpec, inputs):
    """The GPU-numerics reference: {stage: {"in": {...}, "out": {...}}} in STAGES order, plus "env"
    (every named array). Each stage's inputs are its predecessors' outputs or the layer's inputs."""
    env = dict(inputs)
    trace = {}
    for name, _fn, ins, _outs in STAGES:
        out = run_stage(spec, name, env)
        trace[name] = {"in": {k: env[k] for k in ins}, "out": out}
        env.update(out)
    trace["env"] = env
    return trace


def ideal(spec: LayerSpec, inputs):
    """The same step in float64 from the same stored inputs: no intermediate rounding, natural-base
    softmax with scale 1/sqrt(head_dim), exact RoPE angles. Returns the named intermediates."""
    I = {k: np.asarray(v, dtype=F64) for k, v in inputs.items()}
    d, H, hd = spec.d_model, spec.n_heads, spec.head_dim
    def rms(v, g):
        return v / np.sqrt(np.mean(v * v) + spec.norm_eps) * g
    h1 = rms(I["x"], I["g1"])
    qkv = h1 @ I["wqkv"]
    q, k, v = (qkv[i * d:(i + 1) * d].reshape(H, hd) for i in range(3))
    inv_freq = spec.rope_base ** (-np.arange(0, hd, 2, dtype=F64) / hd)
    cos, sin = np.cos(spec.kv_len * inv_freq), np.sin(spec.kv_len * inv_freq)
    def rot(t):
        x1, x2 = t[:, :hd // 2], t[:, hd // 2:]
        return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=1)
    q, k = rot(q), rot(k)
    k_all = np.concatenate([I["k_cache"], k[:, None, :]], axis=1)
    v_all = np.concatenate([I["v_cache"], v[:, None, :]], axis=1)
    s = np.einsum("hd,hnd->hn", q, k_all) / math.sqrt(hd)
    p = np.exp(s - s.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    o = np.einsum("hn,hnd->hd", p, v_all)
    attn = o.reshape(-1)
    h = I["x"] + attn @ I["wo"]
    h2 = rms(h, I["g2"])
    gate, up = h2 @ I["wgate"], h2 @ I["wup"]
    act = gate / (1.0 + np.exp(-gate)) * up
    out = h + act @ I["wdown"]
    return dict(h1=h1, qkv32=qkv, q=q, q16=q * (LOG2E / math.sqrt(hd)), k_new=k, v_new=v, o32=o, attn=attn, h=h, h2=h2, gate32=gate,
                up32=up, act=act, out32=out, out=out)


def error_report(got, want):
    """max abs error, max relative error (to max |want|) and the elementwise relative error."""
    got = np.asarray(got, dtype=F64).ravel()
    want = np.asarray(want, dtype=F64).ravel()
    diff = np.abs(got - want)
    scale = float(np.max(np.abs(want))) or 1.0
    elem = diff / np.maximum(np.abs(want), 1e-3 * scale)
    return dict(max_abs=float(diff.max()), max_rel_to_max=float(diff.max() / scale),
                max_rel_elem=float(elem.max()), mean_abs=float(diff.mean()), max_abs_want=scale)


# ---------------------------------------------------------------------------------------------------
# the plan: the decomposition into dispatches, and what covers each today

AVAILABLE, NEEDS_CLASS, MISSING = "AVAILABLE", "NEEDS-CLASS", "MISSING"
REGION_ALIGN = 256


def _dt(storage):
    return {"half": "f16", "bfloat": "bf16"}[storage]


def buffers(spec: LayerSpec):
    """The step's buffers and byte regions: weights (one buffer), the KV cache (per head, capacity
    rounded to whole 16-key blocks), and the activation scratch. {buffer: {region: (offset, bytes,
    dtype, shape)}}."""
    d, H, hd, f = spec.d_model, spec.n_heads, spec.head_dim, spec.ffn_dim
    s16 = _dt(spec.storage)
    cap = spec.key_blocks * KEY_BLOCK
    layout = {
        "weights": [("g1", s16, (d,)), ("wqkv", s16, (d, 3 * d)), ("wo", s16, (d, d)), ("g2", s16, (d,)),
                    ("wgate", s16, (d, f)), ("wup", s16, (d, f)), ("wdown", s16, (f, d)),
                    ("rope_cos", "f32", (hd // 2,)), ("rope_sin", "f32", (hd // 2,))],
        "kv": [("k_cache", s16, (H, cap, hd)), ("v_cache", s16, (H, cap, hd))],
        "act": [("x", s16, (d,)), ("h1", s16, (d,)), ("qkv32", "f32", (3 * d,)), ("q16", s16, (H, hd)),
                ("o32", "f32", (H, hd)), ("attn", s16, (d,)), ("h", "f32", (d,)), ("h2", s16, (d,)),
                ("gate32", "f32", (f,)), ("up32", "f32", (f,)), ("act", s16, (f,)),
                ("out32", "f32", (d,)), ("out", s16, (d,))],
    }
    size = {"f16": 2, "bf16": 2, "f32": 4}
    out = {}
    for buf, regions in layout.items():
        off, table = 0, {}
        for name, dt, shape in regions:
            nbytes = int(np.prod(shape)) * size[dt]
            table[name] = (off, nbytes, dt, shape)
            off += -(-nbytes // REGION_ALIGN) * REGION_ALIGN
        out[buf] = table
    return out


def plan(spec: LayerSpec = MILESTONE):
    """The decode step as 8 dispatches. Each op: its stages, GEMM extents (a decode row is padded to a
    16-row tile), buffer regions read and written, the existing class that covers it, its status
    (AVAILABLE / NEEDS-CLASS / MISSING), the lane that owns the missing class, what is missing, and
    how the host-orchestrated pipeline (tools/g17decodestep_gpu.py) runs it today."""
    d, H, hd, f, n = spec.d_model, spec.n_heads, spec.head_dim, spec.ffn_dim, spec.kv_len
    B = buffers(spec)
    def reg(*names):
        out = []
        for name in names:
            for buf, table in B.items():
                if name in table:
                    off, nbytes, dt, shape = table[name]
                    out.append(dict(buffer=buf, region=name, offset=off, bytes=nbytes, dtype=dt, shape=list(shape)))
        return out
    blocks_max = 8                  # runtime.ATTENTION_MAX_BLOCKS
    grid_blocks_max = 17            # runtime.ATTENTION_GRID_CAPACITY (phase grid, MM 25.135)
    grid_ok = hd == 128 and H in (1, 2, 4, 8, 16)          # runtime.ATTENTION_GRID_HEADS
    def route(N, K, blocks=1):
        grid_n, G, L = projection_route(N // blocks, K, spec.k_chunk)
        return dict(blocks=blocks, N=N // blocks, grid_n=grid_n, G=G, launches=L)
    def gemm_covered(r):
        return "N-tiled grid (25.134)" + (" + split-K, folded on the GPU (row 0; the residual on the host)" if r["G"] > 1 else "")
    def gemm_missing(N, K, r, extra=()):
        miss = []
        if r["G"] > 1:
            miss.append("the split-K fold runs on the GPU (tensorreduce.emit_split_k_fold at M_live 1, MM 25.132.2/4) "
                        "as its own dispatch after the %d-way partials; a fold that also adds the residual C, or a "
                        "fold fused into the GEMM, is not built (the residual stays on the host)" % r["G"])
        if r["blocks"] > 1:
            miss.append("N %d is not a power-of-two tile grid (the column offset is a shift): %d launches of N %d"
                        % (N, r["blocks"], r["N"]))
        if r["launches"] > 1:
            miss.append("K %d exceeds one launch's K %d (the K loop's 256 slices): %d launches over contiguous K "
                        "ranges whose partials join one fold" % (K, MAX_LAUNCH_K, r["launches"]))
        return miss + list(extra)
    def gemm_today(N, K, r):
        return ("gpu: %d launch(es) of gemm_generic 16x%dx%d half (the token in row 0), grid_n %d (%d columns per "
                "threadgroup) x split_k %d = %d threadgroups, K loop; %s"
                % (r["blocks"] * r["launches"], r["N"], K // r["launches"], r["grid_n"], r["N"] // r["grid_n"],
                   r["G"] // r["launches"], r["grid_n"] * r["G"] // r["launches"],
                   "the (G*16) x N partials folded on the GPU (row 0), a separate dispatch" if r["G"] > 1 else "fp32 out"))
    rq, ro, rg, rd = route(3 * d, d, 3), route(d, d), route(2 * f, d, 2), route(d, f)
    ops = [
        dict(op="attn_norm", stages=["attn_norm"], kind="row_norm", shape=dict(N=d),
             reads=reg("x", "g1"), writes=reg("h1"), status=AVAILABLE, lane="Set A (MM 25.136)",
             covered_by="tools/g17decodeops.py rmsnorm, half row (one simdgroup; rsqrt rounded once by the exact "
                        "midpoint test; bit-exact on hardware at d 2048, MM 25.136)",
             missing=[], today="gpu: 1 x decodeops rmsnorm (half row); the host stub is the fallback"),
        dict(op="qkv_proj", stages=["qkv_proj"], kind="gemm", shape=dict(M=16, N=3 * d, K=d), reads=reg("h1", "wqkv"),
             writes=reg("qkv32"), covered_by=gemm_covered(rq), status=NEEDS_CLASS,
             lane="Set C (split-K fold with the residual)", missing=gemm_missing(3 * d, d, rq),
             today=gemm_today(3 * d, d, rq), route=rq),
        dict(op="rope_append", stages=["rope_append"], kind="elementwise", shape=dict(heads=H, head_dim=hd),
             reads=reg("qkv32", "rope_cos", "rope_sin"), writes=reg("q16", "k_cache", "v_cache"),
             status=AVAILABLE if (hd % 64 == 0 and H % 4 == 0) else MISSING, lane="Set A (MM 25.136)",
             covered_by="tools/g17decodeops.py rope_append (4 threadgroups; the 1-row append at the runtime "
                        "length word into a P9-layout per-head cache; bit-exact on hardware at 16 x 128, MM 25.136)",
             missing=[] if (hd % 64 == 0 and H % 4 == 0) else
                     ["rope_append admits head_dim a multiple of 64 and heads a multiple of 4 (%d x %d)" % (H, hd)],
             today="gpu: 1 x decodeops rope_append; the host stub is the fallback"),
        dict(op="attention", stages=["attention"], kind="attention",
             shape=dict(heads=H, head_dim=hd, value=hd, keys=n + 1, key_blocks=spec.key_blocks, rows=1, q0=n),
             reads=reg("q16", "k_cache", "v_cache"), writes=reg("o32"),
             covered_by=("attention class phase grid (MM 25.135)" if grid_ok else
                         "attention class P7 phase attend / P9 phase step (MM 25.129, 25.131)"),
             status=AVAILABLE if (grid_ok or (hd == 64 and spec.key_blocks <= blocks_max)) else NEEDS_CLASS,
             lane="Set A (P7/P9, phase grid)",
             missing=(["one query row in a 32-row score tile: 31/32 of the QK and PV MMA work is padding (a cost, "
                       "not a gap)"] +
                      ([] if spec.key_blocks <= grid_blocks_max else
                       ["%d blocks: one dispatch holds %d; the rest runs as a chain of dispatches (resume), and the "
                        "counted key-block loop is not built" % (spec.key_blocks, grid_blocks_max)])
                      if grid_ok else
                      ([] if hd in (64, 128) else ["head %d: the class admits QK head 64, and 128 in phase grid" % hd]) +
                      (["%d heads: phase grid launches 1, 2, 4, 8 or 16" % H] if hd == 128 else
                       ["value %d: the head-64 phases admit value width 16 (a 16-column V slice per dispatch is exact)" % hd,
                        "one head per dispatch at head 64: the head grid is phase grid's (head 128)"]) +
                      ([] if spec.key_blocks <= blocks_max else
                       ["%d keys = %d blocks: head-64 straight-line code with immediate offsets admits 8"
                        % (n + 1, spec.key_blocks)])),
             today=("gpu: %d x attention phase grid (%d heads as %d threadgroups, value %d, one row, %s key "
                    "offsets), Q and the K/V cache written by the host, causal q0 = kv_len; O fp32 out, narrowed "
                    "on the host" % (-(-spec.key_blocks // grid_blocks_max), H, H, hd,
                                     "register" if spec.key_blocks > blocks_max else "immediate")
                    if grid_ok else
                    "gpu: %d x attention phase attend (one per head and 16-wide V slice), K/V cache and "
                    "Q written by the host, causal q0 = kv_len; O fp32 out, narrowed on the host" % (H * hd // 16)
                    if hd == 64 and spec.key_blocks <= blocks_max else
                    "not runnable on the GPU at this shape (head %d, %d heads, %d keys): the host stub" % (hd, H, n + 1))),
        dict(op="o_proj", stages=["o_proj"], kind="gemm", shape=dict(M=16, N=d, K=d),
             reads=reg("attn", "wo", "x"), writes=reg("h"), covered_by=gemm_covered(ro),
             status=NEEDS_CLASS, lane="Set C (split-K fold with the residual)",
             missing=gemm_missing(d, d, ro, ["the residual as the GEMM's accumulate add: gemm_generic admits "
                                             "accumulate for int8 only, and a split-K partial never accumulates "
                                             "(C belongs to the fold)"]),
             today=gemm_today(d, d, ro) + "; the residual is the fold's C, added last on the host", route=ro),
        dict(op="ffn_norm", stages=["ffn_norm"], kind="row_norm", shape=dict(N=d), reads=reg("h", "g2"),
             writes=reg("h2"), status=AVAILABLE, lane="Set A (MM 25.136)",
             covered_by="tools/g17decodeops.py rmsnorm, fp32 row (bit-exact on hardware at d 2048, MM 25.136)",
             missing=[], today="gpu: 1 x decodeops rmsnorm (fp32 row); the host stub is the fallback"),
        dict(op="ffn_gate_up", stages=["ffn_gate_up", "ffn_swiglu"], kind="gemm", shape=dict(M=16, N=2 * f, K=d),
             reads=reg("h2", "wgate", "wup"), writes=reg("act"), covered_by=gemm_covered(rg),
             status=NEEDS_CLASS, lane="Set C (split-K fold with the residual)",
             swiglu="tools/g17decodeops.py swiglu: a separate elementwise body after the GEMM, bit-exact on "
                    "hardware at 8192 (MM 25.136); a fused epilogue reading gate and up is not built",
             missing=gemm_missing(2 * f, d, rg, [
                 "silu(gate) * up as a GEMM epilogue: no epilogue step reads a second tile (the separate "
                 "decodeops swiglu body runs it instead, MM 25.136)"]),
             today=gemm_today(2 * f, d, rg) + "; SwiGLU as 1 x decodeops swiglu (host stub the fallback)", route=rg),
        dict(op="ffn_down", stages=["ffn_down"], kind="gemm", shape=dict(M=16, N=d, K=f),
             reads=reg("act", "wdown", "h"), writes=reg("out32", "out"), covered_by=gemm_covered(rd),
             status=NEEDS_CLASS, lane="Set C (split-K fold with the residual)",
             missing=gemm_missing(d, f, rd, ["the residual as the accumulate add (as o_proj)",
                                             "a half out after a K loop: narrow_out_half excludes the K loop"]),
             today=gemm_today(d, f, rd) + "; the residual (the fold's C) and the narrowing on the host", route=rd),
    ]
    for op in ops:
        if op["kind"] == "gemm":
            K, r = op["shape"]["K"], op.pop("route")
            op["split_k"] = dict(
                route="split_k" if r["G"] > 1 else "none", G=r["G"], slice_k=K // r["G"], grid_n=r["grid_n"],
                blocks=r["blocks"], launches=r["launches"],
                contract="gemm_reference(A, B, C, split_k=G): contiguous K slices, single-chain partials, "
                         "ascending fp32 left fold, C last",
                fold="host stub (tools/g17decodestep_gpu.py fold_split_k) pending Set C's reduce kernel"
                     if r["G"] > 1 else None,
                why=("a one-token K %d chain on %d threadgroups is K-chain-bound (2.9x its floor, Piece B "
                     "MM 25.124.6): split-K partitions K" % (K, r["grid_n"])) if r["G"] > 1 else
                    ("%d threadgroups already reach 1.2x the floor (Piece B MM 25.124.6): no split-K"
                     % r["grid_n"]))
    traffic = op_bytes(spec)
    p11 = p11_plan(spec)
    for i, op in enumerate(ops):
        op["dispatch"] = i
        op["bytes"] = traffic[op["op"]]
        op["floor_us"] = op["bytes"]["total"] / DRAM_READ_BYTES_PER_S * 1e6
        op["p11_stage"] = P11_STAGE[op["op"]]
        op["p11"] = None if p11 is None or op["p11_stage"] is None else p11["stages"][op["p11_stage"]]
    return ops


# THE PROJECTION ROUTE (tools/g17decodestep_gpu.py dispatches it; MM 25.132, 25.134)
TG_COLUMNS = 128                          # columns per threadgroup: 8 tiles, 25.134's receipted K=2048 width
CORES = 20                                # M5 Pro GPU cores (25.134: grid_n 16 leaves cores idle)
MAX_SPLIT_K = 8                           # G = K/256 at K 2048 (plan()'s slice), capped so a K 8192 grid stays 128
MAX_THREADGROUPS = 256                    # tlower's threadgroup index mask is eight bits
MAX_LAUNCH_K = 4096                       # PER-THREADGROUP K (the K loop's 256 slices; runtime.TensorSpec since
                                          # #200): a longer K / G is several launches joining ONE ascending fold
LONG_K = 2048                             # a projection with K above this takes split_k K / LONG_K (Set C's one-shot)


def projection_route(N, K, k_chunk=0):
    """The launch for one projection block of N columns contracting K: (grid_n, split_k G, launches).
    grid_n gives each threadgroup TG_COLUMNS columns (at least 2 threadgroups, so no single-threadgroup
    C[0,0] += 1 tail). split_k: a spec's k_chunk fixes it (G = K / k_chunk, so the whole-step reference
    agrees); otherwise G = min(8, K/256) where the column grid alone is below the core count (the 2048-wide
    projections: K-chain-bound at 2.9x their floor, Piece B 25.124.6) and 1 where it already fills the GPU
    (the FFN gate/up on 64 threadgroups: 1.2x its floor, no split-K needed); a K above LONG_K takes
    G = K / LONG_K in one dispatch (the down projection). A per-threadgroup K above MAX_LAUNCH_K is
    `launches` launches over contiguous K ranges, each with split_k G/launches, so the G partials are the
    same contiguous slices in the same ascending order as one launch would write."""
    if N % 16:
        raise ValueError("projection_route: N %d is not whole 16-wide tiles" % N)
    grid_n = max(2, N // TG_COLUMNS)
    tiles = N // (16 * grid_n)
    if grid_n & (grid_n - 1) or N % (16 * grid_n) or tiles & (tiles - 1):
        raise ValueError("refused: N %d does not split into a power-of-two grid of power-of-two tile columns "
                         "(the N-tiled grid's offset is a shift, MM 25.134); split it into blocks" % N)
    if k_chunk:
        G = K // k_chunk if k_chunk < K else 1
    elif grid_n >= CORES:
        G = 1
    elif K <= LONG_K:
        G = min(MAX_SPLIT_K, K // 256)
    else:
        # A LONG K (the FFN down projection, K 8192): ONE dispatch at split_k K / LONG_K, each threadgroup a
        # 2048-long K loop. Set C's one-shot down (#199/#200: N 2048 K 8192 split_k 4 grid_n 16, bit-exact,
        # 136-141 us idle, 1.14-1.18x the floor) replaced 25.132.1's two K-4096 launches at split_k 8.
        G = K // LONG_K
    # the K loop bounds each THREADGROUP's K (runtime.TensorSpec, #200): K / G above it is several launches
    launches = -(-(K // max(1, G)) // MAX_LAUNCH_K)
    G = max(1, G, launches)
    if K % (16 * G) or G % launches:
        raise ValueError("refused: split_k %d does not split K %d into whole 16-wide issues over %d launch(es)"
                         % (G, K, launches))
    if grid_n * (G // launches) > MAX_THREADGROUPS:
        raise ValueError("refused: grid_n %d x split_k %d = %d threadgroups > %d"
                         % (grid_n, G // launches, grid_n * G // launches, MAX_THREADGROUPS))
    return grid_n, G, launches



# this plan's dispatches -> Piece B's P11 decode stages (tensorsched.decode_step): the FFN is one P11
# stage and two dispatches here; rope_append has no P11 stage (its bytes are activations only)
P11_STAGE = {"attn_norm": "attn_norm", "qkv_proj": "qkv_proj", "rope_append": None, "attention": "attention",
             "o_proj": "o_proj", "ffn_norm": "ffn_norm", "ffn_gate_up": "ffn", "ffn_down": "ffn"}


def p11_plan(spec: LayerSpec, dtype=None):
    """Piece B's P11 decode plan (agxforge.g17.tensorsched.decode_step: per stage bytes, floor_ms,
    predicted_ms, at its bw_dram of 280 GB/s, weights and KV only) when that API is present on this
    checkout, else None (the plan then carries only this module's own byte count)."""
    try:
        from agxforge.g17 import tensorsched
        DecodeStep, decode_step = tensorsched.DecodeStep, tensorsched.decode_step
    except (ImportError, AttributeError):
        return None
    dt = dtype or {"half": "bf16", "bfloat": "bf16"}[spec.storage]     # P11 prices 16-bit weights as bf16
    dp = decode_step(DecodeStep(d=spec.d_model, heads=spec.n_heads, kv=spec.kv_len, hidden=spec.ffn_dim,
                                dtype=dt, kv_dtype="bf16"))
    return {"dtype": dt, "floor_ms": dp.floor_ms, "unpriced": list(dp.unpriced),
            "stages": {st.name: dict(bytes=st.bytes, floor_ms=st.floor_ms, predicted_ms=st.predicted_ms,
                                     note=st.note) for st in dp.stages}}


# MM 25.121: DRAM read 269.6-275.7 GB/s at a known clock; the floor of a dispatch is its bytes at 270
DRAM_READ_BYTES_PER_S = 270.0e9


def op_bytes(spec: LayerSpec, weight_bytes=None):
    """Per dispatch, the bytes it must move: weights, KV cache and activations, read and written
    (compulsory traffic, each byte once). `weight_bytes` per weight element defaults to the storage
    size (2; 1 prices an fp8 weight). The attention reads the kv_len + 1 live keys, not the padding."""
    d, H, hd, f, n = spec.d_model, spec.n_heads, spec.head_dim, spec.ffn_dim, spec.kv_len
    s, wb, f32 = 2, (2 if weight_bytes is None else weight_bytes), 4
    kv_row = H * hd * s                       # one token's K (or V) over every head
    rows = {
        "attn_norm": dict(weights=s * d, kv=0, act_read=s * d, act_write=s * d),
        "qkv_proj": dict(weights=wb * 3 * d * d, kv=0, act_read=s * d, act_write=f32 * 3 * d),
        "rope_append": dict(weights=f32 * hd, kv=2 * kv_row, act_read=f32 * 3 * d, act_write=s * d),
        "attention": dict(weights=0, kv=2 * (n + 1) * kv_row, act_read=s * d, act_write=f32 * d),
        "o_proj": dict(weights=wb * d * d, kv=0, act_read=s * d + s * d, act_write=f32 * d),
        "ffn_norm": dict(weights=s * d, kv=0, act_read=f32 * d, act_write=s * d),
        "ffn_gate_up": dict(weights=wb * 2 * d * f, kv=0, act_read=s * d, act_write=s * f),
        "ffn_down": dict(weights=wb * f * d, kv=0, act_read=s * f + f32 * d, act_write=s * d),
    }
    for r in rows.values():
        r["total"] = r["weights"] + r["kv"] + r["act_read"] + r["act_write"]
    return rows


def step_floor(spec: LayerSpec, weight_bytes=None):
    """The whole step's compulsory bytes and its floor at DRAM_READ_BYTES_PER_S (ms)."""
    rows = op_bytes(spec, weight_bytes)
    tot = {k: sum(r[k] for r in rows.values()) for k in ("weights", "kv", "act_read", "act_write", "total")}
    tot["floor_ms"] = tot["total"] / DRAM_READ_BYTES_PER_S * 1e3
    return tot


def blocker_table(spec: LayerSpec = MILESTONE):
    rows = ["| # | op | MB | floor us | status | lane | missing |",
            "| ---: | --- | ---: | ---: | --- | --- | --- |"]
    for op in plan(spec):
        rows.append("| %d | %s | %.2f | %.1f | %s | %s | %s |" % (
            op["dispatch"], op["op"], op["bytes"]["total"] / 1e6, op["floor_us"], op["status"],
            op["lane"] or "unowned", "; ".join(op["missing"]) or "-"))
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", choices=sorted(SPECS), default="milestone")
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--plan", action="store_true", help="print the plan as JSON and the blocker table")
    args = ap.parse_args(argv)
    spec = SPECS[args.spec]
    if args.plan:
        print(json.dumps(plan(spec), indent=1))
        print(blocker_table(spec))
        print(json.dumps({"storage_weights": step_floor(spec), "fp8_weights": step_floor(spec, 1),
                          "p11": p11_plan(spec), "p11_fp8": p11_plan(spec, "fp8")}, indent=1))
        return 0
    inputs = make_inputs(spec, args.seed)
    trace = reference(spec, inputs)
    ide = ideal(spec, inputs)
    report = {"spec": spec.as_dict(), "seed": args.seed,
              "reference_vs_ideal": {k: error_report(trace["env"][k], ide[k])
                                     for k in ("h1", "qkv32", "o32", "attn", "h", "h2", "act", "out32", "out")}}
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
