# M5 Neural Accelerator through Metal 4 TensorOps: reconnaissance batch 1

Campaign assigned directly by Spencer (2026-09-16): use Metal 4 TensorOps as a controlled
compiler-differential generator and determine whether the recovered G17 tensor instructions are
the instruction interface to the M5 Neural Accelerator, then work down the list (operand routing,
fragment representation, accumulator semantics, datatypes, synchronisation, partial tiles,
multi-SIMD-group, postfix ops, exact accumulation, configuration state). This document records
batch 1: the software layering, the single-instruction differential, and the first hardware
measurements.

**A subject-organized synthesis of this entire record is `docs/g17-tensorops-machine-model.md`.** It states
the current machine model once, with the superseded claims of this document listed as such; where the two
disagree, the synthesis is the current state and this document is the provenance. Everything is under `results/g17-tensorops-recon-v1/` (ignored; `manifest.json`
pins every script, source, object and record by hash), compiled by Apple's toolchain (Xcode 27.0, metal 32023.921) and run on
this machine's H17s through `spike/accel/libaccel.dylib` with single-simdgroup dispatches
(32 threads, one threadgroup). No production code was touched.

## 1. Where TensorOps lives in the toolchain (read from the SDK, not inferred)

- `mpp::tensor_ops::matmul2d<descriptor, execution_simdgroups<N>>::run` is header code
  (`MetalPerformancePrimitives.framework/Headers/__impl/MPPTensorOpsMatMul2dImpl.h`, 16,754 lines)
  that calls `extern "C"` functions in section `air.externally_defined`, named
  `__tensorops_impl_matmul2d_op_run_<left>_<right>_<dest>[_v2]` (dv/tg/cooperative operand kinds).
  So the AIR of a TensorOps kernel carries a CALL, not the matmul.
- Those functions are resolved at native compile time from
  `/System/Library/Frameworks/MetalPerformancePrimitives.framework/Resources/libTensorOps.rtlib`,
  an `ar` archive (50 MB, 1,970 members) of air64_v28 bitcode: 1,475 `matmul2d_op_run_*`
  members plus `__loweringlib.internal.N` helpers. The toolchain's `applegpu25-nt/tensor.metallib`
  is only the tensor-descriptor runtime (`air.get_extent_*`, `agx.generate_emask_*_tensor`, ...).
- Inside a member (`__tensorops_impl_matmul2d_op_run_dv_f16_dv_f16_dv_f32`, disassembled with
  `metal-opt -S -disable-verify`): `air.supports_family(i32 1010)` selects
  `air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f16.v8f16.v8f32(<8 x half> A,
  i1 transA, <8 x half> B, i1 transB, <8 x float> C) -> <8 x float>`; the `supports_family(1009)`
  branches use the legacy `air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f16.v64f16.v64f32`.
  1010 is `MTLGPUFamilyApple10` (M5 / A19 Pro); 1009 is Apple9.
- The full intrinsic surface of the new path, from the rtlib's strings: the float family
  `16x16x16_multiply_accumulate.f.f.v8f32.{v8f16|v8bf16|v8f32}.{v8f16|v8bf16|v8f32}.v8f32`
  (nine A/B combinations, fp32 accumulate only) and the integer family
  `16x16x16_widening_multiply_accumulate.{s|u}.{s|u}.v8i32.v8i8.v8i8.v8i32`. Every other
  TensorOps datatype (int4/int2/fp8/fp4, bfloat/half destinations) is converted in AIR around
  these. `metal-opt` refuses to verify the 16x16x16 intrinsic, but the driver compiles it.

So the accelerator's compiler-level interface is one AIR intrinsic with per-thread 8-element
fragments; the 17-instruction `op.run` expansion measured on 2026-09-02 is the rtlib's tile
loop around it.

## 2. Single-instruction differential: hand-written AIR, Apple's native compiler

`gen.py` writes AIR modules from Apple's own template (`template.metal` -> `metal -S -emit-llvm`)
whose body loads one 8-element fragment per lane for A, B and C, calls the intrinsic once (or
chained), and stores the result. `build.py` compiles each through `metal-as -> metallib ->
libaccel ac_archive (driver native compile) -> metal-lipo -> metal-source` and decodes the object
with `agxforge.g17.model.decode`. Sixteen variants compiled; the whole kernel is ~160 bytes and the
matrix instruction is isolated between ordinary loads and stores:

| kernel | matrix instruction | bytes |
| --- | --- | --- |
| new f16.f16 nn | op5106 sched 172 `2f 00 25 12 22 02 a4 42 00 04` | 10 |
| new f16.f16 nt / tn / tt | same, byte 8 = `08` / `04` / `0c` | 10 |
| new bf16.f16 nn | byte 6 `a4 -> a0` | 10 |
| new f16.bf16 nn | byte 7 `42 -> 02` | 10 |
| new bf16.bf16 nn | bytes 6 and 7 both | 10 |
| new f16.f32 nn | **op5104** `2f 00 25 01 22 82 a4 02 00 04` | 10 |
| new f32.f32 nn | **op5098** `2f 00 25 01 22 82 a0 02 10 04` | 10 |
| new i8.i8 (s.s / u.u / s.u) | **op10384 sched 170** `2f 00 25 0a 22 0a a1 02 40 04` | 10 |
| new f16.f16 x2 (chained) | first issue `2f 00 05 12 2a 00 a4 42 00 04`, last as single with byte 0 `27` | 10 |
| legacy 8x8 f16.f16 | **op2862 sched 118** `2f 02 05 ea 20 20 af 02 28 08 14 00 00 01` | 14 |

Findings from the encodings:

- **The recovered `tensor.mac` (op5106, `isa/tensor-isa.toml`) IS the native form of the
  accelerator intrinsic.** Its signature (`byte4 & 0xf7 == 0x22`, `byte6 & 0xfb == 0xa0`,
  `byte7 & 0x1f == 0x02`, `byte8 == 0x00`) and the two dtype bits already recovered causally
  (byte6[2] = A is half, byte7[6] = B is half) reproduce exactly here from the intrinsic's type
  suffixes. The legacy `simdgroup_matrix` path is a different instruction (op2862, 14 bytes,
  scheduling class 118) with no shared bytes. Two matrix families exist on this chip and the
  repository's tensor ISA is the Neural Accelerator one.
- **Transpose flags are bits of the same instruction**: byte8[2] = transA, byte8[3] = transB.
  The ISA file's `byte8 == 0x00` signature and "inert byte8[2]" note reflect that matmul2d never
  set them in the measured shapes; they are not inert.
- **Operand widths select the opcode**: fp32 operands are a different MCInst opcode (5104 for
  A16.B32, 5098 for A32.B32) with byte5[7] set when B is fp32 and byte8[4] set when A is fp32;
  int8 is op10384 in scheduling class 170 with byte6 = `a1` and byte8[6] set. The 10-byte shape
  and byte4 = `22` are common to all of them.
- **Chained issues differ from a writeback issue**: an MMA whose result feeds the next MMA
  carries byte2 `25 -> 05`, byte4 `22 -> 2a`, byte5 `02 -> 00`; the last issue has the single
  kernel's bytes; byte0 `2f` on the first issue of a sequence and `27` on later ones. These are
  the bytes the 2026-09-04 census called coupled routing/`acc` fields; here they are the
  compiler's register-vs-accumulator routing, measured semantically in section 4.

## 3. Hardware: fragment layout (P1-P5 preregistered, 6,651 dispatches, 0 findings)

`layout.py` one-hot probes on the f16.f16 nn kernel. All five predictions held: zero operand
preserves C; an A one-hot lights exactly one 16-slot row class; a B one-hot one column class; row
and column classes intersect in single slots (a bijection onto 16x16); the reduction index pairs
A columns with B rows in 16 classes of 16. The measured maps (`layout-new_f16f16_nn.json`) are
closed-form, lane `l` (0..31), slot `j` (0..7):

    A (transA = 0) and D:   row = 8*(l>>4) + 2*((l>>1)&3) + (j>>2)      col = 8*((l>>3)&1) + 4*(l&1) + (j&3)
    B (either flag):        k   = 4*(l>>4) + ((l>>1)&3) + 8*(j>>2)      col = 8*((l>>3)&1) + 4*(l&1) + (j&3)
    A (transA = 1):         k   = 4*(l>>4) + ((l>>1)&3) + 8*(j>>2)      row = 2*(j&3) + 8*(l&1) + ((l>>3)&1)

A lane holds a 2x4 block of A/D (rows 2m, 2m+1; four consecutive columns); B's lane holds rows
k and k+8 of four consecutive columns; the transposed-A register fragment holds A^T in yet another
arrangement (`alayout-new_f16f16_tn.json`). With these conventions all four flag combinations
compute exactly `D = C + A.B` on random integer matrices (`transpose-*.json`, 4 trials each): the
flags change how the register fragment is READ, not a mathematical transpose of the nn layout
(`tnprobe-*.json` shows the nn/nt readings with the two candidate layouts).

## 4. Hardware: exact accumulation arithmetic of one MMA (f16 x f16 + f32)

Products of two f16 values are exact in fp32, so stimuli on D[0][0] (row 0 of A, column 0 of B,
C[0][0]) isolate the reduction order and rounding. `arith.py` (13 preregistered stimuli against 36
hypotheses in RNE and RZ), `tree.py` (1,680 triples: 1 at i, eps at j, eps at k, all orderings),
`cplacement.py` (C placement), `chain.py` (x2 and x3 chained kernels, 17 stimuli each):

- RZ is rejected on every stimulus. RNE at every intermediate is consistent with everything.
- The reduction is NOT a balanced tree and NOT sequential. Fitted on 1,680/1,680 triples and all
  stimuli, with k the A-column index of section 3:

      p_i = RNE32(a[2i]*b[2i] + a[2i+1]*b[2i+1])        i = 0..7   (adjacent pairs)
      q_j = RNE32(p_j + p_{j+4})                         j = 0..3   (pairs interleaved by 8)
      acc = C;  acc = RNE32(acc + q_0); RNE32(acc + q_1); RNE32(acc + q_2); RNE32(acc + q_3)
      D   = acc

  C enters FIRST (four stimuli with 3-eps quads discriminate 1+8eps from 1+6eps; measured 1+8eps).
  A balanced tree over natural order fits 976/1,680; interleave-by-8 balanced 1,392; the fitted
  topology 1,680.
- Whole tile: on 20 random f16/fp32 inputs chosen so that pair and quad sums round, the model
  matches all 5,120 output elements exactly (`fulltile.json`); sequential (2,493), balanced tree
  (2,764) and exact-once (2,742) do not. The structure is uniform across the tile.
- Chaining: the x2 kernel returns the model applied twice through the rounded fp32 accumulator
  (17/17 stimuli), x3 three times (17/17); an unrounded/wide accumulator between issues is
  rejected (8/17 and 12/17). So a K=64 matmul2d is four such MMAs composed through fp32, in
  whatever K order the rtlib loop feeds them. This is the exact fractional accumulation rule the
  tensor campaign lacked, at the instruction level; composing it with the tile-load order of a
  given matmul2d shape is the remaining step for that gap.

## 5. Batch 2: sibling opcodes, the legacy datapath, and the library level

(Section 33 corrects one statement below: fp32 operand subnormals are NOT flushed; the low 13 bits of
the pattern are cleared, so a subnormal keeps whatever lies above mantissa bit 12.)

Correction first: batch 1's fp32 and int8 hand-written kernels indexed the fragment buffer as if
every operand were 16-bit (a `half` GEP), so their arithmetic runs were invalid; `gen.py` now
addresses fragments by vector element type, the f16/bf16 kernels and every encoding above are
unchanged, and the f32/int8 layouts re-measured below (6,651 dispatches each) are the SAME as
f16's for A, B and D.

- **fp32 operands are truncated to 10 mantissa bits** (op5098 f32.f32 and op5104 f16.f32,
  `precision-*.json`): `1 + 2^-m` survives for m <= 10 and collapses to 1 for m = 11..23, the
  dropped bits are truncated toward zero (`-(1 + 3*2^-12) -> -1`, `2 - 2^-23 -> 2 - 2^-10`),
  the fp32 exponent range is kept (`2^100 * 2^-100 = 1` exactly, `1 * 2^100 = 2^100`), and fp32
  subnormals flush to zero. So the accelerator's "fp32" is a tf32-like operand (8-bit exponent,
  10-bit truncated significand); products of two such operands are exact in fp32 and the K=16
  reduction then follows the same pair / interleave-8 / sequential structure as f16.
- bf16 x bf16 (op5106 with both dtype bits clear): products exact, same structure, same
  results as f16 on every stimulus (`arith2-new_bf16bf16_nn.json`).
- int8 (op10384): exact 16-term sums, `C + 2^31 -> -2^31` (**wraps**, no saturation), signed and
  unsigned variants agree with exact integer arithmetic (`arith2-new_i8i8_*.json`).
- **The legacy simdgroup_matrix MMA (op2862, via MSL `simdgroup_multiply_accumulate`, `legacy.py`,
  168 eps-triples + 6 stimuli) is a plain SEQUENTIAL fp32 chain**: `acc = C; for k in 0..7: acc =
  RNE32(acc + a_k*b_k)` (168/168; pairs-first fits 144, balanced tree 112). Two structurally
  different reductions - a chain of eight FMAs against a 2-2-4 dot-product tree - is the strongest
  evidence so far that op2862 and op5106 execute on different datapaths, not one unit with two
  encodings.
- **Zero-accumulator siblings**: `air.simdgroup_matrix_16x16x16_multiply_accumulate` with a
  constant-zero C lowers to **op5107** (sched 173; `2f 00 05 10 22 00 a4 42 02 00`), which ignores
  the C register entirely (P1 fails, P2-P5 and all layouts identical to op5106); the fp32-operand
  form is op5099. So the family is {5106/5107, 5104/5105, 5098/5099} x {with, without C}, plus
  op10384 for int8.

Library level (`mm_matrix.py`, 30 compile-only matmul2d descriptors, `mm_matrix.json`):

| descriptor change (base M32 N32 K64 half->float, 1 SG) | MMAs | note |
| --- | --- | --- |
| base / relaxed / k128 / bf16 / a_bf16 / sg_single | 16 x op5106 | 2 x 2 x 4 sub-tiles; K=128 loops the same body |
| m64 / n64 | 32 | m16 / n16: 8; partial n40: 24 (padded to 48), m24: 16, k48: 12 |
| k16 | 4 x **op5107** | no accumulate chain; `multiply_accumulate` at K=16 adds C with ALU adds |
| ta / tb / tatb | 4 x op5107, 45 instructions | a different (looping) schedule, not flag bits on the 16 MMAs |
| **f32 (exact precision)** | **128 x op2842 (legacy family, sched 118)** | the library does NOT use the accelerator for fp32 |
| f32 + relaxed_precision | 16 x op5098 | relaxed = the tf32-like accelerator path |
| i8 / u8 | 16 x op10384 | tiles through ordinary loads, no op12674 |
| sg2 / sg4 | 16 x op5106, 2 SR reads, op11456 x2 + op17258 x4 in sg4 | SIMD-group coordination instructions, unmeasured |
| coop / coop_relu | 16 x op5106 | cooperative destination costs nothing extra; the ReLU is ALU on the fragment |
| coop_sg4 | 4 x op5106 per SG | the tile is divided across the four SIMD groups |

- **Cooperative-tensor coordinates**: Apple's `get_multidimensional_index` for a 16x16 single-SG
  destination (`cooplayout.json`, capacity 8) equals the measured **B** fragment layout with index
  0 = column and index 1 = row: a lane owns rows k and k+8 of four consecutive columns. Relative to
  the intrinsic's D layout (rows 2m, 2m+1) the row bits are rotated (`y = ((r&7)<<1) | (r>>3)`);
  the library's tile loads absorb the permutation, so memory results are correct and the
  register-level layout is the intrinsic's.
- **matmul2d 16x16x16 in memory coordinates** (`mm16_tree.py`, 1,680 triples): the reduction over
  memory k is exactly the intrinsic's structure with the identity map k_memory = k_intrinsic
  (1,680/1,680; the row rotation touches M only).
- **matmul2d K=64 in memory coordinates** (`mm64_chain.py`, 8 stimuli on the 32x32x64 base
  kernel): D[0][0] equals the intrinsic model applied to the four 16-wide K chunks in ascending
  memory order, chained through the rounded fp32 accumulator, starting from 0 (mode multiply);
  chunk orders 3210 / 0213 / 1032 and single-rounding are each rejected by at least one stimulus.
  This is the complete exact-arithmetic rule for half x half -> float matmul2d at this shape.

## 6. Batch 3: SIMD-group cooperation, partial tiles, quantized formats

- **execution_simdgroups<2> / <4>, with and without a cooperative destination** (`sg_compare.py`,
  64/128-thread single threadgroups): the 32x32 result region is bit-identical to the
  single-SIMD-group kernel on three random half inputs (same region hashes for base, sg2, sg4,
  coop, coop_sg4), nothing outside the region is written. Dividing a tile across SIMD groups
  changes which lanes issue which MMAs (4 per SG in `coop_sg4`), not the per-element arithmetic.
  The sg4 code adds op11456 (sched 282, one def; unnamed) and op17258 (a store form); their roles
  are not measured.
- **Partial tiles** (`partial.py`: M=24, N=40, K=48 against 32/48/64-padded instruction
  schedules): only the declared MxN region is written (sentinel elsewhere intact); poisoned
  padding rows/columns (value 1000 beyond K, M, N) never leak into the result (max error vs float64
  at fp32 level); the eps-chain stimulus for K=48 equals the intrinsic model over three ascending
  16-chunks. Masking is done by the library's masked tile loads/stores, not by the MMA.
- **Quantized operands**: int4 / uint4 B (`mm_quant.json`) still emit 16 x op5106; the nibbles are
  unpacked and converted by ordinary ALU instructions (op11183 x64, shifts op435/437, converters
  op776/799/801 or op862/864) before the MMA; int8 x int4 emits op10384 with its own unpack
  sequence. There is no quantized-operand MMA form on this chip; the accelerator consumes 16-bit
  or int8 fragments. fp8 (e4m3) and fp4 (e2m1) formats compile only under `-std=metal4.1`, and
  this OS (macOS 26.6.2) refuses to load a 4.1 library ("language version 4.1 which is not
  supported on this OS"); int2 formats are not accepted by the tensor_inline constructor. Bounded
  negatives, not capability claims. *(Section 137: the language gate is confirmed and joined by three more blocks of the API path; fp8 e4m3 and e5m2 operands are nevertheless reachable on this OS from hand-written AIR, unpacked by op17642 and multiplied by a bf16 op5106; fp4 is not.)*

## 7. Batch 4: K-loop order, half destinations, and the instruction's operand structure

- **K=128 (runtime loop over 64-chunks)**: seven asymmetric chunk probes (3eps, 3eps, 1 at chunks
  a<b<c) all give 1+6eps, the ascending-order value; descending would give 1+8eps
  (`k128chain.json`). The 127-eps stimulus equals the ascending chain. So the memory-K order is
  ascending throughout, loop or unrolled.
- **Half destination** (`mm_half_out`, `halfout.json`): the fp32 accumulator is rounded to half
  once at the store with round-to-nearest-even (ties to even in both directions, `65520 -> inf`,
  `2^-25 -> 0`); RZ is rejected.
- **Operand structure of the MMA, from Apple's own instruction tables** (`model.decode` values,
  which the earlier bit census could not see): op5106 is
  `D:GPR32tup8, imm, imm, A:GPR32tup4, imm, imm, B:GPR32tup4, imm, imm, C:GPR32tup8, imm, imm`
  and op5107 the same without the C triple. In every kernel here D and C are the same 8-register
  tuple (in-place accumulate: R0-R7), A and B are 4-register tuples (R12-R15, R8-R11), so one
  16x16 f16 fragment costs 4 registers per lane and the fp32 accumulator 8. The per-operand
  immediates behave as scheduling flags rather than routing: the imm after A or B is 16 exactly
  when that register tuple is not read again (last use) and 0 otherwise, across x2, independent,
  store-and-chain and B-from-result variants; the first imm after D has bit 31 set on the first
  MMA of a kernel and bit 24 on a second MMA that follows an independent first one (observed
  correlations, not decoded semantics). The 2026-09-04 ISA file's `acc` one-hot (byte 1) and
  "coupled routing codeword" are these immediates and register numbers, which is why single-bit
  flips looked coupled.
- **Tensor tile loads/stores** (`mm16_mul`, Apple tables): op12674 `dst:GPR32tup2, imm, imm,
  base:GPR32tup2(reloc expr), imm, index:GPR32, imm, offset imm, elem imm, mask imm` loads half a
  16-bit fragment (two loads per operand, offsets 0 / 256), trailing (2, 15) = element code and a
  static mask of 15; op17257 stores a 4-register tuple with (4, 15). The register-mask forms
  op12675 / op17258 carry a `GPR16` mask operand and appear exactly in the partial-tile variants
  (M=24, N=40) and for the runtime-bounded loads, i.e. `agx.generate_emask` feeds them. Static
  full tiles use the immediate form.

## 8. Batch 5: what metadata/resource state marks accelerator use

`metadata_slot44.json` (57 objects, every kernel built in this campaign, read with the repository's
`gpumd`/`mdgen.describe`):

- **Per-kernel slot 44 is present exactly when the code contains an accelerator MMA**
  (op5106/5107/5104/5098/5099/10384): all 50 such kernels carry it; the plain template, the two
  hand-written legacy op2862 kernels, the MSL `simdgroup_matrix` kernel and the exact-fp32
  `matmul2d` (op2842) do not. Nothing else in the section, the binding records or the LD/ARCH
  sections distinguishes an accelerator kernel from a plain one (the MSL legacy kernel alone has a
  40-byte `__GPU_ARCH_LD_MD` instead of 32).
- Its value is `0xRR010101` with RR equal to the slot-0 register count in every kernel with a
  register count of 28 or less (eleven kernels: 9, 13, 17, 18, 20, 21, 25, 26, 28) and 0x01 in
  every kernel with 32 or more; byte 2 is 0x03 exactly in the kernels that also carry slot 33
  (the looping library kernels) and 0x01 otherwise. The two constant low bytes are 0x0101. This
  is an observation over 50 objects, not a decoded field. The repository's measured `TENSOR`
  class (`mdgen.py` 54-58) already carries slot 44 as a one-byte field with value 1, so authored
  tensor images do set it; the scalar/buffer classes do not, and the four-byte form seen here in
  driver-compiled objects is the same low byte followed by the inert bytes measured below.
- **Mutation (causal, `mutate44.py`, archives patched in place, 14 pipelines)**: with slot 44's
  low byte cleared (`11010100`, `00000000`, and `01030100` on the 32x32x64 library kernel) every
  accelerator MMA produces ZEROS - the pipeline builds, the dispatch completes without error, the
  D tile is all zero (single, x3-chained and library kernels alike). With the low byte nonzero
  and everything else changed (`00000001`, `11010001`, `11010102`, `11010103`, `11010180`,
  `11010111`, `12010101`, `01030101`, `ffffffff`, `ff030101`) the results are bit-identical to
  the original. So slot 44's low byte is the per-kernel accelerator ENABLE - the metadata state
  that selects the tensor unit - and a kernel lacking it would execute its tensor.mac
  instructions as silent no-ops. The other three bytes are inert for these
  shapes (register counts 17 and 72); whether they matter for heavier register use is untested.
- Slot 32's low byte equals the register count in the same objects, and slot 33 (`0x01010301`)
  appears only in the looping library kernels; both are outside this campaign's scope.

## 9. Batch 6: the legacy fp32 datapath and bfloat destinations

- **Legacy fp32 `simdgroup_float8x8` MMA (op2842, `legacy-old_msl_8x8_f32.json`)**: a sequential
  FUSED multiply-add chain in k order with C first at full fp32 precision - `1 + 2^-23` survives,
  `(1+u)(1+u) - (1+2u)` returns the `2^-46` residual (fused, not separately rounded), 168/168
  eps-triples sequential. So the two families differ on every axis: the legacy op2842/op2862 is an
  FMA chain on the SIMD ALU at full precision; the accelerator op5106 family is a 2-2-4 reduction
  tree on 10-bit-truncated operands. That is why the library keeps exact fp32 on the legacy path.
- **bfloat destination** (`mm_bf16_out`, `bf16out.json`): the fp32 accumulator is rounded to
  bfloat once at the store with round-to-nearest-even (five tie/above/below stimuli; RZ rejected),
  the same rule as the half destination.

## 10. Batch 7: synchronisation, and a census of the MMA's flag immediates

- **No explicit wait instruction exists between an MMA and its consumer.** In every kernel here
  the store or ALU instruction that reads the D tuple directly follows the MMA (`dep_alu_consumer`:
  eight op1006 adds on R0-R7 immediately after op5106; the single kernel: the two stores), and
  every hardware result was correct, so from the program's point of view the instruction is
  synchronous: dependency tracking is carried in the instruction's own immediates, not by a
  separate wait/barrier op. Multi-SIMD-group cooperation adds op11456/op17258 but no barrier-class
  opcode either.
- **Census of the immediates over all 512 MMA instances in the 57 objects** (`mma_flag_census.json`,
  Apple-decoded operands): the first immediate after D has a ONE-HOT top byte (values 01, 02, 04,
  ..., 80 or 00) and a low byte of 00 or 20; the first MMA of a kernel carries 80 in 51 of 55
  kernels. It is the field the 2026-09-04 census called `acc` (byte 1, one-hot); with eight
  one-hot values assigned in sequence it reads as an issue-slot tag for the accelerator's
  in-flight results rather than a routing field, but its rule was not fitted here. The second
  immediate after D is 1 or 9. The immediates after A and B are 16 or 0 and, in the dependency
  variants, 16 exactly when that source tuple is not read again (154 / 130 / 116 / 112 instances
  of the four combinations). D and C are the same tuple in 481 of 481 with-C instances (in-place
  accumulation is the only form the compiler emits).

## 11. Batch 8: masks, the multi-SIMD-group opcode, slot 44 under heavy register use, the 0x20 flag

- **op11456 is the mask generator for masked tensor stores/loads.** It appears in exactly the
  kernels that use the register-masked tensor store op17258 (partial M=24: 2 / 6, partial N=40:
  3 / 12, sg2 and sg4: 2 / 4) and in none of the others; its operands (Apple tables) are
  `dst:GPR16, imm, imm, src:GPR32, imm, imm, mask:GPR16, imm, imm`, consuming the GPR16 mask that
  the register-masked tensor load op12675 also reads (R41L in `mm_sg4`) and a bound register, and
  producing the GPR16 that the following masked accesses use. In the multi-SIMD-group kernels the
  second system-register read is `SR_SIMD_GRP`, so per-SIMD-group sub-tile ownership is realised
  as per-SG store masks, not as separate code paths - consistent with the bit-identical results.
  This is the native form of `agx.generate_emask`; its immediates are not decoded.
- **The MMA's low flag byte 0x20 means "more accumulation follows into this tuple"**: it is set on
  every MMA whose D tuple is accumulated into again by a later MMA and clear on the last MMA into
  each accumulator (x3: 20, 20, 00; the library's K-step pairs: 20 then 00; `coop_sg4`'s four
  issues into one tuple: 20, 20, 20, 00; the two independent MMAs: 00, 00). The one-hot top byte is
  assigned in issue order within a phase (80 after a store/phase boundary, then 01, 02, 04, ...)
  with zeros on issues whose result is consumed only by the immediately following MMA; a full
  rule is not fitted.
- **Slot 44 at register count 116** (`mm_m64`): `00000001` and `01010101` are bit-identical to the
  original `01030101`; `00000100` (low byte clear) yields zeros. The inert bytes stay inert under
  the heaviest register use in this campaign.

## 12. Batch 9: tensor load/store addressing (Apple-decoded operands, `addr_variants.json`)

Eight 16x16x16 `matmul2d` kernels varying only tensor base offsets and strides, operands decoded
with Apple's tables:

- op12674 / op17257 operand list: `dst, mode imm, flag imm, base (relocation expr = the tensor's
  buffer), imm, index:GPR32, last-use imm (16 on the final access through that index register),
  BYTE OFFSET imm, element code (2 = 16-bit, 4 = 32-bit), mask (15 = full)`.
- The byte-offset immediate is exactly the sub-tile's byte offset: A + 16 halves moves the two A
  loads from (0, 256) to (32, 288); A + 32 halves to (64, 320); C + 16 floats moves the stores from
  (0, 512) to (64, 576). The second load/store of each operand sits at 8 x row stride bytes (256
  for a stride-16 half tile, 512 at stride 32, 1024 at stride 64; 1024 for a stride-32 float C):
  each tensor access moves an 8-row x 16-column sub-tile (4 elements per lane, two registers for
  16-bit, four for 32-bit), rows 0-7 then 8-15, which is why the register layout puts memory rows
  r and r+8 in the two slot halves (the "B layout" of section 3).
- A non-default stride changes the mode immediate from 241 (0xF1) to 2289 (0x8F1), switches the
  per-lane index register to a computed one (R0/R2 instead of the shared R8) and adds ALU code
  (270 -> 318 bytes): the row stride is not an instruction field; it is folded into the per-lane
  index register, and bit 11 of the mode word marks that form.
- Convolution2d was not re-examined: the repository's `g17-convolution-is-the-same-primitives`
  ledger (2026-09-05) already found eleven conv kernels lowering to op5106/5107, op12674/12675 and
  op17257/17258 with op5107 as the initialising MAC, consistent with everything measured here.

## 13. Batch 10: the wait bit, and the tag words as a dependency mechanism

- The MMA's byte0 `2f` vs `27` difference (section 2) is byte0[3], which the repository's
  `g17-alu-load-use-wait` ledger established causally as the ALU load-use wait bit: it is set on
  the first MMA after the fragment loads and clear on MMAs whose operands are already resident,
  exactly as in the scalar ISA. No new mechanism.
- The packed first immediates form a producer/consumer pairing visible in `dep_independent2`:
  the four loads feeding the first MMA carry `0x800000` (bit 23) and the last of them
  `0x2000800000` (bit 37 added), the first MMA carries tag bit 31; the two loads feeding the
  second MMA carry `0x100000` (bit 20) and `0x2000100000`, the second MMA carries tag bit 24; the
  stores carry `0x10` and the tensor stores `0x40000000010` (bit 34 added). The exact bit-to-tag
  correspondence and the repository's byte6[4:3] rotating counter on tensor loads
  (`g17-load-counter-is-not-a-scoreboard-slot`) are both open; the observation is that every
  load group, MMA and store group carries a distinct one-hot word, consistent with the hardware
  tracking accelerator results by tag rather than by an explicit wait instruction.

## 14. Batch 11: the tag byte is an eight-slot result ring

**RETRACTED in part - see section 26.** bit 31 of the flags word is byte0[3], the repository's load-use wait bit (804/804 MMAs), so the ring has SEVEN tag slots (bits 24-30) and the '80' on a kernel's first MMA is the wait bit, not a tag.


Kernels with 3, 4, 6, 8, 9 and 10 INDEPENDENT MMAs (`indep*.ll`, each with its own accumulator,
all stored afterwards): the one-hot tag byte is assigned in issue order 80, 01, 02, 04, 08, 10,
20, 40 - eight distinct values - and with nine or ten issues the sequence rotates and the surplus
issues carry 00 (nine: 01 02 04 08 10 20 40 00 80; ten: 01 02 04 08 10 20 00 00 80 40). So the
accelerator tracks up to eight in-flight results by a one-hot slot tag that the compiler allocates
round-robin; an issue with tag 00 is one whose result is not tracked by a slot. byte0[7]
alternates 1, 0, 1, 0 across consecutive MMA issues in these kernels (the "high bits" of byte 0
noted on 2026-09-02), a parity-like bit whose role is not measured.

## 15. Batch 12: how a store waits for an MMA result (straight-line rule)

**RETRACTED in part - see section 26.** byte2[7:6] of the store is bits 5:4 of its SOURCE REGISTER number (632/632 stores), not a wait counter; the `floor(n/2)` pattern was the compiler's descending register allocation.


`store_wait_counter.json`: in every straight-line kernel (indep3/4/6/8/9/10, the single, x3,
independent, store-and-chain, mm16 kernels: 92 of 92 stores) the store of an MMA result carries in
byte2[7:6] the value `floor(n / 2)` where n is the number of MMAs issued after the producing MMA,
saturating to 0 (the most conservative wait) when n >= 8, i.e. when the eight-slot ring has
wrapped. It is a counter-based wait ("at most this many newer results may still be outstanding")
in the same spirit as the repository's mod-4 load counter on tensor loads. The rule as stated
does NOT fit the looping library kernels (mm_base 4/8, m64 4/16, sg4 2/8), where accumulators are
reused across loop iterations; the loop-carried form is open.

## 16. A bit-exact reference model, validated

`results/g17-tensorops-recon-v1/tensorops_model.py` packages the measured rules: `mma16` (one
issue), `matmul_ref` (ascending 16-chunks through the rounded accumulator), `to_f32_operand`
(10-bit truncation, exponent kept, subnormals flushed), `to_half_rne` / `to_bf16_rne`. Its
self-test replays 120 / 120 retained stimulus measurements, and `validate_model.py` compares it
with the 32x32x64 library kernel on six random half tiles with mixed magnitudes: **6,144 / 6,144
elements bit-exact**; an exact-sum-rounded-once reference matches only 1,948 of them. The model is
usable as the oracle the tensor fractional-arithmetic campaign lacked, for half/bfloat operands
with fp32 accumulation at any M, N and K multiple of 16 under the library's ascending K order.

## 17. The repository's failed fractional tile, replayed

`results/g17-tensor-fractional-runtime-v1` retains the 32x32x64 half tile on which the earlier
campaign's 21 arithmetic hypotheses all failed (`docs/g17-capability-frontier.md`,
`runtime.gap.tensor_fractional_arithmetic`): `a.npy`, `b.npy` and the executed
`measurement-a/query-0.npz` output of the repository's OWN authored tensor program. Replaying it
with `mma16` composed over the four 16-wide K chunks (`fractional_replay.json`): the ascending
order reproduces **1,024 of 1,024 output elements bit-exactly**; the next-best chunk order fits 71
of a 100-element sample, exact-once 27, sequential 21. So the authored program issues its K
slices in ascending order, the unit arithmetic is the one measured here, and the gap was the
missing reduction structure (pair / interleave-8 / C-first sequential quads), not the traversal.

## 18. A coarse throughput differential (not a benchmark)

`throughput.py`: wall clock over 8,192 vs 16,384 single-simdgroup threadgroups of kernels carrying
64 matrix issues each (dispatch overhead differenced out; three repeats, best time). Accelerator
op5106, 64 chained: 8.2e8 issues/s = 6.7 dense-equivalent TFLOP/s; eight independent chains of
eight (`ilp8x8`): 8.4e8 issues/s, no gain from ILP at one simdgroup per threadgroup; legacy op2862,
64 chained: 1.6e9 issues/s = 1.6 TFLOP/s. The accelerator issue does 8x the arithmetic of a legacy
issue at about half the issue rate, roughly 4x the throughput in this deliberately unfavourable
shape (one SIMD group per threadgroup, no tile reuse); the earlier campaign's 25 TFLOPS came from
a properly tiled matmul2d. The numbers are coarse and machine-state dependent; the qualitative
point is that the two families have distinct issue rates as well as distinct arithmetic.

## 19. Batch 13: fused row reductions (postfix operations) are ALU work with a measured order

`reduce_rows` on the cooperative destination (`mm_reduce_sum` / `mm_reduce_max`, 32x32x64): the
MMA count is unchanged (16 x op5106) and the reduction adds only ALU: 36 x op998 (fadd) or 40 x
op9700 (fmax), 8 x op14169 (an unnamed sched-408 op with `IRGPR32, imm, GPR32, imm, imm` - the
cross-lane exchange) and 8 x op595. No tensor-family opcode is involved; postfix operations on the
accelerator's result are ordinary SIMD code over the fragment layout. On hardware the sum
destination's element `i` is the sum of D row `i` (integer inputs exact), and its fp32 order,
mapped by 14,880 eps-triple dispatches over the 32 columns (`reduce_tree.json`) and verified on 32
random rows (`reduce_sum_model.json`, 32/32): each lane sums its eight D elements sequentially in
DESCENDING column order (tile-1 columns 4i+3..4i, then tile-0 columns 4i+3..4i), then lane^1
(columns +4), then lane^8 (columns +8), RNE32 at every step. Alternatives (sequential, balanced
tree, ascending lane order, xor-8-first) fit 6 to 21 of 32.

**Correction (section 122, added later).** The DESCENDING per-lane order stated above is a property of the
compiled object `mm_reduce_sum` (object_sha256 0110b1da...), not of the hardware and not of `reduce_rows`
in general. The identical `reduce_rows` call compiled into a kernel that first reads all 32 elements of the
destination cooperative tensor sums each lane's elements in ASCENDING order (4 of 20 objects tested). The
lane^1 then lane^8 exchange structure was the same in every census run. The 32/32 random-row check above
stays valid for the object it was run on.

## 20. Batch 14: load words, and where the loop-carried wait rule stands

- Ordinary fragment loads (op12709, `loads1..12.ll`): the first immediate carries a 4-bit value in
  bits 20-23 assigned in issue order 8, 1, 2, 3, 4, 5, 6, 7 for up to eight loads, and the LAST
  load of the group additionally carries bit 37 (`0x2000000000`); beyond eight loads the
  assignment is non-monotonic (12 loads: 8 1 2 3 4 4 4 4 7 7 6 5), so it is a scheduler-chosen
  counter, not a positional slot. The tensor loads of the library kernels show the same 8, 1, 2,
  ... sequence per group with bit 37 on the group's last load. Stores carry 0x10 (the last-use
  flag) and nothing else. This is the repository's typed hole (`g17-load-counter-is-not-a-
  scoreboard-slot`) seen from the other side; the tag-to-consumer rule is not fitted here.
- Loop-carried store wait (`mm_base`): the body is a loop of 16 masked/unmasked tensor loads and
  8 MMAs (tags 80 01 02 04 08 10 00 00 over four accumulators), a peeled final iteration, then
  eight tensor stores carrying counters 2, 2, 1, 1, 1, 1, 0, 0 for the accumulators produced by
  MMAs 2, 6, 4 and 8 of the last block. Neither "MMAs after the producer" (3, 1, 2, 0) nor
  "tracked tags after" (2, 0, 1, 0) reproduces that; the loop-carried rule remains open. What IS
  established for an author: the counter's value 0 is the conservative "wait for everything"
  case (the compiler's own choice whenever the ring wraps), so emitting 0 is always safe at some
  cost in overlap.

## 21. What an author now has (checklist)

For a G17 program that uses the Neural Accelerator without copying compiler output:

1. Metadata: per-kernel slot 44 with a nonzero low byte (the `TENSOR` class's `44: 1`); nothing
   else in the section, LD or ARCH distinguishes an accelerator kernel (section 8).
2. Instruction forms and their bit-level encoding (sections 4/7 and 25, `mmaenc.py`): op5106 `D:tup8 A:tup4 B:tup4 C:tup8`
   with D == C (in place); op5107 without C (accumulator starts at zero); op5104/5098 for fp32
   operands (10-bit truncated), op5099 for their no-C form; op10384 for int8 (i32 wraps);
   byte6[2] = A is half, byte7[6] = B is half (else bfloat); byte8[2] = transA, byte8[3] = transB.
3. Register fragments (section 3): A/D lane l slot j holds row `8*(l>>4) + 2*((l>>1)&3) + (j>>2)`,
   column `8*((l>>3)&1) + 4*(l&1) + (j&3)`; B holds k = `4*(l>>4) + ((l>>1)&3) + 8*(j>>2)` at the
   same column formula; transposed A per section 3. Tensor loads deliver rows r and r+8 of a
   16-column sub-tile per lane in two 2-register halves, which is B's layout (section 12).
4. Arithmetic (sections 4-6, 16): `tensorops_model.mma16` per issue; chains through the rounded
   fp32 accumulator; fp32 operands truncated to 10 bits; destinations RNE.
5. Scheduling (sections 10, 11, 14, 20): byte0[3] = load-use wait on the first MMA after its
   fragment loads; seven tag slots (flags-word bits 24-30, assigned in issue order by the
   compiler; 0 = untracked); 0x20 in the low byte on every MMA that will be accumulated into
   again; the per-source "16" on the last
   use of a fragment tuple; consumers carry NO token (section 26: the store field once read as a
   token is bits 5:4 of the source register); inside a loop only the first accumulator's chain
   carries tag bits; no explicit wait instruction exists.
6. Masking and cooperation (sections 6, 8, 12): partial tiles through op12675/op17258 with a
   GPR16 mask from op11456; multi-SIMD-group ownership is per-SG store masks, arithmetic
   unchanged.

## 22. Batch 15: the store token is causal, and 0 is NOT a safe default (correction)

**RETRACTED in part - see section 26.** the mutations redirected the store's source register (bits 5:4) together with its last-use flag; that is why the store's own tile and the tile owning the named registers were both corrupted. There is no token.


`mutate_wait.py` patches byte2[7:6] of the sixteen result stores in the `indep8` archive (eight
independent MMAs, tokens 3 3 2 2 1 1 0 0) and runs each variant five times with random integer
tiles (deterministic across repeats):

| mutation | tiles wrong |
| --- | --- |
| original | none (5/5 runs) |
| every store token -> 0 | all eight |
| every store token -> 3 | tiles 2-7 (tiles 0 and 1, whose token was already 3, stay right) |
| store 0 (tile 0, token 3) -> 0 / 2 / 1 | tile 0 AND tile 6 / tile 2 / tile 4 respectively |
| store 15 (tile 7, token 0) -> 3 | tile 7 |
| store 7 (tile 3's second store, token 2) -> 0 | tiles 3 and 7 |

So the token names a result group (two consecutive MMA issues), waiting on the wrong group leaves
the store's own tile stale AND corrupts the tile of the group it named; the field is a consumer
token with global accounting, not a threshold. Section 20's statement that 0 is the conservative
choice was wrong and is withdrawn. The compiler's own 9- and 10-issue kernels run correctly on hardware,
and their tokens are exactly `floor(n / 2)` saturating to 0 for n >= 8 (`store_wait_counter.json`,
18/18 and 20/20), so the rule covers a wrapped ring too; what changes with a wrap is that 0 then
means "the oldest of more than eight", not a safe default. Authoring rule that is established for
straight-line code: each consumer carries `floor(n / 2)` with n the number of MMAs issued after
the producer (exact on 92/92 stores across twelve kernels, causally verified by the mutations).
The looping library kernels, whose consumers are not in production order, still do not fit
(section 20).

## 23. Batch 16: the token counts down over CONSUMERS, not producers

**RETRACTED in part - see section 26.** the countdown is the register allocator assigning accumulators in descending order; the field is the register number.


Kernels with the same independent MMAs but permuted store order (`order4_rev`, `order4_1032`,
`order8_rev`, `order8_mid`, `order6_2`; the last three run correctly on hardware): the emitted
tokens are 3 3 2 2 1 1 0 0 (P = 8), 2 2 1 1 0 0 (P = 6), 1 1 0 0 (P = 4) in STORE order whatever
producer each store consumes - e.g. `order8_rev` stores producers 6, 5, 4, 3, 2, 1, 0, 7 with
tokens 3, 3, 2, 2, 1, 1, 0, 0. So for P outstanding results consumed by 2P stores, the c-th store
carries `floor((2P - 1 - c) / 4)`: a countdown of result pairs still to be consumed, which equals
`floor(n_after / 2)` only when consumption follows production order (the earlier fits). Together
with section 22 (a wrong value corrupts the store's own tile and the tile whose value it named,
and the compiler's own countdowns are correct under arbitrary consumer order), the field is a
consumption-sequence token the hardware checks against its own accounting, not a wait threshold
or a producer slot. Its microarchitectural meaning is not resolved; the emission rule is.

## 24. Batch 17: the loop-carried rule (closed)

**RETRACTED in part - see section 26.** the store values are register numbers (R32, R24, R16, R8 -> 2, 1, 1, 0); the 'A + 1' rule is void. The loop observation that only the first accumulator's chain carries tag bits stands.


Hand-written AIR loops (`loop_a{A}x{n}.ll`: four trips, A accumulators carried through phis, n
chained MMAs each per trip, fragments reloaded per trip, stores after the loop; the compiler
unrolls by two), read with Apple's tables and run on hardware (`mutate_wait-loop_*`):

- Inside a loop body only the FIRST accumulator's chain is tagged - its first issue 80, its last
  01 - and every other issue in the body carries tag 00 (untracked), for A = 1, 2, 4, 8 and n = 1,
  2. The 0x20 "more accumulation follows" flag and byte0 behave as in straight-line code.
- The store tokens after the loop are the straight-line countdown for **A + 1** results: the r-th
  consumed result's two stores carry `floor((2A + 1 - 2r) / 4)`, saturating to 0 (A = 1: 0; A = 2:
  1, 0; A = 4: 2, 1, 1, 0; A = 8: 0, 3, 3, 2, 2, 1, 1, 0). This is exactly the library kernel's
  2, 2, 1, 1, 1, 1, 0, 0 (`mm_base`, A = 4) that the straight-line rule could not fit: a loop
  leaves one extra result outstanding for the accounting (the tagged chain's loop-carried entry).
- Causal: `loop_a4x1` runs correctly as compiled (5/5), with the straight-line tokens 1 1 1 1 0 0
  0 0 substituted tiles 0, 2 and 3 are wrong, and with tile 0's tokens set to 3 tile 0 is wrong.
  `loop_a8x1` (ring wrapped, first result's token saturated to 0) is correct as compiled.

With sections 20-23 this closes the scheduling side for an author: tags (8-slot ring in
straight-line code; first chain only, 80 ... 01, inside loops), the 0x20 and last-use flags, the
load-use wait bit, and the consumer countdown `floor((2P - 1 - c) / 4)` with P = the outstanding
results, plus one inside a loop.

## 25. Batch 18: the MMA is encodable - field map from the decoder oracle, authored MMAs executed

`fieldmap.py` flips each of the 80 bits of a reference encoding of op5106, 5107, 5104, 5098 and
10384 and records which Apple-decoded operand moves (`fieldmap.json`). Register fields (register
number in units of the tuple size, bit weights as decoded): D = C: byte0[7] (8), byte7[5] (16),
byte2[7] (32), byte2[3] (64), byte2[4] (128); A: byte3[4] (4), byte3[1] (8), byte3[0] (16), byte3[7]
(32), byte3[5] (64), byte3[6] (128); B: byte9[1] (4), byte9[2] (8), byte9[3] (16), byte9[4] (32),
byte5[2] (64), byte9[6] (128). Flags word: byte0[3] = bit 31 (wait), byte1[2..6] = bits 24-28,
byte7[7] = bit 29, byte2[6] = bit 30 (the tag slots), byte4[3] = 0x20 (more accumulation follows),
byte4[6:5] a two-bit code (01 nothing, 00 bit 37, 10 bit 36, 11 invalid), with inverted high bits
at byte6[5], byte6[7], byte0[5], byte4[5]. A-flags: byte2[5] = 16 (last use), byte1[7] = 32;
B-flags: byte5[1] = 16, byte8[0] = 32. Type codes: A byte6[2] and B byte7[6] (2 = half, 3 =
bfloat, 1 = float, 75 signed / 11 unsigned int8), transposes byte8[2] / byte8[3] (+32 in the
type code). Structural bits: byte8[1] selects the no-C form, byte5[7] B-is-fp32, byte8[4]
A-is-fp32, byte6[0] the int8 family. Sixteen bits are inert.

`mmaenc.py` encodes from decoded operand values (800/800 compiler MMAs round-trip byte-exact)
and from semantic fields `mma(D, A, B, C, a_type, b_type, transA, transB, wait, tag, more,
a_last, b_last)` (796/800; the four exceptions carry the unexplained bit-36 code). `author_mma.py`
then REPLACES the compiler's MMA in the single kernel with encoder output and runs it: the
original configuration re-encodes byte-identically; A/B register tuples swapped, both transposes
set, and A read as bfloat - none of which the compiler emitted - all execute exactly as the
layouts and arithmetic model predict (3/3 random trials each). The instruction can be authored.

## 26. Correction: there is no store token

The same decoder oracle applied to the stores shows that byte2[7:6] of op17256/17257/17258 is
bits 5:4 of the SOURCE REGISTER number - `(src >> 4) & 3` on 632 of 632 stores in this campaign -
and that bit 31 of the MMA's flags word is byte0[3], the repository's load-use wait bit (804/804).
Sections 12, 15, 16, 17 and 22-24 read a register field as a scheduling token: the "countdown"
was the compiler allocating accumulators in descending register order; the "ring wrap" was
register numbers above 63 wrapping the two bits; the "A + 1" loop rule was the loop kernels'
register assignment; and the mutation results were the consequence of redirecting a store's
source tuple (with its last-use flag) to another tile's registers. Those sections are retracted
as marked; the load-use wait bit, the seven tag slots (bits 24-30), the 0x20 flag, the last-use
flags and the loop-body tagging observation stand. What an author must emit for synchronisation
is therefore smaller than claimed: the wait bit on the first MMA after its loads, distinct tag
slots for results consumed out of issue order (compiler practice; not proven necessary), 0x20 on
non-final accumulations, and the last-use flags - and no consumer token at all. The lesson is
recorded: decode a field with the oracle before modelling it.

## 27. Batch 19: the whole accelerator family - 132 declared, 10 admitted, one never emitted

Question: does Apple's table declare accelerator forms this compiler never emits, and does the
M5 decode or execute them (fp8, fp4, 16-bit accumulators)? Instrument: Apple's MCInstrDesc table
(`agxforge.g17.model.opcodes`) for the declaration, Apple's AGX3 disassembler (`tools/agx3dis`) for
admission, the single-MMA archive for execution. Files: `family.py`, `family_exh.py`,
`family_cpu.py`, `family*.json`, `author_5100.py`, `bitcensus_hw.py`, `bitcensus_hw*.json`.

**Declared.** Scheduling classes 170-175 hold 132 opcodes, all with tsflags 0x400002001. Read
from the operand classes (nothing here is executed): 5090-5107 are the D:tup8 (32-bit
accumulator) forms over every A/B width in {tup2, tup4, tup8}; 5108-5143 are D:tup4 forms (a
16-bit accumulator) over A/B widths in {GPR32, tup2, tup4, tup8}, a GPR32 fragment being 4 bytes
per lane, the size of a 16x16 4-bit tile; 10386-10389 the class-170 (int) D:tup4 forms; 5144-5618
(classes 174/175) `_shifted` register-class variants with 14/15 operands, GPR16tup3 operands and
an extra GPR16. So the table declares 8-bit-operand, 4-bit-operand and 16-bit-accumulator MMAs.

**Admitted.** From the five compiler-emitted encodings (op5106/5107/5104/5098/10384), every 1-,
2- and 3-bit flip of the 10-byte form and of a 12-byte zero-extended form (1.1 M candidates,
batched through `g17bitprobe.decode_many`), plus a bounded 5-deep BFS and 1-2 flips of 14/16-byte
extensions with 0x00 and 0xff fill: exactly TEN accelerator-class opcodes decode at their own
length: 5098, 5099, 5100, 5101, 5104, 5105, 5106, 5107, 10384, 10385. All ten are D:tup8 with A, B
in {tup4, tup8} or both tup2. Not one of the 122 others (every 8-bit-float, 4-bit, 16-bit-
accumulator and shifted form) was reached. The same sweep under every CPU name libLLVM's string
table carries for the agx3 target (`family_cpu.py`; the decoder is built from a scratch copy of
`tools/agx3dis.c` with the CPU from `$AGX3_CPU`): g17s, g17g and g18 admit the same ten; g17,
g16, g16s, g16g, g15, g15s decode the very same bytes as op3773/op3789 (sched 144, an unrelated
IRGPR32/GPR16 form) and admit no accelerator opcode at all; g14g and an invalid name decode
nothing. So the accelerator family is feature-gated by CPU, and the 122 unadmitted forms are not
gated behind any CPU this libLLVM knows. This is a bounded claim: not reachable within three
flips of these seeds (five deep with a bounded frontier); a form whose base format differs in
more bits is not excluded, and no encoding exists to try on the hardware.

**Executed: op5100, admitted and never emitted.** op5100 is A:tup8 x B:tup4, the mirror of
op5104 (A:tup4 x B:tup8), one flip from op5098 (byte5[7] cleared). No library kernel in this
campaign or in `libTensorOps.rtlib`'s H17s path emits it. Field-mapped from the decoder
(`fieldmap.py`, now covering all ten), added to `mmaenc.mma` (a_type float, b_type 16-bit), and
executed by replacing the op5104 in `new_f16f32_nn` with op5100 that reads the fp32 tuple
(R8-R15) as A and the half tuple (R16-R19) as B. Prediction from the account alone: A read in the
A layout with `to_f32_operand` truncation, B in the B layout, `mma16`. Result: 4/4 random tiles
bit-exact with B read as half and 4/4 with the same bits read as bfloat (`author_5100-*.json`);
the re-authored op5104 is byte-identical to the compiler's and runs 4/4. Control: the same bytes
with byte0[1] flipped (the decoder refuses them) run without fault and leave D == C (4/4) - an
unadmitted encoding here is a no-op on the accumulator, not a trap.

**Correction (section 125, added later).** "No library kernel ... emits it" is wrong for the relaxed forms: with `relaxed_precision = true` the library
emits op5100 for float x half, float x bfloat and float x int8, and op5104 for the mirrored pairs (30 forms measured). The statement held only for the
non-relaxed compilations this campaign had used.

**Hardware single-bit census of op5106** (`bitcensus_hw.py`: one flip per process, random
integer tile, C != 0; outcome classes identical / D==C / zeros / wrong / refused / error; 80/80
dispatched, none refused, none faulted):
- every one of the 16 decoder-inert bits is hardware-inert (identical result);
- every flags-word bit is hardware-inert on a single MMA - the seven tag slots, 0x20, 0x40,
  the byte4 code including the decoder-invalid 11 (`bitcensus_hw-multi-37_38`), and the
  inverted bits 33/41/47 alone and together - EXCEPT byte0[3]: clearing the load-use wait makes
  the MMA read its fragments before the loads land and the result is all zeros (C was nonzero,
  so this is not an unread buffer). That is the causal confirmation of section 25's reading;
- register-field bits do what the map says: D-field bits move the result to another tuple (D==C
  in R0-R7), A/B-field bits read other tuples (D==C when those are the never-written R28+,
  wrong otherwise), byte8[1] gives the no-C product, byte8[3] the transposed-B product;
- bits the decoder refuses (byte0[1], byte0[2], byte2[2], byte4[0], byte4[1], byte6[1]) execute
  as D==C; byte0[4], byte1[1], byte2[1], byte3[3], byte4[7], byte7[3], byte7[4], byte8[7] are
  refused by the decoder but change the result (byte2[1] to zeros), so the hardware reads
  fields there that Apple's decoder does not name for this opcode.

Answer to the batch question: on this OS and decoder, M5's accelerator surface is ten opcodes,
all with a 32-bit accumulator and 16/32-bit or int8 operands; the fp8/fp4/16-bit-accumulator
forms are declared in Apple's table but admitted by no CPU this libLLVM knows, so nothing here
can put an encoding of them in front of the hardware. The one unexposed form that IS admitted,
op5100, executes exactly as the account predicts.

## 28. Batch 20: the load-to-MMA synchronisation is an eight-slot scoreboard, executed

Spencer's caution ("you may be relearning things") sent me to the repository first. The relevant
prior art: `ledger/g17-the-wait-tag-is-a-mask.toml` (corpus statistics: the ALU byte1[2..6] field
is a mask over dependency slots, "not yet executed", naming the discriminating experiment as "author
two loads and a consumer, vary which bits of the mask are set, and see which combinations
stall"); `isa/g17-scalar-isa.toml` op12709 (fields located, "one vector load per program" because
the 8/14-byte length choice was read as an unrecovered wait composite in bytes 4..5);
`ledger/g17-load-counter-is-not-a-scoreboard-slot.toml` (the tensor loads' byte6[4:3]). This batch
runs the named experiment with the MMA as the consumer. Files: `load_census.json`, `memmap.py`,
`memmap.json`, `mutate_code.py`, `mutate_code-*.json`, `waitmask_matrix.json`,
`waitmask_matrix2.json`.

**Length rule (census, 397 op12709 loads in 51 objects).** 14 bytes if and only if the load's
word carries bit 37, 52/52 and 319/319 (8-byte) with 26 10-byte forms (a mode-word extension,
never bit 37). The length is not a wait composite: it is whether the group-end flag is carried.

**Field map (decoder census, `memmap.py`).** op12709's word: the 4-bit value the decoder shows as
8, 1, 2, ... (section 20) is `8 - (byte4[3] + 2 byte4[4] + 4 byte6[4])`; other word bits at
byte2[5], byte5[5..7], byte6[3]; bit 37 at byte8[6] of the 14-byte form. Destination tuple,
index register, displacement and width fields agree with `isa/g17-scalar-isa.toml`.

**Hardware, single flips (`mutate_code.py` on `new_f16f16_nn`, random tile, C != 0).** Every
word bit of every one of the four loads flipped alone, 47 dispatches: all identical. Then all
four loads' values changed together from 8 to 1 (with bit 37 cleared): ZEROS - the MMA read its
fragments before any load landed. So the words are not inert; single flips passed because the
MMA's wait covered the other three loads and the fourth won its race.

**The matrix (`waitmask_matrix.json`, 43 dispatches), loads all set to value v, MMA flags bit
set, MMA byte0[3] on or off:**

    v = 8   byte0[3] on: identical for any tag bit;  byte0[3] off: zeros (bits 24, 27 do not help)
    v = 4   bit 27: identical (byte0[3] on OR off);  every other bit, or none: zeros
    v = 2   bit 25: identical;                       every other bit, or none: zeros
    v = 1   bit 24: identical (byte0[3] on or off);  every other bit, or none: zeros

**Preregistered completion (`waitmask_matrix2.json`, 20 dispatches, prediction written before
dispatch):** value v fills slot v - 1 and the consumer waits on it with flags bit 24 + (v - 1):
v = 3 -> bit 26, v = 5 -> bit 28, v = 6 -> bit 29, v = 7 -> bit 30, each against four wrong bits.
20/20 as predicted.

**Reading.** There are eight dependency slots. A load names the slot it fills in its word (the
decoder's value is slot + 1, so the compiler's first group "8" is slot 7, then 0, 1, 2, ...). A
consumer MMA carries an eight-bit wait mask at flags-word bits 24..31 - byte1[2..6] are slots
0..4, byte7[7] slot 5, byte2[6] slot 6, and byte0[3] is slot 7. The "load-use wait bit" and the
"tag slots" of sections 12 and 25 are one field: sections 14 and 25 read bits 24-30 as a result
ring or tags assigned to the MMA; they are the mask of load slots the MMA waits on, and the
one-hot-per-MMA pattern arose because each MMA in those kernels waited on its own load group.
Bit 37 (group end) was cleared in every v != 8 run and changed nothing here. Unmasked slots are
not waited on: a load in a slot the consumer does not name is a race, which the single-flip
census won every time and the four-load move lost every time.

**What this gives the compiler owner.** A second, third, ... vector load is authorable: give each
load group a slot (any of 0..7), carry bit 37 (14-byte form) on the group's last load as Apple
does or omit it (inert in these runs), and set the consumer's mask bit for every slot it depends
on; length is not chosen by what is in flight. The ledger's discriminating experiment is executed
for the MMA consumer. Not tested: the ALU consumer's byte1[2..6] (in `dep_alu_consumer` the
op1006 fadd's byte1 bits are its immediate - flipping them changed the constant - and byte0[3]
is an opcode bit there, so the fadd after the MMA carried no wait and still read the result:
MMA-to-ALU is interlocked or the fadd's form waits implicitly); slot reuse and lifetime; whether
a store's word bits 24/25/27 (present in `memmap.json`) are the same mask.

## 29. Batch 21: independent tensor lowering - generated GEMM kernels bit-exact against matmul2d

The repository's tensor compiler is a registry ("a tensor matmul cannot yet be built from its shape";
`docs/archive/g17-tensor-common-handoff.md`: "independent complete tensor lowering remains open"). This batch
builds complete GEMM bodies from the account and proves them against Apple's library. Files:
`memenc.py` (+ `memenc_roundtrip.json`), `faddenc.py`, `gen_gemm.py`, `lower.py`, `run_lowered.py`,
`oracle_matmul2d.py`, `oracle_child.py`, `lowering_matrix.json`, `lowered-*.json`, `vs_*.json`,
`tmpl_*/`, `or_*/`.

**Encoders, all from field maps, none from copied instruction bytes.**
- `memenc.py`: op12709 (8-, 10- and 14-byte forms) and op17256 (8- and 14-byte) from `memmap.json`:
  destination/source tuple, index register, displacement (the 14-byte forms extend it to 65535
  bytes), width code (byte7[6:5]: 0 = 16, 1 = 1, 2 = 4; width 2 = code 0 + byte8[7] in the 10-byte
  form; code 3 unwitnessed), scoreboard slot, byte1 = 4 x binding rank (the decoder shows it only
  as an `expr`). Round trip: 822/822 loads and stores in every object of this campaign, including
  the templates. CORRECTION to section 28: the slot field is `byte4[3] + 2 byte4[4] + 4 byte6[4]`
  (value shown by the decoder = slot + 1), not `8 - (...)`; the 14-byte load is selected by bit 37
  OR a displacement above 255, not by bit 37 alone (the census's 52 fourteen-byte loads all carried
  bit 37 because their displacements were small; the templates have 14-byte loads with
  displacements 512..3584 and no bit 37). The 14-byte word is not linear across bit 37, so the
  encoder keeps one reference per case.
- `faddenc.py`: op998 fadd and op10282 integer add, 12-byte three-register forms, bit positions from
  the repository's certified ledger (`isa/g17-contract.jsonl`; register slot bits are indices in
  16-bit register units). Round trip 36/36 for op998; op10282's unmapped byte10/byte4 mode bits
  come from the library's own int32 C-add instance, and the encoder is claimed only for that form.
- `mmaenc.py` as in section 25, now with the eight-bit wait mask set directly.

**Generator (`lower.py`).** Input: M, N, K (multiples of 16), operand types (half, bfloat, float,
int8, uint8), transposes, mode. Output: the whole body of the kernel - load groups with slot
assignment (K-step k in slot k mod 8), MMA chains with in-place accumulation (op5107 opens a chain,
op5106/5104/5100/5098/10384 continue it; op5100, which no library kernel emits, carries the
fp32 x 16-bit cases), wait masks from the slot rule, last-use flags, register allocation with
double-buffered A/B tiles and accumulator groups (the load/store register fields are seven bits,
so every tuple they touch is below R128; 64x64 runs as two groups of eight tiles), the fp32 or
int32 C add for Apple's accumulate order, stores, END and nop padding. Inherited from a
compiler-built template (`gen_gemm.py`): the prologue before its first load (argument fetch and
the lane-index register), the container and the metadata (register count from slot 32's top byte,
else slot 44's; slot 44 enable). The inherited byte range is recorded in every plan.

**Memory layout.** Fragment-major buffers: A tile (mi, k) at (mi KT + k) 512 bytes, B tile (ni, k)
at (ni KT + k) 512, C tile as two 512-byte halves; fp32 operands as two halves; int8 fragments
padded to 16 bytes per lane. Addressing is lane-index x 16 + displacement, one index register.
matmul2d's row-major tensor addressing is not reproduced - the oracle kernels read row-major
buffers holding the same logical matrices, so the comparison is on values, not on addresses.

**What matmul2d's accumulate mode is.** With C != 0 the library computes the product from zero
(24 op5106 as in multiply mode), then loads C (op12710) and adds it with op998 fadds (op10282 for
int32), then stores: one extra fp32 rounding at the end, not C inside the MMA chain. The lowering
offers both: `accumulate='last'` (Apple's order, the oracle comparison) and `accumulate='first'`
(C loaded into the accumulator and the chain run on it: the hardware path the library does not
use, one rounding fewer; validated against the arithmetic model only).

**Proof (`lowering_matrix.json`, 41 cases x 3 random trials, every element compared).** Generated
kernel == matmul2d == fp32 reference model on all 41: M, N in {16, 32, 48, 64} x K = 64 (16
shapes); K in {16, 32, 128, 256} at 32x32; half/bfloat/float/int8/uint8 operand pairs at 32x32x64
(nine); transA, transB, both at 48x32x64; accumulate with half, bfloat, float, int8, uint8 and
transposes; 64x64x64 accumulate and 64x64x128. Mixed-sign int8 x uint8 runs and matches the
reference; matmul2d refuses to compile it, so it has no oracle. Bounded by: template body room
(64x64x256 needs a longer template), single simdgroup, whole tiles, my packed layout.

## 30. Batch 22: partial tiles by masked stores, multi-simdgroup ownership, and decode-back guards

Files: `maskprobe.py/.json`, `lower.py` (partial, sg), `oracle_matmul2d.py`, `gen_gemm.py` (sg), `peer_nine.py`,
`lowering_matrix.json` (now 63 cases), `retired/`.

**Masked store field map and mask semantics.** op17258 (10- and 16-byte forms) censused into `memmap.json`
and added to `memenc.mstore`: same tuple/index/displacement/width fields as op17256, a GPR16 mask operand
(byte4[4], byte4[6], byte5[0..6]), word bit 42 set, and the wait-mask bits 24/25/27 at byte7[5..7] (the
10-byte form carries no bit 26/28-30 field). Round trip 1145/1190 over every load/store in the objects; the
45 that differ are a masked-store sub-form with index flags 146 (byte7 = 0xe1) that this encoder does not
claim. Hardware (`maskprobe.py`, generated body: A, B loads, op5107, a per-lane mask word loaded from the C
buffer into R8-R11, two masked stores with mask register R8L waiting on the mask load's slot): mask bit j
(0..3) enables word j of the four-word store for that lane, bits above 3 are ignored, the written words hold
the correct values, unwritten words keep their sentinel - six mask patterns, 32 lanes each.

**Partial tiles in the lowering.** `lower(..., partial=(Mp, Np))`: M and N are rounded up to tiles for the
computation; every store of a tile touching the boundary becomes a mask-word load (slot 1, from a host
table at C + 16384 laid out per (tile, half, lane): bits j with row < Mp and column + j < Np, from the
measured D layout) plus a masked store waiting on slot 1. K is zero-padded by the host (a zero product adds
exactly zero in this arithmetic, and the library's own partial-K path zero-fills). 17 partial shapes run
against the reference with sentinels outside the true extent required untouched: 24x32x64, 32x40x64,
48x40x48, 16x24x80 acc, 32x8x16, 64x48x16 int8 transB, 16x40x64 transA (all also bit-exact against
matmul2d), and 24x40x48, 50x37x80 acc, 17x19x16 int8 transA, 64x63x64, 64x62x64, 1x1x16,
60x60x100 bfloat transB, 40x24x32 acc transposed, 32x65x17, 48x32x100 acc fp32, 48x24x100 acc fp32 -
shapes matmul2d REFUSES to compile ("At least one of M or N must be a multiple of 16", "n % 8 == 0", "K
must be dynamic or a multiple of 16"), reference only. The generated kernels have no shape restriction
beyond the register and body budget.

**Multi-simdgroup.** `gen_gemm.py sg=S` builds templates taking the simdgroup index as a fifth argument;
the compiler's prologue then holds per-buffer index registers (A and C: lane + s x tiles-per-simdgroup
offset; B: lane), which `Template` reads per binding from the template's own loads and stores. The
generated body is SIMD-uniform and covers MT/S M-tiles; dispatch is 32 S threads. 64x32x64 on two
simdgroups equals matmul2d `execution_simdgroups<2>`, 64x64x128 on four (half, and int8 transA) equals
`execution_simdgroups<4>`, 3/3 each. Templates whose compiler defines an index register inside the body
rather than the prologue are refused by name (tmpl_sg4_64x64x64 and tmpl_sg2_64x32x128 were; the
pre-refusal 0/3 receipt is kept in `retired/`).

**Decode-back guards (after the compiler session's finding that `agxforge.g17.auth.encode` rounds an odd
register index silently).** Every encoder here now decodes its output with Apple's decoder and compares
opcode, length and every requested operand before returning; a mismatch raises. Tested with deliberately
unrepresentable operands: displacement 300 in the 8-byte load (refused by the residual check), index
register R200 (no such name), fadd destination R300 (accepted by the field writer, CAUGHT by the guard:
decodes as R44), an unaligned MMA tuple at R6, a store tuple at R130 - all five refused. Round trips
unchanged (memenc 1145/1190 as above, mmaenc 2226/2226, faddenc op998 36/36).

**Matrix now (`lowering_matrix.json`).** 63 cases; 53 have a matmul2d oracle and all 53 are bit-exact
against it and the reference on every element of every trial; the 10 without an oracle are shapes the
library refuses, bit-exact against the reference with the outside region untouched.

## 31. Batch 23: the unit, by timing - one 16x16x16 slot per SIMD, 22 ns per issue, 33 TFLOPS on 20 cores

Instrument: generated bodies (`ubench.py`: C independent accumulator chains x L dependent MMAs, operands loaded
once, no loads in the chain) in the `tmpl_64x64x256` container, timed on the GPU side
(`gputime/gputime.mm`: GPUStartTime..GPUEndTime of the measured command buffer, best of 9, each preceded by a
22 ms saturating dispatch of the same pipeline so the clock is up; the cold and warm numbers agree within
noise). Every chain body was checked to store L x A.B exactly (L = 4, 100, 250). Wall-clock timing
(`ubench_run.py`, `ubench_results.json`) was tried first and is kept: at sub-millisecond dispatch times it is
dominated by launch overhead and clock ramps and its L = 0 baselines are not additive. Receipts:
`unit_timing.json` (the table below), `ubench-*.plan.json`, `overlap.py`. Machine: Apple M5 Pro, 20 GPU
cores (system_profiler). The GPU clock is not readable from here; where cycles are quoted they assume the
clock implied by the saturated rate (below).

**Issue interval per simdgroup.** One simdgroup, one threadgroup: a dependent chain of L op5106 costs
22.2 ns per MMA (slope of L = 200..1400: 5.96, 10.50, 19.50, 32.58 us; the first ~100 issues hide in the
~4 us kernel start). Eight independent accumulators (C = 8, L = 175, the same 1400 issues): 31.96 us, the
same. So a simdgroup issues one 16x16x16 MMA every ~22 ns whatever the dependency structure, and the
result latency is at most that interval - the chain never waits beyond the issue slot. There is no
deeper pipelining per simdgroup to exploit with ILP.

**Four slots per core.** 1, 2 or 4 simdgroups in one threadgroup (one core) each running the 1400-chain:
32.8, 32.4, 32.8 us - no contention. 8: 57.5 us; 16: 112.6 us; 32: 224.3 us - linear beyond four. A core
runs four MMAs concurrently: one accelerator slot per SIMD (the core has four 32-wide SIMDs), and the
slot is occupied for the whole 22 ns.

**Peak.** 16,384 threadgroups x 4 simdgroups x 1400 issues = 91.75 M issues in 22.77 - 0.33 ms:
4.09 x 10^9 issues/s = 33.5 TFLOP/s fp16/bf16 -> fp32 dense (8192 flop per issue), 1.67 TFLOP/s per
core, and the same rate for the dependent-chain body. Per SIMD that is one issue per 19.6 ns. If the
slot takes 32 cycles per MMA (128 MAC per cycle per SIMD, exactly 4x a 32-lane FMA), the implied clock is
1.63 GHz, which is where this part's GPU is expected to run; the "4x AI compute" Apple states for M5 is
this ratio. With 1 or 2 simdgroups per threadgroup the saturated rate is the same (5.70 / 11.43 ms for
1/2 of the work).

**Forms.** bf16 x bf16, fp32 x half, half x fp32, fp32 x fp32 (op5104/op5100/op5098) and transposed A:
identical to half x half at every point (32.5-32.9 us single chain, 22.75-22.79 ms saturated) - the fp32
operand forms cost nothing, consistent with their 10-bit truncation feeding the same datapath. int8
(op10384): 21.6 us single chain (13 ns/issue), 20.6 us with ILP, and 11.50 ms saturated - exactly 2x the
fp16 rate: 8.1 x 10^9 issues/s = 67 TOPS int8 -> int32, 256 MAC per cycle per SIMD at the implied
clock. op5107 (no accumulator) issued back-to-back INTO THE SAME destination costs 34 ns per issue (a
write-after-write stall); into eight destinations, or at saturation, it is the ordinary 22 ns / 22.78 ms.
In-place accumulation into the same tuple is free.

**Legacy op2862 (8x8x8 fp16) and overlap.** A dependent chain of op2862 costs 16.3 ns per issue (slope
of L = 250..1000: 4.71, 8.38, 16.50 us): 512 MAC in 16 ns against 4096 in 22 ns, a 6.1x throughput ratio
per simdgroup. Interleaving one op2862 with each op5106 (independent registers) costs 31.8 ns per pair
(8.67, 16.63 us at 250/500 pairs) - more than the 22 ns of the accelerator alone, less than the 38.5 ns
sum: the two overlap only partly, so the legacy path is not a second independent unit; it is issued
through the same per-SIMD slot with some pipelining between the two. Four independent fadds interleaved
with each MMA cost 40 ns per step against 22 ns (MMA alone) and 30 ns (the fadds alone, latency-bound at
four chains): ALU work is also only partly overlapped with an MMA in flight.

**Unit model (measured).** Per core: four accelerator slots, one per SIMD, each accepting one 16x16x16
MMA every 32 cycles (fp16/bf16/truncated-fp32, 128 MAC/cycle) or every 16 cycles (int8, 256 MAC/cycle),
with the dependent-chain latency no longer than the issue interval, in-place accumulation free, a WAW
stall on rewriting a destination without accumulating, and the legacy 8x8 MMA and ALU instructions
sharing the SIMD's issue with partial overlap. Chip: 33.5 TFLOPS fp16 / 67 TOPS int8 dense on 20 cores.

## 32. Batch 24: the tensor memory path - per-lane gathers at ALU-computed addresses, no hardware layout unit

Question B: do the library's tensor loads/stores (op12674/op12675, op17257/op17258) transform a row-major tile
into the fragment layout in hardware? Files: `idxprobe.py/.json`, `memmap.json` (tload12/16, tstore10/16),
`memenc.py` (tload/tstore), `lower_rm.py`, `oracle_rm.py`, `rm_*.json`, `orm_*/`, `op612_probe*.json`,
`op612_bound_*.json`, `maskgen_probe_*.json`.

**What the index register holds (`idxprobe.py`).** The ALU sequence of mm_k16's prologue (SR_SIMD_ELEM read,
two op17016, two op423, op10283, op426, two op10279, op10286, op10282 - 122 bytes, position-independent) was
copied into a carrier body and its result register R32 stored per lane: R32 = m(l) x 128 + col(l) with
m(l) = 4 (l >> 4) + ((l >> 1) & 3) and col(l) = 8 ((l >> 3) & 1) + 4 (l & 1), i.e. element (m, col) of a
row-major tile with leading dimension 128. The eight op12674 that follow read [R32 x 2 + disp] with disp =
0, 2048 (+8 rows), 32 (+16 columns), 4096, 6144: two words (four halves) per lane per load. So the "tensor
load" is a per-lane 8-byte gather at an address computed by ordinary ALU instructions once per kernel; the
fragment layout is produced by that arithmetic, and the hardware does no transposition or scatter.

**Row convention.** The fragment layout measured in section 3 puts rows 2m and 2m+1 of the operand in one
lane; the library loads matrix rows m and m+8 of a 16-row tile into those two slots (disp 0 and +8 rows),
and stores the accumulator's slots 0-3 and 4-7 to rows m and m+8 (op17257 at disp 0 and +8 rows x 4 bytes)
- a fixed row permutation of A and D within each 16-row tile, which leaves the product unchanged because
B's rows (k) are not permuted (the B fragment already holds rows k and k+8). Transposed operands presumably
use a different address formula; not measured.

**Field maps and encoders.** op12674 (12 and 16 bytes, the 16-byte form also with the group-end bit),
op17257 (10 and 16 bytes) censused and added to `memenc.tload/tstore`: two-word destination / four-word
source, index register, displacement (16-bit signed in the long form), width (byte10[7] on the load: 2 or
16; byte9[6:5] on the store), element mask at byte5[3:0] (15 = all four), the scoreboard slot/wait-mask
word as on op12709 (a load can carry a wait mask of its own - used below to wait for the index register's
load). Round trip 3419/3793 over every load and store in the objects; the 374 that differ are variants this
encoder does not claim (mask-0 forms with another code, index flags 146).

**Row-major GEMM without host packing (`lower_rm.py`).** The generator now reads A (M x K) and B (K x N)
half row-major with compact leading dimensions and writes C (M x N) fp32 row-major - the SAME bytes Apple's
matmul2d reads and writes, with its tensors declared on those extents (`oracle_rm.py`, `orm_*`). Per-lane
index registers (idxA = m x lda + col, idxB = m x ldb + col, idxC = m x ldc + col) are loaded once from a
16-byte-per-lane host table; every tile is then a displacement; the first tile loads carry wait-mask bit 7
for the table load. 16x16x16, 32x32x64, 48x32x128, 16x64x256 and 32x32x64 accumulate: generated ==
matmul2d == reference, 3/3 each, on identical buffers. 64x64 needs accumulator groups in this path (not
implemented here; the packed path has them). Computing the index registers in the body instead of loading
them needs encoders for the library's extract/and/or forms (op17016, op423, op426), which the certified
ledger carries; not done in this batch.

**The mask generator is op612, not op11456 (correction to section 11, batch 8 - not section 10 as this
line originally cited; a peer session's extraction tool traced the mismatch back to this doc, see
section 86).** The registers the masked loads
and stores read (R7L, R4L in mm_partial_m24) are written by op612 (sched 29, 12 bytes: `dst:GPR16, word,
src:GPR32, 16, lo, hi`). Probed in a carrier with per-lane sources and mutated immediates
(`op612_probe_*.json`, `op612_bound_*.json`): bit j (0..3) of the destination = [lo <= src + j < hi],
unsigned - with hi = 32: src 29 -> 0b0111, 30 -> 0b0011, 31 -> 0b0001, 32 -> 0; hi = 24 and 40 move the
edge to 24 and 40; lo = 8 clears the elements below 8 (src 6 -> 0b1100). The immediate 16 marks a register
source (0 turns the source operand into an expression form and the result is 0). So a partial tile's masks
are range predicates on the lane's element coordinate, computed on the GPU once per bound; my mask tables
are exactly what op612 would produce. op11456 (batch 8's candidate) executes and writes its GPR16
destination but produced 0 for every stimulus tried (six source patterns x three second-register values,
both witnesses, `maskgen_probe_*.json`); its role stays open.

**Answer to B.** There is no tile-memory unit: the accelerator's memory path is the ordinary per-lane
load/store machinery (two- and four-word gathers with a four-bit element mask and the same eight-slot
scoreboard), addressed by ALU arithmetic that the library does once per kernel and reuses through
displacements. The row-major lowering above is the generator using exactly that path.

## 33. Batch 25: numerics at the edges (subnormals honoured, IEEE special values, fp32 operands are a 13-bit mask)

`numerics.py` on a generated 16x16x16 kernel with C loaded into the accumulator (`numerics_{half,bfloat,float,
int8}.json`, 51 cases, every raw result recorded, 51 as expected after the two corrections below):

- **Subnormal operands are honoured.** half 2^-24 x 1 = 2^-24; bf16 2^-133 x 1 = 2^-133, which is an fp32
  subnormal (0x00010000) and is kept; a bf16 product landing in the fp32 subnormal range (2^-140) is kept;
  an fp32 accumulator subnormal (2^-140) passes through unchanged; half 2^-14 x 0.5 (subnormal in half,
  normal in fp32) is exact. No flush anywhere in the fp16/bf16 path or in the accumulator.
- **fp32 operands: the low 13 bits of the bit pattern are cleared, nothing else.** CORRECTION to section 5
  ("subnormals flushed"): 2^-130 x 1 = 2^-130 (0x00080000, mantissa bit 19), 2^-136 (bit 13) kept, 2^-137
  (bit 12), 2^-140, 2^-149 give 0, and 2^-130 + 2^-140 gives 2^-130. That is exactly a mask of the raw
  32-bit pattern to its top 19 bits - sign, 8 exponent bits, 10 mantissa bits - applied identically to
  normals and subnormals: TF32-style truncation, not rounding (1 + 2^-10 + 2^-11 -> 1 + 2^-10). Section 5's
  "flush" came from stimuli whose surviving bits lay below mantissa bit 12. `tensorops_model.to_f32_operand`
  and `run_lowered.trunc_f32` are corrected (self-test still 120/120; the fp32 x fp32 oracle case still 3/3).
- **IEEE special values propagate.** NaN x 1, NaN x 0 (including a NaN in an A column whose B row is zero -
  so K padding must be zeros, never garbage), Inf x 0, Inf + (-Inf), NaN in C: quiet NaN 0x7fc00000; Inf x
  1 = Inf; half 65504 x 65504 = 4.29e9 exact; sixteen such products sum exactly to 6.865e10; bf16 3e38 x
  3e38 = Inf; C = fp32 max + 1 = fp32 max.
- **Signed zero** follows the measured tree: (-0) x 1 with everything else zero gives +0 both with C = +0 and
  with C = -0, because the pair stage adds the +0 product of the neighbouring column first and (-0) + (+0)
  = +0 under RNE. The model predicts this; a naive "(-0) + (-0) = -0" does not.
- **int8:** 16 x 127 x 127 + INT_MAX wraps to -2147225585 (i32 two's complement, no saturation, section 2
  confirmed with C in the chain); 16 x (-128)^2 = 262144.

## 34. Batch 26: what slot 44 gates, and the rest of the table

**Slot 44 is an availability gate the instruction stream feels.** With the enable byte cleared on the timing
bodies of section 31 (`native.lowered-ubench-*_slot44off.metallib`, GPU-timed as before): the 1400-MMA chain
takes 14.75 us instead of 32.0 (8.4 ns per issue instead of 22), the eight-chain body 13.9 us, and the
saturated dispatch 0.587 ms instead of 22.7 ms - 39x faster. So with the gate off the op5106 instructions
still issue and retire (at roughly an ALU instruction's cost), writing the zeros section 8 measured, and the
accelerator does no work: the byte enables the unit for the kernel (a per-kernel power or clock enable the
driver derives from the code's opcodes, which is why it is present exactly when accelerator opcodes are),
and an MMA issued to a disabled unit is a fast zero-producing no-op rather than a fault or a stall.

**The rest of the table.** Section 27 already answers the enumeration: 132 declared opcodes in the
accelerator classes, ten admitted by Apple's decoder under g17s, g17g and g18 alike and none under
g17/g16/g15, none of the 8-bit-operand, 4-bit-operand, 16-bit-accumulator or `_shifted` forms reachable
within three flips of any compiler-emitted encoding, so no encoding of them exists to put in front of the
hardware and their presence in the table says only that the MCInstrDesc is shared across targets. The g18
entry in this libLLVM decodes exactly as g17s for this family. What the hardware would do with such an
encoding is untestable from here; the decoder-refused bits that DO change op5106's result (section 27) are
the only trace of unnamed fields.

## 35. Batch 27: the general lowering - indices, masks and accumulator groups computed in the body

Goal deliverables 1-3 (`tlower.py`, `indexgen.py`, `ledgerenc.py`, `alusem*.json`, `srlanding.py`,
`indexgen_check_*.json`, `run_tlower.py`, `tl_*.json`, `tlower_matrix.json`).

**ALU forms from the certified ledger, semantics measured.** `ledgerenc.py` encodes any opcode whose bit
positions the contract ledger records (register 'slot' fields in 16-bit units), decodes back and refuses a
mismatch. Measured in the carrier (`alusem_*.json`, four stimulus patterns x 32 lanes each): op423 `d = s &
imm` (32-bit), op426 `d = sL & imm`, op17016 `d = sL >> n`, op17013 (32-bit source, from the ledger), op14391
`d = s << n`, op10279 `d = s + imm` (8-bit immediate), op10282 `d = a + b`, op11667 `d = a - b`, op437
`dH = aH & bH` (its second source field counts 32-bit registers with H fixed), op612 `dH bit j = [lo <= s +
j < hi]` (8-bit lo/hi). The 4-byte op14060 reads SR_SIMD_ELEM into R<n>L and FILLS A SCOREBOARD SLOT like
a load (byte3[7:5]); its first consumer must wait on it (the library's does).

**The source-liveness flag.** The 16/32 immediate that rides on every ALU source operand (operand 3 of and,
operand 4 of the shifts and add-imm, operand 3 of op612) is a release flag: 16 = last use. `srlanding.py`:
an op426 carrying 16 as the first reader of the SR-written lane register made every later reader see 0 and
the register itself read 0 at the store; with 32, or with a shift or 32-bit and as the first reader, the
value stayed. The generated code keeps sources live (32) unless the value is dead. This is the same flag
the loads carry as "index_last" and the MMAs as last-use.

**Index registers in the body** (`indexgen.prologue`): lane from SR_SIMD_ELEM, m = 4 (lane >> 4) + ((lane
>> 1) & 3), col = (lane & 8) + ((lane & 1) << 2), idx = m x ld + col by shift-add over ld's bits, one register
per leading dimension. Per-lane check against the formula and against the library's own register
(`idxprobe.json`) for ld = 128: 32/32; also ld = 64, 100, 17, 19, 37, 256: 32/32.

**Masks in the body.** For every tile that touches a boundary, per load or store: row predicate through the
x4 trick (op612 on 4 (m + 8p) against 4 x rows-left, so the four bits agree), column predicate on col
against columns-left, both TILE-RELATIVE so the eight-bit bounds always suffice, ANDed by op437; masked
loads op12675 for A (rows >= M, columns >= K), B (rows >= K, columns >= N) and C in accumulate mode
(fp32: two masked two-word loads on a doubled index with half-granular masks), masked stores op17258 for
C. Buffers beyond A and B are filled with NaN in the harness, so an unmasked out-of-range read would poison
the result; C beyond the extent carries sentinels that must survive.

**Accumulator groups and displacement bases.** Tiles are grouped to the register budget (load/store fields
reach R127), double-buffered A/B tuples when they fit, and when a tile's displacement would exceed the
16-bit field a base register idx + rows x ld is built by shift-add (`based`).

**Matrix (`tlower_matrix.json`, 21 cases x 3 trials, half operands, compact row-major buffers, no host
packing, no host tables):** 16x16x16, 32x32x64, 48x40x48, 32x8x16, 64x64x64, 64x64x128 acc, 128x96x64,
128x128x128 equal matmul2d on identical buffers and the reference; 1x1x1, 1x1x16, 17x19x16, 33x65x17,
7x130x3, 50x37x80 (+acc), 64x63x64 (+acc), 96x80x50, 48x32x100, 100x100x100, 200x40x40 - shapes the
library refuses - equal the reference with sentinels intact. 21/21. Remaining limits: the container
(body room and metadata still from an Apple-compiled template - deliverable 5), operand types other than
half (int8 needs a one-word load form), transposes (deliverable 4).

## 36. Batch 28: our own container - the first tensor kernels in this project with no Apple-compiled byte

Goal deliverable 5 (`ownimage.py`, `run_own.py`, `hybrid.py`, `own/<shape>/{program.o, program.lib.metallib,
program.arc.metallib, contract.json, abi.json, plan.json, field-ledger.json, result.json}`, `own_matrix.json`,
`gputime/gputime.mm` gained `gt_run` with a binding base).

**Path.** The general lowering's code is authored through the repository's public entry
`scanlink.link(Kernel, emission ABI, binding_offsets, program_contract)` with a typed `ProgramABI` built by
`from_dict` (version 1, code digest and size, entry 64, the END + filler header as prologue, bindings,
every instruction's offset/length/opcode from Apple's decoder, `execution = {simd_width 32, tensor true}`,
empty constant pool, an `argument_state` naming per_kernel_slot_1 as not stated) and an emission ABI shaped
like the compiler's own (`results/g17-tensor-common-delivery-v5/abi.json`: abi_version 5, forms, register
count, system register 130 for the SR_SIMD_ELEM read, pk_extra/pk_values 15/16, launch block). The linker
then derives __GPU_METADATA (the TENSOR class: register count in slot 0, slot 44 enable, binding records),
__GPU_LD_MD, __GPU_ARCH_LD_MD and __GPU_STATS_MD, builds the object, the library and the archive. Nothing in
the image is copied from an Apple compile: the header is the repository's convention, the body comes from the
field-map and ledger encoders (their witnesses are measured bit positions and, for op998/op10282, an
instance's unmapped mode bits - the account, as for every certified ledger form), the metadata from `mdgen`.
The image is run through `gputime/libgputime.dylib` with buffers at Metal indices 1, 2, 3.

**Two conventions the repository holds that the first attempt got wrong, recorded because the refusals
were named:** `Kernel.code` is the program AFTER the entry and `text_of` prepends the prologue, so passing
header + body produced a text whose entry word was END (the kernel retired at once and wrote nothing -
localised with `hybrid.py`, which packaged the same code with Apple's metadata and with the repository's
metadata through `g17obj/g17mtlb/g17arc` directly and ran both correctly); and the tensor metadata class is
measured for binding indices (1, 2, 3) at pointer offsets (0, 2, 4) - the descriptor bytes in the code are
4 x rank either way, so only the launch-side binding indices move (`tensormetadata.py` refuses (0, 1, 2)
by name: "unmeasured indexed tensor binding signature").

**Two limits found.** (1) A plan that allocated up to R127 executed wrongly (50x37x80: 0/3 at budget 128,
3/3 at 127, 126 and 124); the lowering's budget is 126 here. Batch 31 (`regcount_arms.py`) separates the
two readings: the DECLARED count is not the cause - with the code using R0..R123, counts 124, 127, 128,
160 and 224 patched into slot 0 and slot 32 all execute 3/3, and slot 32's low byte (0x01/0x02) is inert -
so the boundary is in the use of the top registers R124..R127, not isolated further. (2) Programs shorter than the measured tail
refuse: "a tensor program of 25 instructions carries no slot 32; no witness has that shape" (16x16x16 and
16x16x32) - the class will not extrapolate the layout to a shape no Apple witness has, which is the right
refusal and leaves the smallest kernels to the template path.

**Matrix (`own_matrix.json`, 14 images x 3 trials, half operands, compact row-major buffers, NaN beyond
A and B, sentinels beyond C):** 32x32x64, 48x40x48, 64x64x128 acc, 128x128x128 equal matmul2d on identical
buffers and the reference; 1x1x1, 17x19x16, 7x130x3, 24x40x48 acc, 33x65x17 acc, 50x37x80, 64x63x64,
96x80x50, 100x100x100, 200x40x40 - shapes the library refuses to compile - equal the reference with the
sentinels intact. 14/14. The headline holds for the half path: arbitrary non-multiple-of-16 row-major
GEMMs are lowered and executed by this repository's own image with no host packing and no
Apple-generated native code. Still open toward the goal's acceptance: operand types other than half,
more than one simdgroup, transposes (deliverable 4), the handoff (6).

## 37. Batch 29: transposed operands - the library's way, reproduced

Goal deliverable 4. **Study.** `mm_ta`, `mm_tb`, `mm_tatb` (matmul2d with transpose_left / transpose_right on
the 32x32x64 base) decode to the same instruction count as `mm_base` (45 non-filler instructions) with two
differences only: the transposed operand's op12674 loads keep the SAME index register and the SAME row-pair
displacements (0 and 2048 = +8 rows, then +32 = +16 columns, +4096/+6144 = +16/+24 rows) as the other
side's untransposed operand - the stored K x M matrix A^T is read exactly as an untransposed B would be,
rows k and k+8 into one lane - and the MMA's type code for that operand is 34 instead of 2 (the +32
transpose bit of section 25). No layout-shuffling instruction appears; the conversion from the B-style
fragment to the A operand happens inside the unit under the transpose bit.

**Lowering.** `tlower.lower(..., transA, transB)`: with transA the A-tile loads take rows 16k + 8p of the
stored matrix at column 16mi (the roles of the row and column bounds swap in the masks), with transB the
B-tile loads take rows 16ni + 8p of the stored B^T at column 16k, and `mmaenc.mma` sets the transpose bits.
Verified on identical row-major buffers against matmul2d with the descriptor's transpose flags (the
oracle's tensors declared on the stored extents): 32x32x64 tA, tB, tA+tB and 48x32x64 tA - 3/3 each; and
against the reference where the library refuses the shape: 17x19x16 tA+tB, 50x37x80 tB - 3/3 (files
`tl_*_tA*.json`, `tl_*_tB*.json`, run before the duplicate-session collision noted below). So the
transposed forms cost nothing beyond the type bit, and my section 3 remark that a transposed-A fragment
needs a strided gather described the register-side view, not the memory-side load the library uses.

**A process note that affects the receipts.** On 2026-09-17 around 00:28 a second instance of this
session, resumed from the same transcript, began editing the same files in `results/g17-tensorops-
recon-v1/` (tlower.py, memenc.py, run_tlower.py, ownimage.py, run_own.py) and running the same harness
in the same directory, whose scratch files and archive names are shared - so runs in that window
returned 0/3 for cases that had passed minutes earlier. The two instances split the work by message:
that instance keeps the lowering and its receipts (operand types and 1-4 simdgroups in the own-image
path), this one the documents and the handoff. Nothing from the collision window is claimed here; the
manifest is regenerated by the instance that owns the directory.

## 38. Batch 30: the acceptance matrix - types, simdgroups, transposes and partial shapes as repository images

Run by the second instance of this session after the split of section 37 (it owned `tlower.py`, `ownimage.py`,
`run_own.py`, `oracle_rm*.py` and the directory; this instance verified the receipts by file and wrote this
section). Receipts: `own/<label>/result.json` (35 images), `own_matrix.json`, `manifest.json` (3,568 files).

**Lowering as delivered.** `tlower.lower` now carries operand types (half/bfloat: one two-word load per row;
float: one four-word op12709 per row, or two masked two-word loads on a doubled index at a boundary; int8/
uint8: one one-word op12655 per row on a halved index, unmasked - so int8 refuses K not a multiple of 16
and odd leading dimensions, and reads only inside the buffer for partial M/N), transposes (section 37),
Apple's accumulate order (fadd, or iadd for int32), and 1-4 simdgroups: a 4-byte read_sr of SR_SIMD_GRP
into slot 1, and simdgroup s adds s x 16 x MT/sg rows (elements when transA) to its A and C index
registers; refused when M is not a multiple of 16 sg or the tile count per simdgroup is not a power of two.
Register budget 126 (plan up to R125). Two bugs found and fixed on the way, both by a 0/3 that named its
arm: the int8 base register added rows x ld to an index that counts halves (ld // 2), and the transA
simdgroup offset was rows x lda where it must be columns.

**Images (all through `scanlink.link`, no Apple-compiled byte, buffers at Metal indices 1, 2, 3; 3 random
trials each, every element compared, sentinels beyond C required intact, NaN beyond A and B).**
Equal to matmul2d on identical buffers AND the reference (20): 32x32x64 in half, bfloat x bfloat, float x
half, half x float, float x float, int8, uint8, bfloat tA+tB; 48x32x64 float tA; 48x40x48; 48x40x64
int8 (partial M/N with masked stores); 64x64x128 acc; 64x64x64 int8 acc; 128x128x128; 32x32x64 on two
simdgroups, 64x64x64 on four, 64x32x64 tA on two, 64x32x128 bfloat on two, 128x64x64 int8 on four,
64x64x128 float x half tA+tB on four. Equal to the reference where the library refuses the shape (15):
1x1x1, 7x130x3, 17x19x16, 17x19x16 bfloat x half tA, 24x40x48 acc, 33x65x17 acc, 33x65x17 acc half x
bfloat, 50x37x80, 50x37x80 float, 64x63x64, 64x63x64 acc float x half tB, 96x80x50, 100x100x100,
200x40x40, and 32x32x64 int8 x uint8 (mixed sign, which matmul2d does not compile). 35/35.

**Acceptance line, as stated in the goal:** arbitrary row-major GEMM -> tensor form chosen -> indices
computed -> masks computed -> accumulator groups allocated -> our image with slot 44 set -> the common
dispatcher -> hardware, bit-exact against Apple's matmul2d where it compiles the shape and against the
corrected arithmetic model everywhere, on non-multiples of 16, all four operand types, accumulate mode,
one to four simdgroups and transposes: met on this matrix. What the goal does not claim: the smallest
kernels (the class refuses programs shorter than its measured tail), int8 partial K, register counts
above 127 (section 36; the three-arm test - 127 and 128 and 224 with slot 32's low byte at 0x02 - is not
run), and integration itself, which is the compiler owner's and root's (`docs/g17-tensor-lowering-
handoff.md`).

## 39. Batch 31: the R124-R127 boundary closed - it is a known hardware limit, reproduced independently

Goal item 3b. `results/g17-tensorops-recon-v1/topreg_arms.py`: a 32x32x64 GEMM body with a probe appended
before END - a four-word load into a tuple based at register T, an ALU op on it (waiting on its scoreboard
slot), the GEMM's own last MMA storing into a separate 8-tuple at T8, then stores of both tuples to C's
tail - authored with a declared register count of 128 and executed. Loading AND storing through a tuple
based at T = 116, 120, 122 round-trips correctly (`topreg_T*.json`); at T = 123 (register window
R123-R126) and T = 124 (R124-R127) every word of the probe tuple reads back as the untouched sentinel -
not corrupted data, a store that did not happen. Isolated further: the failure is in the STORE, not the
load (loading at T = 124 and storing the same data from T = 120 round-trips; loading at T = 120 and
storing from T = 124 does not) or the accumulator write (the GEMM's own MMA into T8 = 120 and a separate
store from that same T8 both succeed). So the boundary is precisely "storing a tuple that reaches R126
or R127", independent of what wrote those registers.

This is `WIDE_MAX = 125` in `agxforge/g17/cc.py`'s register allocator, already measured by this repository:
`spike/accel/re/hireg.py` pinned one value into each of several high registers and read it back, one
dispatch per process, 107 preregistered every time - r124 and r125 correct, r126 and r127 WRONG, over
924 objects whose highest register is R125. My probe reproduces that result independently, in the
tensor-accelerator path with a declared count of 128 (not just at the allocator's own ceiling), and
narrows the earlier reading from "the declared count above 127 misbehaves" (section 36, retracted by the
`regcount_arms.py` arms) to "registers R126 and R127 do not round-trip a store, at any declared count".
The lowering's budget of 126 (register use through R125) was already correct; this section is why.

## 40. Batch 32: items 3a, 3c, 3d closed or bounded

**3a, int8 masked one-word load - bounded, refused, form named.** `int8_mask_probe.py`: every declared
opcode carrying a GPR16 operand in the neighbourhood of op12655 (the one-word load this lowering's int8
path uses) - op12657, 12658, 12666, 12667, 12684, 12690 (single- or narrow-word writes) and 12681, 12699
(two-word) - decodes with an EXPLICIT base register PAIR (GPR32tup2, e.g. R0_R1) as its address operand,
not the byte1 = 4 x rank binding descriptor that op12655 and every load/store this lowering emits uses.
op12681 carries a genuine Apple corpus witness (the only one of the eight that does) and confirms the
same addressing family even though it is the two-word form, not the int8 target. None of the single-word
candidates has a corpus witness; all are bit-certified only. So the masked one-word load this int8 path
would need does not exist in the addressing family this lowering implements; it exists in a family that
addresses through an explicit register pair, which needs a primitive this campaign has not built (loading
a bound buffer's device pointer into a register pair - `ArgumentState` names the pointer's location,
`pointer_words=2`, but nothing here has authored the load). Refused, with the form (op12657/12658) and
the missing primitive named, per the goal's own allowed outcome; not attempted further.

**3c, kernels shorter than the TENSOR class tail - refusal kept, no witness found.** No search for an
Apple-compiled tensor kernel under ~40 instructions was productive; real GEMMs are not written at that
scale (the smallest measured witness bound is the class's own tail, section 36/38). The refusal stands
as the repository's own guard ("no witness has that shape") and is the correct behaviour, not a gap in
this lowering: `tensorlower.lower_gemm` on a shape that small should and does raise from `k.image()`
rather than author bytes the class cannot certify. Bounded, not blocking.

**3d, masked-lane special values - closed by the existing matrix.** Every generated and every
repository-authored run in this campaign (`tl_*`, `own_matrix.json`) fills A and B beyond the true M/K
and K/N extent with type-appropriate NaN sentinels (0x7e00 half/bfloat, 0x7fc00000 fp32) before
dispatch, specifically so a masked-off element that leaked into a product would poison the result -
and 21 + 35 shapes passed bit-exact, including boundaries that cut a fragment's four-element mask into
a mixed pattern (17x19x16's N = 19 is not a multiple of 4, so some lanes carry three-of-four bits set).
So masked lanes are already measured to contribute nothing under a NaN-filled buffer, across every
operand type this lowering supports (half, bfloat, float, int8/uint8 with whole-tile masking only, per
3a). What was never isolated on its own - a single masked lane's contribution in ISOLATION, without a
whole-matrix GEMM around it - is not needed: the matrix result is the more general test and it already
passed. Closed.

## 41. Batch 33: correction to 3a (executed, not refused); 3c closed with a witness

**Correction to section 40's 3a.** The "explicit register pair (GPR32tup2)" that reading named as op12657's
address operand is the same allocator-printed `expr` operand every device load/store in this campaign has
at that position (agx3dis.c's own comment: an unresolved expr operand prints as the address the decoder's
allocator happened to place it at, not a register). Decoding `memenc.tload1w`'s own output (the unmasked
sibling, already in use throughout this campaign) at the same operand index gives `('expr', <address>)`,
identically to `memenc.mload1w`'s: `mload1w(8, 121, 0, mask, disp=0, width=1, slot=0)` decodes operand 3 as
`('expr', 48570081312)`, not a register. The earlier reading mistook one CENSUS WITNESS's raw bytes (which
happen to decode that unresolved slot as a register-shaped value on THAT witness) for a structural field;
it is decoder noise, the same category section 27's `agx3dis.c` documents for every load in this family.

The masked one-word load executes correctly on hardware, not "not attempted": `run_tlower.py 24 24 20
dt=int8,int8` and `20 24 24 dt=uint8,uint8 acc` (re-run fresh for this correction) give 3/3 against the CPU
reference on random operands - a wrong address register would not produce the exact mathematical answer
across three independent trials. `memenc.mload1w` (op12657) is field-mapped, encoded with the standard
byte1 = 4 x rank descriptor exactly like every other form in this family, wired into `tlower.py`'s int8
row loader, and int8/uint8 partial-K GEMMs (plain, accumulate, transposed) run bit-exact:
`tl_24x24x20_int8.int8.json`, `tl_20x24x24_acc_uint8.uint8.json`, `tl_24x24x24_tA_int8.int8.json`. Item 3a
is CLOSED, not refused.

**3c closed with a witness (`padwitness.py`), correcting "no witness found, bounded".** The class's tail
rule is exact and already in the repository (`mdgen.slot32_for`: no slot 32 at <= 30 instructions, slot 32
= 1 at 31, measured on 43 witnesses including both boundary values) - it does not need a new Apple witness,
it needs a program that reaches 31 instructions. Padding a small GEMM's body with the compiler's own
two-byte nop filler (already used as tail padding throughout this campaign) to 31 instructions before END
authors and executes correctly through the same `scanlink` path: 16x16x16 (25 -> 31 instructions, 6 nops),
16x16x32 and 16x32x16 (30 -> 31, 1 nop each), all 3/3 against the reference; 16x16x16 accumulate needs no
padding (35 instructions on its own, already covered by `own_matrix.json`). Every shape this lowering can
build now has a path through the repository's own image, including the smallest kernels.

**3d**: section 40's finding stands; `maskedspecial.py` adds a direct probe (NaN placed exactly in the
out-of-range half of a masked boundary tile's row-pair) confirming the same result with a narrower,
targeted stimulus rather than only the matrix's general NaN-sentinel coverage.

Items 3a-3d are closed.

## 42. Batch 34: composition - GEMM+scalar in one kernel (closed), GEMM feeding GEMM (bounded negative), K
remainder under a chain (already general, isolated once more)

Goal deliverable 2. `results/g17-tensorops-recon-v1/composition.py`, `composition.json`.

**A. GEMM followed by scalar code in the same kernel - closed.** A 16x16x16 GEMM body trimmed to end
exactly at its last MMA (no store, no END - `gemm_body` locates the MMA by scanning the plan rather than
trusting a fixed offset), followed by eight hand-encoded op998 fadds doubling the fp32 accumulator in
place (waiting on the MMA's own scoreboard slot the way any consumer would), then the ordinary tensor
store. The scalar code sees the live accumulator through exactly the state the lowering left - no new
load, no fresh wait beyond the one slot the MMA itself used - and the doubled result is bit-exact, 3/3.
This is the composition the goal names ("occupied scoreboard slots, live registers, last-use flags at the
boundary") and it holds with no special handling: a GEMM's tail is an ordinary instruction stream and
ordinary code can extend it.

**B. GEMM whose C feeds the next GEMM through registers - a real, isolated negative result, not a
refusal from lack of trying.** D1 = A1.B1 (16x16x16, fp32 accumulator) confirmed correct in isolation
(stored immediately after the MMA, bit-exact against the model - the control that makes the next result
mean something). Feeding those same eight registers directly as the fp32 A-OPERAND of a second MMA
(D2 = D1.B2, B2 loaded normally, no store/load of D1 in between) does NOT reproduce D1.B2, and does not
reproduce D1^T.B2 either (tried with the MMA's own transA bit). So an MMA's fp32 accumulator (D-fragment)
and an MMA's fp32 A-operand are not interchangeable byte-for-byte the way the row-major loader's A and D
addressing are - some difference between the two fragment conventions (word order within the eight-word
tuple, or something else) is real and not isolated within this batch. Composing two GEMMs without a host
round trip therefore needs one of: a repeated store/reload through registers via ALU (a per-lane
permutation of the 8 words, cheap and not yet built), or the actual fp32-A fragment order measured
directly (a `carrier.py`-style probe reading a known accumulator pattern back through an A slot, not
attempted here). Recorded as open, with the isolating control, rather than left as an untried refusal.

**Correction (section 125, added later).** The negative above (a hand-authored register-direct D -> A feed does not reproduce D1.B2) is superseded for
the library-compiled form: on one `matmul2d` object the library feeds op5106/5107's accumulators directly to op5100/5101's A operand with no instruction
in between and the results are exact (section 125). The hand-authored failure recorded here was not reproduced or explained; its cause was not isolated.

**C. K remainder under a chain - already general, isolated once more.** Every partial-K case in
`tl_*.json` already chains multiple 16-wide K groups with a genuine remainder in the last one (masked).
One more explicit receipt isolates just the K dimension: 16x16x100 (7 K-groups, the last 4 wide) and its
accumulate variant, both 3/3 against the reference (matmul2d refuses non-multiple-of-16 K). Closed by the
general lowering; nothing special was needed for this deliverable beyond citing it.

**"Two tensor programs in one worker" - named as unmeasurable with this session's instrument, not
guessed at.** The GPU-timed harness this campaign built (`gputime/gputime.mm`) enforces one Metal
compute pipeline per process by explicit design (`ac_pipeline_from_archive`'s own guard: a second
pipeline in one process returns Apple's cached, unpatched one - a defect this project hit and named
twice before, `ledger/g17-branch-conditional-back.toml`). Building a second pipeline in the same process
to test two tensor programs sharing a real Metal command queue is a new harness component this batch did
not build. What IS answered here, and is probably the fact this item actually wants: within ONE kernel
body (one pipeline), two independent tensor operations compose exactly as A and B above show - the
metadata (slot 44, register count) is per-KERNEL, not per-operation, so two GEMMs in one body share one
declaration, and B is the discriminating case for whether their DATA composes. Whether two SEPARATE
pipelines can be resident in one process at once is the common runtime's own question, for the runtime
owner to answer or measure with its own harness; not attempted here.

## 43. Batch 35: status table (goal deliverable 5)

Per the goal's own categories, each row measured / compiler-owned / bounded / open, with its receipt.
"Full tensor programming model" stays NOT COMPLETE until deliverables 1 and 4 (integration and the
awkward-workload campaign, both outside this session's boundary) report back.

| Area | Status | Receipt |
|---|---|---|
| M5 accelerator microarchitecture (slots/core, issue interval, latency) | MEASURED | section 31, `unit_timing.json` |
| MMA forms and rates (fp16/bf16/fp32-trunc/int8, legacy op2862) | MEASURED | section 31 |
| Row-major feed path (indices, addressing) | COMPILER-OWNED | section 32, `idxprobe.json`, `indexgen_check_*.json` |
| Partial-tile masking (op612, masked loads/stores) | COMPILER-OWNED | sections 32, 35, `op612_*.json` |
| Accumulator allocation (tile groups, register budget) | COMPILER-OWNED | section 35, `tlower.py` |
| Transposes | COMPILER-OWNED | section 37, `tl_*_tA*.json` |
| Metadata enable (slot 44) | MEASURED, COMPILER-OWNED at authoring | section 34; `TENSOR` class already sets it |
| Native tensor image generation (no Apple-compiled byte) | COMPILER-OWNED | section 36, `own_matrix.json` |
| Arbitrary row-major GEMM lowering | ACHIEVED | sections 35-38, `tlower_matrix.json`, `own_matrix.json` (35 images) |
| Exact fractional/edge numerics | MEASURED | sections 4, 16, 17, 33 (NOT open, despite earlier informal tallies) |
| int8 masked partial-K load | CLOSED | section 41, `memenc.mload1w` |
| Register window R124-R127 | CLOSED (measured cause) | section 39, `topreg_*.json`, `WIDE_MAX=125` in `agxforge/g17/cc.py` |
| Class-tail refusal for small kernels | CLOSED (padding witness) | section 41, `padwitness.py` |
| Masked-lane special values | CLOSED | section 41, `maskedspecial.py` |
| Composition: GEMM + scalar code, one kernel | CLOSED for compiling and this campaign's own dispatch; OPEN for the production authoring path | section 42, `composition.json` (A); section 44, `compose_scalar_*.json`; section 48 (compiler-owned, byte-checked); section 61 - `scanlink.author` refuses every composed cell on the measured-singleton-SR130 rule, reproduced directly |
| Composition: GEMM feeds GEMM without a host round trip | CLOSED through memory (register feed: bounded negative) | section 44, `compose_chain_*.json`, `compose_shift_*.json`; section 42 (B) |
| Composition: K remainder under a chain | CLOSED (already general; measured under a chain) | section 42, `tl_16x16x100*.json`; section 44, `compose_*_17x32x19_32.json`, `compose_*_40x48x50_48.json` |
| Composition: two tensor programs in one process | MEASURED (8/8, two libraries, two pipelines) | section 44, `twoprog.json`, `gputime.mm` (`gt_pipeline_slot`) |
| Declared-but-unadmitted opcode forms (fp8/fp4/int4, 16-bit accumulate) | BOUNDED | section 27 |
| Integration: emit_gemm arm content (byte-exactness, hardware execution, both fixes) | MEASURED, byte-exact | section 46, `emitter_verify_rows*.jsonl`; peer `branch_arm_34c4e3f4.json` |
| Integration: caller wired | CLOSED | section 47, `75315b44` (supersedes `c01e92ee`) - `_select_op`'s tensor branch routes to `emit_gemm` when the registry refuses, and only its documented `"refused: "` ValueError is absorbed as a named refusal; any other exception (a real defect) now propagates instead of being silently reported as "not supported" |
| Integration: legacy multiply program's two refused strides (A 18 rows, B 64 rows) | MEASURED - not a limit of the general lowering; **LIFTED on main 2026-09-28, bounded** (the route serves strideA up to 1 MiB and strideB up to 512 KiB, the sweep's range; `cc.STRIDE_MEASURED_MAX`) | section 47, `stride_evidence.json` (3/3 exact, both), `stride_sweep.json` |
| Integration: 16x16x16 padding ported to the candidate's authoring path | OPEN | section 46 (the one item still refused there) |
| Integration: default-stride numerical defect (int8/float addressed wrong bytes) | CLOSED, hardware-confirmed | sections 55-57; root's replay, 3/3 both dtypes, 0 mismatches, `470ebeda`/`bc8d175b` |
| Integration: ABI facts for row-spliced stores, allocator pool after a refusal | CLOSED | section 52, root's `91c3f9fd`/`c0ad2c32` findings, both independently reproduced |
| Integration: three more stale registry refusals (N=64, K=48, K=256, bfloat) | CLOSED | section 54, all five variants reproduced exactly |
| Integration: capability inventory (195 -> 199 -> rehashed at the stride fix) | CLOSED | sections 49, 54, 55 - counts and diffs confirmed against git blobs each time |
| Integration: main-based branch published, pushed for review | CLOSED | section 60, `codex/g17-tensorops-integration-latest`, spot-checked fresh (hashes, inventory, CPU suite) |
| Integration: common-runtime dispatch, merge to main | with root | section 46 (merge is root's call, not this session's); the sequencing gap sections 57-59 tracked is resolved (section 59) |
| Awkward-workload campaign from main | with root/Codex | RUNNING - sections 48, 52, 54, 55, 57: multiple real compiler-owned defects found and fixed, 183/183 regression cases passing on root's own main-based checkout; no defect has needed a measurement from this session that the compiler owner could not supply themselves |
| **Spencer's milestone**: composed GEMM+scalar, ordinary compiler, authored through the ordinary path, dispatched through the ordinary runtime, validated on hardware from a cold `main` checkout, SR130 derived from a measured rule not a hard-coded special case | **ACHIEVED**, independently reproduced end to end on real hardware | section 80, `codex/g17-tensorops-main-integration` (based on current `main`, not merged): the re-keying landed (`ca99db69`, `130: 52` in the measured map, `(130,)`/`(130,156)` accepted and nothing else), a composed program now authors (reproduced: `scanlink.author` returns a real image), and the common-runtime dispatch (`1ddd9626`) is independently reproduced on this session's own fresh worktree - 3/3 bit-exact, `gpu_dispatched: true`, output hash matching the documented cold-checkout receipt exactly. Section 80's one bounded caveat (the map's byte1-only key is ambiguous for 13 of 385 byte1=130 forms) closed within the hour: the milestone's own dispatched program was decoded and confirmed to use the width-4 form where the entry is unambiguously correct, for a structural reason (this backend never emits the width-8 form here), independently reproduced. The byte1-key's generality limit remains, for a future arbitrary-program consumer. Scope explicitly narrow (one shape, two composition forms) and not yet merged - both stated by the branch's own docs, neither this session's to close |
| Full tensor programming model | NOT COMPLETE | pending the milestone above |

## 44. Batch 36: composition through memory (closed) and two programs per process (measured) - second instance

A second instance of this session (triad-b5) ran deliverable 2 concurrently with section 42 in the same
directory; this section records what it measured, and section 43's two affected rows are updated below.
Receipts: `compose.py`, `compose_{chain,shift,scalar}_*.json`, `own/compose_*`, `twoprog.py`,
`twoprog.json`, `gputime/gputime.mm` (extended), `tensorlower_smoke.py`, `tensorlower_smoke_*.json`.

**GEMM feeding GEMM without a host round trip - closed through MEMORY (section 42 B's own named
alternative), one kernel, one dispatch.** `tlower.lower` gained `binds`, `offsets` and `end`: which
Metal binding each operand reads or writes, a byte offset into that buffer (folded into every
displacement, or into the base register when it exceeds the 15-bit field), and whether to emit END.
`compose.py` concatenates two bodies: GEMM 1 (`A . B1 -> C1`, half operands, fp32 C1 stored into the C
buffer at offset 0, no END) and GEMM 2 (`C1 . B2 -> C2`, `a_type='float'`, `binds=(2, 1, 2)` so its A
operand is read back from the C BUFFER as fp32, B2 from the B buffer at an offset, C2 stored at an
offset), then END. No fence or barrier instruction of any kind is emitted between the two: GEMM 2's
loads simply follow GEMM 1's stores in program order, on the same scoreboard. Three arms:

- `chain`: C1 read back by the same lane mapping that stored it (each lane consumes its own stores).
- `shift`: GEMM 2's A offset is one row of C1 (`4 x N` bytes), so every lane consumes a row a
  DIFFERENT lane stored - the cross-lane store-to-load order is what this arm tests; the reference is
  `C1[1:] . B2` with a zero last row. A load that ran before the other lane's store would read the
  buffer's zero fill and fail the comparison (the check can fail: C2 would be a different matrix).
- `scalar`: as `chain`, with scalar code between the two GEMMs - a four-word load of this lane's C1
  words (slot 6), two op998 fadds doubling words 0 and 1 (waiting on slot 6), a four-word store of the
  tuple to a scratch region - so ordinary loads, ALU and stores sit between two tensor operations,
  sharing registers (R100..R103 above both plans) and the scoreboard.

Shapes (M x N x K1 -> K2 = N): 32x32x64->32, 17x32x19->32, 40x48x50->48 (K remainders 19 and 50 under
the chain, partial M and N under both GEMMs). Every arm and shape: C1 and C2 bit-exact against the
fp32 reference (`gemm_ref` with the 13-bit truncation of C1 as GEMM 2's fp32 A operand), 3 random
trials each; the scalar arm's scratch words exact for all 32 lanes. So the K-remainder and the
GEMM-feeds-GEMM items are closed by measurement; section 42 B's register-feed negative (an fp32
D-fragment is not an fp32 A-fragment) stands and is now bounded rather than blocking: the memory route
costs one store and one load per tile and needs nothing the lowering does not already emit.

**Two tensor programs per process - measured, not unmeasurable.** The one-pipeline guard in
`gputime.mm` was this campaign's own choice (the comment cites `spike/accel/accel.mm`'s function-hash
caching). `gt_pipeline_slot(lib, archive, kernel, slot)` and `gt_select(slot)` were added: each slot
holds its own `MTLLibrary`, binary archive and pipeline state, built with
`MTLPipelineOptionFailOnBinaryArchiveMiss` exactly as before. `twoprog.py` loads the 32x32x64 `chain`
image into slot 0 and the 17x32x19 `shift` image into slot 1 - both kernels named `k`, both from
`scanlink.link` - and dispatches them alternately, four rounds, one process: every dispatch produced
its own program's answer (C1 and C2 both exact against each image's reference, 8/8 dispatches). The
failure this arm could have shown - Metal returning slot 0's pipeline for slot 1's same-named function -
would have computed the 32x32x64 program on the 17x32x19 buffers and failed C1. It did not happen when
the two images come from two distinct libraries; the earlier hazard was two archives against ONE library
function. (The first run of `twoprog.py` failed C2 for the second image; the image on disk had been
built before `compose.py`'s buffer offsets were moved inside the displacement field, so the checker read
the wrong offset. Rebuilt, 8/8. Kept here because it is the shape of failure the arm is for.)

**Budget.** `tensorlower.REGISTER_BUDGET` in the directory was 124 (from the displacement arms in
`topreg_T124_*`, which see wrong data below displacement 16384 and no store at or above it); section
39's narrower result - a tuple that reaches R126 or R127 fails, `topreg_T122` exact - is the one the
acceptance matrix was measured under, so the budget is 126 (plan through R125) with the comment
rewritten to cite section 39. `tensorlower_smoke.py` re-runs `lower_gemm` end to end at that budget:
50x37x80, 96x80x50 accumulate, 128x128x128 float x half, 3/3 each (`tensorlower_smoke_*.json`; the
smoke's first reference used C as the initial accumulator and failed 0/3 on the accumulate shape - the
lowering follows Apple's order, product first then one fadd of C, as section 38 states).

## 45. Not yet done

The tensor loads' (op12674) slot field CLOSED BY MUTATION, section 90; the byte6[4:3] counter
CLOSED, section 91 (it is slot % 4); slot reuse/lifetime and the store's mask (section 28); byte0[7] CLOSED,
section 88; op11456 CLOSED COMPLETELY (role, HI, both remaining immediates - sections 82/85/86/87/89);
the tensor load's mode word and index-register formula CLOSED, section 92 (operand1's own further
tail bits left explicitly unresolved there); fp8/fp4 once the OS accepts Metal 4.1 libraries (section 27: no CPU this
libLLVM knows admits an fp8/fp4 MMA opcode, so the 4.1 gate may be the only route); the
lowering's next steps: index registers computed in the body (op17016/op423/op426 encoders), op612 masks
in the body, accumulator groups in the row-major path, dynamic K loops (sections 29-32);
transposed operands' address formula CLOSED (re-confirmed), section 93 - a separate mm_base/mm_ta
MMA-count discrepancy found there is flagged, not resolved; the
multi-MMA role, if any, of flags bits 33/41/47 and the byte4 code CLOSED, section 94: still inert,
tested on a real chained accumulation this time, not just a single MMA.

## 46. Deliverable 1: the real candidate identified and verified, independent of a peer instance

`origin/claude/g17-isa-cartography` (dfe29e63) is superseded, not the branch going forward. Three
independent readings agree: mine (`verify_encoder_copy.py`, 4/8 files hash-mismatched), a peer's
byte-level diff of the same four files, and the compiler owner's own account, all resolve to one
story - that copy predates batches 32-33 (no `memenc.mload1w`/`tload1w`, no `padwitness` fix, no
`tlower` `binds`/`offsets`/`end`, and a registry docstring that still refused int8 partial tiles
after section 41 closed them). A peer's hardware run of the arm as landed there
(`branch_arm_dfe29e63.json`, `run_own`'s dispatch-and-compare, 12 shapes) got exactly the two
refusals staleness predicts (`48x40x64` int8 partial tile, `16x16x16` class-tail) and nothing else -
consistent with "old copy," not with a defect in the emitter arm itself.

The candidate that matters is `origin/claude/g17-tensorops-emitter` @ `34c4e3f4`, based on
`origin/main` @ `70efbfe6` (confirmed: `git merge-base origin/main origin/claude/g17-tensorops-emitter`
== `70efbfe6`; `git merge-base --is-ancestor ... origin/main` fails - not yet merged). It touches
`agxforge/g17/tensor.py` (+57/-2, the emit_gemm arm) plus new `faddenc.py`, `fieldmap.json`,
`indexgen.py`, `ledgerenc.py`, `lower.py`, `memenc.py`, `memmap.json`, `mmaenc.py`, `ownimage.py`,
`tensorgemm.py` (`tensorlower` renamed), `tlower.py`, and `docs/archive/g17-tensorops-emitter-candidate.md`.

Verified independently here (detached temp worktree at `34c4e3f4`, read-only, nothing under
`agxforge/` edited, worktree removed after): `agxforge.g17.tensor.emit_gemm(...)` against this campaign's
own `tensorlower.lower_gemm(...)`, identical arguments, 12 shapes spanning half/bfloat/float/int8
(both a full and a K-partial int8 tile), accumulate, transA, transB, and simdgroups 1/2/4 - **12/12
byte-identical bodies**, including both int8 cases (the stale branch failed exactly those two; this
one does not, because the re-copied `memenc.py` carries `mload1w`/`tload1w`). Also verified:

- The `dis.py`/`resource.py` sys.path-shadow fix holds: `python3 -c "import numpy; import
  agxforge.g17.tensor"` succeeds standalone from the candidate's worktree (rc=0). The prior pattern
  (inserting `agxforge/g17` itself onto `sys.path`) would have let the local `dis.py` shadow the
  stdlib module the moment numpy or anything importing it ran; this candidate imports
  package-relative instead and the crash does not occur.
- The `ownimage.build_from` fix holds: `emit_gemm(32, 32, 64, lda=33, a_type='int8',
  b_type='int8')` gives a body hashing `4c425fc7ff3b82e6` (1028 bytes), and its `.image()` contract
  now reports `code_sha256` beginning `4c425fc7ff3b82e6...` - the authored program matches the body
  the caller was given. Before this fix the fallback path (`ownimage.build()` recomputing
  `lda=K, ldb=N` from `M, N, K` alone) would have silently authored the *contiguous* program instead
  (hash `470ebeda...`) under the non-contiguous caller's name. `build_from` implements what
  `Lowered.image()` was written to mean; no rework needed from this side.
- Still refused on this candidate, unchanged from the stale branch: `16x16x16` at authoring time -
  `Missing: 'a tensor program of 25 instructions carries no slot 32; no witness has that shape and
  its tail is unmeasured'`. The `padwitness.py` nop-padding fix (section 41, batch 33) has not been
  ported into this branch's `ownimage`/authoring path. This is the one concrete, bounded, still-open
  item on the candidate itself; a peer instance reproduced the identical refusal independently on the
  same sha.

What acceptance still lacks, beyond the candidate's own content, is three-part and belongs to the
compiler owner and root, not to this session: (a) no caller - nothing under `agxforge/`, `tools/` or
tests on this branch invokes `emit_gemm`, so "ordinary tensor IR -> emit_gemm" is not wired to
anything upstream of it; (b) every hardware run so far (this session's and the peer's) dispatches
through this campaign's own `gputime`/`run_own` harness at bindings 1-3, not the common runtime; (c)
`origin/claude/g17-tensorops-emitter` is a branch, confirmed not an ancestor of `origin/main`.
Receipts: `emitter_verify_rows.jsonl`, `emitter_verify_rows2.jsonl` (this session); a peer's
`branch_arm_34c4e3f4.json`, `candidate_lda_34c4e3f4.json` (independent, convergent).

## 47. The two strides the legacy multiply program refuses are not a limit of the general lowering

The compiler owner wired a caller (`origin/claude/g17-tensorops-emitter` @ `c01e92ee`, supersedes
`34c4e3f4`): `_select_op`'s tensor branch tries the registry first and authors through
`tensor.emit_gemm` -> `tensorgemm.lower_gemm` only when the registry refuses, emitting the body as
`tensor.wholekernel` rows and dropping the trailing END (the compiler emits its own for `ret`).
This closes gap (a) from section 46 - `17x19x16` and `50x37x80` compile through `emit()` to
programs byte-identical to this campaign's own lowering (`371b0a8d`, `7ebefc28`), an independent
check of the 12/12 body comparison from the caller's end of the pipeline rather than a direct call.
`test/test_g17tensorwholekernel.py` (new on this commit) pins three refusals by name, one of them
already known here (the `16x16x16` class-tail, not closed - the padwitness port is still absent, as
section 46 says) and two new ones: `strideA=576` bytes (18 rows of the legacy multiply program's
32-byte tile) and `strideB=2048` bytes (64 rows), both refused with "six-bit" in the message. The
source is `agxforge/g17/tensorlower.py` (the pre-existing, unrelated module the registry's OTHER
multiply path uses - not this campaign's lowering, which the compiler owner renamed `tensorgemm.py`
on this branch to avoid exactly this collision): `MULTIPLY_IMM_MAX = 63` and a comment measured
there already - "15 rows (60) generates, 18 rows (72) does not" - so both named strides genuinely
overflow that program's six-bit immediate. The question put to this session: the general lowering
forms addresses by shift-add, not that immediate, so is it actually bound by the same limit, and is
either stride in the acceptance matrix already? It was not (checked; neither appears in
`own_matrix.json`/`lowering_matrix.json`).

Measured directly (`stride_evidence.py`, one child process per case - `gt_pipeline` is one-pipeline-
per-process, same convention as `build.py`): `tlower.lower(32, 32, 64, lda=288, ...)` (strideA=576,
the byte-stride the IR carries divided by 2 for half, matching the conversion
`g17-tensorops-emitter` already does) and `tlower.lower(32, 32, 64, ldb=1024, ...)` (strideB=2048),
each authored through the same `scanlink.link` path as `ownimage.build` (verified byte-identical to
`ownimage.build`'s own output at the natural stride before trusting the strided runs), buffers laid
out at the true physical stride with an NaN-sentinel-filled pad beyond the logical K/N columns of
every row (an address bug that read into the pad would poison the result, not merely mismatch it) -
dispatched on hardware. **Both: 3/3 exact against the fp32 reference, 0 mismatches.** (First attempt
read a `float16` array into a `uint16` buffer by direct assignment rather than `.view(np.uint16)` -
numpy casts values instead of reinterpreting bits, which silently corrupts the operands and was
caught by an all-NaN, not a near-miss, result before this number was reported.) This is not a
matmul2d comparison - the oracle kernel template here is not stride-parameterised - but the CPU/
arithmetic reference is independent of physical layout, and exact agreement over random operands at
a stride the addressing had never been run at is the discriminating measurement: an off-by-row-
length or wrong-window read would not produce a bit-exact match by chance. So: the general lowering
is not bound by the legacy multiply program's six-bit immediate at either named stride; the refusal
is specific to that other path, and the registry can route both to the tensor path instead of
leaving them refused, without new evidence beyond what is here. Receipt: `stride_evidence.py`,
`stride_evidence.json`.

The compiler owner did not lift either refusal unilaterally - `g17regress.
_tensor_contract_delivered` asserts both raise by name, and rewriting another owner's pinned
assertions is not this session's call either. Correction to how this got recorded here the first
time: their message to root did not start out scoped. The first version recommended lifting both
outright; it was narrowed to three options (a narrow allowlist of exactly the two tested points, a
lift once a sweep states a measured range, or the wholesale lift they no longer recommend) only
after the point below about their own guard being a REASON match, not a two-value allowlist - so a
wholesale lift would extrapolate from two points to every stride sharing that refusal text. The
narrowing was theirs once the gap was named, not something this session's evidence handed them
pre-scoped, and the record should read that way rather than as if the caution was there from the
start. Once narrowed, the three limits named so root's rule (whichever option they pick) is no
wider than what was measured: one shape only (32x32x64, not a sweep at either stride), this
campaign's own harness rather than the common runtime (the other open gap, unchanged), and that the
3/3 followed a self-caught false negative (the `.view(np.uint16)` fix) rather than a clean first
pass - offered so root can weigh it, not hidden. They also checked their own caller for the same
bit-reinterpretation pattern (none - its buffers are integer-typed throughout, forced by an earlier
finding that `op998` fadd flushes integer bit patterns) and separately confirmed `16x32x64` stays
refused by shape, not swept in by the stride result.

**The sweep, run rather than deferred.** Not more passes above the two already-measured points -
those confirm without discriminating - but a hunt for the first failure, aimed at the one place in
this campaign's own lowering where a stride's address computation changes mechanism:
`tlower.py`'s `based()`, `DISP_MAX = 32767` bytes. Below that a load/store uses a plain index +
displacement, exactly what the rest of this campaign already exercises constantly; above it, the
excess is folded into a base register by bit-by-bit shift-add (`const_times_ld`), a path far less
exercised and therefore the likelier place for a latent bug - and neither of the two points already
measured (576, 2048 bytes) came anywhere near it. Six points, three per operand, each in its own
process (`stride_sweep.py`; `gt_pipeline` is one-pipeline-per-process): a stride that straddles
`DISP_MAX` *within the same kernel* (row 30 of A at 1080 B/row stays under it, row 31 crosses - the
generated code visibly grows, 74 to 80 instructions, confirming the fold actually fired, not a
no-op path being silently reused), then two points deeper into base-register territory per operand
(64 KiB, and 1 MiB for A / 512 KiB for B). Shape held at 32x32x64 half/half throughout, to isolate
the stride variable. **All six: 2/2 exact against the reference, 0 mismatches - no failure found up
to a 1 MiB stride.** This does not prove there is no limit at all (the shift-add decomposition still
lands in a 32-bit hardware register somewhere, so a real ceiling exists near there in principle,
untested and not close to anything a real GEMM's leading dimension would reach) - it says the
mechanism transition itself, the specific place most likely to hide a bug, does not, up to three
orders of magnitude past the two originally-refused points. Receipt: `stride_sweep.py`,
`stride_sweep.json`.

*Decided 2026-09-28 (root): the guard is lifted, bounded by this sweep. `agxforge/g17/cc.py` routes a registry
refusal to the general lowering only when strideA <= 1 MiB and strideB <= 512 KiB (`STRIDE_MEASURED_MAX`);
past that the refusal stands and names the range. The two former six-bit points now compile through the
general lowering (`test_g17tensorwholekernel`, g17regress `_tensor_contract_delivered`). The 32-bit folded-
address ceiling stays untested, and the evidence is the campaign harness at 32x32x64 half/half, not the
common runtime.*

The caller advanced again to `75315b44` (supersedes `c01e92ee`): narrows the exception handling so
only the lowering's own documented `ValueError('refused: ...')` is absorbed as a named refusal;
anything else (a real defect) now propagates instead of being reported as "not supported" - a root
catch. Spot-checked here (three shapes including one int8, in a fresh detached worktree at
`75315b44`): bodies unchanged (`371b0a8d`, `7ebefc28`, `470ebeda`, matching section 46's numbers
exactly) - the exception-handling change did not touch what the lowering emits.

## 48. Composition landed on the candidate itself, and the awkward-workload campaign is live

The candidate advanced twice more: `6a63019d` (one tensor body composes with ordinary scalar work
under one allocator and one final END - the row-splice slice) and `81753cb2` (a float type on an
integer ALU form was an unexplained `KeyError` from a width table; it is a named refusal now).
Both independently verified in a fresh detached worktree, not just read.

**The row-splice mechanism.** The route's rows carry pre-encoded bytes with fixed physical
registers and no virtual defs, so the allocator could not see what they occupied and would colour
surrounding scalar values into the same registers - two writers, one register, no diagnostic. Fixed
by reading the physical registers out of the body's own decoded operands and publishing them as
`_occupies` on the first row; `Alloc.run` excludes that set from its pool for the run and restores
it in a `finally`. This is deliverable 2's ground - GEMM plus scalar code in one kernel - now done
through the actual compiler allocator rather than only this campaign's own `composition.py`
harness. Checked directly, not taken on the commit message's word: extracted the
`tensor.wholekernel`-tagged bytes from all three composed cells (`gemm-residual`, `gemm-activation`,
`tensor-scalar-pressure`, all built on `17x19x16`) and compared against
`tensor.emit_gemm(17,19,16).body` with its own END stripped (the route drops the lowering's END and
emits its own for `ret`, same as section 46 already established) - **all three: byte-identical**,
1182 bytes each. The splice changes nothing about the tensor computation itself; it only makes
room for scalar work around it. A second tensor operation is still refused by name (two complete
kernels cannot both be the whole program) - `chain-same`, `mixed-shapes` and `shared-buffers` are
unchanged and still open, which matches this campaign's own composition matrix (section 44):
GEMM-feeds-GEMM without a host round trip works through memory, not through one allocator's
registers.

Ran the test suite myself rather than trusting "Ran 25 tests OK" as reported: 24/25 pass in a fresh
worktree; the one failure (`test_the_witnessed_shape_emits_no_wholekernel_row`) is a
`FileNotFoundError` for `results/g17-tensor-common-witness-v1/tensor-common.o` - a gitignored
witness data file this campaign's own worktree convention never carries into a fresh checkout, not
a code defect. Suite takes ~66 seconds (real lowering work, not a hang - it looked like one under a
60s timeout the first time, which is itself worth recording precisely rather than reporting a false
"hangs").

Fixed at `271d3c80`: the case now skips, by name, when the witness fixture is absent, rather than
erroring - and says explicitly that "the witnessed shape is untouched by the route" goes unverified
when it does, rather than reading like a pass. The compiler owner also disclosed, unprompted, that
every prior "N OK" they had reported (here and to root, including an earlier 62-of-69 figure) came
from an environment quietly made non-fresh by a symlinked `results/`, so the numbers were true but
not representative of root's fresh-checkout replay. Verified both directions independently rather
than taking the fix on its own commit message: a fresh worktree with no `results/` gives `OK
(skipped=1)`; copying the real witness data in gives `OK` with the case actually running, zero
skipped. The skip is genuinely conditional, not a disguised permanent pass.

**The KeyError fix.** Its own commit message names the source precisely: "Root's matrix found a
real compiler-owned failure on the requested GEMM-plus-residual workload" - **the awkward-workload
campaign (deliverable 4) is running.** `add` on two f32 values reached `cc._width`, which covers
i16/i32 only, and raised an unexplained `KeyError('f32')` several frames from its cause. Correctly
classified as the program being malformed (float value through an integer ALU form), not the route
being limited - the same workload written with `fadd` composes (1238 bytes, 101 rows, one END,
verified). Refused by name in `_width` itself, for every integer form, so the diagnostic is uniform;
an unrecognised type still raises `KeyError` deliberately, so a selector bug does not get dressed as
a user error - exactly the discrimination the `75315b44` exception-narrowing (section 47) was for,
applied one level deeper. This one reached the compiler owner directly and was fixed in their own
file without needing a measurement from this session, which is the campaign design working as
intended - not everything that fails needs to come here.

Status table's "awkward-workload campaign from main" row was still saying "not yet run"; it is
running and finding real, correctly-classified, compiler-owned defects. Updated below.

## 49. Two of root's three library-compatibility blockers fixed; the third is named and left to integration

`dfa36485`: root's own serial reference gate found three library-compatibility rules the encoder
copy broke. Two fixed on the candidate - verified independently in a fresh worktree, not taken on
the commit message:

- **No module under `agxforge/` may touch `sys.path`.** All nine copied modules had an insert; the
  `dis.py`/`resource.py` shadow fix (section 46) had only removed the worst of it. Checked directly:
  `grep sys.path` across all nine now matches only the comment explaining the rule, no code: `ROOT`
  is a plain constant, script execution moved to `python -m agxforge.g17.<name>`.
- **No non-Python files under `agxforge/`.** `fieldmap.json`/`memmap.json` moved to
  `isa/g17-tensor-mma-fieldmap.json` / `isa/g17-tensor-mem-memmap.json`. Checked directly: both load
  from their new path with the claimed key counts (10, 23).
- Regression spot-check after both moves (the kind of edit that breaks imports silently): the same
  four bodies checked repeatedly since section 46 - `d0f00e7d309739c7` (32x32x64),
  `371b0a8d6d8b1986` (17x19x16), `7ebefc28918989c3` (50x37x80), `470ebeda6a7a0b46` (32x32x64 int8) -
  unchanged.

**Not fixed, by design, and named rather than hidden:** `test_g17capcompiler`'s committed-refusal-
count check (195 committed, 199 actual - the four refusals this candidate's row-splice and f32 work
added) needs `tools/g17capcompiler.py build`, which needs
`results/g17-source-conversion-runtime-v1/campaign.json`, currently absent. The compiler owner
correctly declined to generate it from their own worktree: 486 of 488 entries under their `results/`
are symlinks into the shared checkout, so writing there risks another session's owned state - the
same class of hazard this campaign has hit before with concurrent writers to shared `results/`
(section 1's duplicate-session protocol). Left for root's own integration step, which already
extracts that archive; expected count stated in advance (199) so whoever runs it has something to
check against, not just a number to produce.

## 50. A path constant root caught on a clean checkout, backported so the branch is correct on its own

`c6a1abb7`: `ownimage.OUT_ROOT` was an absolute module-level path into the compiler owner's own
`results/g17-ownimage`, which existed in their worktree only because their own probe runs had
created it - invisible there, flagged by the library's own unresolved-path-constant check on a
genuinely clean checkout. Root caught it independently on `codex/g17-tensor-awkward-workloads`
(`efc4823b`) and the compiler owner backported the same fix to this candidate rather than leaving
it to integration to paper over, on the reasoning that otherwise whichever branch is taken decides
whether the bug ships. Verified directly in a fresh worktree rather than taken on the diff: building
32x32x64 through `ownimage.build` now lands at `<checkout>/results/g17-ownimage/32x32x64` - resolved
against the actual checkout root, not a leftover absolute path - with `code_bytes=974`, matching the
unchanged body this campaign has checked at every prior round.

Named but explicitly not this session's to chase: the compiler owner flagged, and did not claim to
explain, that a clean `origin/main` checkout's path-constant scan lists `model.META`/`regs.META` as
unresolved while a clean checkout of their candidate does not, despite neither `model.py` nor
`regs.py` being touched. Outside tensor lowering entirely (a general library-compatibility scanner
artifact, not a refusal or a numeric claim this campaign owns) - recorded here only so it isn't lost,
not investigated.

The third time the same root cause showed up in one evening (the witness-file skip, the 62-of-69
fixture count, now this): a worktree with `results/` populated by symlink or by its own probe runs
verifies less than it appears to, because root's replay checkout has only what git carries. Worth
carrying forward as a standing caution for this campaign's own worktree too, not just the
compiler owner's - `results/g17-tensorops-recon-v1/` here is real (not symlinked), but the general
point, that a convenient local environment can quietly stand in for a clean one, is not specific to
either session.

## 51. Applying the peer's refinement to this campaign's own tool: verify_from_main.py had exactly one such path

The peer's own generalisation of the OUT_ROOT finding (section 50) - "which paths does this check
read that git does not carry" is the question, symlinks and probe output being two ways to answer it
wrongly, not the only ones - pointed at this campaign's own `verify_from_main.py` directly: it named
one script worth checking, since it would "pass for you and fail for root exactly as [their] suites
did" otherwise. Checked, and it was right to ask.

`verify_from_main.py`'s acceptance-matrix slice did `from run_lowered import gemm_ref, trunc_f32`
purely to reach two pure-numpy functions, dragging in `run_lowered.py`'s own imports of
`transpose.py` and `lower.py` (and theirs) - four more files, all in this campaign's gitignored
`results/`, none of them git-carried, all needed only because the acceptance-matrix branch had never
actually run (0/4 present, every round so far, so the import was never exercised). Separately,
`gemm_ref`/`trunc_f32` were imported but never called: the slice only checked that `emit_gemm`
did not raise, not that its output was numerically correct - the docstring's claim of "comparing
against matmul2d and the arithmetic model" was not yet true of the code.

Fixed both: `gemm_ref`/`trunc_f32`/`mma_np` are now defined verbatim in the file itself, and the
acceptance-matrix slice authors each case through the installed `emit_gemm(...).image()`, dispatches
it on hardware (one process per case, `gt_pipeline` being one-pipeline-per-process as always in this
campaign), and compares against `gemm_ref` for real. One dependency remains and is named rather than
removed: `gputime/libgputime.dylib`, the compiled dispatch harness - not a probe artifact or
convenience symlink, but the thing this tool actually needs to run anything on hardware, copied
alongside the script the same way this campaign's other dispatch tools always have been.

Tested the actual intended use, not just the code in isolation: a fresh detached worktree of
`origin/claude/g17-tensorops-emitter`, with only `verify_from_main.py` and the `gputime/` directory
copied in at the matching relative path (nothing else from this campaign's `results/`) - **all four
presence checks OK, all five acceptance-matrix cases `eq_reference: 3/3`.** Also re-ran the negative
case (this campaign's own branch, where nothing is present) to confirm the honest "0/4, NOT YET"
path still holds after the rewrite. One bug caught in the process, the same shape as section 46's
first one: `k.image(name=...)` with a custom name changes the KERNEL FUNCTION baked into the image,
not just a folder label, so dispatching against the hardcoded function name `k` failed with
"computeFunction must not be nil" until the custom name was dropped.

## 52. A real ABI defect from root's own reproducer, fixed and independently reproduced

`4b448977`: disposition of root's fresh-main reproducer (queue commit `91c3f9fd`) - a genuine
compiler-owned defect, not a refusal. `17x19x16`, `17x19x19` and `50x37x80` all compiled and
correctly emitted tensor stores, but the ABI layer captured `has_stores=False`,
`writes_buffer=False`, `register_count=0`, `pk_values={}` for every one of them, so
`ProgramABI.contract()` refused the (false) binding/write inconsistency and none of the three
shapes could be authored at all, despite being authorable all along.

Two causes: (1) the write predicate matched tensor stores by form name + phase string
(`tensor.authored` + phase `"readout"`, the registry path's row), while row-spliced stores carry
phase `"general lowering"` - fixed by classifying any tensor row whose *opcode* is a tensor store,
per this file's own stated rule; (2) `register_count` read only `_defs`/`_uses`, which a row-spliced
body has neither of (it publishes `_occupies` instead), so the count came out 0 for programs naming
60+ registers. Also folded in: root's separate finding (`c0ad2c32`) that the occupancy refusal in
`Alloc.run` raised before the `try`/`finally` that restores the pool, so a reused `Alloc` after a
refusal carried a permanently shrunken register pool - a defect in the row-splice mechanism section
48 verified the success path of but not the refusal path of.

Independently reproduced rather than taken on the commit message's table: built each of the four
shapes through `g17cc.compile_function` in a fresh detached worktree (with the common witness copied
in so the registry-control case runs too) and read `.abi_inputs()` / `.contract()` directly -
**every field matches exactly**: `17x19x16` 1186 bytes/True/True/76/`{15:1,16:1}`/OK, `17x19x19`
1882/True/True/84/same/OK, `50x37x80` 5382/True/True/124/same/OK, `32x32x64` (registry control)
1276/True/True/72/same/OK - the control's 72 is untouched, confirming the fix is scoped to the
row-splice route only. Ran `test_g17tensorwholekernel.py` fresh: 32/32 pass in ~92s (the two new
test classes, `TheGeneralRoutePublishesCompleteABIFacts` and `TheAllocatorPoolSurvivesARefusal`, are
part of that count, so the refusal-path pool restoration is covered by the same run rather than
taken separately on faith). `test_g17tensorlower.py` still fails outside a fully-populated
environment (needs witness fixtures beyond the one directory copied in) - the same class of gap as
section 48, not a regression from this commit, which does not touch that file.

Remaining failures named as unrelated to this fix and left to root's integration, per the commit:
`librarycompat`'s `corpus.LIBACCEL` (an extraction/environment baseline root asked be kept, not
papered over), `pinnedfiles`' stale `cc.py` digest, and `capcompiler`'s extraction-dependent cases
including the still-199-vs-195 refusal count (section 49) - unchanged by this fix, which adds no
refusals.

Correction to how this got characterised in this session's own reply, since the compiler owner
pushed back on it precisely: this was not a case of review missing an untested edge. The compiler
owner wrote the row-splice test suite (section 48) themselves, and it exercised only the success
path (`test_the_allocation_is_coherent...` compiles three cells that all succeed) - so the `finally`
looked correct to a reader because no test asked what happened on the branch it did not cover. A
test suite that only confirms can make code look reviewed when the failure path was never built at
all, which is a sharper claim than "a reviewer would miss this" and belongs in the record precisely.
Root's finding came from asking what the other branch does; the compiler owner's fix added a
success-path companion alongside the refusal test for the same reason - without it, the refusal
case alone could be satisfied by an implementation that never narrows the pool in the first place.

## 53. The candidate's own provenance table corrected to read from git blobs, not a working tree

`fe299c14`: root caught that `docs/archive/g17-tensorops-emitter-candidate.md`'s provenance table listed
hashes taken at copy time from the compiler owner's own (gitignored, unreadable-by-anyone-else)
working tree, presented as the candidate's current contents - true of the two untouched census
files, false of the nine modules edited since. Regenerated from git blobs at the candidate tip
(`4b448977`) instead, so the table is reproducible by anyone with the ref. Cross-checked against
root's own independent hash of `faddenc.py` (`4b1af19d7b4f38a0`) before publishing, not just
recomputed alone.

Spot-checked here, independently, on four entries: `git show 4b448977:<path> | sha256` for
`faddenc.py`, `tlower.py`, and both census files at their new `isa/` paths - all four match the
corrected table exactly, including the two "byte-identical" census files. Purely a documentation
fix; no code changed.

## 54. Root's awkward-workload assignment: three more stale refusals found, and the inventory rebuilt properly

Two more commits from the same thread, both independently reproduced end to end.

**`795c25b5`.** Root's queue assignment reported "exactly one check failed: `{'N': 64}` compiled"
against the old `_tensor_matmul_precise_refusal` test, which asserts five shape variants all refuse
and raises on the FIRST that does not - so `N=64` was only ever the first stale assertion after the
still-correctly-refused `M=16`, not the extent of the gap. Measured directly here, in a fresh
worktree with the common witness copied in, all five variants against `g17cc.compile_function`:

    {'M': 16}            REFUSED (unchanged - the class-tail shape)
    {'N': 64}            COMPILED  1750 bytes  21ef4228ac8cee29
    {'K': 48}            COMPILED   836 bytes  9f1d130b94d4df80
    {'K': 256}           COMPILED  2974 bytes  ae361232b6ac8da9
    {'a_dtype': 'bfloat'} COMPILED  974 bytes  3938602d834a9bae

Every byte count and hash matches the commit's own table exactly. The mechanism is right, not just
the numbers: `K=48`/`K=256` were refused for REGISTRY reasons (not a step-32 tile multiple; a loop
bound field overflow) that do not apply to the general route, because the general route does not
loop at all - it emits a fully unrolled body, so the field in question is never written. Confirmed
the discriminating control myself: the witnessed `32x32x64` shape still takes the registry route
(`forms = {tensor.authored, tensor.inherited, end}`, no `tensor.wholekernel`), so the two routes
remain genuinely distinguished rather than the new assertions being vacuously true of everything.
Also notable, and recorded as the compiler owner corrected it rather than as this session first
characterised it: the point of trying the forbidden evasion (adding the shape to a pinned denylist
instead of fixing the check) was not to test their own discipline against the assignment's wording -
it was to test whether the CHECK ITSELF catches the evasion, since the next person to hit this case
will not have read the queue doc that forbids it. Confirmed the denylist gives "0 of 1 preserved"
with the refusal text quoted, then reverted - a property of the check that outlives the person who
wrote it, not a demonstration of resisting a shortcut. This closes the three previously-unnoticed
served shapes without weakening the one refusal that is real (section 46's `16x32x64`/class-tail
family).

**`455fc764`.** The inventory count named in section 49 (195 committed vs 199 actual) is now
rebuilt, not retyped - `isa/g17-capabilities-compiler.json`'s `refusals.count` reads `199` at this
commit (`195` at the prior one, checked directly against both git blobs) and the diff is exactly
the four new refusal messages (`795c25b5`'s and section 52's), zero removed - checked by grepping
each string's presence at `455fc764` and absence at `fe299c14`, all four confirmed both ways.
Notable for how the compiler owner got there, not just that they did: `build` fails hard in their
own worktree (`evidence.extract` correctly refuses a symlink-escaping destination rather than
writing through 426 of 488 `results/` entries that are symlinks into shared state - the guard doing
exactly its job), so the rebuild ran in a separate clean detached checkout populated only by
extracting the repository's own committed `evidence/g17-generality.zip` (12,943 members) - making
the document reproducible from the repository alone, the property a fresh-main replay actually
needs. Not re-run here in full (that extraction is heavy and the diff-by-message check above is
sufficient corroboration for what this campaign needs to know), but the reasoning and the checked
diff are independently sound.

**One finding surfaced, correctly left unfixed and unclaimed by the compiler owner:**
`tools/g17frontier.py` wraps its entire GPU-receipt loop in a bare `except Exception`, so one
unextracted archive (the same missing evidence this commit's own build needed) silently drops
EVERY retained receipt from the frontier's population rather than just the one archive's - a
masked-failure pattern one level worse than a missing fixture, since it changes what a ranking is
computed over without saying so. Not this candidate's defect (the affected program's identity is
unchanged, sha-confirmed both ways) and not this session's file to fix; recorded here only so
it is not lost, exactly as the compiler owner recorded the `model.META` discrepancy in section 50.

## 55. A real numerical defect: default row strides assumed halves for every dtype

`0ace23d9`, disposition of a hardware failure root's own run found (the integration's default-stride
`32x32x64` int8 case disagreed with the reference on 1024 of 1024 elements, three trials). The
defect: `cc.py` built the default A/B byte strides as `K * 2` / `N * 2` unconditionally - correct for
half/bfloat, wrong by 2x for float, wrong the other way for int8/uint8 - and wrong *silently*: a
too-small stride still compiles, passes every structural check, and publishes a valid ABI, so nothing
short of a hardware run against a real reference could have caught it. Fixed by computing the default
from the declared element width (a single module-level width table, promoted out of two local copies
that already existed for the ragged-stride check and the leading-dimension conversion) rather than a
hardcoded `2`.

**Compile-identity, independently reproduced.** Built all five dtypes through `g17cc.compile_function`
with no explicit stride (the exact code path the defect and fix both live in) in a fresh detached
worktree: `half` -> `472570faa7d227ee`, `bfloat` -> `265241da8b926e18`, `float` -> `bc8d175bf6df0513`,
`int8` -> `470ebeda6a7a0b46`, `uint8` -> `73052d818a6f4be5` - **all five match the commit's table
exactly**, including the two dtypes that do not move (half, bfloat - confirming the fix reaches only
the dtypes that were actually wrong) and the int8 hash landing on this campaign's own lowering's
value (`470ebeda6a7a0b46` - the same body `emitter_verify_rows.jsonl` and `verify_from_main.py` have
each already dispatched and matched 3/3 against the reference, independently, more than once this
session).

**What that does and does not establish, stated as precisely as the compiler owner stated it.** The
`470ebeda6a7a0b46` code bytes are proven identical, and identical machine code necessarily executes
identically on the same hardware - so the fixed compiler's output for this exact case is covered
*transitively* by this session's prior hardware runs of that body. That is real corroboration, not
nothing. It is not the same claim as dispatching *this program, authored through cc.py's own image
path*, which `G17Program.to_image` needs a reference container for that this session does not have
on hand - that part of the gap is real. The other reason first given here was not: reading
`contract().bindings` off a test IR built with buffers declared at 0/1/2 (this campaign's own scratch
convention, borrowed from the existing test file's helpers) showed indices 0/1/2, read as a
structural mismatch against this campaign's own 1/2/3 scanlink convention. Wrong inference, checked
by declaring the SAME test at 1/2/3 instead: `Binding.index` simply echoes whatever slot the IR
declares, so it is not cc.py's convention at all, and it agrees with this campaign's own once the
same slots are used. The compiler owner also measured, and this session did not, that `element_type`
does not track `a_dtype` for good reason - this backend accesses 2-/4-byte scalars only, so an int8
tensor operand is carried in a wider declared buffer by construction, not by a bug, and that is now
pinned by a case in their own suite. Corrected here rather than left standing. What remains open is
only the reference container `to_image` needs - root's acceptance explicitly still wants that rerun,
and the compiler owner explicitly did not claim it either ("compile identity... does not establish a
runtime result"); left open here for the same reason, not forced under time pressure into a
reconstruction of a container this session does not have.

`c24c7c46`: the inventory rebuilt again in the same clean-checkout-from-committed-evidence way as
section 54 - refusal count unchanged (199 -> 199, checked directly against both blobs, matching the
commit's own stated invariant: a stride fix changes what the compiler computes, not what it
declines, so a rebuild that added a refusal would mean a numerical bug had been turned into a
refusal instead of fixed).

**`fad86df5`, a correction the compiler owner made about their own prior report in this same
section.** The six new stride-regression test classes 0ace23d9 reported as passing ("32 whole-kernel
tests OK (6 new)") were appended below `if __name__ == "__main__": unittest.main()` and were never
collected - the 32 was the pre-existing count, unchanged, which is what should have been the tell.
Moved above the guard: 32 -> 38, all OK, independently reproduced here fresh (38 tests, ~96s). What
stands regardless, and is unaffected by this: the five-dtype hash table was produced by compiling
and hashing each case directly, not by those tests, and this session (section 55, above) and root
each reproduced all five independently before this correction landed - the substance was never
resting on the tests that turned out not to run.

## 56. The reference container was never missing - packaging confirmed, only dispatch remains

Correction to section 55's own closing line, which said this session and the compiler owner had
"agreed not to rush" a reconstruction because `to_image` needed a reference container neither had.
That was wrong on the facts: `~/.cache/agxforge/agx` (a shared machine-level build cache, not a
project-gitignored directory) holds 25 Apple-compiled tensor containers, including
`abi-v1-tensor-common-9e3449b707`, the one section 55's int8 program needs. `c5e8641a` measured the
packaging directly rather than continuing to treat it as unavailable; independently reproduced here,
byte for byte:

```
container exists: True
code sha: 470ebeda6a7a0b46  len: 1002
materialised image: 30576 bytes
window: (64, 1002) - matches probe.entry + len(code) exactly
compiled bytes found in the image, byte-identical at that offset
implied text base matches the container's own text head
```

Also reproduced the near-miss the compiler owner disclosed rather than hid: their first check
indexed the materialised file directly at `probe.entry` (a text-*relative* offset) and got a false
`False` on two containers, briefly reading as "packaging is broken" before they found the text
section actually starts at file offset `0x3ba0`, not 0. The corrected case asserts the relationship
(window equals entry+length, and the offset implied by that relationship lands on the container's
own text head) rather than one offset that happens to work - the same shape of self-caught
measurement bug as this campaign's own numpy `.view()` mistake and the compiler owner's earlier
`float32`-into-`uint16` near-misses, worth naming for the same reason each of those was: it would
have been reported as a defect in the wrong thing.

Ran the suite fresh with the same shared cache available here too: 39/39 (the new case runs, not
skipped, since this machine has the same `~/.cache/agxforge/agx`).

**What is left, now genuinely narrow.** Not the container, not the packaging - only a dispatch of
the packaged image and its comparison against a reference. The compiler owner declined to run that
dispatch unilaterally, and this session is not attempting it either, for the same reason: root holds
the specific harness that produced the original int8 default-stride failure (1024 of 1024 elements
wrong), and that before/after comparison - not just any dispatch of any correctly-packaged image -
is the one that actually closes the finding. The instruction bytes remain covered transitively (this
session has dispatched `470ebeda6a7a0b46` 3/3 against reference, independently, more than once
tonight, under this campaign's own packaging); what stays open is a runtime result for *this*
packaging, through *that* harness.

## 57. Root ran the dispatch: the stride fix is hardware-confirmed, closing the thread this campaign opened

Root's own queue (`origin/codex/g17-tensor-awkward-workloads`, `00fd16de` and `5501afb6`) closes
what section 56 left open. They cherry-picked `0ace23d9` and `c24c7c46` onto their main-based
integration checkout and reran the exact harness that produced the original failure: the formerly
failing default-stride `32x32x64` **int8 and float** controls both now match their independent
reference in all three hardware trials, 0 mismatches per trial, preservation true, at code hashes
`470ebeda...` and `bc8d175b...` - the same hashes section 55 and section 46 have independently
carried all night. Then `tools/g17regress.py --skip-suite`: 183 of 183 regression cases preserved.
Root's own words, worth keeping rather than paraphrasing: "these are candidate-checkout results;
they do not establish main, common-runtime, or release acceptance" - appropriately scoped by the
same owner who will decide the merge.

This is the comparison section 56 named as the one that actually mattered - not any correctly-
packaged dispatch, but a rerun of the specific instrument that produced the 0/3 failure. It did, and
it passed. The instruction bytes' transitive coverage (this campaign's own repeated dispatches of
the same hashes) and the packaging measurement (section 56) are no longer standing in for a runtime
result; there now is one, from the harness that matters, and this session's and the compiler owner's
shared decision not to reconstruct a substitute dispatch was the right call precisely because it
turned out to be unnecessary.

**One sequencing gap, flagged by the compiler owner as their own defect reaching root's run, not
root's.** Root's focused slice reports "32 wholekernel tests" - confirmed directly against `00fd16de`'s
own text - which is the count *without* the six stride-regression tests, because root cherry-picked
`0ace23d9`/`c24c7c46` but not the later `fad86df5` (which moved the guard, 32 -> 38) or `c5e8641a`
(38 -> 39). The hardware result is untouched by this - it came from dispatching, not from those
tests - but root's replay as it stands has no regression guard on the stride defaults. Named here so
it is checkable if root's next pass still reports 32: if so, the guard has moved back and the six
cases are silently unguarded again in exactly the way `fad86df5` closed once already.

**The pattern, generalised.** Three near-misses in one evening (this campaign's `float16`-into-
`uint16` bit-cast, the compiler owner's text-relative-offset read, and the compiler owner's own
naming of it) share one shape: *reading correct bytes through the wrong base, stride or element type
yields a confident wrong answer that reads as a finding about the subject, not the reader.* The
sharpest addition, from the compiler owner directly: repeating a wrong read against a second subject
can feel like a control and is not one, if the reader itself is the constant - two containers giving
the same false mismatch corroborated nothing, because the same bug produced both. The operational
rule worth carrying forward in this campaign's own work too: locate a payload by searching for it
and assert the *relationship* that makes an offset meaningful, not the offset itself - and name which
base an offset is relative to at the point it is computed, not just in a comment above it.

Status table: composition and integration content are now fully closed on the merits (hardware-
confirmed on both sides of the campaign boundary); only the merge itself, the sequencing gap above,
and the two items already named in section 46 (16x16x16 padding port, common-runtime dispatch)
remain.

## 58. The sequencing gap materialised exactly as flagged, and root diagnosed it correctly

Root's queue (`f58e4c96`, immediately after `5501afb6`): having also cherry-picked `fad86df5` and
`c5e8641a`, `test_g17capcompiler.py` now fails 5 of 39 with 1 error, all at the same check -
`evidence hash mismatch: agxforge/g17/cc.py`. This is the sequencing gap section 57 named, materialised
in a slightly different shape than predicted (an evidence-hash mismatch rather than a stale test
count, since `test_g17capcompiler.py` checks file hashes directly rather than a test count).

Root's own diagnosis is correct, confirmed by reading the actual diff: `fad86df5` added seven lines
to `agxforge/g17/cc.py` - a comment, no functional change, explaining exactly the binding/`a_dtype`
distinction section 55's correction above now also states - placed at the point the strides are
derived. A comment-only edit still changes the file's bytes, and the capability inventory pins exact
file hashes as evidence, so the inventory `c24c7c46` built (before this comment existed) no longer
matches `cc.py`'s current content once `fad86df5` is applied on top. Root names this precisely: "an
inventory refresh issue, not a new compiler behavior," and the fix is mechanical - regenerate
`isa/g17-capabilities-compiler.json` from the corrected checkout once more before integration, the
same rebuild the compiler owner has now done twice (sections 49, 54).

Not a new finding about the compiler; a confirmation that the sequencing gap this campaign flagged
was real and would surface, and that it surfaced exactly the way a hash-pinned evidence system
should - loudly, at a specific named file, rather than silently passing on stale evidence. Relayed to
the compiler owner in case root's queue branch is not already on their watch list, since the fix is
theirs to run and they are the one who has run it before.

Root's own follow-up (`a364d0dc`) confirms the rebuild itself is clean (51 capabilities, 57 gaps, 344
evidence - the standard figures) and, correctly, has not committed it: the ledger's `base_commit`
needs to name the actual landing tree, not a temporary cherry-pick tip assembled for testing. Nothing
left open on this specific gap beyond the merge itself deciding what that tree is.

## 59. The compiler owner's own sequencing gap, closed a third time, verified fresh - and a near false alarm of this session's own

`e4ebfe70`: the compiler owner rebuilt the inventory a third time (base `c5e8641a`, 51/57/344,
refusals 199) - not because I asked, but because they read their own `fad86df5` and recognised it
had committed exactly the sequencing gap they'd flagged in root's replay one commit earlier: seven
comment-only lines added to `cc.py`, which still moves the file's pinned evidence hash. Confirmed
directly: `git show e4ebfe70:isa/g17-capabilities-compiler.json`'s `base_commit` reads `c5e8641a`,
refusals `199`, capabilities `51`, gaps `57` - exact.

**This session's own near-miss in verifying it.** First run of `test_g17capcompiler.py` in a fresh
worktree (only the tensor-common witness copied in, as in every prior round) failed 7 of 39 with 5
errors - which would have read as "the rebuild is broken" if reported at face value. It was not: the
suite also needs the full evidence extraction (`evidence/g17-generality.zip`, 12,943 members, the
same archive the compiler owner's own clean rebuilds use), which this session's usual spot-check
setup has never carried before because no prior round needed `test_g17capcompiler.py` to actually
pass, only the inventory's own counts to match. Extracted it properly (`agxforge.g17.evidence.extract()`,
read-only, symlink-escape-checked, refuses to overwrite differing content - the same guard section
54 already described) rather than reporting the failure: **12,943 members verified, 12,932
extracted, 39/39 OK in ~71s.** Caught before writing anything down as a finding, not after - but
worth recording as a fourth instance of tonight's pattern (section 57): a failing check first read
as a fact about the subject, and was actually a fact about an incomplete reader.

**One near-miss the compiler owner disclosed from their own side of this same commit**, kept here
for completeness rather than left to their queue alone: their first rebuild attempt ran after a
`git checkout` that had printed "Aborting" (a dirty working tree from the previous rebuild's leftover
file) - HEAD silently stayed at `0ace23d9`, the build still succeeded, and it printed the identical
"status passed, 51 capabilities, 57 gaps, 344 evidence" line a correct run would print. Nothing in
that output said it had built the wrong commit; only reading the `git log --oneline -1` requested in
the same command surfaced the wrong subject line. `base_commit` is now checked explicitly rather than
inferred from `build` succeeding - the general fix, not just a rerun.

## 60. Root published a main-based integration branch - pushed for review, not yet merged

`4f0860c9` and `6a9cee79` on root's queue: a new branch, `origin/codex/g17-tensorops-integration-latest`,
carries the full candidate through `c5e8641a` plus the regenerated compiler/integration/unified
ledgers, rebased onto current `main`. Confirmed directly rather than taken from the queue doc: its
merge-base with `origin/main` equals `origin/main`'s own tip (genuinely current, not stale), and it
is NOT an ancestor of `main` (not merged - root's own words: "pushed for review... no cold-gate result
has been claimed").

Spot-checked in a fresh detached worktree of the branch itself: the three body hashes this campaign
has carried all night are unchanged (`371b0a8d6d8b1986`, `7ebefc28918989c3`, `470ebeda6a7a0b46`), and
the capability inventory reads `refusals: 199, capabilities: 51, gaps: 57` - consistent with every
prior rebuild, on root's own `base_commit` (a cherry-pick, so a different SHA than the candidate's,
same content). Also ran the newly-named CPU runtime suite fresh: `python3 -m pytest -q
test/test_triad.py` - **38 passed in 5.69s**, matching root's reported 38/38 in 5.78s. Did not rerun
every suite root named (wholekernel/tensorlower/package-compat/surface-ledger/pinned-files/compact-
compile/compact-image/the 183-case regression) - this campaign has already independently verified the
tensor-specific ones repeatedly across sections 46-59, and re-running the full battery again on a
branch that is itself a cherry-pick of content already checked would not add proportionate
information.

This is the closest the campaign has been to the acceptance line stated in the original goal: ordinary
tensor IR routes through `emit_gemm`, on a branch built from current `main`, with hardware-confirmed
numerics and a passing regression suite. What remains is exactly what section 46 named at the start
and nothing new: the merge decision itself, which is root's and not this session's to make.

The compiler owner independently checked the same rebase claim (main tip equals the branch's
merge-base with main, 19 commits ahead, not merged) and went a step further than this session did:
a field-by-field diff of root's regenerated ledger (`e4040967`) against their own (`e4ebfe70`) -
identical apart from `base_commit`, zero of 51 capabilities differing, no refusal message differing
in either direction. Two independent checkouts, two different cherry-pick lineages, the same content
- which demonstrates the inventory build is a pure function of its source rather than merely having
been observed to pass twice.

**Closing synthesis, the compiler owner's own words kept rather than paraphrased weaker:** "every
one of [tonight's defects] was found by an instrument disagreeing with another instrument, never by
either of us reading our own output more carefully... that's the argument for keeping three lanes
checking each other rather than one lane being careful." The stride defect needed root's hardware
against an independent reference; the ABI-facts defect needed the form/opcode distinction disagreeing
with a phase-string check; this session's own near-misses (the numpy bit-cast, the incomplete
evidence extraction) each needed a second run from a state the first one did not have. None of them
were caught by care alone. Recorded here as the one methodological finding that outlasts any single
commit in this section.

## 61. Composed programs compile correctly but the production authoring path refuses them - a real, bounded gap this campaign's own dispatch route does not share

Root's awkward matrix (`13bf80de`) on `codex/g17-tensorops-integration-latest`: the row-splice
composed cells (GEMM+activation, GEMM+residual, f32 residual, GEMM+eight scalar adds) all compile
with the tensor and scalar rows present and one final END - matching section 48 exactly - but
`scanlink.author` refuses every one of them "at the measured singleton SR130 metadata class." The
pure cells (32x32x64, 17x19x16, 17x19x19, 50x37x80, float, int8) all author fine.

**Reproduced directly, not taken on the queue doc.** Built the same `residual(1)` program section 48
used, compiled it, and called `scanlink.author(program)` - the public single-call authoring entry
(`author(compiled_program)`, which derives the kernel and binding offsets from the contract itself
rather than three separately-stated descriptions that nothing checks agree): `Missing 'tensor
metadata class requires measured singleton SR130'`. The pure case, same call, authors cleanly. Read
the actual check (`agxforge/g17/tensormetadata.py::layout`): it requires `system_registers == (130,)`
exactly - a MEASURED rule, its own comments say, from witnesses that are all pure single-tensor
kernels. Confirmed why the composed case fails it: `program.abi_inputs()['system_registers']` is
`(130, 156)` for the residual case, not `(130,)` - the scalar `threadgroup_position_in_grid` builtin
reads its own system register (156) alongside the tensor body's 130, and the singleton rule has
never been measured against a program that reads both.

**Not the same finding as the 16x16x16 class-tail gap (section 46), though the same shape**: that
one is about a body too short to carry a slot; this one is about a body that reads more than one
system register. Both are cases where the metadata-authoring layer's rules were measured only from
pure-tensor witnesses and have not yet been extended to what row-splice composition newly makes
possible - not lowering defects, authoring-layer gaps one level downstream of where sections 46-52
already checked.

**Why this campaign's own dispatch of composed programs (section 48, `composition.py`) was not
already blocked by this**: this campaign's authoring path (`ownimage.py`, and this section's own
`stride_evidence.py`/`verify_from_main.py`) hand-builds its `abi` dict directly rather than calling
`tensormetadata.layout()`, so it never exercised this specific measured-singleton check. That is why
section 48's hardware dispatch of composed programs succeeded on this campaign's own harness while
the production authoring path refuses the same compiled bytes today - two different authoring
routes to the same instructions, only one of them gated by a rule this narrow. Composition is
correct (bytes, and this campaign's own dispatch); it is not yet *authorable through the production
path*, which is the more precise claim.

Not this session's to fix (`agxforge/g17/tensormetadata.py` is production code); relayed to the
compiler owner as a bounded, precisely-located gap rather than left for them to rediscover from the
queue doc alone.

## 62. Correction to section 61: not a lagging rule, a missing measurement - and the compiler owner is right to leave it refused

`a1f825bc`: the compiler owner corrected this session's own framing precisely, and the correction
changes the disposition, not just the wording. Section 61 called the SR130 singleton check "a rule
that hasn't caught up with what row-splice composition now makes the compiler emit," which reads as
"widen the tuple check." That would be wrong, and the compiler owner's reason is structural, not
cautious: the metadata class's slot-29 fill writes exactly ONE entry, at one position, value `52`,
measured on eight independent single-system-register witnesses. Nothing measures what a SECOND
entry looks like, or whether the first entry's value changes when a second is present. Widening the
check would author a program declaring one system register while reading two, with the second slot
simply unwritten - a wrong image that authors cleanly, which is worse than the refusal it would
replace. Root's own standing instruction (the composed-image refusal must not become a compiler
success without a measured metadata class) is exactly right for this reason, and the compiler owner
left the refusal in place rather than relaxing it.

What changed is the message: it now states the mechanism (one measured entry position, not a
policy) and what would lift it (nine slot-29 entry values - 156, 157, 158, 160, 161, 162, 164, 165,
166 - each measured reading its register ALONE; none measured two-together), rather than only the
requirement.

Reproduced directly, not taken on the commit message: `tensormetadata.layout()` with
`system_registers` in `{(130,156), (130,130), (156,), ()}` all refuse with the new message text;
`(130,)` still authors. Confirmed the inventory claim too - `tensormetadata.py` does not appear
anywhere in `isa/g17-capabilities-compiler.json` (grepped the whole file), so the "no fourth
rebuild needed" claim holds. Ran the suite fresh: 41/41 in ~125s (count moved from 39, the two new
cases - the pure `(130,)` arm asserted first, against ground truth rather than an empty population,
and the four-input refusal).

Section 61 updated in spirit by this section rather than edited in place, so the record shows the
correction happened rather than reading as though the framing were right from the start.

## 63. Correction to section 62: the claim it recorded from the compiler owner was itself false, and the compiler owner found this before this session did

`d09c2a77`: the compiler owner's own message in section 62 ("no witness measures what a SECOND
entry looks like... two-entry vectors are not measured") is false, refuted by this project's own
committed measurement, and they corrected it themselves within the hour rather than leaving it
standing. Reproduced directly:

```
mdgen.MEASURED_SYSTEM_REGISTER_COUNTS == (1, 2, 3)
mdgen.slot29_entries((160, 161)) == [80, 81]
mdgen.slot29_entries((160,))     == [80]
mdgen.slot29_entries((161,))     == [81]
```

Two-entry (and three-entry) slot-29 vectors are witnessed - measured 2026-09-09 from five compiled
four-buffer controls, and the entry follows the REGISTER, not the position (161 alone gives `[81]`,
not the second position's `[80]`). Section 62's own confident restatement of "no witness shows two
together" was wrong at the time it was written, and this session did not catch it - the compiler
owner did, by running the one check that would have refuted it before publishing rather than after.

**The real gap, narrower than section 62 stated:** SR130 has no entry in the map at all -
`slot29_entries((130,))` refuses by name, listing exactly the nine registers it covers
(`156, 157, 158, 160, 161, 162, 164, 165, 166`), none of them `130`. The tensor class works around
this by hardcoding its one entry's value to `52` rather than deriving it. So the composed-program
refusal needs one number, SR130's own slot-29 entry value, not a new two-entry layout - and that
number is sitting unread in the build cache: 1,318 already-compiled objects, including Apple's own
compilation of the witnessed `32x32x64` shape, read exactly `(130, 156)`. Confirmed the map and the
refusal directly:

```
slot29_entries((130,))       -> refuses by name, lists [156,157,158,160,161,162,164,165,166]
layout(system_registers=(130,156)) -> still refuses, message now names the real gap
layout(system_registers=(130,))    -> still authors
```

Nothing authors that did not before - the compiler owner explicitly did not add SR130 to the map,
correctly treating that as a reviewed decision rather than something to slip in while correcting a
different mistake. Ran the suite fresh: 42/42 (count moved from 41, asserting the corrected reason
and the two facts that make it honest - the two-entry layout measured, SR130 absent by name - so
that adding SR130 later would need to touch this test, not bypass it silently).

**Worth keeping for its own sake, since the compiler owner named it as a pattern in their own
behaviour rather than a single slip:** three absences asserted in one session that a check
overturned - the reference container (section 56), this one, and one that happened to hold anyway.
The shared shape is writing the absence into the artifact before running the one-line check that
would refute it. This session's own equivalent pattern (sections 46-59) was reading correct bytes
through the wrong base or an incomplete environment; the compiler owner's is asserting a negative
without checking it first. Different failure, same remedy: run the check before publishing the
claim, not after.

## 64. Spencer's precise milestone, and a lead toward SR130's measurement rather than a claimed value

Spencer stated the immediate milestone exactly: a composed GEMM+scalar program produced by the
ordinary compiler, authored into a repository-owned image, dispatched through the ordinary runtime,
and validated on hardware from a cold `main` checkout, with the system-register metadata derived
from a measured rule rather than a hard-coded tensor special case. This supersedes the scattered
phrasing this doc had been using for the same handful of items (section 46's "common-runtime
dispatch," section 57's "cold checkout," sections 61-63's SR130 gap) with one compound, falsifiable
target. Recorded here as the acceptance line this campaign now tracks; relayed to the compiler owner
and root in those exact terms.

**What is this session's to contribute toward it, and what is not.** The measurement itself -
reading a real object's slot-29 bytes to find SR130's entry value, and deciding whether to add it to
`SLOT29_BY_SYSTEM_REGISTER` - touches `agxforge/g17/mdgen.py` and is exactly the kind of "deliberate
reviewed decision" the compiler owner has twice now declined to make casually. What this session
can contribute without guessing is a lead, checked rather than asserted:

`~/.cache/agxforge/agx/ac2-32x32x64/s.metal` is a real, already-compiled (`metal`/`metallib`, not this
campaign's own authoring) kernel - confirmed by reading its source directly - that runs a `matmul2d`
(reads SR130) and also reads `tg.x` from `threadgroup_position_in_grid` (SR156), storing three
derived values. Its sibling `ac-32x32x64` declares the same `tg` parameter but never reads it, so it
should read SR130 alone. `ac2-32x32x64`'s `__GPU_METADATA` is 488 bytes against `ac-32x32x64`'s 412
(`ac-32x32x64`'s 412 matching `TENSOR`'s declared base size exactly, per `classbytes.metadata()`,
read directly from both objects here) - a real, concrete confirmation that the witness the compiler
owner named ("1,318 objects read exactly (130, 156)") is not a guess: this one object, read and
checked, is one of them.

**What is NOT safe to conclude from this pair, and why the naive next step was not taken.** The two
objects' metadata bytes diverge starting at offset 164 - well before slot 29's own region - with
several fields shifted by different amounts (12, 8, 12, 16 bytes at neighbouring positions, checked
directly), not one field moved by four. `ac2`'s kernel does more than read one extra system
register: it also loads `pad1[tg.x]`, computes two derived values, and writes three extra results,
so its larger metadata plausibly reflects several structural changes at once (register count,
binding record shape, possibly the slot32/tail selection this doc's own mdgen.py comments describe
elsewhere), not a single isolated slot-29 widening. Picking two offsets from `TENSOR`'s base layout
and reading them out of `ac2`'s bytes would have produced a specific-looking number from a confounded
comparison - the exact failure shape this project's own ledger warns about repeatedly ("one
branchless witness cannot separate confounded candidates") and has retracted findings over before.
Not done here. A clean measurement needs either a genuinely minimal pair (same instruction count,
same binding shapes, differing only in whether SR130 or SR156 is read) or an explicit accounting of
every simultaneous difference - work for whoever holds this ledger's full history, not a byte-diff
rushed under a milestone's time pressure.

## 65. SR130's slot-29 entry measured directly from the population - the tensor class's hardcode was right, the map's KEY was the defect

`9730ca86` (`origin/claude/g17-isa-cartography`, not the tensorops-emitter branch - this is a
general ISA measurement, landing where that kind of finding belongs): the compiler owner did not
need the confounded `ac2`/`ac` pair section 64 stopped short of. Instead of differencing two
programs, they selected the population - every build-cache object with exactly one `read_sr` and
one slot-29 entry, so each entry has an unambiguous source - and read SR130's entry directly:
**372 such objects declare byte1=130 (the register `decode_sr` reports) and every one's slot-29
entry is 52.**

**What this also found, bigger than the number itself:** `mdgen.SLOT29_BY_SYSTEM_REGISTER` is keyed
on `decode_sr`'s `sr` field alone - byte1, whole and alone - and byte1 does NOT identify the
register. Three real forms share byte1=0x82 (130) and disagree: byte3=0x38 gives entry 52 (the
common case, 99 of these objects), byte3=0x20 gives entry 50 (9 objects), byte3=0x40 gives entry 10
(4 objects). A number-keyed map cannot express a register whose entry depends on a byte the key
throws away - which is exactly why the tensor class hardcodes 52 rather than deriving it: deriving
it from the existing map was never going to work, for any register, not just this one.

**Independently reproduced end to end, not taken on the commit message:**

- Read `isa/g17-slot29-entries.json` directly: keying on `(byte1, width, byte3)` gives 35 keys, 0
  ambiguous; keying on `(byte1, width)` alone gives 20 keys, 4 ambiguous - the control that makes
  "zero ambiguous" a real claim rather than an artifact of a key fine enough to separate everything.
  All nine of the committed map's registers appear in the population and all nine agree (156->0,
  157->1, 158->2, 160->80, 161->81, 162->82, 164->48, 165->49, 166->50 - checked row by row).
- Read the raw per-key rows rather than only the summary: byte1=130 has FOUR rows, not one -
  `(width=4, byte3=6)` 273 objects -> entry 52, `(width=8, byte3=56)` 99 objects -> entry 52,
  `(width=8, byte3=32)` 9 objects -> entry 50, `(width=8, byte3=64)` 4 objects -> entry 10. The two
  entry-52 rows sum to exactly 372 - the figure the compiler owner quoted - confirming it is a sum
  over two forms of the same register reading the same way, not a single row's count taken at face
  value. The width=4 form (`op14059/4`, this project's own 32-bit read_sr form per
  `results/g17-readsr8-v1`) is the one relevant to the tensor class's actual usage.
- Reran the tool myself against the full, live build cache: `python3 tools/g17slot29census.py
  check` -> "regenerable: g17-slot29-entries.json matches a fresh census" - the committed artifact
  is reproduced from the 23,029-directory population on this machine too, not merely internally
  consistent with itself.
- Ran `test/test_g17isamap.py` fresh: 57/57 in ~66s.

**What is still correctly not done, per the compiler owner's own account and confirmed by reading
the diff**: `mdgen.py`'s map is untouched, and the tensor class still refuses `(130, 156)`. The
census gives the entry value; re-keying the general map (and deciding whether byte3's actual bit
layout needs isolating first, which the artifact explicitly leaves `not_claimed`) is the reviewed
step Spencer's milestone still needs, and it is the compiler owner's file to change, not this
session's.

## 66. Correction to section 65's own number, and a precise scoping of what "measured" still needs

`7a7fc79b`: the compiler owner corrected the very summary this session had just relayed and folded
into section 65 - "372 objects say 52" was itself a coarse-key collapse, the same mistake the census
exists to detect. Confirmed directly: byte1=130 is four keys (matching section 65's own table), and
372 is `273 + 99`, two DIFFERENT forms of the register agreeing on entry 52 - real corroboration,
but a sum quoted as one population, not one. Recorded in the artifact itself as `worked_example`
(read directly, matches) rather than left in a commit message only, and it credits this session's
own read of the raw rows by name. Section 65 above already stated the two-row split rather than the
bare sum, so nothing here needed correcting on this session's side - recorded for the record's
completeness and because the compiler owner's own artifact update is itself worth having.

**More important: a precise statement of what "measured" buys, which narrows section 65's own
framing.** A tensor-class rule (not just a number) needs three things, per the compiler owner's own
accounting: (1) the slot-29 entry value - measured, 52; (2) the slot-29 VECTOR LENGTH at two entries
- measured only for the four-buffer/SEPARATE class (`MEASURED_SYSTEM_REGISTER_COUNTS` includes 2),
never for the tensor class specifically; (3) the tensor-class SECTION SIZE at two entries - entirely
unmeasured, and this session's own `ac`/`ac2` finding (section 64) is exactly why: that pair is
confounded, so a genuinely clean measurement of the tensor class's two-entry size needs either a
minimal pair this session did not find or a comparison against an authored section Apple's own
compiler produced. Re-keying `mdgen.py` would deliver (1) alone; (2) and (3) would still be missing
before a composed program could actually author. Section 65's status-table entry is corrected below
to say this precisely rather than let "one item closed" read as more than it is.

## 67. Items 2 and 3 closed the same way item 1 was - by reading structure, not by differencing

`ed1cc463`: the compiler owner closed the two remaining pieces of the three-part accounting
(section 66) - the slot-29 vector length and the tensor-class section size at more than one
entry - and the method is the same insight twice: this question, like SR130's entry value, was
never answerable by differencing two programs, because no clean pair for it can exist.

**Independently reproduced from scratch, not taken on the ledger's table.** Wrote my own scan
(`machobj.parse` on each object's raw `object/0-0` bytes - no need for `locate`'s archive-matching
step - `model.decode` on the `__TEXT,__text` section to test for the tensor MAC opcodes 5106,
5107, 10384, 10385; `g17gpumd.kernel_table`/`table_at` plus a direct read of the slot-29 vector's
own count word, exactly as `g17slot29census.py`'s `slot29_of` does) over every `ac-`, `ac2-`,
`abi-v1-tensor-` and `tsr-` object in the live build cache:

```
89 tensor-MAC objects checked
entry counts: {1: 24, 2: 17, 3: 48}     <- exact match to the ledger's table
```

Went further than a bare count match: read the actual entry VALUES for samples of each count and
found them fully coherent with everything measured so far - one-entry objects read `[52]` (SR130
alone); two-entry objects read `[0, 52]` (SR156's known entry 0, then SR130's 52, ordered by entry
value as the map's own rule states); three-entry objects read `[0, 52, 53]`. Also checked that
every vector's `4 + 4*count` span fits inside its own metadata section (0 of 89 overflow) - a
sanity bound the ledger's own claim implies but does not itself restate.

**Why no pair could have answered this, confirmed by the same logic that stopped section 64's
attempt:** reading one more system register needs one more `read_sr` instruction, so an object's
text length moves with its entry count *by construction* - there is no pair of tensor objects that
differ in entry count without also differing in code, which is exactly the confound `ac`/`ac2`
hit. The fix is not a better pair; it is reading each object's own structure directly, the same
move that got SR130's entry value from the 372-object population instead of a differential.

**What this still does not license, read directly from the ledger rather than summarised past it:**
the vector's *extent* (4+4n) is now measured on the tensor class itself, but the SERIALIZER change
that would write a second entry - shifting every metadata target after the vector by four bytes per
extra entry - is code that does not yet exist and is not a fact these 89 objects state on their
own. `tensormetadata.layout()` is unchanged; `(130, 156)` still refuses. Three measurements now
support a reviewed re-keying; they do not themselves perform it.

Test suite rerun fresh: 57/57 in ~119s, unchanged count, matching the commit's own claim.

## 68. A near-miss caught by a fuller check, and the actual instrument: describe/build_from, not a pair

`bdff0cd4`, appended to the same ledger: the compiler owner nearly recorded a wrong conclusion and
caught it themselves before publishing. `abi-v1-tensor-common-9e3449b707` (one entry) and
`ac2-32x32x64` (two entries) are the same size, slot 29 sits at the same position in both, and
every vector after it shifts by exactly four bytes - a pair that "looked decisive," about to be
written up as "the extra entry shifts everything after it by four and changes nothing else." A
FULL descriptor diff (every table's `vtpos`/`vlen`/`tlen`/slot set, not just the vector positions)
found five differences the vector-only check would have missed, including one table's slot SET
changing shape entirely (`[0, 1]` against `[0]`). Searched all 175 comparable one-vs-two-entry pairs
in the population scored by how many fields a plain `+4` rule leaves unexplained: the best residual
is 2, on a different pair entirely. No pair in the population is clean - the same shape as item 1's
finding, one level up: reading another register needs another instruction, so a class that reads
more registers differs from one that reads fewer in more than just the vector.

**The pivot is the substantive result, and it is independently confirmed here, not taken on
faith.** `mdgen.describe()`/`build_from()` - decode a whole `__GPU_METADATA` section into a
structured description and re-emit it - sidesteps differencing entirely: if the round trip is
byte-identical, the description missed nothing, and a two-entry tensor object can be fully
characterised from ONE object, no pair required. Reproduced directly:

```
describe(ac2-32x32x64's metadata)                  -> build_from -> byte-identical (488 bytes)
describe(abi-v1-tensor-common-9e3449b707's metadata) -> build_from -> byte-identical (488 bytes)
```

Also ran my own (naive, offset-keyed rather than semantically-aligned) diff of the two descriptions
directly and found real structural differences beyond a uniform shift - a `tail` field of different
length between the two objects (12 bytes / three words against 16 bytes / four) - which corroborates
the "more than +4" finding in substance, though not field-for-field identical to the ledger's own
more careful accounting.

**What follows, specified but explicitly not done here:** express the two-entry tensor class as its
own descriptor, the same pattern `SEPARATE_FOUR_REGISTERS` already uses (`dict(SEPARATE, ...)` with
shifted offsets and its own `word_vectors`), selected when the declared register set has two
entries, entries sourced from `isa/g17-slot29-entries.json` rather than the hardcoded 52, with the
single-entry case's continued byte-identity as the control that makes the change reviewable. Root's
standing instruction gates it; the compiler owner left it as its own future commit rather than the
tail of a measurement, which is the right place to draw that line.

## 69. Attempted independent verification of the 171/38 field split - inconclusive, and why, named precisely

`12777e7c`: the compiler owner's next step from section 68, measured - of the 17 two-entry tensor
objects, 171 fields are constant across all of them (safe to take from a witness) and 38 vary, all
38 inside the single per-kernel table at offset 152, which is what makes a class/program split
usable rather than merely asserted.

**Attempted a from-scratch check here and it did not reproduce cleanly - recorded honestly rather
than claimed either way.** Found the same 17 objects (count==2 via the same scan as section 67),
described each, and diffed field by field. First pass (naive, keyed on raw byte offsets) found far
more variation than claimed and was wrong to trust: absolute table offsets shift whenever an
upstream value (register count, held in table 152) changes, so comparing by raw offset key confuses
"this table's address moved" with "this table's content changed." Redone aligning tables by their
RANK in each object's own `.order` list instead of by address - the correct comparison - and it
still showed variation outside index 1 (table 152's own rank). Before concluding anything from that,
checked the more basic structural facts first and found the real problem: **these 17 objects are not
even the same size** (`{460, 480, 484, 488}`) and do not all have the same number of tables
(`order` length 8 or 9 across them). A population that varies in overall structure is not the single
homogeneous class the 171/38 split describes, so this session's own object selection (four family
name prefixes, filtered only by `slot29 count == 2`) is coarser than whatever population the
compiler owner actually compared - not a contradiction of their finding, a gap in reproducing their
exact population. Left open rather than resolved under time pressure, and not reported as either
confirming or disputing the split - the same caution the compiler owner asked for after section 68's
partial corroboration was at risk of reading as more settled than it was.

## 70. Withdrawn: "no unknowns left" - the section-size heterogeneity section 69 found independently was real

`bed4770a`: the compiler owner withdrew their own previous commit's "no unknowns left" within the
hour, on two grounds, and the second one is exactly what section 69's inconclusive attempt had just
surfaced independently.

**First**, the derived-vs-independent split for the 38 varying fields: a test grouping the 17
two-entry objects by shared program facts (instruction count, back edge) and checking whether
varying fields stay constant within each group returned "37 of 37 derived" - but 10 of the 12 groups
have exactly one member, where the test cannot fail by construction (a field cannot vary within a
group of one). Only two groups (sizes 5 and 2, seven objects total) are actually informative.

**Second, and this is the one section 69 already found by itself**: the tensor class's section SIZE
is not a class constant. Across the same 17 objects it is 488 (13), 480 (2), 484 (1), 460 (1) - the
exact same four-value set, with the exact same population, this session's own section 69 reported
independently before this commit existed, from a completely separate script. Reproduced the group
structure too: grouping by instruction count alone (a proxy for the compiler owner's
instruction-count-plus-back-edge key) gives one group of 5, one of 2, and ten singletons - the exact
`[5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]` distribution quoted in the commit.

So section 69's "inconclusive, population heterogeneity found but not understood" was not a gap in
this session's method after all - it was the same real fact the compiler owner independently
rediscovered and used to correct themselves. Recorded as vindication of the caution rather than
quietly folded into a "confirmed" - the honest report was to flag it as unresolved, and it stayed
unresolved on both sides until the same population fact surfaced twice.

**Status table correction, following the compiler owner's own retraction rather than lagging it:**
section 68's "all three sub-measurements closed" is walked back. Item 3 (section size at two
entries) is open again - "size does not move" was true of one pair and not a class fact - and the
derived/independent split for the 38 varying fields needs more two-entry objects sharing an
instruction count than currently exist in the cache, which means compiling new ones rather than
finding them.

## 71. Resolved: the 17 were four classes, and the real one needs almost nothing per program

`64ff7520`: the compiler owner traced section 70's withdrawal to its root cause and closed item 3
properly, crediting this session's failed reproduction as the signal ("the linker lane failing to
reproduce it and saying so rather than reporting a number"). The 17 two-entry objects were never one
class - by structural signature (table count, section size) they are four: 13 objects at 9
tables/488 bytes, 2 at 9/480, 1 at 9/484, 1 at 8/460. Section 69's population was exactly this same
mixture, found the same way.

**Reproduced independently, and it holds up well beyond a count match.** Regrouped this session's
own 17 objects by `(len(order), len(md))`: `{(9,488): 13, (9,480): 2, (9,484): 1, (8,460): 1}` -
exact match. On the 13-object class, diffed every field: at leaf granularity (this session's finer
count, roughly double the compiler owner's per-field count throughout tonight, consistent with
counting each `(offset, width, value)` tuple's three parts separately rather than as one field) it
is 212 constant against 2 varying, and one of the two is exactly the field named -
`tables.152.fields.0`, the per-kernel table's slot 0, the register-count declaration. Read the
values directly from all 13 objects rather than trusting the prose list: `32, 48, 48, 72, 72, 72,
72, 72, 76, 80, 84, 116, 117` - five 72s, not four as the commit's own recount had it (a minor
transcription slip, not a substantive one - the objects themselves are what was read, and larger or
more irregular shapes need visibly more registers, exactly "register pressure" as claimed).

**One thing this session's finer count surfaced that is worth naming rather than smoothing over:**
a second varying leaf, `.tables.252.tail`, that the compiler owner's 102/1 accounting does not
mention. This may be an intentional exclusion (a genuinely unstructured padding tail, not a named
field in their schema) or a real second varying field their per-field granularity does not resolve
into a leaf; not chased further here, named so it can be checked rather than left silently absorbed
into "one varying field."

**The cross-compiler corroboration, confirmed from this session's own earlier work rather than
newly checked:** `ac2-32x32x64`'s register count is 72 (read directly above); this campaign's own
`g17cc.compile_function` on the witnessed `32x32x64` shape reported the identical `register_count:
72` in section 52's independent verification, hours before this commit existed. Two different
compilers, the same program, the same one number - which is exactly the corroboration the compiler
owner cited, available without a new hardware run because this session had already produced it for
an unrelated reason.

**Status, corrected once more:** item 3 (section size / layout at two entries) is genuinely closed
now, on the one real homogeneous class - not "no unknowns left" (withdrawn, section 70) but "on 13
matched-signature objects, 102 fields fixed and one field is the register count the compiler already
computes." The caveat the compiler owner kept: the derived-vs-independent split still rests on a
group of five sharing an instruction count, not the whole thirteen - solid for what it claims, not
yet exhaustive.

## 72. A scoping note, not a new measurement: the remaining open item doesn't block a conservative implementation

`a0a7f595`, closing this thread for tonight: the group-of-five caveat from section 71 (whether the
one varying field is truly independent, established only on five objects sharing an instruction
count) does not need to be resolved before a first authoring change is safe, because this project's
existing class descriptors are already written as a layout *plus entry conditions* - a program that
doesn't match is refused, not authored wrongly (`ONE_WRITTEN_ZERO` refuses anything but its own
witnessed register set; the four-register class demands dense bindings and no promotion). The
proposed conservative shape: accept exactly the one measured signature (9 tables, 3 vectors, 488
bytes, the 102 constant fields), supply the register-count field from `abi_inputs()`, take slot-29
entries from `isa/g17-slot29-entries.json`, and refuse everything else by name - including the other
three signatures section 71 found mixed into the original 17. Under that contract the open
measurement bears on how much the class can be *widened* later, not on whether a narrow version is
correct now.

This is a design note, not an empirical claim, so nothing here to independently reproduce - recorded
for the record because it is the reasoning that will presumably shape whatever the eventual reviewed
commit looks like. No authoring change has been made; `tensormetadata.layout()` is unchanged.

## 73. The second varying field this session flagged is real, explained, and confirmed byte for byte

`bad45b1d`: the compiler owner credited and closed the `.tables.252.tail` flag from section 71 -
two varying fields in the 13-object class, not one. Read the raw tail bytes directly rather than
trusting the correction: all 13 share the same 64-byte tail up to an embedded
`agc.main.constant_program` marker and a length prefix, then one 4-byte little-endian value - `0`
in eleven objects, `01000000` (1) in `ac2-32x32x256`, `80010000` (384) in `ac2-32x32x384`. Exact
match, byte for byte, to the commit's claim. 384 is explained (the loop-bound compare immediate an
8-bit form cannot hold, so it moves to the constant pool - a real, named mechanism); the `1` for the
K=256 case is not, and the compiler owner named that honestly rather than inventing a story for it.
Revised count: 102 constant, 2 varying - the register count and this constant-pool literal, both
program facts, neither safe to copy from a witness.

The same commit also found the earlier "atomics needs `__GPU_LD_MD` generation" blocker was smaller
than stated - `describe`/`build_from` already round-trips it byte-for-byte on three real objects -
and identified the flatbuffer slot (24) that discriminates device-atomic objects cleanly (511 of 511
present, 0 of 21,463 others). This is a separate thread from the tensor/SR130 work Spencer's
milestone tracks; not independently reproduced here since it does not bear on that milestone, noted
for completeness only.

Continuation, `f9c681c9`, same non-tensor thread: a stale claim ("this backend does not build
`__GPU_LD_MD`", carried forward from an already-retracted ledger without rechecking) corrected at
its source; the atomics gap narrowed to three specific flatbuffer slots (two measured - a constant
declaration flag and a program-varying value shared with 20,493 non-atomic objects - one
unexamined), using the same structural-signature-first method the tensor work above required.
Nothing authored or dispatched. Noted for the record only; still no bearing on Spencer's milestone.

Follow-up, `c0e32ddf`, worth keeping for the general pattern rather than the atomics content: the
compiler owner found their own previous commit had claimed a map edit that a heredoc failure had
silently dropped, with every downstream check (regeneration, `--check`, the 57-test suite) passing
against the unedited file because nothing in that chain could distinguish "applied" from "not
applied" - the same shape as the session's earlier silently-failed `git checkout`. Confirmed the
described end state directly: `isa/g17.yaml` no longer contains the stale "does not build the
section at all" phrase (0 occurrences, checked against both commits; this session's own grep found
11 instances before the fix and 0 after, not the single instance the commit's prose cites - a minor
scope difference in how each grep was run, not chased further since it doesn't change the outcome).
Named the fix correctly as a habit (read an edit's own output before trusting the verification that
follows it), not a check that can be automated away.

## 74. The 1,318 count corrected to 1,039 - root caught it, independently reproduced, plus an out-of-sample validation of the map

`3bbbcf65`, back on the candidate branch itself (`agxforge/g17/tensormetadata.py`'s refusal message and
comment, not just the isa-cartography ledger): root recounted the witness population independently
and could not reproduce 1,318. The compiler owner traced it to their own prose - one census had
produced two numbers (objects reading SR130 *and at least one other* register: 1,318; objects whose
set is *exactly* `(130, 156)`: 1,039), and the refusal message quoted the superset count for the
subset claim. Their own earlier table had the right number; the sentence built from it didn't.
Corrected in the actual error message and comment (confirmed by reading the diff: the refusal logic
itself - `registers != (130,)` still raises - is unchanged, this is text only, no authoring change).

**Independently reproduced with a full cache scan, not taken on either party's word.** Wrote a
from-scratch scan over the entire build cache this time (all 23,029 directories, not just the four
tensor-family prefixes used for the slot-29 census) - decode every object's text, collect the set of
`decode_sr`'s byte1 values (`asm.decode_sr`'s own field, `u[1]` on each `read_sr` instruction's raw
bytes) per object, count exact matches:

```
objects checked: 22,037
exact system-register set == {130, 156}:  1,039     <- exact match to the correction
objects containing both 130 and 156:       1,111     <- root's own recount says 1,121; close but
                                                          not identical, not chased further
```

The headline number - the one actually cited in the refusal message and the one this session's own
independent full-cache scan targeted - matches exactly. The secondary "contains both" figure differs
by 10 between three independent countings (root's 1,121, this session's 1,111) and neither matches
the retracted 1,318; worth naming rather than silently rounding away, though it doesn't touch the
corrected claim.

**Root's out-of-sample corroboration, the part worth more than the correction cost:** breaking the
1,121-object "contains both" population by which additional registers appear predicts every group's
slot-29 vector correctly from a map fitted only on single-register objects - `(130,156)` -> `[0,
52]` (1,039), plus `SR_SIMD_GRP` -> `[0, 52, 53]` (51), plus two more registers -> `[0, 52, 58, 59]`
(16), and so on for two more groups. Five register-set combinations the fitting population never
contained, all five vectors correct. Not independently re-derived here (would need the named-SR
lookup table this session has not located), recorded as reported since the mechanism - entries
ordered by value, one per declared register - is already independently confirmed in sections 65-67.

Test suite fresh: 42/42 in ~101s, matching the commit's own count.

**The pattern, now three times in one ledger by the compiler owner's own count, plus this
session's own instances tonight:** "372 objects say 52" (a sum quoted as a population), the 171/38
field split (four classes pooled), and now this (a superset count quoted as a subset) - every one a
correct measurement described by a sentence that did not survive contact with someone re-deriving it
from the raw rows rather than reading the prose.

## 75. The three-way count gap section 74 flagged, recorded as unresolved rather than picked

`845230ed`: the compiler owner named the exact discrepancy section 74 raised - three independent
counts of the same population (theirs, root's, this session's full-cache scan) agree exactly on the
headline figure (1,039 for `(130, 156)` exactly) but give three different numbers for "contains both
registers" (1,121 / 1,111) and the different-boundary 1,318 - and recorded it as genuinely
unreconciled rather than silently choosing one. This is the right disposition: the number that
matters to the refusal message is settled three ways; the one that doesn't (yet) is named as open,
with the reason stated (the population is defined slightly differently each time) rather than
guessed at. Nothing to independently verify here beyond what section 74 already ran - recorded for
completeness, since the ledger now correctly cites this session's scan as one of the three.

The same commit's atomics-thread continuation (t44 slot 2 examined and found NOT the device-atomic
discriminator, 504 of 511 with 7 exceptions and 42 false positives correlated with source-level
`atomic_load`) remains outside Spencer's milestone; noted, not independently checked.

## 76. A near-miss on the atomics thread worth flagging for severity, still outside Spencer's milestone

`4ea870e0`: not part of the tensor/SR130 work, but severe enough to record with more than a
one-liner. The compiler owner nearly wrote a candidate atomic metadata section that would have
passed every structural self-check available - `describe`/`build_from` round-trip, size, vtable,
slot presence - while actually corrupting the entry PC, because their layout's slot 6 occupies
bytes (20-23) that overlap where the candidate placed a new slot 24 entry (22), a collision that
does not exist in Apple's own layout (slot 6 at 24 there, leaving 22 free). Their own account:
"a metadata section that walked, read back exactly and passed a verifier scored on 30,548 tables
still hung the GPU and killed WindowServer" is describing a known failure class in this project, not
a hypothetical. What caught it was comparing an inferred field WIDTH against Apple's actual value
(one field read back as `(22, 2, 1)` against Apple's `(22, 1, 1)`) rather than checking the candidate
section only against itself - no self-consistency check, round trip included, would have caught it.
Nothing written to `ldmd.py`, nothing dispatched. Recorded here for its severity and because it is
the same general lesson as tonight's tensor-side near-misses (round-trip completeness is not the
same claim as correctness against the real target) at a much higher cost if it had gone wrong. Not
independently verified - outside this session's tracked milestone - but worth the record knowing
what was avoided.

## 77. The atomics near-miss became a proper measured mode - opt-in, default path untouched

`82648f95`, on the candidate branch itself: the compiler owner turned section 76's near-miss into a
correct, conservative addition - `ldmd.build(atomic=True)`, an opt-in mode alongside the existing
default, every one of its nine written slots checked field-by-field against Apple's own atomic
section rather than the candidate's own self-consistency (the exact discipline the near-miss showed
was necessary), with slot 1 and t44 slot 2 deliberately left out and a case that fails if they are
added without being measured first - the same "accept exactly what's measured, refuse the rest"
shape as the tensor work's own conservative-implementation note (section 72).

Spot-checked only for regression risk to the tracked milestone, not the atomics content itself: ran
`test_g17atomicldmd.py` fresh (6/6) in a fresh worktree. Did not rerun the 42-test tensor suite
again - this diff touches only `agxforge/g17/ldmd.py` and a new test file, not `tensormetadata.py` or
`mdgen.py`, and that suite has already been confirmed stable and unaffected repeatedly tonight.
Still outside Spencer's milestone; noted for the record that the near-miss two sections ago produced
a real, careful fix rather than being quietly worked around.

## 78. Atomics thread, briefly: a self-caught default-swallows-request bug, still opt-in, still not dispatched

`675857e6`: the compiler owner found their own `if atomic and size == 216: size = ATOMIC_SIZE`
silently overrode an explicit caller request the same way a `.get(dtype, 2)` would have (the exact
pattern the stride fix refused by name earlier tonight, section 55) - fixed so an explicit size is
honoured and a too-small one is refused rather than silently widened. Also established the atomic
section fits the one real splice target it would need (`tb-host3wide`'s `__GPU_LD_MD`, 216 bytes,
zero gap to the next section) without any Mach-O container edit. Spot-checked for regression risk
only: `test_g17atomicldmd.py` fresh, 8/8. Still outside Spencer's milestone, still nothing authored
or dispatched.

## 79. Atomics thread: a real crash, correctly diagnosed and contained, and a null-arm probe defect

`35b25d1c`, still outside Spencer's milestone but flagged at more than the usual one-liner because
a real crash happened, not another near-miss. Two things:

**The probe's control arm was never valid.** `atom12.py`'s `--null` arm - the unmodified carrier,
no atomic anywhere - was run for the first time today and itself returned the seed value the probe
treats as "no effect happened," meaning every one of the nine single-variable eliminations run
against it is uninterpretable: the instrument couldn't distinguish "this variable doesn't matter"
from "nothing this probe does ever writes memory." Diagnosed to a binding-declaration mismatch (the
probe patches bytes into a vendor host container that doesn't declare the rank being written to),
not the harness or the image-authoring path, both independently controlled and cleared.

**Fixing the corrected atomic layout's size crashed the host.** Section 77-78 had settled on a
216-byte atomic section by construction; writing the previously-measured 224-byte field layout into
that 216-byte container put every field eight bytes past where its data actually lives, and
dispatching it **segfaulted the host process in `AGX::DynamicLoader` at pipeline creation** - before
any GPU submission, no GPU event recorded. Diagnosed correctly and without guessing: census by
section size found 20 real Apple-toolchain objects that are 216 bytes, `tlen 40`, and DO declare the
atomic - byte-identical and field-identical to what this project's own 216-byte section should be.
The atomic declaration is four bytes, not eight; `ldmd.build(atomic=True, size=216)` now matches the
real vendor section exactly, the non-atomic default path is unaffected, and unwitnessed sizes are
refused rather than guessed at again.

Not independently reproduced or re-run here - deliberately not attempting to trigger a host crash
myself to "verify" one that already happened and was already diagnosed and fixed with a real,
byte-identical vendor comparison. Recorded at this length because a segfault during development is
real information worth Spencer having visibility into even on a thread this session isn't tracking
for its own milestone, not because there is anything to check on this session's side.

## 80. Spencer's milestone, achieved and independently reproduced end to end on real hardware

A new branch appeared, `origin/codex/g17-tensorops-main-integration` (based on current `main`, not
merged), and it closes every piece of the milestone as stated: *a composed GEMM+scalar program
produced by the ordinary compiler is authored into a repository-owned image, dispatched through the
ordinary runtime, and validated on hardware from a cold main checkout, with the system-register
metadata derived from a measured rule rather than a hard-coded tensor special case.*

**The re-keying (`ca99db69`), verified directly.** `mdgen.SLOT29_BY_SYSTEM_REGISTER` now reads
`{130: 52, 156: 0, 157: 1, ...}` - ten entries, confirmed by reading the file. `tensormetadata.
layout()` accepts exactly `(130,)` or `(130, 156)` and refuses everything else by name - precisely
the conservative shape section 72 specified (accept the measured signature, refuse the rest, widen
later), not a general relaxation. Ran it myself: the `residual(1)` composed program that has refused
at authoring every time this campaign touched it all night now calls `scanlink.author(program)` and
returns a real `Image` (6,000-byte archive) instead of raising. The three body hashes this campaign
has checked at every round since section 46 are unchanged.

**The common-runtime dispatch (`1ddd9626`), independently reproduced, not read from the doc.** Ran
`tools/g17tensorcommonruntime.py --prepare` then `--run` myself, in a fresh detached worktree of this
branch's tip: compiled the measured composed class (17x19x16 half GEMM, `threadgroup_position_in_grid`
read, FP32 add, store) through `cc.compile_function`, authored it through the public
`scanlink.author`, built and loaded the real `tools/g17commonworker.m` native worker, and dispatched
three queries through actual Metal - **all three `passed`, `reference_bit_exact: true`,
`gpu_dispatched: true`, output SHA-256 `ec43027260f5a0b8dd59526c17102ffe00066852f10af42a91103c36e778faf6`
for all three - matching the documented run's hash exactly.** This is real hardware, real dispatch,
compared bit-for-bit against an independent instruction-level RNE32-MMA-plus-FP32-add reference, not
this campaign's own `gputime` harness.

**The cold checkout, satisfied by construction.** My reproduction ran from `/tmp/g17-main-integration-
verify`, a worktree freshly detached from the branch tip with nothing carried over from this
session's persistent state - the same shape of test the milestone asks for, and it produced the
identical output hash the branch's own documented cold-checkout receipt (`execution-cold.json`,
source revision `f938d13b`) records. Three independent instances of the same measurement (their
original run, their own cold-checkout confirmation, and this session's fresh worktree) agree exactly.

**The broader awkward-workload matrix, also authoring now**: `docs/archive/g17-tensorops-main-status.md`
(read directly) shows GEMM+activation, GEMM+residual, and GEMM+eight-scalar-adds all now
authoring (not just compiling) with `system_registers=(130,156)`, while the multi-GEMM composition
cells remain correctly refused by name - the boundary this campaign established stays exactly where
it was measured to be.

**Test suites, fresh:** `test_g17tensorcommonruntime.py` 4/4, `test_g17fourbinding.py` 15/15
(2 skipped, expected without the build cache/witness), `test_g17tensorwholekernel.py` 43/43 in
~145s - one more than the last count this campaign checked, consistent with the new authoring paths
adding coverage rather than displacing it.

**One caveat, found on the isa-cartography branch (`0f1a482b`) and worth carrying forward rather than
smoothing over:** the new map entry `130 -> 52` is correct for 372 of 385 known byte1=130 forms and
wrong for 13 - `SLOT29_BY_SYSTEM_REGISTER` keys on `decode_sr`'s byte1 alone, which section 65-67
already established cannot distinguish sub-forms sharing a register. Checked and found latent, not
active: every current consumer of the map (including this milestone's own dispatch) resolves an
explicit, known register rather than an arbitrary decoded one, and nobody has yet confirmed which
exact sub-form the dispatched tensor programs use - plausible that it's the common one (372-object
population), not confirmed. Named as a real, bounded residual risk on a shared map, not a defect in
what was just verified.

**That residual closed within the hour (`2abd0424`), independently reproduced.** The compiler owner
built the exact milestone program (`g17tensorcommonruntime.build_program()`) and decoded every
system-register read in it rather than reasoning about which form it should be. Reproduced here from
scratch, same tip: the 1,232-byte program's two `read_sr` instructions are `04821006` (byte1 130,
width 4, byte3 6) and `9c9c1006` (byte1 156, width 4, byte3 6) - exact match. `(130, width 4, byte3
6)` is precisely the 273-object census key whose entry is 52, and the two forms that give the wrong
entries (50, 10) are both width 8, a form this backend's tensor-reading code does not emit. So the
milestone's own dispatch sits on the correct side of the byte1-ambiguity for a structural reason (this
compiler only ever emits the width-4 form here), not by chance. The generality caveat itself is
unchanged and still real: byte1 alone still cannot tell 52 from 50 from 10, so a future consumer that
decodes arbitrary programs still needs the full `(byte1, width, byte3)` triple. One phrasing note from
the compiler owner worth keeping precise: "372 of 385" is a count over byte1=130's own forms
specifically, not a statement about the shared map as a whole - this doc already said "known
byte1=130 forms" throughout, and the correction is recorded here as confirmation that phrasing
survived rather than as a fix.

**What the milestone does not yet claim, stated by the branch's own docs and left that way**: one
measured shape and composition class only (17x19x16, half, one simdgroup, two scalar-composition
forms); not int8/bfloat/float, partial tiles, multiple simdgroups, multiple tensor programs per
worker, or the full cold release gate re-run on this exact candidate. Named follow-on work, not this
session's to do.

This is the closest this campaign has come to full closure of the goal it was set. The remaining
items are the same as ever - the merge decision, and the scope the branch's own docs already name as
not yet covered - and none of them are this session's to perform.

## 81. A backward-compatibility fix on `tensormetadata.py` itself, verified no regression

`e140414c`/`c82bb531`, back on `main-integration`: `ca99db69`'s ledger-text update had
unconditionally changed the singleton `(130,)` case's wording alongside the new `(130,156)` case's,
which would fail any existing byte-identical receipt still expecting the old exact text. Fixed by
branching the ledger text on the declared register set, old wording preserved for the singleton case.

Verified directly rather than trusting the "no regression" framing: spot-checked the three body
hashes (unchanged) and the composed `residual()` authoring case (still returns a real image,
6,000-byte archive, unchanged) in a fresh worktree. `test_g17tensormetadata.py` first showed 4 of 6
failing - all four traced to this session's own incomplete worktree (missing
`g17-tensor-common-witness-v1` and `g17-tensor-variant-witness-v1`, neither copied into this
particular ad hoc worktree), not a regression. Copied both in: 6/6 OK.

## 82. op11456's role: a conditional pass-through gate, `dst = mask if src < HI else 0` - found only after correcting a wrong destination register

Spencer set a new goal ("go after more metal ISA stuff") after the tensor-runtime milestone closed.
Section 45's open list still named op11456's role and immediates as unrecovered, and batch 24
(section 32) had it "executing and writing its GPR16 destination but produced 0 for every stimulus
tried" - six source patterns, three second-register values, both known witnesses. That negative
result turned out to be the reader, not the instruction.

**The bug.** Batch 8's original probe (`maskgen_probe.py`) assumed the destination (`reg426`) was
R0H and read `out[:512].view(uint32).reshape(32,4)[:,0] >> 16`. Resolving the MCRegister id through
the project's own `tools/g17layout.py:slot()` (read-only, not modified) instead of trusting that
assumption: `slot(426) == 2`, i.e. **R1L**, not R0H. Every probe of this opcode to date - batch 8's
and this session's own first attempt (`maskgen_probe2.py`, which fixed the mask-operand coverage gap
but initially kept the same R0-word readout) - was reading a register op11456 never writes. The same
check on the other two register operands confirms `slot(110) == 10` (R5, the GPR32 src, as assumed)
and `slot(432)/slot(429) == 14/8` (R7L / R4L, the GPR16 "mask" operand, as assumed) - only the
destination was wrong.

**Corrected readout, both known witnesses (10-byte forms, `mm_partial_m24`'s `128b25c0020e05022080`
and `120b0dc0020805022080`).** `dst == mask` for every stimulus tried (per-lane values, 0xffff/
0xaaaa/0x5555, `1<<lane`, one-hot) as long as `src < 64`; `dst == 0` for every stimulus once
`src >= 64`, including `src` values that leave the mask register itself untouched. Bisected on a 32-
point src sweep and a single-bit-of-src sweep: the boundary is exact and unsigned - 63 passes, 64
fails, `0x7fffffff`/`0x80000000`/`0xffffffff` all fail. A finer test read `dst` bit-by-bit (mask =
`1<<k` for k=0..15) against src values straddling the boundary (58, 59, ..., 65) to distinguish a
per-bit range predicate (op612's own shape: bit `j` = `[lo <= src+j < hi]`, which would show a
partial pass/fail split near the boundary) from a scalar gate: **the pass/fail is all-or-nothing
across all 16 bits simultaneously at every src tested**, ruling out the per-bit hypothesis. So
op11456 is not a mask generator (op612 already is one, section 32) and not a per-element predicate -
it is a **scalar conditional pass-through**: `dst = mask if src <u HI else 0`, one comparison gating
the whole 16-bit register.

**HI, and what's still open.** Both witnesses measured HI = 64 despite differing in the *other* two
immediates (`32, 9` before src / `32, 64` after src for the first form; `0, 9` / `16, 64` for the
second) - varying those first-pair values didn't move the threshold in either witness, so HI is read
from the fixed immediate directly preceding the mask operand, and the two witnesses agree on its
value though this is only n=2. What the first immediate pair and the trailing `imm 0, imm 0` do
remains unmeasured (they were held at their witnessed values throughout, not varied) - op11456's
immediates are now one comparison away from fully named, not one register value.

**A stale peer artifact worth flagging.** `origin/claude/g17-isa-cartography`'s
`isa/g17-peer-claims.json` carries this same opcode as an open, ambiguous item (`d3_peer_reported`,
width 10 vs 14, "no instrument records which") - but its `pinned_commit`
(`c675569fabaf97d3cf29d307b08d038f432fd1ec`) is 71 commits behind this branch's current tip and
predates batch 24's own correction (`op11456 is not the mask generator, op612 is`), let alone this
section. The tool's own labelling is exactly right to be cautious about this
("EXTRACTED, UNREVIEWED, and NOT VERIFIED HERE... their measurement is theirs to defend") - it is not
a bug on that branch, just a stale snapshot. No live channel to that session was available this
round (`ListAgents` showed no isa-cartography instance, and the `/tmp/cc-socks/*.sock` addresses from
earlier in this campaign are all several days stale); recording it here since the git-committed doc
is the channel that tool actually re-pins and quotes from.

**Corpus census: HI is a real per-instance immediate, not a constant that happens to read 64.**
`isa/g17-corpus-programs.jsonl` (same branch, read-only) has 76 real op11456 instances across 58
distinct encodings. Cross-tabulated by decoded operand: the two witnesses measured above (10-byte
forms) are the majority (58/65 tensor-store witnesses) at HI=64, but 7 of 65 decode HI=8 with the
same operand shape; the 8-byte forms (7 distinct, 10 instances) are a separate sub-family where
`dst == mask` (an in-place form, which is presumably why the encoding is 2 bytes shorter - no
separate mask operand to carry) and HI reads 0 or 8. A third witness pulled from the HI=8 group
(`d28f2588c21a05122080`, dst=mask=R77L, src=R7) was carried into a new probe to hardware-confirm the
gate reads 8 rather than 64 for this encoding; the carrier itself (a second load/store reaching
R76-R79, needed because R77 sits outside the R4-R7 window the earlier carrier used) hit a genuine
plumbing bug of its own - the second load only reached even-numbered SIMD lanes, confirmed as a
carrier bug rather than an op11456 behaviour by a control run with op11456 removed entirely, which
reproduced the identical even/odd split. Not resolved this round; the corpus fact (HI takes at least
three values across real compiler output: 64, 8, 0) stands on its own as evidence the field is read,
independent of the not-yet-repeated hardware confirmation.

## 83. op612 is already named (batch 24) - a direct answer to isa-cartography's `efce2b47`, "recorded as unreachable rather than a determination"

`origin/claude/g17-isa-cartography` at `efce2b47` ran three hardware probes of op612, got a constant 0
across all four inputs, correctly refused to call that a determination (an absorbing fit that any
function mapping the input set to one value would satisfy), and stated precisely what would settle
it: "a record that varies op612's OTHER operand," since the batch varied only the source register.

This campaign already has that answer, hardware-verified, in section 32 (batch 24): op612
(`dst:GPR16, word, src:GPR32, 16, lo, hi`) computes `bit j (0<=j<4) of dst = [lo <= src+j < hi]`,
unsigned, **only when the immediate right after src is 16** ("the immediate 16 marks a register
source"); when that immediate is 0 instead, "the source operand [becomes] an expression form and the
result is 0" regardless of src. Decoding this session's own two witnesses confirms the shape exactly:
`270200122120a11224001c18` (`op612_probe.py`'s `r5_16_32`) carries that marker as **16** and produces
the real range predicate; `370200022120a11a20001c18` (`r1_0_32`) carries it as **0** and returns
constant 0 even though its src operand (R1 = R5+0) varies with the lane exactly like the other
witness's does. So a hardware batch that (like this campaign's own `r1_0_32` control) happens to hold
that one immediate at 0 will see exactly the constant-zero result `efce2b47` reports, for a reason
that has nothing to do with lo/hi or the base being left at zero - it's a third operand, not the
"other operand" being src's own base. The "other operand" to vary is that immediate (16 vs. 0), not
necessarily lo/hi.

No live channel to relay this directly was available (as in section 82); recording it here since
that branch's own tooling re-pins and quotes this doc.

## 84. Cross-check on `claude/g17-tensorops-emitter`'s untracked-witness finding: the milestone's own dispatch path is clean, a later regression check of this session's own shares the same defect class

`5f215f91` (that branch) found 6 of 26 awkward shapes blocked not by a real compiler refusal but by a
compile-path dependency on `results/g17-tensor-common-witness-v1/tensor-common.o` and
`results/g17-tensor-variant-witness-v1/tensor-a-stride-96.o` - both under the gitignored, untracked
`results/` (`.gitignore` line 30), so a genuinely fresh clone cannot produce them at all. Checked
whether Spencer's milestone (section 80) shares this dependency: `tools/g17tensorcommonruntime.py`
and its test (`test_g17tensorcommonruntime.py`), both on `codex/g17-tensorops-main-integration`,
contain no reference to `witness` or `results/` anywhere - the milestone's dispatch path builds its
composed program from the compiler itself, not from a pre-built cached object, so the "cold main
checkout" claim stands as recorded.

`test_g17tensormetadata.py`, the SEPARATE regression check this session ran in section 81 (not part
of the milestone itself), does carry the identical dependency - `raw=(ROOT/'results/g17-tensor-
common-witness-v1/tensor-common.o').read_bytes()` and a `glob('*.o')` over both witness directories.
Section 81 already reported the 4-of-6 initial failures as this session's "own incomplete ad hoc
worktree," which was correct as far as it went; this is the same fact stated at the scope the other
branch's finding earns it - not a one-off gap in one improvised worktree, but a real property of that
test file, true of any fresh checkout including a correctly-assembled one, since `results/` is
untracked by design and neither witness directory is.

## 85. op11456's HI hardware-confirmed as a per-instance operand, not a fixed 64 - on the second attempt, with the first attempt's failure mode honestly separated from the result

Section 82's carrier read the mask/dst register (R1L) directly off a plain store, no confound. Section
84's attempt to reach a second, HI=8 witness (`d28f2588c21a05122080`, dst=mask=R77L, outside the R4-R7
window) added a second `memenc.load` at a different displacement and found only even-numbered SIMD
lanes were reached - a carrier bug, confirmed by a control run with op11456 removed.

**Second attempt, avoiding a second load entirely**: move the mask value into R77 with a register ALU
copy (`R77 = R6 + R9`) from an already-correctly-loaded low register, following the same pattern
`op612_probe.py`'s own `r1_0_32` case uses ("R9 = 0"). The boundary came out clean and reproducible:
src 0..7 give one constant readback, src 8..15 give a different one, the transition exactly at src=8 -
matching this witness's encoded HI=8, not the 64 measured on the other two witnesses. **That threshold
generalises across two independently hardware-tested witnesses with two different encoded HI values**,
which is the fact worth recording.

**The individual VALUES from this carrier are not trustworthy**, and this is worth stating precisely
rather than silently dropped: isolating R9 alone (`R6 = 0 + R9`, no op11456 anywhere in the body) reads
back `0x2468acf0` - not zero. `op612_probe.py`'s "R9 = 0" was evidently true in that script's own
kernel layout, not a portable constant; this kernel's own register allocation leaves something else in
R9 (structurally suspicious - it equals `2 x 0x12345678`, this session's own filler word - so it is
very likely comes from an address-computation intermediate in the tile-load prologue, not true
uninitialised state, but that is not established here). Because R9 is a fixed additive pollutant
independent of `src` (confirmed separately: constant across all 32 lanes with no per-lane variation
fed into it), the STEP still locates the true boundary correctly even though the two readback constants
either side of it do not cleanly decode to "mask" and "0" the way section 82's clean carrier did. The
exact value relationship for this witness is left unconfirmed rather than reverse-engineered through an
uncontrolled pollutant.

## 86. Two corrections to op612 from a live peer exchange: the bound is `lo+w` not `hi`, and the "register-source marker" claim is refuted by a single-bit-flip control

A peer session (via `uds:/tmp/cc-socks/19108.sock`) followed up the section 83 relay with two
corrections, both worth taking seriously and both checked here rather than accepted on report.

**Correction 1, accepted as measured: the second immediate is a WIDTH, not an absolute upper bound.**
Section 32 read op612 as `bit j = [lo <= src+j < hi]`. The peer's hardware batch (24/24, two
preregistered records) found `bit j = [lo <= src+j < lo+w]` - a base and a COUNT. My own witnesses
could not see the difference because all but one sat at `lo=0`, where the two readings coincide;
their falsifying pair is `lo=8, w=32` at `src=30` and `src=32`, which read 15 and 15 where an absolute
bound of 32 requires 3 and 0. Reproduced here independently rather than taken on their word: the
`lo=0, w=32` sweep (`src` 0..31 through the exact witness `270200122120a11224001c18`) gives 15 for
every `src <= 28`, then 7, 3, 1 for 29, 30, 31 - matching both readings at `lo=0` and confirming the
carrier and witness are sound before trusting the falsifying case, which was not re-run on this
machine. **op11456 is unaffected**: it is a scalar gate, not a 4-bit range predicate (section 82 ruled
that out directly with a bit-by-bit sweep), and its own first immediate does not shift its threshold -
the same value it was witnessed with (32) sits under both a 64 threshold and an 8 threshold on the
same field (section 85), so op11456's first operand does not function as op612's `lo` does.

**Correction 2, tested and confirmed: my own "operand 3 = 16 marks a register source, 0 forces a zero
result" claim does not survive a controlled single-bit test.** The peer could not test this
themselves - their record went through the actual compiler, whose liveness pass rewrites that operand
after the encoder, so a byte-level claim about it cannot be checked past that pass - and named exactly
the experiment that would settle it. This session's methodology sidesteps that: every op612 witness
here is hand-patched into a carrier and executed directly (`lower.patched_archive`), with no compiler
pass between the bytes and the hardware. Using `isa/g17-contract.jsonl`'s own field map for op612
(`3:raw`: bits at byte1[7] (32), byte8[2] (16), byte11[6] (8)), a single bit was flipped - byte8 bit2 -
on the working `270200122120a11224001c18` witness (`lo=0, w=32`, src=R5, dst=R4L, operand3=16),
producing `270200122120a11220001c18` (operand3=0, decode-verified: only that one field changed,
src/dst/lo/w identical). Run back-to-back on the same `src=0..31` sweep: **byte-for-byte identical
output**, `[15]*29 + [7, 3, 1]` both times. Operand 3 does nothing to the result at these two values,
under a test that changes nothing else.

That refutes the causal claim in my own section 32 ("0 turns the source operand into an expression
form and the result is 0"), though the underlying *measurement* it was built on now has a clean
explanation rather than standing as a mystery: the original `r1_0_32` witness computed its src as
`R1 = R5 + R9` (a "dummy consumer," `op612_probe.py`), and section 85 of this same document
independently found - for an unrelated reason, on the same day - that `R9` is not reliably zero in this
harness (it read back `0x2468acf0` in a different kernel). A src register offset by a large,
lane-independent constant would push every lane's effective `src` far outside any `[lo, lo+w)` window
tested, which produces exactly the robust, stimulus-independent "always 0" result originally reported
- for a reason having nothing to do with operand 3's value. Two independent problems in this session's
own carriers, both involving an assumed-zero register, converged on the same wrong conclusion; this is
now corrected rather than left as a name to defend. This is also consistent with (though does not by
itself prove) the peer's repository's own stored fact that this operand is a lifetime carrier for the
source register (`g17auth.lifetime_operand(612, 2) == 3`), which would explain a real field that
affects scheduling/liveness bookkeeping but not the computed result - exactly what a null single-bit
test would show.

**Noted, not independently pursued**: the peer's finding that op621 is the same operation as op612
(identical structure, obeys the same `lo/w` relation, corpus counts differ by two orders of magnitude)
was independently reproduced and published on `claude/g17-isa-cartography` (`8332b1c9`) before this
section was written; no need to duplicate that work here.

Also fixed in this pass: section 32's correction cited "section 10," but the claim it withdraws is
printed under section 11 (batch 8) - the peer's extraction tool caught the mismatch and now detects
supersession language directly rather than trusting citations; the citation itself is fixed above.

## 87. The peer's falsifying pair for op612's `lo+w` reading, independently reproduced on this session's own hardware - and the real cause of sections 84/85's "carrier bug," found along the way

Section 86 accepted the peer's `lo+w` correction on the strength of a non-discriminating reproduction
(the `lo=0` ramp) plus their reported values. Their next message named this directly and offered the
two falsifying records without spending this session's hardware time: `w612.win37` (`lo=8, w=32`,
src 37/40/36 -> 7,0,15 - a lower-only or absolute-bound reading would give 15,15,15 or 0,0,0) and
`w612.lo16` (`lo=16, w=8`, src 21/24/16/13 -> 7,0,15,8 - an absolute bound is empty here and predicts
all zeros). Reproduced both directly rather than trusting the JSON.

**First attempt failed, and not for the reason sections 84/85 gave for a similar-looking failure.**
The peer's witnesses encode `src` as `reg121` = R16 (resolved via `tools/g17layout.py:slot()`). Built
straightforwardly - load R16..R19 from the buffer, wait, run op612, store R0..R3 - the result was
garbage (one lane read 0, the rest read the untouched fill constant). The instinct was to blame the
same "second load only reaches some lanes" pattern diagnosed in sections 84/85, but checking first
rather than reusing that diagnosis found something else: `tmpl_16x16x32`'s own `t.index_reg` maps
**every** binding to register 16 - `t.index_reg == {0: 16, 1: 16, 2: 16}` - so loading test data into
R16 overwrites the template's own output-address index register, corrupting the store regardless of
what op612 does. Confirmed by decoding the carrier's own store instruction: its index operand resolves
to `reg121`, the identical register the probe had just clobbered.

**This forces a re-read of sections 84 and 85's "even/odd lane" and "R9 pollutant" findings.** Both of
those carriers also used a register above R7 (R76/R77, R16) reached by a second `memenc.load` or a
register copy from an unproven-zero source, and both blamed a *load-side* defect (a stride mismatch,
an assumed-zero register). Section 85's boundary-location conclusion doesn't depend on this - the
threshold it found (src=8) came from a register copy (`R77 = R6 + R9`), not from a second load, and is
unaffected by the index-register collision found here (R77, not R16). Section 84's abandoned carrier
did use a second load at a different width/displacement into R76-R79, a genuinely different register
range from R16, so it is not simply the same bug re-found - but this session did not re-examine section
84's specific carrier for an index-register collision before writing this section, and is not claiming
to have ruled that out; section 84 already correctly declined to trust its own readback, so nothing
downstream relied on it being right for a specific reason.

**Second attempt: re-encode the identical `(lo, w)` values onto known-safe registers.** Using
`isa/g17-contract.jsonl`'s own field map (the same one section 86's bit-flip used) to patch only
fields 4 and 5 of the already-working `r5_16_32` witness (`dst=R4L/reg429, src=R5/reg110` - proven
safe by `op612_probe.py` and never touching a template-reserved register), producing
`270200122120a11224021c18` (`lo=8,w=32`) and `270200122120a1120c041c18` (`lo=16,w=8`), both
decode-verified to change only the intended fields. Run against the peer's exact source values:

    win37 (lo=8,  w=32): srcs [37, 40, 36]         -> got [7, 0, 15]      MATCH
    lo16  (lo=16, w=8):  srcs [21, 24, 16, 13]      -> got [7, 0, 15, 8]   MATCH

Both match exactly. **The `lo+w` reading is now independently confirmed on this session's own
hardware by both of the cases that actually discriminate it from `hi`**, not just the non-separating
ramp reported in section 86.

## 88. byte0[7] (section 45's open item) closed: not an independent flag, it's bit0 of D's tuple index - already measured by this campaign's own batch 18, never cross-referenced back

Section 45's open list and section 14 (batch 11) both carry byte0[7] as an unresolved question -
section 14's exact words: "a parity-like bit whose role is not measured," observed alternating
1,0,1,0 across consecutive independent MMAs. Section 25 (batch 18)'s decoder-oracle field census
already assigns this same bit to D's register field (`GPR32tup8_alignedrc`, delta 8 in register-
number terms) - but nothing in this document ever went back and closed section 14's claim against
that later measurement, so the doc has carried a "not measured" line for a bit this project measured
several batches later.

**Confirmed two ways.** (1) A direct single-bit flip on the campaign's own reference encoding
(`2f0025122202a4420004`, batch 18's own reference), read through `tools/g17layout.py:class_index()`
(the resolver the project's own register-model notes say is the correct one for aligned tuples,
not `slot()`): flipping byte0[7] alone moves D from `R0_R1..R7` (member index N=0) to `R8_R9..R15`
(N=1) - i.e. it is bit0 of N, nothing else changes. (2) A census of all 40 independent-MMA
instructions actually emitted across the six `indep3`/`indep4`/`indep6`/`indep8`/`indep9`/`indep10`
compiled objects already sitting in this campaign's own `results/` (no new compilation needed):
`byte0[7] == (N & 1)` in 40 of 40 cases, zero exceptions, across six independently-verified objects.

So the "parity-like" alternation section 14 saw was never an independent flag at all - it is fully
explained by which physical accumulator tuple the compiler assigned to each successive independent
MMA (these kernels happen to allocate `N=0,1,2,3,...` or `N=0,2,4,...` in sequence depending on
count, and the LSB of that sequence is what alternates). Section 45's open-item line for byte0[7] is
removed below; nothing about D's register field itself needed correcting - the field map in section
25 already had this bit right, this section only closes the stale cross-reference.

## 89. op11456's last two immediates closed: it is a true `csel`, `dst = mask if src <u HI else imm_f`, not "else 0" - the compiler simply never emitted a nonzero else-value

Section 82 left two immediates unmeasured: index7 (`imm_e`, called "trailing" there) and index8
(`imm_f`), both witnessed as 0 in every real compiler-emitted instance seen so far and never varied.
`isa/g17-contract.jsonl`'s field map for op11456 is for its 14-byte form and doesn't apply byte-for-
byte to the 10-byte form this campaign's own witnesses use, so a fresh single-bit-flip census
(mirroring batch 18's own method) was run directly on the 10-byte witness to locate the physical
bits: `imm_e` is `byte4[7]` (weight 16) + `byte5[7]` (weight 32), a 2-bit field (0/16/32/48); `imm_f`
is `byte8[7]` (weight 128) + `byte9[0..6]` (weights 1-64), a full 8-bit field (0-255) - both hand-
patched with `isa/g17-contract.jsonl`'s same weighted-bit encoding used in sections 82/86/87, and
decode-verified before dispatch.

**`imm_e` (4 values, both branches tested): no effect.** `dst == mask` when `src < HI` and
`dst == 0` when `src >= HI`, identically across all four possible values of `imm_e`, on real
hardware. Left as a measured, checkable negative rather than an assumption - this is the field
`isa/g17-contract.jsonl` names `"1:raw"` bit 41 in the 14-byte encoding, a different physical
location, so no claim is made here about what it does in that longer form.

**`imm_f`: this is the "else" value, and it is not always zero - the compiler's own witnesses were
simply the degenerate case.** Ten values (0, 1, 3, 7, 15, 31, 63, 64, 127, 128, 255) each dispatched
in their own kernel, `src` fixed to force each branch (0 -> pass, 100 -> block): **in every case,
`dst == imm_f` exactly when `src >= HI`**, and `dst == mask` exactly when `src < HI`, with no
dependence between the two. A follow-up varied `mask` across five values (0xffff/0xaaaa/0x5555/
0x0/0x1234) at a fixed nonzero `imm_f=200`: the pass branch tracked `mask` exactly and the block
branch stayed at exactly `200` regardless of `mask` - ruling out any interaction between the two
operands in either branch.

**Corrected statement of op11456, superseding section 82's "else 0":**

    dst = mask if src <u HI else imm_f

matching the opcode's own name in `isa/g17-contract.jsonl` (`"name": "csel"`) exactly - a genuine
conditional SELECT between two operands, not a masked zero. Every real compiler-emitted witness in
this campaign's own corpus census (section 82's `isa/g17-corpus-programs.jsonl` cross-tabulation)
happened to carry `imm_f=0`, which is why the "else 0" reading survived as long as it did - nothing
in the corpus ever exercised the general case. This closes section 45's op11456 line completely: role,
HI, and now both remaining immediates are measured.

## 90. The tensor load's (op12674) slot field feeds the SAME scoreboard as the ordinary load - closed by mutation, not by analogy

Section 45 named this the one remaining "by analogy" item: batch 20 (section 28) proved by mutating
a real hardware dispatch that op12709's slot field is read by the MMA's flags-word wait mask, but
this campaign's own general lowering (`lower.py`) never uses the library's tensor load (op12674) at
all - it does per-lane gathers with the ordinary load exclusively (section 32) - so no prior batch
had a tload-based kernel on hand to run the same mutation against. `memenc.py`'s own `tload()`
encoder has assumed the identical word layout since it was written, unverified.

**Found a clean single-MMA case already sitting in this campaign's own corpus.** `mm16_mul`
(Apple/matmul2d-compiled, `mm16_tree.py`'s own target) decodes to exactly one op5107 MMA and four
op12674 loads, all four at slot 7 (`word=0x800000`), with the MMA's wait word `0x80000000` - bit31
set, every other wait bit clear - a single, isolated slot-7 dependency, the cleanest form of the
exact experiment batch 20 ran for the ordinary load.

**First attempt at the flip used the wrong physical bit and is worth naming, since it repeats
sections 84/87/88's exact shape of mistake.** Assumed the decoded value's bit31 lives at byte3 bit7
of the instruction (a positional reading) - flipping that bit changed nothing. A direct bit-flip
census on the real instruction (mirroring what closed byte0[7] in section 88) found the actual
physical location: **byte0 bit3**, which toggles the decoded word by exactly `-2^31` - matching this
campaign's own already-published field map (`byte0[3] = bit31, the load-use wait bit`, section 25) for
the ORDINARY load/MMA pair, now confirmed to hold for op12674 too.

**The mutation, on real hardware, in a fresh dedicated harness (not reusing `mutate_code.py`'s A@B+C
comparison, which is wrong for `mm16_mul`'s no-accumulate `op5107` - checked and built correctly
before touching anything, matching this campaign's own data convention from `mm16_tree.py`, not the
fragment layout used for hand-authored MMA carriers elsewhere in this doc).** Baseline (unmodified
`native.metallib`): 10/10 dispatches match `A @ B` exactly, confirming the harness itself before
mutating anything. Mutated (byte0 bit3 flipped, clearing the MMA's only wait bit, leaving all four
loads' own slot-7 fields untouched): **50/50 dispatches across five separate process runs returned
exactly zero**, not scattered garbage - the identical deterministic signature batch 20 found for the
ordinary load ("the MMA read its fragments before any load landed").

**This closes section 45's item by mutation rather than by analogy.** op12674's slot field is
confirmed, causally, to feed the same 8-slot scoreboard the MMA's flags word already reads for
ordinary loads - not merely consistent with it by construction. The tensor store side (op17257)
was not re-tested here: `mm16_mul`'s own two stores follow the MMA with no intervening wait field,
consistent with the already-established straight-line rule for ALU/MMA-fed stores (section 15/26),
so nothing about this result calls that into question.

## 91. The tensor loads' byte6[4:3] "counter" closed: it is `slot % 4`, not an independent field - and this explains, rather than contradicts, an older negative result

Section 45's other tensor-load item, and older than this campaign:
`ledger/g17-load-counter-is-not-a-scoreboard-slot.toml` (2026-09-04, a different lane, checked
before probing per this campaign's own standing rule) measured op12674/op12675's `byte6[4:3]` as a
mod-4 counter that "advances in groups" and, with a properly-controlled reuse-after-consumption
test, found it **indistinguishable from an arbitrary mod-4 rotation** - a clean, correctly-reasoned
negative (rejecting "this is itself a scoreboard slot"), dated before this campaign's own section 28
pinned down the REAL slot field (word bits 20-27, closed by mutation in section 90).

**Direct test: does `byte6[4:3]` simply equal the real slot, mod 4?** Checked against every real
op12674 instance in `isa/g17-corpus-programs.jsonl` (claude/g17-isa-cartography, read-only): decode
each instance's word field to get its actual scoreboard slot (the field section 90 just confirmed by
mutation), take `slot % 4`, and compare against `byte6[4:3]` directly.

    op12674 (tensor load):        7,853 / 7,853 instances match, exactly - zero exceptions
    op12675 (masked tensor load): 4,452 / 4,452 instances match, exactly - zero exceptions

**`byte6[4:3]` is the low two bits of the already-known slot field, not a separate mechanism.** This
also explains the 2026-09-04 ledger's own result rather than conflicting with it: that ledger tested
whether `byte6[4:3]`'s OWN values behave like a valid scoreboard slot under a reuse-after-consumption
test, at a time before the real 8-valued slot field (word bits 20-27) had been separately identified.
A mod-4 projection of the true slot aliases two real slots together (0 and 4 both read 2 as low bits;
1 and 5 both read 1; etc.), which degrades exactly the kind of reuse-timing test that ledger ran -
consistent with getting a score statistically identical to an arbitrary rotation, since half the
real distinctions the test needed were already erased by the projection before the test began.

**Not re-tested here**: whether `byte6[4:3]` is a genuinely separate hardware field that happens to
be redundant with slot's low bits (an encoding/alignment artifact), or whether it is LITERALLY the
same physical bits as part of the slot field's own encoding (i.e., not two fields at all, just one
read twice under two different names in this campaign's own prior censuses). The 100% correlation
across 12,305 real instances doesn't distinguish those two readings, though it settles the practical
question either way: nothing downstream needs to track `byte6[4:3]` independently of the slot field
section 90 already closed. `op17257`/`op17258` (tensor stores) were checked and excluded from this
count - they don't carry a `(slot+1)<<20`-style field in the same word position, consistent with the
already-established straight-line rule for stores (no explicit wait/slot on the consumer side).

## 92. The tensor load's mode word and index-register formula, closed by a full single-bit census - both are ordinary fields, and one genuinely open detail found along the way

Section 45's last named encoding gap for op12674: "the mode word and index-register formula." A
full single-bit-flip census of the 12-byte `tload12` reference (`070047400800780a10808008`,
`memmap.json`'s own reference witness) - flip every one of 96 bits, decode, record which of the
nine named operands moves and by how much, the same method that closed byte0[7] (section 88) -
answers both cleanly, plus surfaces one genuinely open detail not to be glossed over.

**Mode word: exactly one bit, nothing more.** Only `byte11[3]` (weight 2048) moves operand2 (the
"mode" field, base 241) at all - matching this doc's own prior finding (section 12: "bit 11 of the
mode word marks that form") exactly, and now confirmed as the FIELD'S ENTIRE variable content: no
other bit within the reference's fixed `0xF1` base is independently variable. The five set bits of
`0xF1` are part of op12674's own fixed identity (consistent with the census's own opcode-
discriminating bits: dozens of single-bit flips elsewhere in the same instruction reach neighbouring
opcodes - op12675, op12677, op12656, op12646, op12079, op12353, op12665, op12647, op12710,
op13483/2-byte, op17221 - i.e. this instruction sits in a dense, single-bit-adjacent opcode
neighbourhood, and "241" is where op12674 lives in it, not a semantically-decomposable flag byte).

**Index register: an ordinary 7-bit register-id field, no special formula.** `byte3[1..7]`
(weights 1, 2, 4, 8, 16, -32 inverted, 64), base 137 = R16 (matching `t.index_reg`'s own convention
used throughout this campaign's carriers). Section 12's "the row stride is not an instruction field;
it is folded into the per-lane index register" already correctly located WHERE the row-stride
arithmetic lives (in the register's own precomputed value, done by separate ALU instructions in the
kernel body); this census confirms there is no additional per-instruction "formula" beyond a
standard register selector - the arithmetic itself is outside op12674's encoding entirely, which is
consistent with, not a correction to, what was already known.

**Bonus, not previously fully censused: the destination tuple's own register field.** `byte0[4..7]`
(1,2,4,8), `byte2[6]` (-16, inverted), `byte2[7]` (32), `byte7[4]` (64) - a clean 7-bit field,
same shape as the index register's.

**Byte offset: confirmed as a clean 8-bit field**, `byte8[5..7]` + `byte9[0..4]` (weights 1-128),
matching section 12's "BYTE OFFSET imm" with exact bit positions now identified for the first time.

**One detail found, corpus-checked, and left open rather than overclaimed.** The single-bit census
also touched the "last-use" (operand6), "element code" (operand8) and "mask" (operand9) fields, and
two of the three reached values this campaign's own corpus (`isa/g17-corpus-programs.jsonl`, 7,853
real op12674 instances) never shows: element code is 2 in 7,853 of 7,853 real instances (never 4,
despite section 12's "4 = 32-bit" claim for the opcode generally - that value may belong to a
different form or opcode entirely, not established here); mask is 15 or 0 in 7,853 of 7,853 (never
any intermediate value); last-use is 0 or 16 in 7,853 of 7,853, matching the documented convention
exactly, with the census's second bit on that field (`byte2[4]`, weight 32) also unwitnessed. So the
census found real, decodable encodings beyond what Apple's compiler ever emits for this opcode - a
fact worth recording, not a claim about what those encodings mean, since nothing here observed them
executed.

**Explicitly not resolved here**: operand1 (the already-closed "word"/slot field, sections 25/28/90)
showed sensitivity to several further bits in this same census, at `byte7[5..7]`, `byte8[3..4]`, and
a wide tail at `byte10[3,5,6]` and `byte11[0,1,4,5,6,7]`, with deltas in the billions/trillions -
values far too large to be a meaningful part of the same field as currently understood, and
overlapping the same byte range as the mode/element-code/mask fields just closed above. This reads
as either a genuinely wide decoder-level field only a small low slice of which is ever used, or an
artifact of how the model computes operand1 from bytes it shares with other operands - not decided
either way here, and no claim in this section depends on resolving it.

## 93. Transposed operands' address formula: section 37's claim re-confirmed by a cleaner comparison, and a separate, flagged-not-solved oddity found while picking the comparison pair

Section 45's "transposed operands' address formula" pointed back at section 37, which already
states the answer (transposed loads reuse the untransposed side's exact index register and
displacement pattern) but verified it through `tlower.py`'s own hand-authored lowering against the
oracle, not by diffing the compiler's own `mm_ta`/`mm_tb` objects directly. Did that diff here.

**First attempt picked the wrong comparison pair - caught before publishing anything from it.**
Compared `mm_ta` (`matmul2d_descriptor(32,32,64,true,false,false,...)`) against `mm_base`
(identical shape, `false,false,false`) expecting near-identical byte counts per section 37's own
"same instruction count" line. They are not close: `mm_base` decodes to 1330 bytes / 16 MMAs,
`mm_ta` to 626 bytes / 8 MMAs - a real, unexplained 2x discrepancy between two objects whose
`.metal` sources differ by exactly one boolean. **Not resolved here** - `results/` is entirely
untracked (`.gitignore` line 30), this campaign's own section 37 documents a same-day concurrent-
rebuild collision in this exact directory, and there is no git history to check whether these two
specific compiled objects were ever actually a matched pair. Flagged rather than chased down, since
it doesn't bear on the address-formula question below.

**The clean comparison: `mm_ta` against `mm_tb`.** Both compile to exactly 626 bytes and 8 MMAs -
a genuine apples-to-apples pair (`mm_ta`: `transA=true`; `mm_tb`: `transA=false,transB=true`,
otherwise identical). Decoded every op12674 load's `(slot, displacement, index register)` in both:
**all 16 loads in each object share the identical index register (id 149)**, and the displacement
PATTERN swaps exactly the way section 37 describes - `mm_ta`'s slot-0 load group uses displacements
`(4096, 6144)`, which is precisely the pattern `mm_tb`'s slot-1 group uses, and `mm_tb`'s slot-0
group uses `(32, 2080)`, precisely `mm_ta`'s slot-1 pattern. Transposing an operand does not change
which register or displacement its loads use; it relocates the SAME load pattern from one operand's
usual slot group to the other's, and the MMA's own type-code bit (section 37, already established)
carries the rest. This is a stronger confirmation than a byte-count comparison would have given,
since it directly shows the two transposed variants trading displacement patterns symmetrically
rather than merely having similar totals.

An ad hoc attempt to also confirm this functionally (dispatching `mm_ta` directly against a guessed
buffer layout) did not match a hand-computed reference - not published as a finding, since nothing
here established that the guessed dispatch parameters (buffer extent, slice semantics, thread grid)
were themselves correct, and section 37's own oracle-compared `tlower.py` verification already
covers functional correctness for this case through a harness with those parameters actually
nailed down. Left as unresolved effort, not a claim.

**More precisely characterised before leaving it, since a vague "discrepancy" is less useful to the
next person than a specific one.** `mm_base`'s LLVM IR (`input.ll`) differs from `mm_ta`'s at
exactly the same one point as the `.metal` source (`_ZTA...matmul2d_descriptorELi32ELi32ELi64EEE`'s
`i8 0` vs `i8 1` transA byte, nothing else) - so the two really were compiled from a genuinely
matched, transpose-only-differing source, at least up to the point where the `matmul2d` intrinsic
call is still opaque; the divergence happens entirely inside the library's own lowering of that
call. And the loads are not simply "more of the same": `mm_base`'s 22 loads and `mm_ta`'s 16 loads
reach the identical SET of byte-offset displacements (`{0, 32, 2048, 2080, 4096, 4128, 6144,
6176}`, both topping out at 6176) - `mm_base` does not read further into either buffer than `mm_ta`
does, it just issues six additional loads that repeat four displacements already used earlier in
the same object. So this is not "mm_base computes more of the K dimension"; it is some other
difference in how many times the library re-issues the MMA over the identical fragment set,
and resolving which behaviour is correct (or whether both are, via a fusion this campaign hasn't
identified) needs someone who can dispatch and check numerically against a properly-derived
transposed reference - attempted here twice without confidence in either result, not pursued
further this round.

## 94. The multi-MMA role of flags bits 33/41/47 and byte4's code: still inert, now tested where a role would actually matter

Section 45's last open item. Section 27's hardware census (`bitcensus_hw.py`) already showed bits
33/41/47 and the byte4 code are hardware-inert on an ISOLATED single MMA - but that census only had
a single-MMA object to test, so a role that only matters when a SECOND MMA reads the first one's
result (the natural place a "more coming" or synchronisation signal would live) was never actually
exercised. `mm_k32` (K=32, matmul2d-compiled, already in this campaign's corpus) is a real 2-step
chain into one accumulator: the first MMA carries the documented 0x20 "more accumulation follows"
flag, the second doesn't, and the second's C operand is the first's D - exactly the situation where
these bits could matter and never had a chance to be wrong before.

**Bits located fresh on this specific instruction** (not assumed positionally - the standing lesson
from sections 87/88/90 today): a full single-bit census of the chain's first MMA found bit 33 at
byte0[5], bit 41 at byte6[5], bit 47 at byte6[7] - matching the documented bit numbers exactly, at
physical locations specific to this instruction's own byte layout. The same census surfaced two
further bits not previously enumerated in this doc's flags-word account: byte4[5] (contributes bit
37 to the decoded word) and byte6[4] (weight 64, i.e. bit 6 - a much lower, previously unlisted
position).

**All five flipped individually on the chain's first MMA, redispatched, K=32 result checked against
`A @ B` on real hardware.** Baseline (unmodified): 5/5 correct. Each of bits 33, 41, 47, 37 and 6,
flipped alone: **5/5 correct in every case** - the chained accumulation is bit-for-bit unaffected
whether the second MMA of the pair still finds its predecessor's result correct or not. This
extends section 27's single-MMA negative to the one place a role was actually plausible and finds
the same answer: these bits are inert, not merely unobserved-as-nonzero. The byte4 two-bit code
itself was only reachable at one of its two component bits here (byte4[5]/bit37); the other half
of that pair was not independently isolated in this witness's own census and is not claimed either
way.

## 95. Spencer's milestone is merged into `main` - PR #31, `00dfc051`, 2026-09-18

Root merged `codex/g17-tensorops-main-integration` (the branch this campaign independently
reproduced end to end in section 80, `c82bb531`) into `main` via PR #31. Confirmed directly on
`origin/main` rather than taken from the PR title: `tools/g17tensorcommonruntime.py`,
`docs/archive/g17-tensorops-common-runtime.md`, and `agxforge/g17/mdgen.py`'s re-keyed
`SLOT29_BY_SYSTEM_REGISTER` (`130: 52`, with the file's own comment now citing "1,039 exact
(130,156) objects") are all present on `main` as of this commit.

The merge itself was verified by root's own process rather than re-running the full gate: the
merged tree is byte-identical to the tree already certified by the cold serial gate at
`89c58134af799f4649d8ad2bc286b5eac02cef19` (5,187 unit tests, runtime/ledger/pin checks, 183/183
regressions), so the gate was not rerun for the merge itself - a reasonable inference from an empty
diff against an already-certified tree, not a new claim this campaign re-verified independently.

**Spencer's exact milestone specification - "a composed GEMM + scalar program produced by the
ordinary compiler is authored into a repository-owned image, dispatched through the ordinary
runtime, and validated on hardware from a cold main checkout, with the system-register metadata
derived from a measured rule rather than a hard-coded tensor special case" - is now live on `main`**,
not just on a research branch. The PR description states the scope precisely, matching this
campaign's own section 80 framing exactly: the measured `(130,156)` class specifically, not general
TensorOps coverage; the 13 latent SR130-sharing forms found and closed within the hour back in
section 80 are named again here as explicitly outside the claim, consistent all the way through.

This closes the loop this campaign opened on 2026-09-16. What remains - other dtypes/shapes,
multiple simdgroups, the full cold release gate re-run on a wider TensorOps surface - was already
named as future work by the branch's own docs and by this campaign's memory notes, and stays out of
scope here as before.

## 96. The next queue: multi-GEMM composition (root's PR #33), and re-verifying this campaign's own prior art before anyone re-derives it

Root's PR #33 (`b2401ce4`, 2026-09-18) opened `docs/archive/g17-root-assignments-current.md`'s "Authoritative
queue: multi-GEMM TensorOps composition," starting from released baseline `main@8f23d314`. The
compiler owner's target is `GEMM1 -> scalar activation/transform -> GEMM2` "where GEMM2 consumes
GEMM1's stored result through the ordinary tensor-load path," through the real production compiler.
This session's own boundary is stated explicitly in the same queue text: "hand-authored measurement,
`results/`, `isa/`, and `docs/` only... must not edit `agxforge/` or `tools/`" - confirmed directly
against Spencer, matching root's own separate confirmation relayed the same day.

**Section 42 already answers the question the compiler owner is being asked to reproduce through the
real compiler - re-verified fresh today, not merely cited from an old JSON.** Section 42A's `arm='scalar'`
case in `compose.py` is exactly `GEMM1 -> scalar epilogue -> GEMM2` via the ordinary memory path (C1
stored, read back through a normal tensor load as GEMM2's A operand), authored through this
campaign's own `ownimage.py`/`scanlink.link` (not the production compiler frontend, but the same
downstream ABI/linking machinery). Re-ran it today, live, on current hardware:

    compose.py scalar 32 32 64 32   -> c1: [T,T,T]  c2: [T,T,T]  scalar: [T,T,T]
    compose.py scalar 17 32 19 32   -> c1: [T,T,T]  c2: [T,T,T]  scalar: [T,T,T]   (K1=19, a genuine
                                                                                    K-remainder shape)

Both still pass exactly as recorded when first built. This is the queue's own "memory-mediated
positive" this section is asked to keep documenting - already true, freshly reconfirmed, and now
explicitly the thing the compiler owner's task is to reproduce through `cc.py`'s real IR/ABI path
rather than an open question to solve from scratch.

**Section 42B's negative result has a precise root cause, found by the isa-cartography peer, not by
this campaign.** Feeding an op5106/5107 MMA's fp32 D-accumulator (an 8-word `GPR32tup8_alignedrc`
tuple) directly as the SAME opcode's fp32 A-operand (a 4-word `GPR32tup4_alignedrc` tuple) cannot
work regardless of byte ordering or transpose bit - half the fragment is structurally never read by
an operand four words wide. This replaces this section's earlier "some difference between the two
fragment conventions... is real and not isolated" with an exact, checkable cause (confirmed directly
against this campaign's own `mmaenc.py`: `a_n = 8 if a_type == 'float' else ... 4`, matching exactly).
The queue's own text already reflects this framing correctly ("a retained negative control; it is
not evidence that the normal store/reload composition route is impossible"), so no correction to the
queue itself is needed - this is a correction to this doc's own account of section 42B.

**The dimensionally-matching form the peer identified: op5098 (fp32 A x fp32 B, both A and D are
8-word tuples), attempted as a follow-on, not part of the queue's own ask.** `mmaenc.mma(a_type=
'float', b_type='float')` already selects op5098 directly (`a_n=b_n=8`), built and field-mapped in
this campaign's own batch 18/25 - unlike the peer's own contract-based authoring table, which cannot
reach this opcode (named as an authoring-table gap on their side). A one-hot D-fragment sweep (feed a
D with exactly one non-zero element as A, observe which output element moves, 8 dispatches recovers
the whole permutation) is the peer's proposed protocol for asking 42B's original question on a form
where it is dimensionally answerable. Attempted separately from, and not blocking, the queue's actual
ask above.

## 97. Section 42B's question, answered cleanly on the dimensionally-matching form: op5098's A and D share the IDENTICAL fragment layout - register-direct D-to-A chaining works, no permutation needed

The peer's proposed one-hot-D-sweep (section 96) turned out not to be necessary - a much cleaner
answer was already sitting in this campaign's own `layout-new_f32f32_nn.json` (an earlier pass,
6651 dispatches), just never compared against `transpose.py`'s `pos()`/`pos_b()` in the right
direction.

**First attempt used the wrong precedent and is worth naming, since it repeats today's own recurring
shape of mistake.** Assumed op5098's fp32 A operand (8-word tuple) uses the half-precision `pos_b()`
layout, by analogy with op5100's validated test (A:float, B:half) - checked before trusting it, and
it was wrong: `A_row`/`A_k` from the measured table disagree with `pos_b()` at 254/256 and 224/256
of the 256 (lane, slot) positions under either argument order, not a near-miss. An identity-matrix
dispatch built on this wrong assumption gave neither the "wrong" nor the "right" predicted answer,
which was itself the tell that the layout hypothesis, not just the register wiring, was the problem.

**The actual answer, found by comparing the measured table against itself rather than against a
half-precision precedent: `A_row == D_row` and `A_k == D_col`, exactly, at all 256 positions.**
op5098's fp32 A operand and the fp32 D/C accumulator are the SAME layout - not merely close, not a
permutation, identical. Confirmed independently with two hardware dispatches on the compiler-emitted
`new_f32f32_nn` object (unpatched - this object already emits op5098 as its natural op), each
checked element-by-element against a full 16x16 reference product, not a summary statistic:

    A = identity(16), D-layout bytes fed directly as A     -> D2 == B exactly
    A = a cyclic permutation (non-symmetric, rules out a    -> D3 == A_perm @ B exactly
        symmetric-input false positive), D-layout bytes
        fed directly as A

**This answers section 42B's original question on the one form where it is dimensionally askable: a
GEMM's D-accumulator CAN feed directly into the next GEMM's A-operand with no store/reload and no
permutation, when both operands are fp32 (op5098).** This does not extend to the ordinary half-
precision path (op5106/5107) that real GEMMs use - there, A is a 4-word tuple and D is 8-word, a
structural width mismatch no layout fix can repair (the peer's own diagnosis, section 96). This is a
narrow, real result about the fp32-fp32 accelerator form specifically, not a general finding about
register-direct tensor composition - stated at exactly that scope, matching this campaign's own
"the operand nobody wrote"/scope-precision standard from today's exchanges.

## 98. The practical bridge (half GEMM's D feeding op5100's A, matching B2) is NOT yet confirmed - a correction to what section 97 would suggest, and an honest stop rather than a forced conclusion

Spencer's sharpened question after section 97: op5098 is fp32/fp32 only, a narrow special case.
The practically useful bridge would be an ORDINARY half-precision GEMM's fp32 D-accumulator feeding
directly into an **op5100/5101** MMA's A operand (fp32-tagged, tup8, same width-selection mechanism
as op5098) paired with a **fresh half B2** - letting GEMM2 stay half-precision-in, not requiring both
sides to be fp32. Section 97's finding (fp32-tagged tup8 A shares D's own layout) would predict this
works too, by the same mechanism.

**Re-reading `composition.py`'s own `probe_B()` - the actual origin of section 42B's "0/3" - corrects
this document's own account of it.** `probe_B()` already constructs exactly this: `mmaenc.mma(d2,
d1base, b2buf, None, 'float', 'half', ...)` is op5100/5101, not op5106/5107. Section 42B's original
negative was NEVER a tup4-vs-tup8 width test - the peer's diagnosis in section 96/97, while correct
as a general fact about op5106/5107, does not explain probe_B's specific result, because probe_B
never used that opcode pair. This is a real correction to sections 96-97's framing, caught by reading
the actual generating code rather than trusting this document's own prior paraphrase of it.

**Re-ran `probe_B()` fresh: still 0/3**, confirming the negative is real and current, not stale.
Tried three things to find the cause, in order, each a genuine attempt with its own bug caught before
trusting the result:

1. Added an explicit wait on GEMM1's own scoreboard slot to the second MMA (probe_B's version only
   waits on B2's load slot) - no change, still 0/3.
2. Replaced GEMM1 entirely with a hand-placed, known fp32 pattern loaded via ordinary loads (ruling
   out anything specific to a FRESH MMA's output vs. a settled register value) - still 0/3, with the
   identical error magnitudes as with a real GEMM1, suggesting the cause is unrelated to timing or
   freshness.
3. Double-checked `author_5100.py`'s own proven op5100 pattern still passes fresh (4/4, unchanged) -
   ruling out a regression in `mmaenc.mma`'s encoding of op5100 in general. That test places A at
   register base 8; `probe_B`'s D1 (and this section's own attempts) sit at base 24. Whether base
   specifically matters was not conclusively isolated before time on this side-investigation ran out.

**Left honestly unresolved rather than forced to a conclusion.** Section 97's op5098 result stands
on its own (confirmed twice, independently, against a direct table comparison). It does NOT establish
that the same holds for op5100/5101 chaining from an ordinary half-precision GEMM - that remains
exactly where section 42B originally left it, an open negative, now with three ruled-out explanations
(wait timing, MMA freshness, general encoder regression) rather than zero. The next useful step, if
picked up again, is checking whether A's register BASE (not just its type tag) affects this specific
form - untested here.

## 99. URGENT: real doubt found about the exact bridge PR #36 is now building toward, before any side has dispatched it on hardware

PR #36 (`codex/g17-multigemm-compiler-20260918`) advanced fast today (commits `66c118e5`, `f7007d46`,
`c7432daf`) to a composed slice chaining a real half-precision GEMM1 (32x32x64) into a
**GEMM2 with dtype (float, half)** - i.e. exactly the op5100/5101 shape section 98 was investigating.
Checked directly: their own new test is named
`test_multigemm_memory_chain_contract_is_admitted_without_dispatch` - **compile/contract evidence
only, no GPU dispatch has been run on their side yet.** This section's own hardware attempts are
therefore the only real data on whether this bridge works at all, and they have NOT reached a clean
positive.

**Ruled out cleanly, with controlled tests, each isolating exactly one variable:**
- Register base (24 vs. the known-good 8): **not the cause** - a clean hand-built op5100 test with A
  fed a known pattern at base 24, own buffer/load conventions fully corrected, passed 5/5.
- A register collision with tlower's own working set: found and fixed a real one (tlower's generated
  body actually uses up to 69 registers, not the ~33 estimated - my own earlier scratch register
  choices at 44/64 were inside that range); moving to 104/112 changed nothing, ruling this out as
  the cause of the chain's own failure specifically.
- MMA-to-MMA timing/synchronisation: added an explicit tag on GEMM1's own MMA and a matching wait on
  the second MMA (bit 30, physically located fresh on each instruction, not assumed) - **identical
  error values, to the decimal**, with and without. This is strong evidence the failure is not a
  race or a missing wait.

**A new, unresolved, and more concerning fact, found while isolating the above:** GEMM1's own real
D-accumulator, read back via the SAME `memenc.store` + `pos()`-layout convention that section 97
confirmed correct for op5098's D (and that this section's own hand-built op5100 test just confirmed
correct for hand-placed data), does **NOT** match the reference when the data comes from a real
op5106/5107 MMA - 224 of 256 elements wrong, cleanly reproduced. The SAME registers, read back via
the library's own tensor store (`tstore`) instead, give the exactly correct row-major result. Combining
both readback methods in one body (to directly map physical (lane,slot) positions against the trusted
`tstore` output) corrupted the `tstore` result too, before yielding a clean answer - not chased
further tonight.

**What this means, stated carefully:** it is not established that op5106/5107's own D-accumulator
uses the identical physical layout that op5098/5100's D and A operands share. Section 97's "A and D
share the identical layout" finding was measured for op5098 specifically (fp32/fp32) and confirmed
via `author_5100.py`'s pre-existing evidence for op5100 - **neither of those tests ever involved a
live op5106/5107 output**, only hand-placed data or op5098's own natural compiled form. The
half-precision-GEMM-into-float-operand chain PR #36 is now building has NOT been shown to work on
real hardware, and this section's own attempts, despite ruling out three plausible causes cleanly,
have not produced a passing chain either.

**Recommendation, stated plainly given the pace of that PR:** do not treat PR #36's multigemm slice
as validated until a real GPU dispatch (not a contract/compile check) confirms the (32,32,64)
half-half GEMM1 into (32,32,32) float-half GEMM2 chain numerically, ideally via
`tools/g17tensorcommonruntime.py`'s own multigemm path once it is actually run rather than only
prepared. This is a second, independent flag beyond P1/P2 - not a new bug in `cc.py`, but an
open question about whether the underlying ISA behaviour the candidate assumes is even correct.

## 100. Section 99 resolved: op5106/5107's D reads via `pos_b()`, not `pos()` - a real, precisely-characterised layout mismatch, not a mystery, and the actionable bridge answer

Assigned lane after Spencer's three-lane framing (root frozen on PR #36, this session on "direct
register-resident GEMM chaining", the ISA session on full-surface closure): resolve section 99's
open question rather than leave it flagged. Resolved cleanly.

**Method.** Two separate, clean dispatches with the identical seed/data (avoiding the
store-mechanism collision that corrupted the combined attempt in section 99): one storing GEMM1's
raw D-accumulator per-lane via ordinary `memenc.store` (no unscrambling), one storing the SAME D1
via the library's own `tstore` (trusted, already confirmed row-major correct). For every one of the
256 `(lane, slot)` positions in the raw fragment, found the exact matching `(row, col)` in the
trusted output by value - a full, unambiguous, zero-collision mapping (`argwhere` on floating point
equality gave exactly one match per position, not an approximate or partial one).

**Result: the mapping is `pos_b()`'s own inverse, exactly - 256 of 256, zero exceptions.** Not a
new formula, not a permutation of `pos()` - `transpose.py`'s existing `pos_b()` (the layout this
whole campaign has always used for the B operand) describes op5106/5107's own D-accumulator when
read through an ordinary per-lane store. Section 97's "A and D share the identical layout" (which
used `pos()`) was correct for op5098 specifically and does NOT generalise: op5098's D uses `pos()`,
op5106/5107's D uses `pos_b()`, despite both being the same `GPR32tup8_alignedrc` register class.
The layout convention is a property of the OPCODE that wrote the accumulator, not just its register
class - a real, load-bearing distinction this campaign had not drawn before tonight.

**Correction (section 129, added later).** The attribution to the OPCODE is not supported. `pos_b(k, c) = pos(rotl1(k), c)`: same lanes and slots, rows relabeled by a one-bit rotation, and the measured
hardware tables give one fragment map for all forms (A == D slot for slot). What differs between the two readings is the row labeling of the kernel that produced the D, not the opcode. I did not re-run this section's
kernel to find which labeling its loads used.

**This fully explains every chain failure in section 99, cleanly, with no remaining mystery.**
Feeding GEMM1's raw registers (arranged `pos_b()`-style) directly into op5100's A operand (which
reads `pos()`-style, established in section 98) reads every element from the WRONG physical
position - not randomly, but through one specific, now-known, wrong permutation. That is why every
attempt gave byte-for-byte identical wrong answers regardless of register base, explicit waits, or
tags: none of those variables mattered, because the actual defect was a static, deterministic
mismatch present from the first attempt.

**Tested and confirmed the naive fix does NOT work, for a precise reason worth stating rather than
leaving implicit.** A same-pattern store-then-reload (store D1 out via ordinary addressing, reload
it back via the identical addressing) round-trips the SAME `pos_b()` arrangement unchanged - no
repack happens, because both sides of that round trip use the same linear `index*width+disp`
addressing, which cannot express a genuine cross-lane permutation. Confirmed on hardware: still
0/1, identical wrong answer to every register-direct attempt.

**The actionable answer for the compiler.** A genuine `pos_b()`-to-`pos()` repack needs an
addressing scheme that computes per-ELEMENT row/col addresses, not a per-lane linear index - i.e.
the library's own **tensor** store/load path (`tstore`/`op12674`), not the ordinary per-lane
store/load this section's negative tests used. This is exactly the path `compose.py`'s own
`scalar` arm already uses successfully (sections 42A/96/98, re-verified fresh this session,
3/3 on two shapes) - GEMM1's result reaches memory in true row-major form via `tstore`, and
whatever reads it next (a scalar op, or another GEMM's tensor load) picks it up correctly because
tensor addressing - unlike a flat register move - already accounts for the layout. **For this
specific opcode pairing (ordinary half GEMM1 into an op5100-style GEMM2), there is no zero-cost
register-direct forward; the memory-mediated path is not merely a fallback, it is the only
demonstrated-correct route, and it is already proven correct.**

**Read on PR #36 specifically, stated carefully.** This does not necessarily mean PR #36's own
approach is broken - an ordinary compiler lowering two separate tensor operations would naturally
route the first body's output to memory via its own tensor store and the second body's input via
its own tensor load (matching normal codegen for independent operations, not a special optimisation)
rather than attempt a bare register alias. The risk section 99 named - trusting a compile-only test
as evidence the numerical chain works - stands regardless: whatever addressing PR #36's lowering
actually emits for the boundary between its two bodies needs a real hardware dispatch to confirm it
follows the tensor-addressed path (known-correct) rather than a plain register carry-forward
(now demonstrated wrong for this exact pairing).

## 101. Section 100's actionable bridge confirmed end-to-end on hardware, not just by precedent - the compiler-realistic path is already proven, and it is already in this repo

Closes the one item section 100 left open ("has not been confirmed end-to-end on hardware"). Rather
than hand-build a new confirming carrier, found that `compose.py`'s `chain` arm already **is** this
exact test, and re-ran it fresh rather than trusting the earlier summary secondhand.

**What it actually tests.** Two SEPARATELY invoked `tlower.lower()` calls - not a hand-rolled
register alias, not a bespoke tensor-op carrier - connected purely through memory, exactly the shape
a real compiler emits for two independent tensor ops: GEMM1 (`a_type='half', b_type='half'`, the
ordinary op5106/5107 path) lowered with `end=False`, its OWN generated store tail writing C1 into
buffer binding 2; GEMM2 (`a_type='float', b_type='half'`, op5100/5101 - the exact pairing sections
98-100 characterized) lowered separately with `binds=(2, 1, 2)` so its A-operand load code reads
directly from the SAME buffer GEMM1 just wrote, at the same offset, via GEMM2's own generated
prologue/`row_load` (an ordinary `memenc.load` per fragment row, addressed through index registers
`tlower.lower` computes from scratch for GEMM2 - not tied to GEMM1's index registers or accumulator
registers at all). No `tstore`/`tload` are even used here for the fp32 boundary specifically -
`row_load`'s `words == 4` branch (`tlower.py`, the fp32 case) uses plain `memenc.load` with a
displacement computed by `based()` from GEMM2's own freshly-generated per-lane index register. The
correctness therefore rests entirely on `tlower.lower` generating BOTH ends (the store after GEMM1,
the load before GEMM2) from its own consistent row-major addressing convention, with no assumption
about what physical register layout GEMM1 happened to leave data in - which is exactly why this
works despite section 100's finding that a raw register carry-forward does not.

**Result, run fresh just now, three independent random trials per case:**
- `chain 16 16 16 16` (plain, square, exact tile): C1 3/3, C2 3/3, bit-exact.
- `shift 16 16 16 16` (GEMM2 reads C1 shifted one row down - every lane's GEMM2 load reads a value a
  DIFFERENT lane's GEMM1 store wrote, a real cross-lane store-to-load stress case, not just a same-
  lane round trip): C1 3/3, C2 3/3, bit-exact.
- `chain 32 16 19 16` (M=32 spans two row-tiles, K1=19 is a genuine non-multiple-of-16 remainder):
  C1 3/3, C2 3/3, bit-exact.

**Conclusion.** The memory-mediated GEMM-to-GEMM bridge for exactly this opcode pairing
(op5106/5107 -> op5100/5101) is not a hypothesis by precedent anymore - it is hardware-confirmed,
cross-lane-store-to-load-safe, and remainder-safe, using nothing but two ordinary, separately
generated `tlower.lower` bodies joined at a shared buffer offset. This is the concrete template for
what PR #36's own lowering should be emitting at its GEMM1/GEMM2 boundary: two independently
generated bodies meeting at a plain memory buffer, not a register alias. Relaying this positive
result alongside section 100's warning, since it is the constructive half of the same finding: the
register-direct path is a proven dead end (section 100), and the memory-mediated path is now a
proven, working replacement (this section), so there is a known-good answer to build toward rather
than an open risk to merely flag.

## 102. Next bridge-matrix cell (op5104/5105, half-A/fp32-B) opened but NOT resolved - honest stop, precedent does not transfer, needs its own derivation

Continuing the ambitious bridge-matrix goal past the op5106/5107<->op5100/5101 pairing (sections
99-101, now closed). op5104/5105 (`a_type='half', b_type='float'`) is the mirror pairing - real and
compiler-emitted per `mmaenc.mma` (no "never compiler-emitted" caveat, unlike op5100/5101) - and had
not been characterized in this campaign at all.

**What was tried.** The standard identity-times-distinct-values probe (A = identity, half, tup4,
laid out via `pos()` - the convention every half A operand in this entire campaign has used
successfully, including every native op5106/5107 GEMM): if A is truly identity, D should equal B
exactly wherever B was placed correctly. Hit and fixed one real bug first (omitted the scoreboard-
wait ALU instruction between the loads and the MMA, the same `faddenc.encode(... word=(1<<24)|...)`
idiom `op5100_regbase_test.py` uses - without it D came back all zero, a harness mistake, not a
finding). With the wait fixed, D is non-zero and genuinely responds to B's placement (pos() and
pos_b() now give different, non-matching results, unlike before the fix), but **neither placement
gives D == B**, and value-matching D's 256 cells against B_test's 256 distinct values (`arange(256)`)
found only 30 unique targets, not the 256-way bijection a true identity-times-permuted-B would
produce. Checked pos()/pos_b() under both row/col orders (4 candidate tables) against the raw
value-match mapping: 2/256, 1/256, 0/256, 1/256 - all consistent with noise, none remotely close to
section 100's 256/256 signal.

**Conclusion: precedent does not transfer, and the honest state is unresolved, not wrong-but-close.**
Either A's own fragment convention is opcode-specific here too (i.e. `pos()` may not describe how
op5104/5105 reads its half A-operand, despite reading identically-typed/classed data as every other
half A operand in the campaign), or B's true layout is some third arrangement entirely uncharacterized
by `pos()`/`pos_b()`, or both are simultaneously wrong and confounding each other (the identity trick
cannot distinguish these cases on its own, since a wrong A-layout turns "identity" into some other,
unknown-to-us matrix, and the resulting D is then no longer expected to equal B under any B-layout).
**Not forcing a conclusion here** - this needs the same rigorous, one-unknown-at-a-time derivation
this campaign used originally to first establish `pos()`/`pos_b()` (pin one operand with a
census/one-hot method, not an identity shortcut that assumes the other operand's layout is already
known). Left open for the next work session; not blocking anything currently in flight (unlike
sections 99-101, this pairing is not what PR #36 is building toward).

## 103. The memory-mediated bridge generalizes to op5104/5105 too - confirmed end-to-end, WITHOUT needing to resolve section 102's open register-layout question at all

Direct follow-on to section 102's honest stop. Rather than keep chasing op5104/5105's raw register
layout, asked the more practically important question first: does the SAME memory-mediated pattern
that section 101 proved for op5100/5101 (two separately-lowered `tlower.lower` bodies joined only
through a shared buffer) also work when the roles are swapped - a prior GEMM's D fed as THIS
opcode's B-operand (fp32, half A) rather than its A-operand?

**Built and ran fresh**: GEMM1 (half x half, ordinary op5106/5107) stores C1 into buffer binding 2.
GEMM2 lowered separately with `a_type='half', b_type='float'` (op5104/5105), A2 fresh half data from
buffer 0, **B read directly from C1's own buffer/offset** (`binds=(0, 2, 2)`) - GEMM2's own generated
B-loading code (`tlower.py`'s `row_load`, `words==4` branch, the same ordinary `memenc.load`
mechanism section 101 already described) reads C1 with no knowledge of, or dependency on, whatever
raw register layout op5104/5105's B-operand actually uses internally.

**Result: 3/3 bit-exact** (`compose_5104_bridge.py`, `M=K1=N=16`, C1 and C2 both exact against the
`gemm_ref`/`trunc_f32`-based reference).

**Why this matters more than it might look.** Section 102 could not pin down op5104/5105's own
fragment layout and explicitly declined to guess. This result shows that doesn't block the practical
bridge-matrix deliverable at all: the memory-mediated pattern is layout-agnostic by construction -
each `tlower.lower` call generates BOTH its own consistent write-side and read-side addressing, so
neither side of a memory-mediated composition ever needs an externally-supplied answer to "what raw
layout does this opcode use." **The generalizable finding for the whole bridge matrix**: for any pair
of opcodes in this family, the memory-mediated path is expected to work by construction and does not
need per-pair register-layout characterization to trust; only the REGISTER-DIRECT path (no store/
reload) needs that characterization, and that is the genuinely hard, opcode-specific, uncharacterized
part (section 102's still-open item, and generally not worth resolving unless there's a concrete
reason to want a zero-copy register-resident chain rather than the always-safe memory-mediated one).

## 104. The compiler-consultable decision table (ambitious goal items 2-3): what is actually known across all 10 opcodes, and the measured cost of the always-safe bridge

Continuing Spencer's ambitious roadmap directly: item 1 (finish op5100 properly) is done (sections
98-101). This section is the honest current state of items 2-3 - a real decision table, not a
width-match guess, and a measured bridge cost, not a bare "works/doesn't."

**Decision table (D = producer's accumulator layout when read via ordinary store; A/B = consumer's
operand-read layout).** `pos()` is D/A's native fragment layout, `pos_b()` is B's; "measured table"
means a closed form was not derived, only a direct lookup table (still exact, just not yet reduced
to `pos()`/`pos_b()` algebra):

| Opcode | D (producer) | A (consumer) | B (consumer) | Source |
|---|---|---|---|---|
| 5098/5099 (fp32.fp32) | `pos()` | `pos()` (== D exactly) | measured table, distinct from `pos()`/`pos_b()` under either argument order | section 97 |
| 5100/5101 (fp32 A, half B) | not measured (never compiler-emitted per `mmaenc.mma`'s own comment - decoder-reachable only) | `pos()` (confirmed via hand-placed data at two register bases) | `pos_b()` (used successfully throughout, e.g. `op5100_regbase_test.py`) | sections 98, 101 |
| 5104/5105 (half A, fp32 B) | not measured | not measured (identity-shortcut inconclusive, needs real census) | not measured (same) | section 102, OPEN |
| 5106/5107 (half.half, the native GEMM path) | `pos_b()` **(not `pos()` - opcode-specific, section 100's headline finding; section 129: the same map as `pos()` with rows rotated one bit, not an opcode property)** | `pos()` | `pos_b()` | sections 97 (contrast), 100 |
| 10384/10385 (int8.int8) | not measured | not measured | not measured | untouched this campaign |

**The eligibility rule this table actually supports today, stated the way a compiler would need it:**
`if producer_op == consumer_op's own native producer AND D_layout(producer) == read_layout(consumer's
matching operand): forward directly; else: bridge.` The only CONFIRMED-good direct-forward case in
this table is trivial (an opcode's own D feeding right back into more of the SAME accumulate chain,
which every K-loop in this campaign already relies on and which `tlower.lower` already generates
correctly). The only CROSS-opcode case actually tested (5106/5107's D into 5100/5101's A) is a
CONFIRMED NEGATIVE. Given real width-matched pairs still fail (section 99's original discovery) and
half the table is simply unmeasured (5104/5105, 10384/10385, and 5100/5101's own D), **the honest
decision table today is: treat every cross-opcode register-direct forward as unverified-therefore-
unsafe by default, until a specific pair is hardware-confirmed** - not a permissive default that only
blocks the one pair caught failing.

**Correction (section 125, added later).** The premise that the only cross-opcode case tested (5106/5107's D into 5100/5101's A) is a confirmed negative,
and the default drawn from it, are superseded for the pairs in section 125: on one `matmul2d` object the library forwards op5106/5107 accumulators
directly into op5100/5101 and op5104/5105 operands with zero instructions between, exactly and with the bridge's numerics. The table's D column is also
wrong for the raw registers: all 23 accelerator forms measured have `pos_b()` as their destination layout, including op5098, op5100 and op5104, and
5100/5101 and 5104/5105 ARE emitted (under `relaxed_precision`). Chains between GEMMs of different descriptors remain unmeasured; the bridge stays the
safe default there.

**Current hardware-backed eligibility table: section 132.** Everything not listed there as eligible stays on the memory bridge.

**Bridge cost, measured, not asserted.** Decoded real instruction counts (not `tlower`'s internal
`ops` bookkeeping, which folds each prologue into one entry): a standalone half/half 16x16x16 GEMM
body is 25 real instructions; a standalone float/half 16x16x16 GEMM body is 26; a standalone
half/float (5104/5105) body is 26. `compose.py`'s actual fused `chain` object (both bodies plus one
shared `END`) decodes to exactly **50** real instructions - almost exactly the sum of the two
standalone bodies (51) plus one `END`, not sum-plus-something. **The memory-mediated bridge adds
no separate repack step and no separate instructions beyond each body's own ordinary store tail and
load prologue** - the 1-2 instruction difference between the naive standalone-sum estimate and the
measured fused total is well within what changing `binds`/`offsets` parameters alone would explain
(different constant-folding in the address-base arithmetic `based()` performs, not a hidden
reformatting cost), not evidence of an extra operation. **This directly answers item 3's "cheapest
bridge, with a real measured cost, not just works/doesn't":** for this opcode family, the
memory-mediated bridge's marginal cost over two independent, already-necessary GEMM lowerings is
approximately zero - there is no cheaper alternative to chase (no subtuple/reinterpret split or
repack op could beat "zero additional instructions"), which resolves that half of item 3 rather than
leaving it as a costed-but-unexplored option.

**What remains of the ambitious plan.** Filling in the table's blank cells (5104/5105's own layout,
5100/5101's own D, all of 10384/10385) needs the original one-hot/census method, not shortcuts -
substantial, standalone work, and not gated on anything currently blocking PR #36 (section 103
already showed the memory-mediated answer does not need these cells filled in to be trusted). Item 4
(generalizing `csel_family_probe.py` into a swept tool, and the "13 latent SR130-sharing forms")
is a different, ISA-authoring-layer question (`scanlink.author`'s byte1-only key ambiguity, section
80/62) rather than a tensor-chaining one - flagged here as explicitly out of this section's scope,
better suited to whoever owns that authoring-path thread, rather than force-fit into this table.

## 105. int8 (op10384/10385): the last decision-table blank attempted, partially confirmed, raw D-layout left honestly open after three real carrier bugs

Per Spencer's steering (move to int8, stop digging at op5104/5105 unless a concrete need arises).

**Confirmed, and it fills in a real decision-table cell.** An old pre-session finding (doc section
5: "the f32/int8 layouts... are the SAME as f16's for A, B and D") turns out to be about the
WRITE-side packing convention specifically: `run_lowered.py`'s own `pack_A`/`pack_B` already place
int8 tiles via `to_frag(..., at=pos)` / `at=pos_b)` - the identical convention as half. Re-verified
fresh: a plain `tlower.lower` int8x int8 GEMM through the proper `pack_A`/`pack_B`/`pack_C` path is
bit-exact against `gemm_ref` (integer=True). So int8's A/B input-side convention is settled: same as
every other opcode in this family, `pos()`/`pos_b()`.

**Not resolved: int8's raw D-layout under ordinary `memenc.store` (the section-100-style question).**
Three real bugs, found and fixed in sequence, before giving up rather than continuing to guess:
1. A hand-rolled carrier using the generic 4-word `memenc.load` for int8's 2-word (tup2) A/B operands
   is simply the wrong instruction - `tlower.py`'s own int8 path uses `memenc.tload1w` (a genuinely
   different, byte-addressed, one-word load), documented elsewhere in this file as "int8 needs a
   one-word load form". Fixed by reusing `tlower.lower`'s own generated int8 loading code instead of
   hand-rolling it.
2. Even with correct loading code, feeding a plain row-major `uint8` buffer as A/B is wrong: int8
   tiles need the padded fragment-block layout `pack_tile` produces (each lane's 8 real int8 bytes
   padded out to a 16-byte block, matching the uniform per-tile block convention every dtype in this
   campaign uses) - not a dense row-major array. Fixed by using `pack_A`/`pack_B` directly (confirmed
   above).
3. With both of those fixed, a hand-appended raw `memenc.store` pair after `tlower.lower`'s own
   (truncated) int8 MMA still gives a degenerate result - only ~33 unique values across 256 raw
   cells (not the full-256-unique bijection section 100 got cleanly for f16, and reproducible at
   that same ~33 count across two different random seeds, so it is a real, structural problem with
   this carrier, not sampling noise). A missing scoreboard wait between the MMA and the appended
   store was the first suspect (every successful raw-store probe elsewhere this session added one)
   but is RULED OUT here: decoding `tlower.lower`'s own full, proven-correct int8 body shows its own
   `tstore` pair follows the MMA with no wait instruction in between either, and `memenc.tstore`'s
   own signature has no wait parameter at all - so an immediate store after this specific MMA is
   the library's own normal pattern, not a hazard needing a manual fence. Printing the actual raw
   fragment instead of just its match count sharpened the diagnosis: only 8 of the 32 lanes carry
   any nonzero data at all (a `lane % 8 in {0, 2}` pattern), and even those 8 lanes are zero in
   slots 4-7 - i.e. the second `memenc.store` call (`disp=512`, meant to capture the accumulator's
   upper 4 registers) is reading nothing, and 3 of every 4 lanes read nothing at all. This is
   unambiguously an addressing/register-span bug in this specific hand-appended carrier - not a
   permutation-vs-`pos()`/`pos_b()` question, and not a property of the hardware's own int8 D-layout
   - but its exact cause (wrong `idxC` stride for a 32-lane span, or `d_base+4` not being the
   accumulator's true upper half) was not chased further, per the decision to stop digging here
   rather than open a fourth bug in the same sub-investigation.

**Left open, honestly, matching this campaign's standing practice** (section 102's precedent) rather
than publish a guessed layout. **Does not block anything**: section 103 already established that the
memory-mediated bridge does not need a raw layout answer to be trusted, and separately, int8's own
output (int32) cannot be fed directly into another int8 MMA as an operand at all - there is no
int32-typed A/B operand in this ISA (`mmaenc.mma` only accepts half/bfloat/float/int8/uint8), so
"direct register forwarding into another int8 MMA" is not merely unverified but not even a
well-formed idea; any real int8-to-int8 chain would need an explicit requantization step (scalar/
vector code narrowing int32 back to int8) before a second MMA, which is a different kind of bridge
than anything characterized so far and out of scope for this stopping point.

## 106. New mission: int32 -> int8/uint8 requantization for chaining int8 MMAs - no native instruction exists, it is a six-opcode scalar sequence, all already-named in the ISA corpus

Section 105 left a real question open: int8's int32 output has no matching MMA input type, so a
genuine int8-to-int8 chain needs an explicit narrowing step. Spencer set this as the next mission.
Method, per the mission's own instruction: start from compiler-generated code, not from guessing.

**No native quantized-matmul API exists.** Checked `libTensorOps.rtlib` (the same rtlib this whole
campaign traces `tensor.mac` through) for any matmul-shaped quantization symbol: `quantized_matmul2d`,
`requantiz*`, `dequantiz*` all return zero hits. The ONLY quantization-shaped type in the whole rtlib
is `quantized_convolution2d_descriptor` - a real Apple type, but for **convolution**, a structurally
different top-level API from `matmul2d`, and not usable here. (A doc named
`g17-tensor-projection-handoff.md` also uses the word "quantiz" 21 times - checked and it is an
unrelated numerical-precision test suite for a different campaign, FP32/FP16/FP64 comparison work,
not integer requantization; noted so it is not mistaken for a lead by anyone reading `grep` output
later.)

**So any int32->int8/uint8 narrowing in a real compiled kernel has to be ordinary scalar/vector Metal
code - built one, using `mpp::tensor_ops::matmul2d` for the GEMM and a plain scalar loop
(`clamp(int32_t(rint(float(v)*scale)), -128, 127)`, the standard quantized-inference idiom) for the
narrowing, compiled through `xcrun metal` (Apple's real front end, via this campaign's own `build.py`,
not a hand-authored carrier) - a genuine compiler differential, not an inference from naming.** The
compiler-generated instruction sequence for the narrowing step, decoded directly:

`load` (op12682) -> `cvt.i2f` (op11179) -> `fmul` (op3290) -> `rint` (op3770) -> `cvt.f2i` (op9320) ->
`clamp`/`select`+`clamp` (op11364, plus op11375 `select` for the signed int8 form specifically -
absent for the unsigned uint8 form, which uses two plain `clamp` calls instead) -> `store` (op17193 /
op17229).

**Every one of these opcodes is already named and characterized in `isa/g17-certification.jsonl`** -
general-purpose scalar ALU instructions this campaign had never needed before, not accelerator-
specific, not new discoveries requiring their own derivation. This directly and conclusively answers
the mission's first question: **Apple does NOT use a dedicated native conversion/packing instruction
for int32->int8/uint8 tensor operands - it is an ordinary scalar sequence of six already-known ALU
ops**, the same sequence any handwritten quantized-inference Metal kernel would compile to.

## 107. Requantization numerical semantics - measured with discriminating values chosen before dispatch, not inferred from opcode names

`rint`'s presence suggested round-to-nearest-even by IEEE default, and `clamp`'s presence suggested
saturation rather than wraparound - but the mission explicitly asks for these to be measured, not
assumed, so both were checked with values chosen in advance specifically to discriminate between the
candidate semantics (`reqz_round_probe.metal`, an isolated probe of just the six-opcode sequence,
independent of any GEMM, so the discriminating inputs are exact and not GEMM-noise-dependent).

**Rounding mode: round-half-to-even, confirmed on all 8 exact half-integer inputs.** With scale =
1/256 (exact in float), inputs 128/384/640/896 and their negatives give exactly 0.5/1.5/2.5/3.5 and
-0.5/-1.5/-2.5/-3.5 after scaling - the only values that can distinguish round-half-to-even from
round-half-away-from-zero from truncation. Hardware result: `[0, 2, 2, 4, 0, -2, -2, -4]` - an exact,
unambiguous match to round-half-to-even (`away` predicts `[1,2,3,4,-1,-2,-3,-4]`, `trunc` predicts
`[0,1,2,3,0,-1,-2,-3]`; both ruled out, not just "close").

**Clamp is genuine saturation, never wraparound, checked at the exact boundary.** Signed int8:
32512 (exactly 127.0 after scale) -> 127 unchanged; 32768 (exactly 128.0, one past max) -> clamped to
127, not wrapped to -128; -32768 (exactly -128.0) -> -128 unchanged; -33024 (exactly -129.0) ->
clamped to -128, not wrapped to 127; 100000 (far out of range) -> 127. Unsigned uint8 (separate probe,
`reqz_round_probe_u8.metal`, clamp bounds [0,255]): -25600 (deep negative) -> clamps to 0, **not**
wrapped to a large unsigned value (e.g. not 156); 65280 (exactly 255.0) -> 255 unchanged; 65536
(exactly 256.0, one past max) -> clamped to 255. All 13 signed and 10 unsigned discriminating cases
matched exactly, both directions of both boundaries, no partial or "close" results.

**Signed vs unsigned is the same six-opcode sequence with different clamp bounds**, not a different
mechanism - the only structural difference decoded was `select`+`clamp` (signed) vs two plain `clamp`
calls (unsigned), still all pre-existing ALU ops either way.

**Zero-point/bias and scale are not native features to characterize separately** - since no dedicated
quantization instruction exists at all (section 106), a zero-point offset (if a real model used one)
would just be one more ordinary scalar add composed into the same sequence, not a distinct hardware
mechanism with its own semantics to measure. Not tested here since this probe's discriminating values
already fully pin the mechanism that WOULD carry it (the same `cvt.i2f`/`fmul`/`rint`/`cvt.f2i`/
`clamp` chain, just with one more `fadd` or `fsub` before/after the multiply).

## 108. Chaining architecture: the high-level matmul2d API needs a real kernel boundary, not just a memory buffer within one kernel - and the full GEMM->requantize->GEMM path is hardware-validated end to end

**Discovery, found while building the natural single-kernel chain:** `mpp::tensor_ops::matmul2d`'s
`run()` method can only be called ONCE per kernel invocation through this high-level C++ API. Proven
with the simplest possible isolation, stripped of every other variable: the exact same `matmul2d` op
object, called twice with the exact same A/B input tiles, writing only to a DIFFERENT destination
slice the second time (`reqz_isolate_repeat.metal`) - the first call is bit-exact, the second call
silently writes all zeros, with no error, no assertion, no dispatch failure. Confirmed this is not
about slice offsets, tensor-object reuse, or barriers by testing each independently (fresh tensor
objects for the second call: same result; explicit `threadgroup_barrier(mem_flags::mem_device)`
between calls: same result). **This is a real, load-bearing distinction from sections 101/103**: the
underlying hardware CAN execute two chained MMAs within one kernel (proven there, at the raw
AIR/instruction level using hand-authored `tlower.lower` bodies), but Apple's own high-level
`matmul2d<>` template does not expose that capability - it appears to hold some single-use internal
resource (the rtlib's own `matmul2d_op_cooperative_destination_*` machinery is the likely carrier,
not chased further since the practical answer does not require it). **So for any real compiler
targeting this high-level API, the GEMM->requantize->GEMM bridge is not a register-resident-vs-memory
choice at all - it MUST cross a genuine kernel/dispatch boundary**, independent of anything about
register layouts or `pos()`/`pos_b()` (sections 99-101's concerns do not even arise here, since there
is no register-resident option on the table to begin with).

**Retraction (section 127, added later).** The claim above is wrong. On these tensors the offsets of `slice<E0, E1>(o0, o1)` are (column, row), so `tC.slice<16, 16>(16, 0)` is columns 16..31 of
rows 0..15; the second call wrote its correct result there and I read the wrong region, as did every "independent test" in this paragraph. Read correctly, `reqz_isolate_repeat` is bit-exact on both calls
(6/6). One kernel can call `run()` repeatedly and hold several `matmul2d` objects; "must cross a kernel boundary" is withdrawn (a single-kernel GEMM -> requantize -> GEMM through device memory is exact, section 127).
The multi-kernel path below remains valid as a working path, not a necessity.

**Full path hardware-validated end to end, against an independent reference, satisfying this
mission's acceptance criterion directly.** Two separately compiled kernels, each its own process (this
dylib's one-pipeline-per-process rule), A2 relayed between them exactly as a real host-orchestrated
multi-kernel graph would: Kernel A runs GEMM1 (int8x int8 -> int32, `mpp::tensor_ops::matmul2d`) then
the confirmed requantization sequence (round-half-to-even, saturating) to produce A2 (int8). Kernel B
runs GEMM2 (int8 x int8 -> int32) on A2 and a fresh B2. Result: C1 exact against `gemm_ref`; A2 exact
against the closed-form round-half-to-even/saturating model applied to C1; **C2 exact against
`gemm_ref` applied to the ACTUAL requantized A2 and B2** - the complete chain, not just its parts,
bit-exact (`reqz_chain_two_kernels.py`).

**Instruction/latency cost.** Kernel A: 71 real instructions (GEMM1's own load/MMA/store plus the
6-opcode requantization epilogue over 8 SIMD-lane iterations of 32 elements each). Kernel B: a plain
single GEMM, matching every other int8 oracle kernel's own instruction count in this campaign's
existing corpus (no new cost there). The requantization epilogue itself is cheap in isolation (6
opcodes per element, running 8-wide across 32 lanes for a 16x16 tile) - the dominant cost of this
bridge is the KERNEL BOUNDARY itself (a full dispatch: pipeline state, grid launch, and the intervening
host-visible memory round trip), not the arithmetic. No waits/hazards were needed BEYOND the ordinary
`threadgroup_barrier(mem_flags::mem_device)` already used everywhere else in this campaign's memory-
mediated compositions (section 96/101/103) between GEMM1's store and the scalar epilogue's read.

## 109. Decision table (int32 -> int8/uint8 requantization for MMA chaining)

| Producer | Requantization op | Result type/layout | Evidence | Hardware-confirmed direct reuse by next MMA? |
|---|---|---|---|---|
| op10384/10385 int32 accumulator | Scalar sequence: `cvt.i2f`, `fmul`, `rint`, `cvt.f2i`, `clamp` (+`select` for signed) - NOT a dedicated instruction (section 106) | int8, signed, round-half-to-even, saturating (not wrap) to [-128,127] | Hardware-confirmed, compiler-differential (`xcrun metal`, real MPP source) + 13 discriminating boundary values | **YES, but only across a kernel/dispatch boundary** - confirmed end-to-end (section 108). Register-resident/same-kernel reuse is refused: not a layout question, `matmul2d` structurally supports one `run()` per kernel |
| op10384/10385 int32 accumulator | Same sequence, clamp bounds [0,255] instead of [-128,127] | uint8, unsigned, round-half-to-even, saturating (confirmed no wraparound at the negative boundary) | Hardware-confirmed, same method, 10 discriminating boundary values | Not separately re-run through a full GEMM->GEMM chain (the signed case already proves the mechanism; the numeric narrowing step itself is identically validated) - **refuse this cell** for direct MMA-chain reuse specifically until run, per this campaign's standing "refuse unresolved cells" discipline |
| Zero-point/bias-bearing quantization | Not a native mechanism - no dedicated instruction exists for ANY int32->int8 narrowing (section 106), so a zero-point offset would be one more composed scalar add, not a separate hardware feature | N/A | Inferred from the absence of any native quantization API, not measured directly (no zero-point test was run) | **Refuse this cell** - not measured |

**Acceptance criterion met**: a full int8 GEMM -> requantization -> second int8 GEMM path is
hardware-validated bit-exact against an independent reference (section 108), and the precise missing
operation is identified with evidence rather than inferred (no native requantization instruction
exists at any level checked - rtlib symbols, decoded opcodes are all pre-existing general ALU ops,
section 106). Architectural rule for the compiler owner, stated as a rule rather than an
implementation: **treat int32->int8/uint8 requantization as an ordinary six-opcode scalar epilogue
(round-half-to-even, saturating clamp) fused onto the producing GEMM's own kernel, and treat the
bridge into the next int8 MMA as a mandatory kernel/dispatch boundary, not a register-forwarding or
same-kernel memory question** - handed to the compiler owner as an architectural finding; not
implemented here, per this lane's standing boundary.

*Correction (section 136 part 5.2): the register-resident (same-kernel) int8 and uint8 chain is built and exact in hand-authored AIR, for shift requantization and for float-scale requantization with per-column bias and zero point, in modes A, B, At and Bt (272 of 272 runs). The refusals in this table concern `matmul2d` (one `run()` per kernel) and the uint8 and zero-point cells that had not been run; those cells are now measured.*

## 110. uint8 chain closed: full end-to-end GEMM->requantize->GEMM, bit-exact, boundary-heavy inputs chosen before dispatch

Closes the mission's first item. Same architecture as section 108 (two kernels, real kernel-boundary
crossing, since section 108 already established `matmul2d` supports one `run()` per kernel regardless
of signedness), same confirmed numerical model (int32->float, scale, round-half-to-even, float->int,
saturating clamp - this time to [0,255]).

**Discriminating inputs, constructed rather than hoped-for.** Random 16x16 uint8 A1/B1/B2 for ordinary
coverage, PLUS two deliberately controlled rows using single-nonzero-term selectors (one nonzero A1
entry scales one whole row of B1 directly, giving exact, hand-picked accumulator values per column
without needing a 16-term sum to coincidentally land on a boundary):
- Row 4: `A1[4,15]=1` -> `C1[4,:] = B1[15,:]` exactly, with `B1[15,:]` hand-set to
  `[0,1,2,4,8,16,32,64,96,128,160,192,224,254,255,8]` - covers 0, 1, and values through 255 as
  accumulator inputs, all staying in-range after scaling.
- Row 5: `A1[5,14]=255` -> `C1[5,:] = 255*B1[14,:]`, with `B1[14,:]` hand-set to the same value list -
  covers the same range but at 255x magnitude, guaranteeing saturation for all but the smallest cases.

With `SCALE = 1/16` (exact in float), row 5 alone produces BOTH in-range and saturated results in the
SAME dispatch: `A2[5,:] = [0, 16, 32, 64, 128, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]`
- satisfying the mission's explicit requirement directly, not by coincidence. Row 4 confirms in-range
rounding behavior at the low end: `A2[4,:] = [0,0,0,0,0,1,2,4,6,8,10,12,14,16,16,0]` (8/16=0.5 rounds
to 0, the even neighbor; 254/16=15.875 and 255/16=15.9375 both round to 16 - consistent with the
already-confirmed round-half-to-even model, not re-derived here, just observed to hold under new data).

**Result: C1 exact, A2 exact against the round-half-to-even/saturating model, C2 exact against
`gemm_ref` on the ACTUAL requantized A2 and a fresh B2 - the full chain, bit-exact** (found one real
bug along the way and fixed it before trusting the result: the kernel's own compiled-in `SCALE`
constant had been left at an earlier value, 1/1024, while the Python-side reference used 1/16 - caught
because row 5's actual output cleanly matched a division by 1024 instead of 16, not because of any
uncertainty about the underlying mechanism).

**Mission acceptance item 1 met: the full uint8 two-GEMM quantized chain is hardware-validated
bit-exact against an independent reference**, using genuinely boundary-heavy, pre-chosen discriminating
inputs (0, 1, near-UINT8_MAX, exact-half-adjacent rounding cases, and both in-range and saturated
results from one scale in one dispatch) - not just typical/random coverage.

## 111. Integrity note: a previously-cited hand-authored carrier does not currently reproduce - flagged, root-caused as far as practical, and the investigation moved to a more reliable authoring path

While building this mission's first dependency probe (a bare `mmaenc.mma()`-encoded op5100/5101
carrier, extending `op5100_regbase_test.py`'s own structure), the ADDED test line changed nothing -
but re-running `op5100_regbase_test.py` ITSELF, unmodified, right now, gives **0/4**, not the 5/5 (or
4/4) this document has cited since section 98. This is stated plainly rather than buried: **the
specific standalone hand-authored confirmation that section 98 relied on does not currently
reproduce**, and the cause was not found (esz sweep, wait=True/False, and an inert-instruction
positional test were all tried and none explained it; every variant gave the same characteristic
~12-22 magnitude error).

**This does not undermine the underlying architectural conclusion** (op5100/5101's A-operand reads
`pos()`-arranged data) **because that conclusion has independent, currently-passing support**:
`compose.py`'s `chain` arm (section 101), re-run fresh just now, is still 3/3 bit-exact, and its own
correctness DEPENDS on `tlower.lower`'s generated GEMM2 body correctly reading a `pos()`-arranged A
operand from a row-major buffer - a compiler-differential confirmation, not a hand-built one. So the
conclusion stands; the specific standalone artifact cited to support it does not currently reproduce,
and this document is not going to quietly keep citing it as if it still does.

**Consequence for method, not just for this one file**: this session's own `mmaenc.mma()`/`memenc`
Python-level byte encoder - used successfully for many things earlier in this campaign - turned out to
be unreliable in THIS session's hands for a from-scratch bare-MMA carrier reading operands from
ordinary loads (not derived from `tlower.lower`), and the cause was not isolated before time ran out
on that specific thread. Rather than keep debugging an unreliable authoring method, the rest of this
mission's dependency work pivots to a DIFFERENT, apparently more trustworthy path already present in
this results directory: hand-written LLVM-IR kernels calling
`@air.simdgroup_matrix_16x16x16_multiply_accumulate` directly (matching `gen.py`'s own established
pattern, compiled through Apple's real `metal-as`/native pipeline via `build.py` - the SAME compiler-
differential machinery section 106-110 already used and trusted this session), checked against the
MEASURED (not assumed) `layout-new_f16f16_nn.json` tables rather than this campaign's `pos()`/
`pos_b()` closed forms directly. This removes an entire class of hand-encoding risk (flag/tag/wait
bits are whatever Apple's own backend emits, not bytes I patched by hand) and, per section 112 below,
gives clean, reproducible results immediately.

## 112. Dependency/synchronization semantics: five bounded, single-factor experiments, real hardware, negative results preserved

Method throughout: an existing, not-yet-numerically-validated corpus of hand-written LLVM-IR kernels
(`dep_alu_consumer.ll`, `dep_independent2.ll`, `dep_b_from_result.ll`, `dep_store_and_chain.ll`,
predating this session, evidently prepared for exactly this question) dispatched fresh via
`dep_run_helpers.py` (new this session), data placed and read back via `layout-new_f16f16_nn.json`'s
measured A/B/D tables - the SAME gen.py-family compiled-object convention, not this campaign's
`pos()`/`pos_b()` assumptions. Each kernel isolates ONE factor.

**1. MMA -> scalar (ALU) consumer, zero wait: WORKS, 5/5 bit-exact.** `dep_alu_consumer.ll`:
`r0 = MMA(A,B,0); r1 = r0 + 1.0; store r1`, no wait instruction anywhere in the source. Reconfirms
section 10's much older finding ("no explicit wait instruction exists between an MMA and its
consumer") with a fresh, clean, discriminating dispatch rather than citing it secondhand.

**2. Back-to-back MMAs, SAME destination via a genuine register-resident accumulate chain, zero
wait: WORKS, exact (0 error).** `dep_store_and_chain.ll`'s `fc` output (`r1 = MMA(A,B,C=r0)`, r0
being the immediately-preceding MMA's own live result, no store/reload) matches
`mma_np(A,B,mma_np(A,B,0))` exactly. This is the register-resident chain-of-accumulation pattern
every K-loop in this campaign already relies on, freshly reconfirmed at the AIR-intrinsic level with
zero explicit synchronization.

**3. NEW, genuinely surprising negative result: storing an MMA's result immediately after the MMA
FAILS when that same result is ALSO consumed register-resident by an immediately-following MMA.**
Same kernel (`dep_store_and_chain.ll`), its OTHER output: `fc2` (`store r0` to memory, right after
the first MMA, with the SAME r0 ALSO feeding the second MMA as its C-accumulator) does NOT match
`mma_np(A,B,0)` - reproduced on 4 independent seeds, consistent ~12-13 magnitude error every time,
never bit-exact. **The intermediate value is correct when consumed by the following MMA (finding 2's
0-error result proves this) but wrong when stored to memory from the exact same point in the
program.** This is a genuine, discriminating, result-readiness-class hazard, isolated from WAW and
from register lifetime (the destination registers are not reused; only one is even the same
physical space) and from pure issue-rate (finding 1 shows a lone MMA's result is readable
immediately by a DIFFERENT kind of consumer, the ALU). The most likely explanation, not confirmed
further: the compiler's own scheduler provides adequate implicit synchronization for a
register-consuming instruction (another MMA) but not for an ordinary store issued from the same
program point when a competing consumer exists - preserved as a negative result rather than
explained away.

**4. NEW, genuinely surprising negative result: two fully INDEPENDENT back-to-back MMAs (no data
dependency at all, separate destinations) - the second one's stored result reads back as exactly
zero.** `dep_independent2.ll`: `r0 = MMA(A,B,C1=0)` stored to `fc`; `r1 = MMA(A,B,C2=0)` (same A, B,
a completely separate C) stored to `fc2`. `fc` is bit-exact (5/5 equivalent, matches finding 1's
clean baseline); `fc2` reads back as all zero, every trial, not garbage - consistent with the second
MMA's result never reaching memory (a scheduling/codegen gap for the multiple-independent-MMA-per-
kernel case specifically), not with a data hazard (there is no data dependency between the two MMAs
at all). Not root-caused further (an already-known, load-bearing accumulate-chain case, finding 2,
DOES work at the AIR-intrinsic level - this failure is specific to independent, non-chained,
back-to-back issuance).

**5. LEFT OPEN, not guessed: an MMA's result used as a SUBSEQUENT MMA's OPERAND (not its
C-accumulator), register-resident, zero wait.** `dep_b_from_result.ll`: `h = fptrunc(r0)` used as
MMA2's B operand. Result does not match the direct model (`mma_np(A, half(r0), 0)`, max_err ~60-93
across 5 seeds), nor a reinterpretation attempt (recasting `r0`'s raw fragment through B's own
measured `B_k`/`B_col` layout instead of D's, max_err worse, ~357). This needs the same kind of
one-hot/census derivation that resolved section 100's D-layout question - genuinely unresolved, and
explicitly left that way rather than reported as a hazard OR as correct.

## 113. Dependency decision table

| Producer -> consumer class | Required wait/sync | Measured hazard | Legal immediate (zero-wait) reuse? | Evidence |
|---|---|---|---|---|
| MMA -> scalar (ALU) consumer | None needed | None found | **YES** | Hardware-confirmed, 5/5, section 112.1 (reconfirms section 10) |
| MMA -> MMA, same destination, register-resident accumulate (C = prior MMA's live result) | None needed | None found | **YES** | Hardware-confirmed, exact, section 112.2 |
| MMA -> ordinary store, immediately, with NO competing consumer | None needed (established elsewhere: sections 10, 101, 103, 108, 110 all store an MMA's result immediately with no wait, always correct) | None found in those contexts | **YES**, when nothing else also consumes the same result | Compiler-differential, many prior sections, not re-derived here |
| MMA -> ordinary store, immediately, WHILE the same result ALSO feeds a register-resident MMA consumer | **Unresolved - refuse to certify as safe** | **YES, confirmed hazard**: the stored value is wrong (~12-13 magnitude error, 4/4 reproductions) even though the register-resident consumer sees the correct value | **NO** - do not schedule a store of an MMA's result concurrently with another consumer of that same result without further characterization | Hardware-confirmed negative, section 112.3 |
| MMA -> MMA, independent destinations, no data dependency, back-to-back | **Unresolved - refuse to certify as safe** | **YES, confirmed hazard**: the second MMA's result never reaches memory (reads back as zero, not garbage) | **NO** for this exact pattern (two independent AIR-intrinsic calls, no compiler-inserted synchronization) - NOTE this does not contradict sections 101/103's proven multi-MMA-per-kernel chains, which use `tlower.lower`'s own hand-crafted tag/wait bits, not bare back-to-back intrinsic calls | Hardware-confirmed negative, section 112.4 |
| MMA result -> next MMA's OPERAND (not C), register-resident | **Refuse - not measured** | Unknown - two candidate models both failed | Refuse | Section 112.5, explicitly open (answered in sections 125 and 129: register-direct works for A and B; the earlier failure was a wrong expected value, hand-authored 20/20) |
| Cross-SIMDgroup MMA dependency (one SIMD group's MMA feeding another's consumer) | **Refuse - not measured** | Unknown | Refuse | Out of scope for this session's bounded experiments; section 11's `SR_SIMD_GRP`/mask-generator findings cover per-SG tile ownership, not cross-SG data dependency specifically |

**Mission acceptance, item 2**: the common MMA<->scalar and MMA<->memory dependency paths now have
hardware-backed rules (rows 1-4), stated as rules a scheduler could consult directly, with two
concrete, reproduced hazards preserved as negative results (rows 4-5) rather than glossed over, and
two genuinely unresolved classes explicitly refused (rows 6-7) rather than guessed.

## 114. RETRACTED: section 112's two "hazards" were a dispatch-scoping artifact, not GPU hardware hazards - the real answer, found by measuring the exact threshold as instructed

Spencer's own follow-on goal asked for the PRECISE minimum distance/state transition that changes
the outcome, rather than accepting a guessed explanation - doing that immediately overturned
section 112's findings 3 and 4.

**What was wrong.** Both "hazards" (MMA result store immediately after the MMA while also feeding a
register-resident MMA consumer; two independent back-to-back MMAs) involved storing to the `fc2`
buffer region (`lane + 32`, the SECOND 32-lane block of the output buffer) - and EVERY test this
session that stored only to the canonical `fc` region (`lane`, the FIRST block - `dep_alu_consumer`,
and a fresh isolated `mma_store_fc_direct.ll` control built specifically to check this: MMA -> bare
immediate store, zero intervening instructions, no competing consumer) was bit-exact, always. That
is the tell: the variable that mattered was never GPU-side instruction spacing.

**The controlled experiment, one factor at a time as instructed.** First varied 0-32 intervening
independent `fadd` instructions between the MMA and the `fc2` store (`sv_sweep_one.py`): the error was
BIT-IDENTICAL at every N from 0 to 32 - conclusive proof spacing was never the variable (a real
GPU-side race would show SOME N where it starts passing; this showed none). That result pointed away
from "GPU instruction ordering" and toward "a fixed property of the fc2 region or the dispatch
itself" - confirmed by a plain load-then-store-unchanged round trip on fc2 alone (no MMA at all)
working perfectly, ruling out a basic buffer-addressing bug too.

**The actual variable, found by sweeping the ONE remaining candidate: `ac_run_ps_es`'s `dim`
parameter** (this campaign's convention for calling the dispatch entry point - `dim, esz, threads`
after the three buffers). Sweeping `dim` on the EXACT SAME kernels that "failed" in section 112:
`dim=16` (this session's default) fails, `dim=22` fails, **`dim=23` passes, and every value through
64 passes** - a hard, bit-exact, one-step boundary, not a gradual effect. `fc2` begins at byte offset
1024 in the output buffer (32 lanes x 8 floats x 4 bytes = 1024, the end of `fc`'s own region);
`23^2 * esz(2) = 1058 >= 1024` while `22^2 * 2 = 968 < 1024` - the threshold falls exactly where
`dim^2 * esz` first covers the byte offset `fc2` starts at. **`dim` scopes how many bytes of the
output buffer the dispatch call guarantees are complete/host-visible when it returns - a host-side
completion/flush parameter, not anything about GPU instruction issue order, register reuse, or
execution-unit scheduling.** Re-ran BOTH of section 112's exact failing kernels with `dim=23`:
`dep_independent2`'s `fc2` (previously "all zero") now matches exactly; `dep_store_and_chain`'s `fc2`
(previously ~12-13 magnitude error) now matches exactly. Both "hazards" are fully, not just
partially, explained by this one parameter - not a coincidence or a partial fix.

**Answering the mission's actual two questions, now that the confound is removed:**
- **What event makes an MMA result store-visible?** Ordinary program-order completion of the store
  instruction itself - confirmed with ZERO intervening instructions and ZERO explicit wait needed
  (`mma_store_fc_direct.ll`), as long as the HOST's own dispatch call is told to wait for enough of
  the buffer (the `dim` parameter, a call-site concern, not a kernel-authoring one).
- **What event permits an independent MMA to retire correctly?** The SAME answer: ordinary issue in
  program order, no additional GPU-side synchronization needed - `dep_independent2`'s two fully
  independent MMAs, zero wait, back-to-back, are both correct once the dispatch is told to wait for
  the whole output region.
- **Are these the same mechanism or two different ones, per the goal's own framing?** **The same
  one** - both apparent "hazards" trace to the identical `dim` root cause, not two separate GPU
  phenomena.

**Section 112/113 correction, stated plainly**: findings 3 and 4, and the corresponding "confirmed
hazard" / "refuse" rows in section 113's decision table, are RETRACTED. Sections 112.1 and 112.2
(MMA -> scalar consumer safe; MMA -> MMA register-resident accumulate chain safe) stand unchanged -
both stored to the canonical `fc` region and were never affected by this. Section 112.5 (MMA result
used as a subsequent MMA's OPERAND, not C) is UNCHANGED and still genuinely open - that experiment
also stored to `fc`, not `fc2`, so this correction does not resolve it; it remains a real, separate,
uncharacterized layout question, not a dispatch artifact.

## 115. Revised decision table (supersedes section 113)

| Producer -> consumer class | Required wait/sync | Measured hazard | Legal immediate reuse? | Evidence |
|---|---|---|---|---|
| MMA -> scalar (ALU) consumer | None | None | **YES** | Section 112.1, unaffected by this correction |
| MMA -> MMA, register-resident accumulate (C = prior MMA's live result) | None | None | **YES** | Section 112.2, unaffected by this correction |
| MMA -> ordinary store, immediately, any consumer configuration (alone, or with a competing register-resident consumer) | None at the GPU/kernel level | **None** (RETRACTED: section 112.3's finding was a dispatch `dim`-scoping artifact, section 114) | **YES** | `mma_store_fc_direct.ll` (0 error) + `dep_store_and_chain` re-run at proper `dim` (exact), section 114 |
| MMA -> MMA, independent destinations, no data dependency, back-to-back | None at the GPU/kernel level | **None** (RETRACTED: section 112.4's finding was the SAME dispatch `dim`-scoping artifact, section 114) | **YES** | `dep_independent2` re-run at proper `dim` (exact), section 114 |
| Any of the above, dispatched via `ac_run_ps_es` with `dim` too small for the true output extent | N/A - this is a HOST call-site bug, not a kernel hazard | **YES, confirmed**: silent wrong/zero data past `dim^2 * esz` bytes, no error returned | **NO** - always size `dim` (and `esz`) so `dim^2 * esz` covers the full output region in bytes, not just the nominal tile | Section 114, exact threshold measured (`dim=22` fails, `dim=23` passes for a 1024-byte offset at `esz=2`) |
| MMA result -> next MMA's OPERAND (not C), register-resident | Refuse - not measured | Unknown, two candidate models failed | Refuse | Section 112.5, unchanged by this correction; answered at the hand-authored level in section 129 (D -> A and D -> B, 20/20 each, table-exact) |
| Cross-SIMDgroup MMA dependency | Refuse - not measured | Unknown | Refuse | Out of scope this session |

**Practical rule for the compiler owner, corrected**: there is no GPU-side MMA->store or
independent-MMA-retire hazard to design around at all, for the cases measured here. The real
operational rule is a HOST/DISPATCH one: whatever launches these kernels must size its own
completion-wait scope to the TRUE output footprint, not a nominal tile size - undersizing it produces
silently wrong data with no error signal, which is arguably a more dangerous failure mode for a
compiler to get right than a genuine instruction hazard would have been.

## 116. Exact MMA arithmetic, stress-tested with genuinely discriminating operands rather than re-asserted from section 4's already-strong prior evidence

Mission: confirm or falsify section 4's fitted arithmetic model (`p_i` = adjacent-pair `RNE32` sums,
`q_j = p_j + p_{j+4}` interleave-by-4, C first, four sequential `RNE32` adds) under adversarial
conditions specifically constructed to separate it from other plausible reduction trees, across all
three operand forms (fp16, bf16, TF32-like/op5098), including accumulation with an existing C.
Built `arith_model_harness.py` (and `_bf16`/`_f32` counterparts) - an executable candidate-model
comparator, not a one-off script: it isolates `D[0,0] = sum_k A[0,k]*B[0,k] + C[0,0]` (only row 0 of A
and column 0 of B matter for that element, so the rest of the tile is irrelevant and left zero),
dispatches a real hardware MMA via the gen.py/AIR-intrinsic corpus this session already validated for
reliability (section 111's pivot away from this session's own mmaenc/memenc encoder), and scores
FOUR candidate models against the SAME hardware result every time: `established_tree` (section 4's
model), `sequential` (plain left-to-right RNE32 accumulation), `balanced_natural` (adjacent-pairing
repeated at every tree level, i.e. NOT interleave-by-4 at the second level), and `exact_once`
(sum everything in exact arithmetic, round once).

**Tree-topology discriminator, constructed to separate `established_tree` from `balanced_natural`
specifically** (both share the same first level - adjacent pairs of the 16 raw products - and differ
ONLY in how the resulting 8 partial sums combine): place a large value (1024.0) in product slot 0 and
four small values (each individually below half the fp32 rounding unit at 1024's magnitude, so each
would be silently absorbed if paired with 1024 one at a time) in product slots 8, 10, 12, 14 - which
`established_tree`'s `q_j = p_j + p_{j+4}` groups as FOUR SEPARATE combinations with 1024's own
group, one per final sequential add (each small value lost individually), while `balanced_natural`'s
adjacent-pairing groups them PAIRWISE FIRST (their combined magnitude clears the rounding threshold
before ever touching 1024). Hardware result: exactly `1024.0` - matches `established_tree` and
`sequential`, REJECTS `balanced_natural` and `exact_once` (both predict `1024.0001220703125`, a
one-ULP difference, not "close"). Reproduced with the sign flipped (`-1024.0` anchor): same
rejection pattern, ruling out a sign-handling coincidence.

**C-injection-point discriminator** (is C genuinely the FIRST operand into the final sequential
chain, or could it enter last?): C=1024.0 (large), all four `q_j` individually tiny (each below
1024's half-ULP). Hardware: exactly `1024.0` - matches "C first" (each `q_j` gets silently absorbed
into the already-1024-valued accumulator one at a time); REJECTS a "sum the q's together first, add
C last" alternative model (which would preserve their combined contribution, predicting
`1024.0001220703125`). Directly reconfirms section 4's "C enters FIRST" finding with a fresh,
independently-constructed adversarial case, not a re-citation.

**Random holdout, fresh seed, not used to construct or tune any model**: 20 trials, fp16, mixed
magnitude scales (some dense/narrow-range for fractional stress, some spanning `1e-3` to `1e3` for
large+tiny stress), random signed C including zero. `established_tree`: **20/20 exact**.
`sequential`: 14/20. `balanced_natural`: 16/20. `exact_once`: 13/20 - the alternatives are not
"almost right," they fail on a real fraction of ordinary-looking random cases once the sample is
large enough, exactly the kind of holdout power the acceptance criteria asked for rather than one
lucky workload.

**One genuine harness bug found and fixed in-flight, documented rather than silently patched**: an
early cancellation test (`100 + (-100) + 0.001`) showed ALL FOUR candidate models simultaneously
wrong by the identical offset (`4.04e-7`) - the tell that it was a shared input bug, not a
tree-structure question (a real tree disagreement shows up as DIFFERENT models disagreeing with each
other, not all of them agreeing with each other and disagreeing with hardware identically). Cause:
the harness computed model predictions from the raw Python float `0.001`, while the hardware always
operates on `0.001` ROUNDED to fp16 first (`0.0010004043579101562`) - confirmed by checking
`float(np.float16(0.001))` against hardware's own output, an exact match. Fixed by rounding every
model's inputs through the operand's own precision before computing (`fp16_round` in the harness);
re-run afterward gave a clean match across all models for that specific (non-discriminating) case.

**bf16 and TF32-like (op5098) confirmed to share the identical tree, not merely assumed to** (the
goal's own explicit instruction: "Compare forms rather than assuming... share internal arithmetic").
Repeated the EXACT SAME tree-topology discriminator construction (1024.0 anchor, four tiny values at
product slots 8/10/12/14, values chosen to be exactly representable in each form's own precision -
powers of two, which need no mantissa bits to represent exactly regardless of format) against
`new_bf16bf16_nn` and `new_f32f32_nn` (gen.py-built, already-compiled objects; TF32-like operands
additionally passed through the established 10-bit truncation before use in the model). Both hardware
results: exactly `1024.0`, matching `established_tree`, rejecting `balanced_natural`/`exact_once` -
the identical discriminating signature as fp16. Backed by fresh random holdouts: **bf16 10/10**,
**TF32-like 10/10**, `established_tree` exact on every trial.

**Subnormal operand** (fp16's smallest subnormal, `2^-24`, added to `1.0`): non-discriminating (all
four models agree, hardware matches) - confirms no crash or special-cased misbehavior, but does not
by itself separate the candidate models (the subnormal value is far below any rounding threshold that
matters for a 1.0-magnitude sum).

**K-dependence**: not independently re-derived here - K=16 is fixed per native MMA issue (an
architectural constant, not a free parameter of a single dispatch); composing K>16 through the
rounded fp32 accumulator across chained MMA issues was already established with its own dedicated
stimuli (section 4's x2/x3 chaining tests, 17/17 each; section 17's 1,024/1,024-element replay of a
real failed program). Not re-run this session since nothing in this session's own testing gave any
reason to doubt it, and re-deriving already-strong prior evidence without a specific new concern
would not be a good use of this session's remaining time.

**Acceptance criterion (1) met**: `established_tree` predicts the exact hardware bits across every
constructed adversarial case (tree-topology x2 signs, C-injection-point, signed cancellation,
subnormal operand) AND a 40-trial combined random holdout across all three operand forms (20 fp16 +
10 bf16 + 10 TF32-like), spanning mixed magnitude scales, signed values, and both `C=0` and `C!=0`
accumulate mode, with zero exceptions - while every alternative candidate model tested fails on a
substantial, non-trivial fraction of the SAME holdout. No correction to section 4's original claim is
needed; this section is additive stress-testing that reconfirms it under conditions section 4 itself
did not construct, not a revision.

## 117. Section 116 addendum: the compositional K=1/K=2 buildup, and tie handling closed cleanly (the earlier attempt was flawed, fixed and reconfirmed)

Two remaining gaps from the goal's own checklist, closed with the same harness.

**Compositional buildup, as instructed ("first one product, then two products, then the smallest K
exposing a tree").** K=1 (a single nonzero product, everything else zero - no accumulation tree
participates at all): `5.5 * 3.25 = 17.875` exact, both `C=0` and `C=100` cases bit-exact against
every candidate model (not discriminating on its own, but confirms product formation and the
C-add path work correctly in total isolation before trusting anything built on top of them). K=2
(exactly two nonzero products, isolating the first tree level's pair-sum in isolation from every
other level): `2*4 + 3*5 = 23` exact. Both steps bit-exact, establishing the base cases the K=16
tree (section 116) is actually built from, rather than jumping straight to the full case.

**Tie handling, corrected.** A first attempt at an exact round-half-to-even tie (scratch-only, never
committed) used `2^20` as the "large" anchor, which silently overflowed fp16's representable range on
cast and produced a meaningless `inf` vs `inf` non-result - caught and discarded here rather than
written up as a finding. Rebuilt correctly, within fp16's actual range: `prod0 = 32*32 = 1024.0`
(`2^10`), `prod1 = 2^-7 * 2^-7 = 2^-14` exactly - their
exact sum, `1024 + 2^-14`, lands PRECISELY halfway between `1024.0` (mantissa all-zero, even) and
`1024 + 2^-13` (mantissa's low bit set, odd), the two fp32 values a real tie must choose between.
Round-half-to-even's rule says pick the EVEN one. Hardware: exactly `1024.0` - the even choice,
confirmed directly at the tree's first level, not inferred from the top-level-only stimuli section 4
originally used to reject round-toward-zero.

**Status against the goal's acceptance criteria, stated plainly**: criterion (1) is met for fp16,
bf16 and TF32-like at K=16 (native single-issue depth) with existing-C accumulation, large+tiny,
signed cancellation, subnormal operands, and now exact ties and the K=1/K=2 base cases - `zero`
falsifying observations against `established_tree` across every constructed and random case run this
session. K-dependence beyond 16 (chained MMA composition) rests on section 4/17's prior, independently
strong evidence (17/17 and 17/17 chaining stimuli, a 1,024-element real-program replay), not
re-derived here since nothing in this session gave a reason to doubt it. No indistinguishable-model
ambiguity remains to name under criterion (2) - a single explicit model, unchanged from section 4,
accounts for everything measured.

## 118. Cross-SIMDgroup TensorOps: a real high-level API limitation, and a real hardware synchronization requirement, cleanly separated

New mission, starting from what section 6 already established (matmul2d with
`execution_simdgroups<2>`/`<4>` cooperating UNIFORMLY on one shared tile is bit-exact, dividing work
across lanes/SGs, not changing arithmetic). The genuinely open question, per this document's own
section 113/115 decision table: does data cross a SIMDgroup boundary correctly when TWO DIFFERENT
SIMDgroups own TWO DIFFERENT, INDEPENDENT tensor operations in one dispatch (not one shared
cooperative op) - a producer/consumer relationship ACROSS SGs, not within one.

**High-level API limitation, found and isolated first.** Attempted the natural way to express this
with `mpp::tensor_ops::matmul2d`: dispatch 2 SIMDgroups (64 threads), guard SG0's `execution_simdgroup`
(or `execution_simdgroups<1>`) - scoped `op.run()` call behind `if (sg == 0)`, leaving SG1 idle or
doing ordinary scalar work. Result: **SG0's own MMA silently returns zero**, even with SG1 doing
NOTHING at all (`sg_control_sg0only.metal`, `sg_control_explicit1.metal` - both fail identically).
Controlled: the IDENTICAL kernel dispatched with only 32 threads (SG1 absent entirely) is bit-exact.
So the failure is not about the `if(sg==0)` guard, not about `execution_simdgroup` vs
`execution_simdgroups<1>` (both fail the same way), and not about SG0's own code - it is specifically
that **the API's execution-scope machinery requires its declared SIMDgroup count to match the total
SIMDgroups actually launched in the dispatch; a scope narrower than the launch silently fails rather
than erroring**. This is squarely a high-level API constraint (the `execution_scope` template
argument's own uniformity contract, matching the header's own "call to run methods must be execution
scope uniform" warning, applied threadgroup-wide rather than just within the target SIMDgroup) - not
evidence about the underlying hardware, and not turned into an ISA rule here.

**Hardware finding, via the raw AIR intrinsic, bypassing the API restriction entirely.** Hand-wrote
LLVM IR calling `@air.simdgroup_matrix_16x16x16_multiply_accumulate` directly (matching `gen.py`'s
established pattern, not `mpp::tensor_ops`), with an explicit `sg` (`simdgroup_index_in_threadgroup`)
parameter: SG0 alone runs a real MMA and stores its result; after `call void @air.wg.barrier(i32 1,
i32 1)` (the AIR form of `threadgroup_barrier(mem_flags::mem_device)`); SG1 alone reads that SAME
memory location with an ordinary scalar load, doubles it, and stores the result elsewhere.
**Result: SG0's own MMA is bit-exact, and SG1's cross-SIMDgroup read of SG0's result is bit-exact**
(`sg_hw_cross_visibility.ll`) - genuine cross-SIMDgroup MMA-result visibility works correctly at the
hardware level, once the high-level API's own restriction is bypassed.

**The barrier is a real requirement, not superfluous - confirmed by removing it, one factor at a
time.** The identical kernel with the `air.wg.barrier` call deleted (`sg_hw_cross_visibility_nobarrier.ll`,
otherwise byte-identical): SG1 reads back exactly zero - the buffer's pre-write initial state, not
garbage, not a partial value - meaning SG1 raced ahead and observed memory BEFORE SG0's store became
visible. So: **cross-SIMDgroup visibility of an MMA-produced result requires an explicit
`threadgroup_barrier(mem_flags::mem_device)` between the producing SIMDgroup's store and the
consuming SIMDgroup's load** - a real, measured hazard, qualitatively different from every SAME-SIMDgroup
case this campaign has measured (sections 10, 112.1, 112.2: MMA -> same-SIMDgroup scalar consumer,
and MMA -> MMA register-resident chain, both need ZERO explicit synchronization). Ordinary Metal
threadgroup-memory-visibility semantics apply; there is no special tensor-specific cross-SG
synchronization mechanism to characterize beyond that.

**Accumulator ownership**: each SIMDgroup's MMA writes only its own destination fragment/registers
(matching every established fact about per-lane, per-SIMDgroup fragment ownership in this campaign);
nothing measured here suggests SIMDgroups share a register file or an accumulator - the barrier
requirement is exactly what ordinary GPU memory-consistency semantics would predict for two
independent hardware execution contexts communicating through device memory, not evidence of a
distinct tensor-unit-specific ownership rule.

## 119. Cross-SIMDgroup decision table

| Producer/consumer scope | SIMDgroup count | Required synchronization | Legal resource sharing | Evidence | Open cases |
|---|---|---|---|---|---|
| One shared matmul2d op, N SIMDgroups cooperating on one tile | 1, 2, 4 (uniform) | None beyond the op's own internal handling | Full - this is the intended, validated usage | Section 6, bit-exact, re-cited not re-derived | N>4 not tested (hardware may not support more per threadgroup) |
| Independent op per SIMDgroup via the HIGH-LEVEL API (`execution_simdgroup`/`execution_simdgroups<N>` scoped to fewer SGs than launched) | 2 (1 active, 1 idle or doing unrelated work) | N/A - not a synchronization question, the API itself refuses this shape | **NO - silently returns zero, no error** | Section 118, `sg_control_sg0only.metal`/`sg_control_explicit1.metal`, both fail identically | This is an API limitation; not tested whether it generalizes past 2 SGs, since the mechanism (scope must match launch count) is already clear |
| Independent op per SIMDgroup via the RAW AIR intrinsic (bypassing the API) | 2 | `threadgroup_barrier(mem_flags::mem_device)` between producer's store and consumer's load | **YES, with the barrier** - confirmed bit-exact | Section 118, `sg_hw_cross_visibility.ll`, hardware-confirmed both directions (with/without barrier) | 3-4 SG producer/consumer chains not tested (see next row for the MMA-to-MMA consumer case, now closed) |
| Cross-SIMDgroup MMA result -> ANOTHER SIMDgroup's MMA C-accumulator | 2 | `threadgroup_barrier(mem_flags::mem_device)` - the same rule as the scalar case | **YES, with the barrier** - confirmed bit-exact | `sg_hw_mma_to_mma.ll`: SG0 computes `D1 = A1.B1`; after the barrier, SG1 loads `D1` from device memory and issues its OWN MMA `D2 = A2.B2 + D1` (D1 as C); D2 matches `mma_np(A2, B2, D1)` exactly | Closed for the C-accumulator-input case; MMA result as a cross-SG A/B OPERAND (not C) untested, and 3-4 SG chains untested |
| Masks / `SR_SIMD_GRP` / per-SG metadata (op11456, op17258) role in the ordinary cooperative case | 2, 4 | N/A (compiler-generated, already correct per section 6) | Full, established | Sections 6/11, cited, not re-derived - their bit-level roles remain formally "unmeasured" per section 11's own honest phrasing, unchanged here | The exact operand semantics of op11456/op17258 remain open; not needed to answer this mission's synchronization question |

**Acceptance met, and exceeded the "at least the common paths" bar**: hardware-backed rules
established for THREE multi-SIMDgroup tensor paths that matter for a compiler - shared cooperative
dispatch (no sync needed, already known), independent-SG scalar producer/consumer via device memory
(sync needed, now measured exactly), and independent-SG tensor-to-tensor producer/consumer via a
cross-SG C-accumulator (same sync rule, also measured exactly) - with the API-vs-hardware distinction
stated explicitly rather than collapsed. Only 3-4 SG chains and a cross-SG A/B-operand (not C) case
remain named-but-open. No correction to any prior claim was needed; section 113/115's "cross-SIMDgroup
MMA dependency: refuse, not measured" row is now superseded by this section for both the scalar-
consumer and the MMA-C-accumulator-consumer cases and should be read alongside it.

*Addendum (section 133): the open cross-SG A/B-operand case is closed there (fragment-order store to threadgroup or device memory, a workgroup barrier, fragment-order load; bit-exact for the A, B and C roles). Any workgroup barrier sufficed in the tests, including one with no memory scope; the simdgroup barrier and no barrier did not.*

## 120. New mission: characterize remaining not-yet-established accelerator forms, prioritizing compiler-capability expansion over census-filling

Surveyed this document's own running tallies (section 27's exhaustive opcode admission census, 43's
status table, 45's "not yet done" list) before picking a target, specifically to avoid re-deriving
already-closed ground. Result: the raw MMA-opcode surface is unusually thoroughly closed already -
section 27 conclusively shows the 122 declared-but-unadmitted forms (fp8, fp4, int4, 16-bit
accumulators) are not reachable through ANY CPU name this libLLVM knows, by exhaustive bit-flip BFS,
not merely "blocked on an OS gate" as an earlier informal note put it; op11456/op17258's GENERAL
roles are closed (sections 82-92); the one remaining named gap, the `mm_base`/`mm_ta` 2x MMA-count
discrepancy (section 93), is a closed-source rtlib LOWERING-STRATEGY question, not an opcode/form
gap, so it is out of this mission's stated scope and was not pursued.

**Genuinely new ground found instead: `mpp::tensor_ops::matmul2d`'s cooperative-destination
`reduce_columns` capability has ZERO prior test coverage in this campaign - only `reduce_rows` (sum,
max) was ever dispatched (section 19).** The rtlib exports BOTH
`matmul2d_op_cooperative_destination_reduce_rows_*` and `..._reduce_columns_*` (checked directly in
`libTensorOps.rtlib`'s own symbol table this session), and the MPP header declares a matching
`get_column_reduction_destination_cooperative_tensor()` / `reduce_columns()` API pair - a real,
compiler-reachable capability nobody had exercised.

**Confirmed working, first.** Built `mm_reduce_columns_sum.metal` (identical structure to the
already-validated `mm_reduce_sum.metal`, swapping `get_row_reduction_destination_cooperative_tensor`
/`reduce_rows` for the column forms). Controlled dispatch (`B[0,0]=1`, `A[r,0]=v_r` for 32 chosen
values, everything else zero, so column 0 of the raw destination is exact and only the reduction
itself can introduce rounding): the reduced value at index 0 matches `sum(v_r)` exactly.
`reduce_columns` is a real, functioning, previously-unverified capability.

**Reduction order is NOT the same as `reduce_rows` - checked, not assumed.** Repeated section 19's
own eps-triple topology census (`C(32,3) x 3 = 14,880` dispatches, the identical method and dispatch
count section 19 used for `reduce_rows`: one `big=1.0` value and two `eps=2^-24` values per dispatch,
recording whether the two epsilons' contributions survive to the final reduced value) against
`reduce_columns`. The resulting 32x32 merge matrix does NOT match
`reduce_rows`'s own merge matrix (a full, non-trivial mismatch, not a few cells) - the two reduction
directions are NOT mirror images of the same order. Tested the most likely single candidate next
(a natural-order balanced binary tree, pairs-then-quads-then-octets over indices 0-31 in address
order): only 320 of 992 off-diagonal cells match - **ruled out, not a fit**, so this is reported as
a real negative result rather than a forced "probably a balanced tree" claim.

**Correction (section 122, added later).** The inference above that the two directions are "NOT mirror
images of the same order" does not survive the full analysis. The merge matrices do differ, but only
because the reduction axis lands on different physical lane bits: it is one rule applied to two axes (a
sequential chain over the lane's own elements, then exchanges over the lane bits that run along the axis in
ascending order: lane^1 then lane^8 for rows, lane^2, lane^4, lane^16 for columns). Comparing merge matrices
in logical-index space could not have shown that. The balanced-tree negative (320/992) stands as a
measurement; its premise, a tree over address order, was wrong.

**Left open, precisely**: the exact closed-form reduction order for `reduce_columns` (sum). What is
established: it works, it is exact for controlled inputs, and it demonstrably differs from
`reduce_rows`'s own order (both facts checked, not inferred from the API's symmetric-sounding
naming). Determining the exact tree would need the same multi-candidate, iterative process section
19 used to close `reduce_rows` (sequential, ascending-lane, xor-first, and other candidates against
the same 1,296-triple census) - a bounded, well-defined next step, not attempted further this
session. `reduce_columns` under `max`/`min` and the int32 accumulator form are also untested; not
pursued here since the sum/half/fp32 case already establishes the capability exists and differs in
order, which is the compiler-relevant fact.

**Closed in section 122** (exact tree reconstructed from a full-coverage census, verified bit-exactly on
hardware); `max`/`min` and the integer accumulator forms remain untested.

**Why this is the higher-value target than remaining opcode census holes**: a compiler wanting to
lower a column-wise reduction (e.g. a softmax-style operation reducing over the OTHER tile axis from
what's already used) previously had zero evidence this path even works on real hardware. It does,
and now has a name for exactly what remains unknown about it (the order) rather than an unstated
assumption that it mirrors `reduce_rows`.

## 121. A blessed compiler API for chaining-compatibility checking, found and independently corroborating this campaign's own hand-measured layout facts - and a real limitation on actually dispatching through it

Surveyed the MPP header further for capabilities beyond `reduce_columns`. Found
`get_left_input_cooperative_tensor()` / `get_right_input_cooperative_tensor()` and their
`is_compatible_as_left_input(src)` / `is_compatible_as_right_input(src)` gatekeepers: an API,
apparently never exercised anywhere in this campaign, for feeding one `matmul2d`'s OWN destination
cooperative tensor directly into a SECOND `matmul2d`'s A or B operand - exactly the register-
resident chaining shape sections 99-101 characterized by hand-authored means (and found to fail for
the specific op5106/5107-D-into-op5100/5101-A pairing, due to the `pos_b()`/`pos()` layout
mismatch).

**The compatibility check itself is real, and it agrees with this campaign's own hardware-measured
findings, from a completely independent source.** `is_compatible_as_left_input<float, half,
float>(cT1)` - checking whether a half x half -> float GEMM1's destination is a legal float A
operand for a float x half -> float GEMM2 (exactly the op5106/5107 -> op5100/5101 pairing) - **reports
FALSE**. The mirror control, `is_compatible_as_left_input<float, float, float>(cT1)` for a float x
float -> float self-chain (op5098, whose D and A this campaign already measured to share the
IDENTICAL `pos()` layout - section 97) - **reports TRUE**. Apple's own compiler, via a completely
different mechanism (presumably a type/layout compatibility trait, not a hardware dispatch), draws
the exact same line this campaign drew by hand-measuring raw fragment bytes: the same-opcode
self-chain is safe, the cross-opcode chain is not. This is independent corroboration of sections
97/100, not a re-derivation of them, and it hands the compiler owner something immediately useful: a
real, queryable compatibility oracle that does not require consulting this campaign's own measured
layout tables.

**Correction (section 125, added later).** The FALSE reported above is not a statement about op5100/5101: the probe's second GEMM had
`relaxed_precision = false`, so `float x half -> float` compiled to the legacy datapath. With relaxed precision on one `matmul2d` object the same check
is TRUE for half x half -> float -> float x half (op5106/5107 -> op5100/5101) and the chain is exact; the "independent corroboration" of a cross-opcode
layout mismatch is withdrawn. The float self-chain TRUE stands.

**But the actual numerical chain cannot be dispatched through this API to confirm it end to end -
for a reason already on record, not a new mystery.** Even in the TRUE-compatible (`float,float,
float`) case, GEMM1's OWN destination - stored plainly, independent of any chaining, as a control -
comes back all zero once a SECOND `matmul2d` object's compatibility-check and cooperative-accessor
calls are also present in the kernel, even though `op2.run()` itself is provably unreachable in this
configuration only when `compat` is false, and the corruption happens regardless of whether that
branch is taken. This is section 108's own finding ("`mpp::tensor_ops::matmul2d` supports only one
`run()` call per kernel via the high-level API") extended one step further: merely constructing a
SECOND `matmul2d`-related object's compile-time machinery in the same kernel is enough to corrupt
the FIRST op's already-completed result, even without ever calling that second object's `.run()`.
Not re-investigated further (the root cause is the same API-level resource/scope machinery section
108 already characterized as a high-level limitation, not a hardware one) - reported as a real
constraint on USING this otherwise-valuable compatibility API, not as a new hardware finding.

**Retraction (section 127, added later).** The corruption reported in the paragraph above is a host-side misreading of slice offsets (column, row), not a property of the API: with the correct reading the debug kernel's
control store of GEMM1 is exact (8/8) and the chained result is exact (8/8), and chains between two objects work (section 127).

**Net assessment for the compiler owner.** `is_compatible_as_left_input`/`is_compatible_as_right_input`
are real and correct wherever they could be checked, and are a genuinely new, independently-useful
capability (a compile-time-ish oracle matching this campaign's hard-won hardware measurements) - but
actually exercising the cooperative chain they gate requires working around the same
one-matmul2d-object-per-kernel limitation already on record; the memory-mediated bridge (sections
101/103, already proven safe and near-zero-cost) remains the correct thing to hand off as the
actionable chaining mechanism, not this cooperative-input path, until that API limitation is worked
around by someone doing compiler integration (out of this lane's boundary).

**Addendum: the corruption is specifically about constructing a SECOND `matmul2d` object, not
about cooperative-tensor operations in general - narrowed with one more, cheaper check.** The MPP
header's last public function, `is_iterator_compatible(source, destination)` (a free function, not a
`matmul2d` method - checks whether two cooperative tensors can be zip-iterated directly rather than
falling back to threadgroup memory, per the header's own worked comment showing a `*it += *dst_it`
bias-add-style pattern), used with a SINGLE `matmul2d` object's own destination tensor and its own
row-reduction tensor (`op1.get_destination_cooperative_tensor()` and
`op1.get_row_reduction_destination_cooperative_tensor()`, both from the SAME `op1` - no second
`matmul2d<...>` object anywhere in the kernel): reports `true` (a destination and its own reduction
destination ARE iterator-compatible, matching the intended usage), and **GEMM1's own result is
STILL CORRECT** - unlike section 121's `is_compatible_as_left_input` test, which corrupted it. This
cleanly isolates the trigger: cooperative-tensor accessor/compatibility functions are safe to use
freely as long as they only ever touch tensors derived from a SINGLE `matmul2d` object; the moment a
SECOND `matmul2d<...>` template instantiation exists in the same kernel - even one whose `.run()` is
never reached - the first object's result is corrupted. A precise boundary, not a vaguer "the API is
unreliable" claim.

**Retraction (section 127, added later).** The trigger isolated in this addendum does not exist: two `matmul2d` objects in one kernel do not corrupt each other (section 127). The addendum's own control read the
right region for the single-object case only.

## 122. reduce_columns: exact lane/tile routing and summation order, reconciled with reduce_rows; the order belongs to the compiled object, not to the hardware

**Question and result.** Where does each element of the 32x32 cooperative destination (four 16x16
accumulator tiles, one simdgroup) live, and in what order does `reduce_columns` combine a column's 32
partials, given that `reduce_rows` (section 19) and `reduce_columns` (section 120) gave different merge
matrices? The routing is four copies of the op5106/5107 D layout `pos_b()` (section 100). Both reductions
follow ONE rule applied to two physical axes: each lane first sums its own elements of the output in one
direction (sequential, RNE32), then a butterfly of lane exchanges over the lane bits that run along the
reduction axis, in ascending lane-bit order. The direction (ascending or descending logical index) is NOT
a constant: it is a property of the compiled object, it flipped between kernels whose front-end IR for the
reduction is identical, and it is recovered per object by a one-dispatch probe. With the direction taken
from each object's own census, the model predicted 215,040 of 215,040 reduction outputs bit-exactly on a
fresh-seed holdout (10 compiled objects, both reductions, four input families).

**1. Routing, read from the lanes' own registers.** `coop_raw_tagged.metal` runs the 32x32x64 half->float
`matmul2d` and writes each lane's register slots `cT[i]`, `rR[i]`, `rC[i]` to memory (`coop_routing_hw.py`,
`coop_routing_hw.json`); the library's index API is dumped beside them only to be compared. With A =
identity and B[k][c] = 32k + c, D[r][c] = 32r + c, so every slot names its own element: the 32 lanes x 32
slots are a permutation of the 1,024 elements. A second, independent encoding (ten dispatches of bit planes,
D = bit b of 32r + c, values only 0 or 1) rebuilds the identical table, 1024/1024. Closed form, 1024/1024:
tile t = 2*(row>>4) + (col>>4) occupies slots 8t..8t+7, and inside a tile the element (k = row&15,
c = col&15) sits where `pos_b` puts it - lane 16*((k>>2)&1) + 8*(c>>3) + 2*(k&3) + ((c>>2)&1), slot
4*(k>>3) + (c&3). There is no cross-tile lane mixing. `get_multidimensional_index` agrees 1024/1024
(ix[0] = column, ix[1] = row), so it is a valid shortcut afterwards but was not the source. Reduction
destinations: `reduce_rows` has capacity 4 per lane, row r at lane l, slot i iff r = 8i + 4*((l>>4)&1) +
((l>>1)&3) (128/128), so each row is held by 4 lanes (lane bits 0 and 3 free); `reduce_columns` has
capacity 8 per lane, column c = 16*(i>>2) + 8*((l>>3)&1) + 4*(l&1) + (i&3) (256/256), so each column is held
by 8 lanes (lane bits 1, 2, 4 free). Replica lanes were bit-identical in every test below (0 disagreements).
Two harness slips were caught on the way and are not findings: counting the buffer's untouched zero cells as
valid slots suggested capacity 64 (a sentinel-filled buffer shows 32), and a first layout gave the
8-slot column destination a 4-slot stride, which showed up as per-lane capacities that differed (8, 30, 31)
and cannot.

**2. The tree is reconstructed, not assumed.** The eps-triple census (one 1.0 and two 2^-24; the pair
survives iff it merged before meeting the 1.0) fixes a rooted binary tree completely, since each triple
yields one rooted triplet. `tree_reconstruct.py` rebuilds the tree by recursive splitting (the clade size of
a pair is 32 minus the leaves that meet it later) and re-checks it against every observation.
`census_all.py` uses A = identity, B = D, so one dispatch tests 32 triples, one per output, and 14,880
dispatches give every output the whole census (6 s). Section 19's `reduce_tree.json` and section 120's
`reduce_columns_tree.json` (each element 0 only) both reproduce 14,880/14,880 under this reconstruction, and
in every full census all 32 rows (or all 32 columns) share one tree, so nothing depends on which output.
Trees, in the objects section 19 and 120 used:

    reduce_rows    mm_reduce_sum          (leaves = columns)
    (((0 (1 (2 (3 (16 (17 (18 19))))))) (4 (5 (6 (7 (20 (21 (22 23)))))))) ((8 (9 (10 (11 (24 (25 (26 27))))))) (12 (13 (14 (15 (28 (29 (30 31)))))))))
    reduce_columns mm_reduce_columns_sum  (leaves = rows)
    ((((0 (8 (16 24))) (1 (9 (17 25)))) ((2 (10 (18 26))) (3 (11 (19 27))))) (((4 (12 (20 28))) (5 (13 (21 29)))) ((6 (14 (22 30))) (7 (15 (23 31))))))

Rows: a lane holds columns {q..q+3, q+16..q+19} and sums them 19, 18, 17, 16, 3, 2, 1, 0; then lane^1
(columns +4), then lane^8 (columns +8). Columns: a lane holds rows {k, k+8, k+16, k+24} and sums them 24,
16, 8, 0; then pairs of k that differ in bit 0 (lane bit 1, mask 2), then bit 1 (lane bit 2, mask 4), then
bit 2 (lane bit 4, mask 16).

**3. Reconciliation, and a correction to section 120.** Section 120 concluded that the two directions "are
NOT mirror images of the same order". The merge matrices do differ, but only because the reduction axis
lands on different lane bits: rows on {0, 3} (each lane holds 8 of a row's elements), columns on {1, 2, 4}
(each lane holds 4 of a column's). It is one rule, and comparing merge matrices in logical-index space
could not have shown that. The balanced-tree negative (320/992) stands as a measurement; its premise, a tree
over address order, was wrong. Corrected where it was written (section 120).

**4. The first holdout failed, and why that mattered.** `reduce_holdout.py` froze the census fit (descending
local chain) and predicted three families (random fp16-exact values, planted +M/-M cancelling pairs, a
signed power-of-two ladder) on both axes. It was dispatched against a new object, `coop_raw_tagged` (the
routing kernel, which also carries both reductions). The frozen model failed: F1 5331/9600 rows and
7585/9600 columns, F2 2552/9600 and 5066/9600, F3 3703/3840 and 3727/3840. One of the six alternatives
listed in advance, "local chain ascending", matched all of it (9600/9600, 9600/9600, 3840/3840 on both
axes). It was chosen after seeing the data among six, so it was not adopted from that run: the census was
re-run on that very object (ascending), and only then was a new holdout registered with fresh seeds. The
failed prediction stays in the record (`reduce_holdout.json`, `reduce_holdout_prereg.json`).

**5. What decides the direction.** Twenty compiled objects, 26 (object, axis) censuses, each one tree for
all 32 outputs and exactly one (local direction, stage order) variant reproducing it. Stage order was
ascending in all 26. Direction:

    ASCENDING (4 objects, 6 censuses)    coop_raw_tagged, coop_raw_tagged_safe, red_rows_dump, red_cols_dump
    DESCENDING (16 objects, 20 censuses) mm_reduce_sum, mm_reduce_columns_sum, red_rows_nostore, red_cols_nostore,
        red_both_rows_first, red_both_cols_first, red_both_nostore, red_rows_dump1, red_rows_dumpafter,
        red_cols_dumpafter, red_rows_idxonly, red_rows_capacity, red_rows_dump_evenonly, and the
        `-fmetal-math-mode=safe` twins mm_reduce_sum_safe, mm_reduce_columns_sum_safe, red_both_rows_first_safe

The four ascending objects are exactly those that read ALL 32 elements `cT[i]` in a loop before the reduction.
Minimal pairs that do not flip it: reading only `cT[0]`, reading 16 of the 32 (even i), reading only the
capacity, calling only the index API, reading all 32 AFTER the reduction, compiling with strict math, calling
`reduce_rows` and `reduce_columns` in either order, adding or removing the final `cT.store`. In the objects
inspected the front-end IR carries no `fadd` at all: the reduction is one external call,
`__tensorops_impl_matmul2d_op_cooperative_destination_reduce_{rows,columns}_f32(32, 32, 64, 0, 0, 0, 0, coop,
red, 0.0f, 0, 268435472, 268435472)`, with identical constant arguments in every variant compared. So the chain is
laid down when the driver compiles the closed-source rtlib routine into the kernel, and reading every
accumulator first changes that outcome. The mechanism (a register-pressure or scheduling threshold is a
guess, not tested) is not established.

**6. Fresh-seed holdout** (`reduce_holdout2.py`, one process per object). Model hash, seeds, the per-axis
rule and the F1-F3 prediction hash are written before the first dispatch, and the direction is read from
that object's own census file, never set by hand. Families: F1 random fp16-exact, F2 cancelling pairs, F3
power-of-two ladder (A = identity, D = B exactly), and F4 dense random A (32x64) x B (64x32), where the
reduction model is applied to the hardware's own fp32 D (full 24-bit mantissas). Ten objects (4 ascending, 6
descending), 4,800 dispatches:

    family   axis   model          opposite direction
    F1       rows   33600/33600    57%
    F1       cols   33600/33600    80%
    F2       rows   33600/33600    26%
    F2       cols   33600/33600    52%
    F3       rows   13440/13440    98%
    F3       cols   13440/13440    98%
    F4       rows   26880/26880    38%
    F4       cols   26880/26880    49%

215,040 of 215,040 outputs bit-exact, 0 replica disagreements. F3 barely separates the two directions
(98%: a power-of-two ladder rarely rounds), so it adds coverage of the routing and butterfly, not of the
direction. Supplementary and outside the claim: the section 116 MMA model (four ascending K slices, C first)
predicted the F4 fp32 D bit-exactly for 204,800 of 204,800 sampled elements.

**7. Instruction-level corroboration.** In the native code of nine objects inspected (`mm_reduce_sum` and its
safe twin, `red_rows_nostore`, `red_rows_dump`, `mm_reduce_columns_sum`, `red_cols_nostore`, `red_cols_dump`,
`coop_raw_tagged`, `red_both_rows_first`) the cross-lane exchange (op14169) carries a final immediate that is 1
or 8 in every `reduce_rows` reduction (8 exchanges = 4 outputs x 2 stages) and 2, 4 or 16 in every
`reduce_columns` reduction (24 = 8 outputs x 3 stages), the masks the measured trees exchange over. The
`fadd` (op998) counts equal the model's: 36 for rows (4 outputs x 7 local + 4 x 2 stages), 48 for columns
(8 x 3 + 8 x 3), 84 in a kernel with both. This is corroboration of the structure; the immediate's semantics
were measured separately in the addendum below.

**8. One-dispatch direction probe** (`reduce_direction_probe.py`, agrees with the 14,880-dispatch census on
all 14 (object, axis) pairs it was run on). Rows: D[r][0] = 1.0, D[r][1] = D[r][2] = 2^-24, all else 0; the
result is 1.0 for an ascending chain and 1 + 2^-23 for a descending one. Columns: D[0][c] = 1.0,
D[8][c] = D[16][c] = 2^-24. A compiler that uses the library reduction should run this per object; one that
needs a bit-stable sum should emit its own chain and `shuffle_xor` exchanges instead.

**9. The model** (`reduce_model.py`, executable; fitted only to the two element-0 censuses, everything else
held out). Element (r, c) sits at `composed(r, c)` (section 1). For an output, each lane holding it sums its
elements in slot order (direction per object), RNE32, starting from its first element; then for each lane bit
that varies along the axis, in ascending order, acc[l] = RNE32(acc[l] + acc[l ^ (1 << bit)]). Rows vary lane
bits 0 and 3, columns bits 1, 2 and 4. Within one object the census identifies the tree uniquely (14,880
rooted triplets), so the only equivalence is commutation inside each add and starting the chain from an
identity 0.0, which differs from starting at the first element only in the sign of an all-negative-zero
sum (not tested).

**10. Not established.** The mechanism behind the direction flip, and whether triggers other than the
full pre-reduction read exist; every one of the 26 censuses fits the two-direction family, but the 20 objects are
near-variants of one 32x32x64 source shape. `max`, `min`, integer accumulators, other tile shapes, 2-4 simdgroups,
`relaxed_precision`, bf16 and fp32 inputs were not run (other shapes, simdgroup counts and max/min: section 123). One chip, one OS, one toolchain: the rtlib is compiled
at native-compile time, so another driver build may lay the chain down differently. The op14169 immediate is measured in the addendum below
(compile-time-constant form only; a runtime mask was not tested).

**11. Corrections recorded where the claims live.** Section 19 (the descending per-lane order is that
object's, not the hardware's) and section 120 ("NOT mirror images", "left open") carry dated correction notes
in place.

**Addendum: the op14169 immediate is the XOR lane mask (measured).** Another Claude session working on the
ISA decode lane pointed out that the corpus cannot separate `lane XOR mask` from `lane + delta`, since every
immediate the compiler emits is a power of two, and that its own harness reads only lane 0's result (the store that
lands is deterministically lane 0's, and at lane 0 an XOR and an addition agree, 0^k == 0+k), so it cannot read the
permutation. `shx_probe.py` closes it with ordinary Metal source: `simd_shuffle_xor(v,
MASK)` with a compile-time mask, lane L holding tag 100+L, and a THREAD-indexed store, so all 32 received
values are read, not just lane 0's. Nineteen kernels (masks 1-8, 12, 15, 16, 17, 31, 32, 33, 48, 63, 64, 255),
each a 34-instruction stream identical to the others apart from the final immediate of exactly one op14169,
which equals the mask (the other two immediates, 2147483648 and 16, and the register operands are constant
across the sweep). For the 13 masks below 32, including 3, 5, 6, 7, 12, 15, 17 and 31 where XOR and addition
disagree, all 32 lanes received `in[L ^ mask]` (32/32 each); the add and subtract maps fit only a fraction (mask
3: 8 of 29 comparable lanes). For the six masks from 32 up the compiler still emits the immediate unchanged and
every one of the 32 lanes received exactly 0.0. That fits source lane = L ^ imm with a source outside 0..31
yielding 0; it is neither a wrap modulo 32 (which would permute the tags) nor an ignored immediate (which would
return the tags unchanged). On 32-lane hardware "out-of-range source returns 0" and "immediates of 32 and up
return 0" are one observation and cannot be separated. The runtime-mask form and `quad_shuffle_xor` are measured in
Addendum 2 below. Not tested: the two constant immediates. The peer's decode-side reading, that op14169 is `simd.shuffle_xor`, agrees with this.

**Addendum 2: `quad_shuffle_xor`, runtime masks and per-lane masks (measured).** Same thread-indexed probe (`shq_probe.py`,
`shr_lane_probe.py`). (1) `quad_shuffle_xor(v, MASK)`, 13 constant masks (1-8, 12, 16, 31, 32, 255): each compiles to one
op13944 whose final immediate equals the mask. Masks 1, 2, 3: every lane receives the value from the lane that differs from
it by the mask inside its own 4-lane quad (32/32; for these masks the 32-lane XOR and a wrap modulo 4 give the same values,
so they are not separable). Every mask from 4 up (4-8, 12, 16, 31, 32, 255): all 32 lanes receive exactly 0.0, which is
neither the cross-quad XOR the 32-lane form gives for those masks nor a wrap modulo 4. The 4-lane bound is therefore
enforced by the hardware and not only by what the compiler emits, and out of range yields zero as in the 32-lane form.
(2) Runtime masks (mask read from a buffer, uniform across lanes): two kernels that differ only in `simd_` versus `quad_`
differ in exactly one instruction each, op14170 (32-lane) and op13945 (quad), the mask being a 16-bit register operand in
place of the immediate (19 instructions each, otherwise identical). One compiled object per width with eleven masks
dispatched through it (1, 2, 3, 4, 5, 8, 16, 17, 31, 32, 255) reproduces the immediate forms exactly. (3) Divergent masks,
each lane passing its own: the quad form follows each lane's own mask exactly (4/4 patterns, 32/32 lanes each); the
32-lane form does not (0/4 patterns). A pattern with one divergent lane fit 32/32 (all 5 except lane 0 = 9), one with two
divergent lanes fit 27/32, and two alternating masks (1 and 2) gave the single mask 3 on every lane; I did not model the
rest. Treat the 32-lane runtime mask as required to be uniform. Not tested: masks above 255, and data types other than
32-bit.

## 123. Beyond 32x32: the destination layout and the reduction rule generalize; the simdgroup grid is a per-shape library decision; max and min

**Result.** Section 122 covered one shape (32x32x64, one simdgroup). Across 46 destination configurations the layout
inside every 16x16 tile is `pos_b()` with no exception. For one simdgroup, tiles sit in row-major slot blocks, and
the section 122 reduction rule (per-lane chain in slot order, then a butterfly over the lane bits along the axis in
ascending order) reproduced the exact summation tree of all 14 (shape, axis) censuses and 184,320 of 184,320
fresh-seed holdout outputs over seven shapes. With 2, 4 or 8 simdgroups the library refuses `reduce_rows` and
`reduce_columns` at compile time (an API rule, not a measured hardware limit), and the tiles are dealt to simdgroups
in a 2-D interleave whose grid shape the library picks per shape by a rule I could not pin down: three candidate
rules failed and are recorded. `reduce_max` and `reduce_min` are exact on random data; the identity is folded into
the result and NaN is ignored.

**1. Method.** `gen_coop_shape.py` writes a kernel for an MxN destination (K = 64) under S simdgroups that stores each
thread's own register slots `cT[i]` (and, for S = 1, `rR[i]`, `rC[i]`) to memory; the index API is dumped only to be
compared. Two dispatches with A = identity and D = B set to the row index and then the column index name every
(simdgroup, lane, slot) exactly (`cs_route.py`, 46 `cs_route_<M>x<N>_s<S>.json`). Reduction destinations are tagged
by two more dispatches (D[r][0] = r + 1, D[0][c] = c + 1).

**2. One simdgroup, seven shapes (16x16, 16x32, 32x16, 32x32, 32x64, 64x32, 64x64).** Every element sits at the lane
and in-tile slot `pos_b(row & 15, col & 15)` gives (0 mismatches over 10,496 elements), 16x16 tile (tr, tc) occupies slot
block tr * (N / 16) + tc (slots 8 * block .. 8 * block + 7), per-lane capacity is M * N / 32. The library index API
agrees on every slot. Reduction destinations: `reduce_rows` capacity M / 8 per lane with 4 replica lanes,
`reduce_columns` capacity N / 4 per lane with 8 replica lanes, in all seven. So the closed form of section 122 is the
general one: `composed(r, c) = (lane, slot) = pos_b(r & 15, c & 15) with slot += 8 * ((r >> 4) * (N >> 4) + (c >> 4))`.

**3. Reduction order across shapes.** `census_shape.py` runs the eps-triple census on every output at once (one triple
per output per dispatch; up to 124,992 dispatches for a 64-element axis), rebuilds each output's tree with no assumed
order, checks it against every one of its observations, and compares it with the shape-general model
(`reduce_model_shapes.py`, a separate file so `reduce_model.py` stays byte-identical to the hash the section 122
holdouts registered). All 14 censuses (both axes of the seven shapes, kernels with no `cT` read, so the descending
context of section 122) give one tree for all outputs, reproduced exactly by the model with the chain descending and
the butterfly ascending. Two censuses (16x16 and 16x32 `reduce_columns`) also fit the ascending chain: a lane holds
only two elements of such a column, a two-element chain is symmetric, and the direction is unobservable there (an
exact equivalence class, not a gap). Fresh-seed holdout (`reduce_holdout_shapes.py`; prediction hash written first;
F1 random fp16-exact, F2 planted +B/-B pairs in every row and column, F3 power-of-two ladder): 11,520, 17,280,
17,280, 23,040, 34,560, 34,560 and 46,080 outputs bit-exact for the seven shapes in the order above (184,320 of
184,320); the opposite direction fails a substantial fraction wherever the chain has more than two elements.

**4. Several simdgroups.** `reduce_rows` and `reduce_columns` do not compile with `execution_simdgroups<2>`: the
header's own `static_assert` says "reduce_rows requires a single SIMD group" (same text for columns). That is a
high-level API rule; nothing here says the hardware cannot exchange across simdgroups. What could be measured is the
destination itself (S = 2, 4, 8; 39 configurations, 16 to 64 rows and 16 to 128 columns): `pos_b` inside every
tile holds with 0 mismatches, each 16x16 tile lies wholly in one simdgroup, and every configuration is reproduced by

    simdgroup = (tr % gr) * gc + (tc % gc),   block = (tr // gr) * ceil(nt / gc) + (tc // gc)     (gr * gc = S, nt = N / 16)

for some grid (gr, gc). Shapes whose tile counts do not divide evenly are padded: the library reports the extra slots
as not valid (for example 16x48 with 2 simdgroups has capacity 16 per thread for 768 elements), so capacity is
not M * N / threads there. What I could not establish is the grid the library picks. Measured (tile rows x tile
columns -> grid), S = 2: 1xn -> (1,2) always; (2,1) (2,2) (2,3) (2,5) (3,1) (3,2) (3,3) (3,6) (4,1) (4,2) (4,3) (4,4)
(4,6) (4,8) -> (2,1); (2,4) (2,6) (3,4) -> (1,2). S = 4: (2,2) for every shape with both tile counts at least 2 except
2x8 -> (1,4); 1xn -> (1,4) (1x2 also fits (2,2)); nx1 -> (4,1). S = 8: (2,4) for 4x4, 2x8, 4x8. Three rules were tried and failed. "Split the
longer dimension" was registered before a 12-configuration batch and reproduced 8 of 12 (missed 32x48 with 2
simdgroups, and three 4-simdgroup shapes with one tile row or column). "Balanced split, squarest per-simdgroup tile
grid" was falsified by the already-measured 48x32 with 2 simdgroups. A rule fitted to all 29 configurations then
measured (2 simdgroups: columns iff N >= 2M; 4: (2,2) unless a tile count is 1) was registered for 14 fresh
configurations and reproduced 9 of 14 (missed 48x64, 48x96, 64x128, 32x80 with 2 simdgroups and 32x128 with 4). A
contrived rule that fits every S = 2 measurement exists (apart from the forced 1xn case) (columns iff the column tile count is even, exceeds the row
count and the destination is at most 12 tiles) but was fitted after the fact and is not offered. Use the measured
table, or the library's own index function, which agreed on every slot of all 46 configurations.

**5. max and min** (`reduce_maxmin.py`, 32x32x64, half inputs, one simdgroup). On 200 random fp16 matrices both axes
equal numpy's max and min exactly (6400/6400 per axis per operation). The identity is folded into the result: rows or
columns whose elements are all -inf give -FLT_MAX for `max` and all +inf give +FLT_MAX for `min`, which are the
header's `lowest()` and `max()` identities. NaN is ignored (fmax-like): a row or column mixing NaN with finite values
returns the finite max or min, and an all-NaN row or column returns the identity. Injection used A = diag(v) with B = ones and A = identity
with B carrying the pattern, so no product other than the intended one is nonzero. For a masked softmax this means a
fully -inf row gives max = -FLT_MAX from this API, not -inf.

**6. Not established.** Simdgroup grid selection (above). Reductions across simdgroups in hardware (only the API
refusal was seen). Integer accumulators, `relaxed_precision`, bf16 and fp32 inputs, K other than 64, a chain longer
than the tiles tested, the direction flip of section 122 in other shapes (the 14 censuses were all taken in the
no-read context). One chip, one OS, one toolchain.

## 124. Do the direction and grid heuristics unlock anything? Direction: yes, and it is fast-math reassociation; grid: no

**Decision.** The user's condition was to recover the direction-selection and simdgroup-grid heuristics only if that unlocks compiler
capability. Direction: the mechanism is reproduced in user code and it reduces to one rule a compiler can act on (emit the reduction
yourself with strict adds and the order is exactly what you wrote, in every context), so no further recovery is warranted. Grid: the
compiler can read any layout from the library's index function or from the measured table, so the heuristic is not needed; it was
not pursued further after three failed rules (section 123).

**1. An explicit reduction has a source-controlled order under strict math.** `gen_own_reduce.py` writes the section 122 reduction
by hand: a per-lane chain over the lane's own `cT[i]` slots in a chosen direction, then `simd_shuffle_xor` stages in ascending lane-bit
order (constants 1, 8 for rows; 2, 4, 16 for columns; slot to (row, col) from the closed form, no index API). Checked against
`reduce_model.py` on 200 fresh-seed matrices (F1 random, F2 planted cancelling pairs; 6,400 outputs per axis; replica lanes 0 disagreements),
with `-fmetal-math-mode=safe`: ascending and descending sources reproduce the model of their own direction exactly (6400/6400 on both
axes), with and without a full pre-reduction read of `cT`. That is 4 of 4 strict objects.

**2. Under default (fast) math the chain is reassociated, and the triggers are the library's.** The default build's front-end IR carries
`fadd fast` and `fadd reassoc nsz arcp contract afn` (the safe build has plain `fadd`). An ascending source with no other use of the elements
ran DESCENDING (opposite direction 6400/6400 on both axes); a descending source stayed descending; an ascending source in which all 32
`cT[i]` are also stored kept its order. Predicted before running and then confirmed: an ascending source with only PARTIAL extra uses (every
even slot, or slot 0 alone) also normalises to descending (6400/6400 each axis). These are the triggers seen for the library routine in
section 122 (read all 32 before the reduce flips it to ascending; reading 1 or 16 does not). The library's own fadds sit in the closed rtlib
body that the driver compiles, which a front-end math flag cannot reach: that is why the strict-math twins of section 122 did not change.
So the reading that fits everything is: the routine's source order is ascending, fast-math reassociation normalises it to descending
unless every leaf has another use. This is an explanation by reproducing the triggers in user code, not an observation of the driver pass.

**3. The read-all trigger is not universal for the library routine.** One-dispatch direction probe on seven single-simdgroup shapes
(`shape_direction_probe.py`): with no extra reads every shape is DESCENDING (16x16, 16x32, 32x16, 32x32, 32x64, 64x32, 64x64; the
16-row column reductions have a 2-element chain and are unobservable); with all `cT` read before the reduce, 32x32, 32x64, 64x32 and 64x64
flip to ASCENDING on both axes, while 16x16, 16x32 and 32x16 (per-lane capacity 8 or 16) stay descending. Not explained.

**4. What a compiler does with this.** (a) Emit the reduction itself with plain `fadd` and `shuffle_xor` (op14169 with the immediate mask, op14170
for a uniform runtime mask): its order is whatever it wrote, in every context; the layout, exchange semantics and rounding are all
established. (b) If it uses the library routine, the default context gives descending in all seven shapes tested, and the one-dispatch probe
reports the direction of any given object. (c) Cross-simdgroup reduction is refused by the API and would have to be written by hand (partials
through threadgroup memory under the section 118-119 barrier rule); not attempted here.

## 125. The library chains an MMA's accumulators directly into the next MMA's A or B, register-resident and exact; one destination layout serves every accelerator form; op5100 and op5104 are emitted under relaxed precision

**Result.** Section 104 told a compiler to bridge through memory unless a pair was hardware-confirmed, and section 121 reported that the
library's own compatibility check refuses op5106/5107 -> op5100/5101. Both were too pessimistic, and the second was mislabeled: the
second GEMM in that probe was not relaxed, so it compiled to the legacy datapath and never was op5100/5101. On ONE `matmul2d` object (the
same descriptor for both GEMMs) the library chains a destination cooperative tensor into the next GEMM's left or right input with
`is_compatible_as_*_input` TRUE in all 18 type combinations tried, the results are exact, the op5106/5107 -> op5100/5101 and
op5106/5107 -> op5104/5105 chains contain zero instructions between the two GEMMs (GEMM2's A or B operand registers are GEMM1's
accumulator registers), and the direct result is bit-identical to an explicit store/reload bridge on 200 random matrices. The layout
reason is that the destination of every accelerator form measured here has one layout.

**1. Destination layout and opcode census over 30 forms** (`gen_form.py`, `form_route.py`; 32x32x64, one simdgroup; the kernel stores each
lane's `cT[i]`, two raw planes name every slot). Twenty-three forms compile to an accelerator opcode and all 23 have the destination
layout `pos_b(row & 15, col & 15)` inside every 16x16 tile with tile blocks row-major, 1024/1024 lane and slot, capacity 32 per lane:
op5106 x16 for half x half -> float or half, bfloat x bfloat -> float or bfloat, and the mixed pairs half/bfloat with int8/uint8 (either
side) and half x bfloat, bfloat x half (the int8, uint8 or bfloat operand is converted in ALU code around the MMA; not inspected); op10384 x16 for int8 x
int8 and uint8 x uint8 -> int32; and, ONLY with `relaxed_precision = true`, op5098 for float x float, op5104 for half, bfloat or int8
x float, and op5100 for float x half, bfloat or int8. The other seven forms (float x float, float x half, half x float, bfloat x float,
float x bfloat, float x int8, int8 x float, all without relaxed precision) contain no accelerator opcode at all: they run on the legacy
datapath and their layout is a different one. This corrects the statements that op5100/5101 are "never compiler-emitted" (section 27
Executed, section 42, sections 98 and 104): the library emits op5100 and op5104 for the relaxed mixed forms, and only for them.

**2. An input layout, measured.** The left-input cooperative tensor of the half x half -> float form has capacity 32 per lane but only
register slots 0-15 exist (16 registers = 4 tiles x 4); a slot's register carries a pair of adjacent-k halves and element access reaches
the low half only. One-hot probing of all 1024 (lane, slot) pairs (`form_in_route.py`; every valid slot lit exactly one operand element)
gives 512/512 agreement with `pos_b(row, k)` at register granularity, tile block 2 * tr + tc, so the A operand and the accumulator
share one layout function. The right operand and the other forms were not enumerated one by one; the chains below test them
functionally.

**3. Same-object chains** (`gen_chain1.py`, `chain1_check.py`, `chain1r_check.py`; 32x32x32; small non-negative integers so every product is
exact in every type; 12 random matrices each; D1 is the destination type of GEMM1 and must equal the element type of the next input). Left
input, GEMM1 -> GEMM2 (all TRUE, all 12/12 exact): half x half -> half -> half; half x half -> float -> float x half (op5106/5107 ->
op5100/5101); bfloat x bfloat -> bfloat; bfloat x bfloat -> float -> float x bfloat; float x float (relaxed) self-chain (op5098/5099);
half x float -> float -> float x float (op5104/5105 -> op5098/5099); float x half self-chain (op5100/5101); float x bfloat, bfloat x float
and half x bfloat combinations. Right input (D1 becomes B of GEMM2 = A2 x D1; all TRUE, 12/12): half -> half, half x half -> float ->
half x float (op5106/5107 -> op5104/5105), float relaxed, bfloat x bfloat -> float, bfloat x bfloat -> bfloat, half x float ->
half x float. Between the two GEMMs: nothing for float destinations feeding float operands (op5106/5107 -> op5100/5101, op5106/5107 ->
op5104/5105, op5104/5105 -> op5104/5105), and 8 conversion instructions (op1016 for half, op1048 for bfloat, fp32 to 16 bits per lane) when
the next operand is 16-bit. Nothing stores D1 or reloads it in any of them; the only other instructions between the GEMMs are fragment loads of GEMM2's other operand. Rounding-sensitive check on op5106/5107 -> op5100/5101
(`chain_vs_bridge.py`, the same object computes GEMM2 both ways from the same `cT1`: direct as left input, and after `cT1.store` +
barrier as an ordinary float tensor): bit-identical on 200/200 random matrices (204,800/204,800 elements), with error against fp64 up to
8e-4 of the largest value, the 10-bit truncation of a float A operand already recorded for the fp32 operand forms (section 5). So the direct chain has the
bridge's numerics, not better.

**4. What this changes, and where the earlier text is corrected.** Section 121 (the FALSE for half -> float A was a non-relaxed second GEMM,
i.e. the legacy datapath, not op5100/5101); section 104's decision table and eligibility rule (the op5100/5101 row said "never
compiler-emitted" and the chain rule said bridge; the direct chain exists for the pairs in section 3); section 112.5 (an MMA result as the next
MMA's OPERAND, register-resident, was "refuse - not measured"; at the library level it is now measured for A and B). Section 100/101's bridge
stays correct and stays the safe fallback; it is no longer the only measured mechanism. Corrected where each claim lives.

**5. Limits.** (a) One `matmul2d` object, the same descriptor for both GEMMs, 32x32x32: a chain between GEMMs of different M, N, K needs a second
object, and section 121's second-object corruption applies; not tested here. **Superseded by section 127:** the corruption claim was wrong; chains between two objects work when GEMM2's N is at least GEMM1's N and are silently wrong when it is smaller. (b) The evidence is the library-compiled instruction stream and its
exact results; a hand-authored AIR replication was not done here (**done in section 129**: D -> A and D -> B, 20/20 each). (c) The older measured tables (`layout-new_f16f16_nn.json`, `layout-new_f32f32_nn.json`)
give D = `pos()` while every raw cooperative slot here gives `pos_b()` (as does section 100's hand-authored store); **Reconciled in section 129**: the tables are hardware coordinates and `pos_b` is `pos` with rows rotated one bit; the tables and the raw slots do not conflict. (d) The input-layout enumeration covers one form and one operand.
(e) Integer destinations cannot feed an int8 input (types differ), so int32 chains do not exist at this API level. (f) One chip, one OS, one
toolchain.

## 126. 16-bit destinations: rounded after every accumulating run() call, and half reductions add in half precision (bfloat reductions do not)

**Result.** A half or bfloat destination cooperative tensor is rounded to 16 bits after each accumulating `run()` call, not only when it is read; a single `run()`
over the whole K keeps fp32 accumulators and rounds once. `reduce_rows` / `reduce_columns` sum on a half destination by rounding every add to half, on a bfloat
destination by adding in fp32 and rounding once at the end, and on an int32 destination exactly. max and min are exact in all three. All of it uses the same summation
tree as section 122 and 123.

**1. Accumulation across run() calls** (`gen_acc16.py`, `acc16_check2.py`; 32x32, multiply_accumulate, K = 16 per call, destination zeroed, then four calls with one
non-zero k per call; every element ends equal). Contributions BASE, 1.5, 1.5, 1.5. Half, BASE 2048 (spacing 2): one rounding of the fp32 sum gives 2052, rounding after
every call gives 2054; observed 2054. Bfloat, BASE 256 (spacing 2): one rounding gives 260, per-call rounding gives 262; observed 262. The same contributions in ONE
`run()` with K = 64 (four internal 16-K slices, `acc16s_*`) gave 2052 (half) and 260 (bfloat): fp32 through the slices, one rounding at the end. With a float destination
and half or bfloat inputs the four calls gave 2052.5, exact. (A first bfloat run with BASE 2048 read 2048 and did not discriminate, since both roundings give 2048; the
BASE 256 design replaced it.) So a K-loop that calls `run()` per K block should use a float destination, or one `run()` over the whole K, or it pays a 16-bit rounding per block.

**2. Reductions on non-fp32 destinations** (`gen_red_dt.py`, `red_dt_check.py`, `red_dt_model.py`; 32x32x64, one simdgroup, kernels with no `cT` read; 100 random matrices per
operation for the exactness rows). Sum, max and min equal numpy exactly on half (3200/3200 per axis per operation), bfloat and int8 x int8 -> int32. Max and min fold the
identity in as they do for float: an all -inf row gives -65504 on half and the lowest bfloat (-3.3895e38) on bfloat, +inf gives the mirrored maxima; int32 max/min are exact
over the full int8 range. Summation precision, by an eps-triple probe at half the destination's spacing of 1.0 (2^-11 for half, 2^-8 for bfloat): on half the eps pair
survives in 165 of 495 triples (exactly one third, the pattern of a rounding tree), on bfloat in 495 of 495 (no intermediate rounding). Models, fresh seeds, 200 matrices
(F1 random, F2 planted cancelling pairs), the section 123 shape model with the chain descending and the butterfly ascending: half with every add rounded to half, 6400/6400 outputs
on rows and 6400/6400 on columns (results that overflow to inf and then give NaN count as equal when both sides are NaN); bfloat with fp32 adds and one rounding to bfloat at the
end, 6400/6400 and 6400/6400. Rounding each half add through fp32 first (double rounding) gave identical counts, so that variant is not separable here. A first run of the half check
counted NaN results as mismatches (NaN != NaN) and read 6368/6400 and 6361/6400; the comparison was fixed and those were all agreements.

**3. For a compiler.** Accumulate in a float destination and convert at the end unless a per-block 16-bit rounding is acceptable; reduce on a float destination when the
sum's accuracy matters (a half reduction rounds every add, a bfloat one does not, and an int32 one is exact). The order is the same rule as sections 122-124.

**4. Not established.** K other than 16 and 64 for the accumulation test; the direction flip of section 122 for 16-bit destinations (all objects here had no `cT` read, all
descending); NaN handling of max and min on 16-bit types; integer overflow (no int32 sum here can overflow with 32 elements of int8 x int8 products); one chip, one OS, one
toolchain.

## 127. Retraction: "one run() per kernel" and "a second matmul2d object corrupts the first" were my host-side misreading of slice offsets; two-object chains work when N2 >= N1 and are silently wrong when N2 < N1

**What was claimed.** Section 108 said `matmul2d::run()` can be called only once per kernel (a second call "silently writes all zeros") and concluded that GEMM -> requantize -> GEMM
must cross a kernel boundary. Section 121 and its addendum said that constructing a SECOND `matmul2d` object corrupts the first object's already-completed result, so a chain could not
be dispatched through the cooperative-input API. Both were relayed to root as API limitations.

**Cause.** On these tensors, `tensor<..., dextents<int, 2>(cols, rows), {1, ld}>`, the offsets of `slice<E0, E1>(o0, o1)` are (dim0 = column, dim1 = row). `tC.slice<16, 16>(16, 0)` is
columns 16..31 of rows 0..15. The second call wrote a correct result there; I read rows 16..31 of the first columns, where nothing was written, and every "independent check" in section 108
(fresh tensor objects, a barrier, a different slice) shared that reading. The same misreading placed the second GEMM's operand at the wrong rows in the section 121 kernels.

**Evidence.** (1) `reqz_isolate_repeat.metal` (int8, one object, two `run()` calls), read at the right place: first call bit-exact 6/6, second call bit-exact 6/6; read the old way, the second
call is 0/6. (2) `mm_coop_input_chain_f32f32_debug.metal` (two objects, the section 121 kernel), correct offsets: GEMM1's plain control store 8/8 exact and the chained result 8/8 exact. (3) A
two-object chain with identical descriptors (32x32x32, `xch_same`): compatibility TRUE, control 12/12, chained result 12/12. (4) A single-kernel int8 GEMM -> requantize -> GEMM
(`oneker_requant_chain.metal`: one object, two `run()` calls, D1 stored to device memory, `threadgroup_barrier(mem_flags::mem_device)`, requantized to int8 in ALU code, second GEMM
on the requantized tile): exact on 20/20 random cases. (5) Sections 118-119 (a scope narrower than the launch returns zero) are not affected: those kernels use offset (0, 0) and the control with 32
threads works. Sections 122-126 read destinations at (0, 0) or through raw slots, and their offset-bearing kernels were checked against this convention.

**What stands and what is withdrawn.** Withdrawn: the one-`run()` limitation, "must cross a kernel boundary", the second-object corruption and its addendum, the statement that the chain "cannot be
dispatched through this API". Standing: the compatibility oracle's answers, the layout findings of sections 97, 100, 121 (the compatibility results), and the multi-kernel requantization path of section 108
as a working path (no longer a necessity). Section 125's limit (a) is answered below.

**Chains between two objects of different descriptors** (`gen_xchain.py`, `xchain_check2.py`; GEMM1 M x N1 x K1 half x half -> float, GEMM2 M x N2 x N1 with GEMM1's destination as the left input
and half B2, both relaxed; small integers, 8 random matrices each; the control store of D1 was exact 8/8 in every case):

    N2 >= N1 : correct 8/8 in all six cases (32,32,32,N2 = 32, 48, 64; 16,32,32,N2 = 32, 64; 64,32,32,N2 = 32;  32,16,32,N2 = 16, 32; 32,32,16,N2 = 32)
    N2 <  N1 : N1 = 32, N2 = 16 : compatibility TRUE but the chained result is WRONG (0/8 whole matrices; some rows right, some zero) at M = 16, 32, 64 and K1 = 16, 32

So `is_compatible_as_left_input` returning TRUE is not sufficient: a GEMM2 narrower than GEMM1 silently produces a partly wrong result (not explained; the smaller-N cases of N1 = 32 all failed). A
descriptor that differs only in mode (`multiply_accumulate` for GEMM2, `xch_diffmode`) worked 12/12. Where GEMM2 is narrower than GEMM1 the memory bridge of sections 101-103 is the safe path.

**Explained in section 132.** The narrower-consumer failure is the library's input tensor exposing only the first min(#D1 tiles, #D2 tiles) blocks of D1 in row-major block order (identity count exactly M * N2), 27 of 27 configurations on the left input and 27 of 27 on the right; hand-authored register chains have no such
hazard. Pad the consumer up to the producer's tile count, or use the memory bridge.

**Effect on the record.** Sections 108 and 121 carry in-place retraction notes; section 125's limit (a) is superseded here. Two statements I relayed to root are wrong and are corrected with this section.
The lesson is a method one: a "silently returns zero" result is a claim about the region I read, so the control is to write a distinct sentinel everywhere and look at the whole buffer, not one expected region.

## 128. Cooperative input tensors (the register-direct chain API) require a single simdgroup

The chain of section 125 was tried on a multi-simdgroup cooperative destination (`gen_msg_chain.py`: one `matmul2d` object with `execution_simdgroups<S>`, GEMM1 = GEMM2's left input,
M = N = K). All five multi-simdgroup kernels (S = 2 at M = 16, 32, 64; S = 4 at M = 32, 64) fail to compile with the header's own `static_assert`: "Input cooperative tensors require a single SIMD group".
The S = 1 control at 32x32x32 built and ran: compatibility TRUE, GEMM1's control store 8/8 exact, chained result 8/8 exact. This is the same kind of rule as `reduce_rows` / `reduce_columns` ("requires a single SIMD
group", section 123): both cooperative-tensor conveniences, input chaining and reductions, are single-simdgroup features of the API. It is an API rule and not a measurement of the hardware; whether a hand-authored
register chain across simdgroups could work is untested. (By reasoning only, not tested: with the 2-D tile interleave of section 123 a simdgroup often holds only part of each row of the previous destination, which a
register-direct A operand needs in full.) For multi-simdgroup GEMM chains the working path is the memory bridge with the barrier rule of section 119.

**Transposed operands.** A cooperative input tensor cannot be transposed either: with `transpose_left = true` (or `transpose_right = true`) on the consuming GEMM the compiler stops at a `static_assert`, "Input cooperative tensor cannot be transposed"
(`xt_left`, `xt_right`; the second GEMM on its own `matmul2d` object with the flag set, its left or right input GEMM1's destination). By the fragment tables of section 129 the hardware A slot at (lane, slot) holds A(D_row, D_col) of that slot, so
D1^T as an A operand needs each element from a different lane and slot; a register-direct transposed chain would need cross-lane exchanges. That is reasoning from the tables, not a measurement; the memory bridge (store, reload as a transposed tensor) is the tested way to chain through a transpose.

**Correction (section 132, added later).** The reasoning above is wrong: the MMA has transposeA and transposeB bits and the hardware transposes the fed operand itself. A register-direct chain of D1^T into A or B is exact, with a fixed one-bit rotation of the row or column labels
(part 3 of section 132, 12 shapes and 5 forms). What the compiler refuses is the library's cooperative input tensor, not the hardware.

## 129. pos and pos_b are the same fragment map with rows rotated by one bit, not an opcode property; a hand-authored register-direct chain needs one fptrunc and a load-time k permutation

**Result.** Sections 99-100 concluded that the D-accumulator layout is "a property of the OPCODE" (`pos()` for op5098, `pos_b()` for op5106/5107) and section 112.5 left "an MMA result as the next MMA's
operand" open. Neither survives. `pos_b(k, c) = pos(rotl1(k), c)` for all 256 (k, c), where `rotl1` rotates the 4-bit row index left by one bit (k = 0..15 -> rows 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5,
7, 9, 11, 13, 15): the two layouts have identical lanes and slots and differ only in how rows are labeled. In the measured hardware tables the A fragment equals the D fragment slot for slot, and B's k is the
row-rotated D row. With those tables alone (no fitted parameter) hand-authored register-direct chains through one `fptrunc` are predicted exactly, D -> A and D -> B, 20/20 each.

**1. The tables** (`layout-new_f16f16_nn.json` and `layout-new_f32f32_nn.json`, which are identical to each other in A, B and D). At every (lane, slot): A_row == D_row and A_k == D_col (256/256, half AND fp32),
B_col == D_col (256/256), and B_K = `rotr1(D_row)` (a function of the D row alone: D rows 0..15 map to B_K 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15). The library's raw destination and left-input slots
of section 125 are `pos_b(row, .)` = `pos(rotl1(row), .)`: in the library's logical coordinates rows of A and D sit at hardware row `rotl1(row)`, B's k is unrotated, so D's logical row equals B's k. That is what makes a
library D -> B chain a zero-permutation chain, and the same rotation on A and D keeps D -> A direct. Feeding raw D registers to an operand that labels rows the other way permutes rows; section 42's negative
(D as fp32 A gave neither D1.B2 nor D1^T.B2) and the section 99 failures fit that mechanism, but I did not re-run them to confirm it.

**2. Hand-authored AIR chains** (`dep_chain_tables.py`, `dep_b2_check.py`; kernels `dep_a_from_result.ll`, `dep_b_from_result.ll`, `dep_b2_from_result.ll`, each `air.simdgroup_matrix_16x16x16_multiply_accumulate`, half x half -> float, one
`fptrunc <8 x float> to <8 x half>` between the two MMAs, no store, no reload, no wait; 20 random tiles each). D -> A: the second MMA's A operand is the truncated D1 vector; result D2 = half(D1) . B, exact on
20/20 (the identity model and the table model coincide, A == D). D -> B: the second MMA's B operand is the truncated D1 vector; the table prediction `B2[B_K][B_COL] = half(D1[D_ROW][D_COL])`, i.e. D1's row r
placed at B row `rotr1(r)`, is exact on 20/20; the identity model that section 112.5 used (B2 = D1) is 0/20, so that failure was a wrong expected value. Recipe for A2 . D1 through a B feed: load A2 with its k index
permuted, `A2'[:, j] = A2[:, rotl1(j)]` (free in the load addressing): the result equals A2 . half(D1) to rounding on 20/20 (worst relative error 9.5e-8) and the hardware bits match the table model on 20/20; with A2
unpermuted it is 0/20. Permuting k changes the association order inside the MMA's k tree, so the result is not bit-identical to an unpermuted product; that is a rounding difference, not an error. (An input-scope note: the
kernels read a second A block from the A buffer, so they must be dispatched with `dim=32`; with `dim=16` the second block is not visible and the result is garbage.)

**3. Corrections** (in place): section 100 (the "opcode-specific" headline), section 104 (the D column and the eligibility rule), section 112.5 (both decision-table rows), and section 125's limits (b) and (c).
Not established: which row labeling section 100's own kernel used when it read `pos_b()` (I did not re-run it, so what produced its 256/256 is unverified), and whether the tables, measured on a library-compiled
single-MMA kernel, hold for every kernel (they match every hand-authored chain here and the library chains of section 125). Extending to a full multi-tile GEMM is per-tile and was not built. (**Built in section 132**: 1x1 to 4x4 grids, four feed modes, five forms, bit-exact.)

## 130. Hardware simd_sum / quad_sum / simd_product / quad_product: a balanced adjacent-lane tree with every operation rounded to the element type

Requested by the ISA-decode peer, who needed the lane-combining order of the four half forms (quad.sum, quad.product, simd.sum, simd.product); measured for half and float with the section 122 method (thread-indexed store, all lanes read).

**Tree.** Eps-triple census in the element type (`hr_census.py`; half: big = 2048, eps = 1, a tie at spacing 2; float: big = 1.0, eps = 2^-24; all 14,880 triples for 32 lanes, all 12 for a quad, all 8 quads per dispatch
compared). The reconstructed tree reproduces every observation and is the same for half and float: `simd_sum` = `((((0 1) (2 3)) ((4 5) (6 7))) ...)`, a balanced binary tree over adjacent lanes (equivalently exchanges over lane
bits 0, 1, 2, 3, 4 in that order), `quad_sum` = `((0 1) (2 3))`; every lane holds the same result. The eps pair survives in exactly one third of triples (4960 of 14,880; 4 of 12), the count of a rounding tree. In half nothing is
accumulated above half precision: an fp32-internal unit with one final rounding would have survived every triple.

**Holdout** (`hr_holdout.py`, fresh seeds, 900 reductions per 32-lane kernel and 7,200 per quad kernel; families random values, random with a planted +2048/-2048 pair; for product values near 1 in three shapes). Model:
balanced adjacent tree, every add or multiply rounded to the element type (half: exact in float64 then RNE to half). Bit-exact on all eight kernels: simd sum half 900/900, float 900/900; quad sum half 7200/7200, float 7200/7200; simd
product half 900/900, float 900/900; quad product half 7200/7200, float 7200/7200; lanes within a group never disagree. Alternatives on the same data: ascending sequential 139-618 of 900 (sum), descending sequential 166-612, tree with the
top level first 225-610, single final rounding (half) 253 of 900; the product alternatives score 163-378 of 900 for simd and 4965-5586 of 7200 for quad.

**The peer's two-point discriminator on hardware.** Lane 0 = 2048 (or the last lane), every other lane 1.0, half: `simd_sum` returns 2078.0 (half bits 0x680f) and `quad_sum` returns 2050.0 (0x6801), for the 2048 in the first or the last lane. The model predicts
both (2078 and 2050). Three predictions written from an exact inner sum (sequential 2048, pairwise 2080 and 2052) were all wrong because the ones are summed with a rounding to half at every level.

**Link to section 122.** The cross-lane stage order of the TensorOps reductions (exchanges over ascending lane bits) is the order of this hardware reduction; a hand-written reduction that uses `simd_sum` over a lane's partials has this tree.
Not established: the forms whose Metal construct is not `simd_sum` / `simd_product` / `quad_sum` / `quad_product` (prefix and shuffle-based forms); integer and other element types; NaN and infinity handling.

## 131. Small closures: int32 accumulation wraps, relaxed_precision is inert for the native forms, and special values are IEEE through the MMA

**int32 accumulation wraps.** `ovf_check.py` (`ovf_i8`, `ovf_u8`; 32x32, `multiply_accumulate`, K = 16, the int32 destination preloaded, one accumulating `run()` adding a single product per element, every slot read). int8: INT_MAX - 100 plus 127 x 127 gives -2,147,467,620, the two's-complement
wrap, not INT_MAX; INT_MIN + 100 plus (-128 x 127) gives 2,147,467,492, the negative-side wrap, not INT_MIN; uint8: INT_MAX - 100 plus 255 x 255 wraps to -2,147,418,724. Controls that do not overflow are exact. So an int8 GEMM accumulated across `run()` calls (or with an
initial C) overflows modulo 2^32 with no saturation. A single K = 16 tile cannot overflow by itself (16 x 16384 = 262,144); overflow inside one MMA was not tested. *(Section 136 part 5.1: overflow inside one MMA wraps as well, L = 1 with the accumulator 1000 below INT_MAX or 1000 above INT_MIN and a step that crosses the limit, and 608 accumulators are exact modulo 2^32 up to L = 1,000,000 in all four sign combinations.)*

**relaxed_precision does nothing for the native forms.** With `relaxed_precision = true` the forms that already compile to op5106 or op10384 (half x half -> float or half, bfloat x bfloat, half x int8, half x bfloat, int8 x int8 -> int32) keep the same accelerator opcode
and the same destination layout (`pos_b`, tile blocks row-major, 1024/1024), and their raw destination slots are bit-identical to the non-relaxed build on 40 random cases each (`rx_dump.py`). It only changes forms with a float operand (section 125): those leave the legacy
datapath for op5098, op5100 or op5104.

**Special values are IEEE.** `ieee_check.py` on the raw destination of half x half -> float (op5106), bfloat x bfloat -> float and relaxed float x float (op5098), constant operands so every slot holds one value: 0 x inf + 1 -> NaN (both operand orders), inf + (-inf) -> NaN, NaN x 1 + 1 -> NaN, inf x 2 + 3 -> +inf,
(-inf) x 2 + 1 -> -inf, in all three. Denormals and overflow: a squared 2^-70 in relaxed fp32 gives 7.17e-43 = 2^-140, an fp32 DENORMAL result kept exactly (no flush to zero); 3e19 squared overflows to +inf; half 60000 x 60000 gives 3.6e9 exactly in the fp32 destination. (The first run of
this check built the 0 x inf case wrongly, as 0 x 1 + inf x 1, and its stride-128 dump included unwritten sentinel cells; both were fixed before the results above.) fp16 subnormal INPUTS were already exercised by the section 122 censuses (2^-24). Not tested: the same on op5100/op5104 and on the 16-bit destination forms, and signed-zero results. *(Section 136 parts 2, 3 and 7: op5100 and op5104 and signed zeros are covered by the exact arithmetic rule over 53 cells, and the 16-bit accumulator forms are conversions around op5106.)*

## 132. Register-resident chaining, closed for the measured space: exact tile and row transforms, the hardware eligibility table, and the library's narrower-consumer rule

**Result.** Hand-authored multi-tile GEMM -> GEMM chains (AIR `simdgroup_matrix_16x16x16_multiply_accumulate`, D1 fed to GEMM2's A or B operand straight from the accumulator registers, one element-wise conversion and nothing else between the MMAs)
are bit-identical to a fragment-table model in every one of 133 runs (kernel, seed, scale) and 1,596 random chained pairs (108 runs in the two main sweeps, 25 more with large grids, seeds, accumulate and range cases), covering four feed modes
(D1 or D1^T into A or B), five operand forms (half, bfloat, fp32, float x half, half x float), tile grids from 1x1 to 4x4, narrower and wider consumers, accumulation, and hardware transpose bits. At the hardware level there is NO narrower-consumer
hazard. The hazard I reported in section 127 is a property of the library's cooperative input tensor and now has an exact rule (part 5). An attention-shaped program (S = Q K^T with K through the transB bit, O = half(S) V, N2 = 32 < N1 = 48) equals a purely
logical tile model bit for bit (12/12). The memory bridge stays the fallback for everything not listed as eligible in part 4.

**1. Method.** `chain_mt.py` writes a straight-line AIR kernel for GEMM1 (M1 x N1 x K1 in 16x16x16 tiles, K-chain per output tile, C first), converts each D1 tile with one `fptrunc` when the fed operand is 16-bit, and runs GEMM2's K-chain on those registers; D1 is also stored
from the same registers as a control. The host model works on fragments with the measured tables only (A and D at `pos`, B at `pos_b`, a transposed A at `pos_at`, `mma_np` for each issue, fp32 operands truncated to 10 mantissa bits), with no fitted parameter, and every comparison is bitwise (any two NaNs
count as equal). The logical statements below (row and k relabelings) are checked separately against the mathematical product of the original operands to rounding. In the decoded code the forms compile to op5106/5107 (half, bfloat), op5098/5099 (fp32), op5100/5101 (float x half) and op5104/5105
(half x float); between the first and last MMA of a 32x32 half or bfloat chain there are exactly 32 instructions, all conversions (op1016 or op1048, one per element of the four D1 tiles), and none for the fp32 forms.

**2. The fragment facts.** `pos_b(k, c) = pos(rotl1(k), c)` (section 129), with `rotl1` and `rotr1` the one-bit left and right rotations of the 4-bit row index. A fragment of a 16x16 tile: A(r, k) at `pos(r, k)`, D(r, c) at `pos(r, c)` (so A and D are slot for slot the same),
B(k, c) at `pos_b(k, c)`, and under the transA bit the fragment holds A^T at `pos_at`. The transB bit does not change the fragment; the hardware uses the transpose of the matrix it represents. Feeding a D fragment as an operand therefore reads:

    fed as A          A_eff[r][k] = D[r][k]
    fed as B          B_eff[k][c] = D[rotl1(k)][c]
    fed as A, transA  A_eff[r][k] = D[rotl1(k)][rotr1(r)]
    fed as B, transB  B_eff[k][c] = D[rotl1(c)][k]

*Scope correction (Set A, 2026-09-23, `results/g17-tensor-feedmodes-v1`, machine-model section 25.103): these relabelings hold for kernels whose loads put A and D at `pos` and B at `pos_b`, which is how this section measured them (Apple-compiled `simdgroup_matrix` chains). They are not properties of the MMA. tlower's feeds of the same op5106 read the LOGICAL D (mode B) or D^T (At, Bt) bit for bit in fp32 and half, and they fail this table by 290 to 477. Row labeling belongs to the kernel that produced the accumulator (section 129).*

**3. Exact transforms (H = the converted D1 tile; tile indices are 16-blocks; the same in every form).**

    mode  GEMM2 tile op                                  operand arrangement the host supplies                 output relabeling           C2 initial value
    A     D2[i][n] += mma(H[i][k], B2[k][n]), k < NT1    B2 as is; A2 = D1                                     none                        as is
    B     D2[r][c] += mma(A2'[r][j], H[j][c]), j < MT1   A2'[:, 16j+m] = A2[:, 16j+rotl1(m)]  (D2 = A2 D1)     none                        as is
    At    D2[a][n] += mma(H[k][a] transA, B2'[k][n]), k<MT1   B2'[16b+j, :] = B2[16b+rotl1(j), :]  (D2 = D1^T B2)  hw row 16b+r = true row 16b+rotr1(r)   hw row r <- true row rotr1(r)
    Bt    D2[r][i] += mma(A2[r][j], H[i][j] transB), j < NT1  A2 as is  (D2 = A2 D1^T)                         hw col 16b+c = true col 16b+rotl1(c)   hw col c <- true col rotl1(c)

D2 has (MT1 x NT2), (NT1 x NT2), (MT2 x NT1) and (MT2 x MT1) tiles for A, At, B, Bt. The conversion is an element-wise `fptrunc` to half or bfloat (round to nearest even; overflow to infinity and half subnormals reproduced exactly, part 4); the fp32 forms need none. A logical A2 . D1 through the B feed needs A2's k index permuted, which changes the association order inside the MMA's k tree, so it agrees with an unpermuted product to rounding, not bit for bit.

**4. Hardware eligibility (all measured; each line is bit-identical on every random case).**

    what                                                        measured
    A, B, At, Bt x half (h)                                    1x1x1x1, 2x2x2x1, 2x2x2x2, 2x2x2x3, 1x2x2x2, 2x1x2x2, 3x2x1x2, 2x3x2x1, 1x3x1x2, 3x1x1x3, 3x3x2x3, 4x2x1x2 (MT1 NT1 KT1 X), plus 4x4x2x4, 3x4x2x3, 4x3x2x3, 3x2x2x3
    A, B, At, Bt x bfloat, fp32, float x half, half x float     1x1x1x1, 2x2x2x2, 2x2x2x1, and 3x2x2x3 (b, fh, hf, f in at least one mode each)
    consumer with fewer tiles than the producer (X = 1)          2x2x2x1 and 2x3x2x1 in all four modes for half; 2x2x2x1 in all four modes for bfloat, fp32, float x half and half x float
    accumulate into a non-zero C2 (h, b, f, fh; A, B, At, Bt)    exact; for At and Bt the C2 must be relabeled like the output (row 16b+r <- rotr1, col 16b+c <- rotl1)
    GEMM1 with transA, transB or both (A, B, At, Bt, h)          exact (the D1 the chain sees is whatever GEMM1 produced)
    a transpose bit on the NON-fed operand of GEMM2              exact at the fragment level (its logical meaning is then the transposed operand, no relabeling claim made)
    range: half overflow (scale 2e4: modes A, B, At), half subnormals (scale 1e-5: modes A, B, Bt)     exact

    Not exact, and why it does not change the table: at bfloat overflow (scale 1e38) GEMM1 itself differs from the model in 1 of 256 elements (a finite hardware value where the model has an infinity) and at fp32 denormal magnitudes (3e-39) GEMM1 differs in 8 of 256 elements by one
    or two denormal steps; in both the model fed with the HARDWARE D1 predicts GEMM2 exactly for bfloat overflow and not for the fp32 denormals (6 of 256), so in the fp32 denormal range the MMA arithmetic itself deviates from `mma_np`.
    The claim is bit-exactness where operands, products and sums are normal numbers. *Correction (section 136): both deviations are errors of `mma_np`, which rounds each product to fp32. The hardware forms the products exactly and rounds each sum once; the rule of section 136 part 2 reproduces every one of these cases, at single-MMA level (53 cells) and at chain level (152 runs at scales 1e38, 3e-39 and neighbouring extremes, `chain_range_matrix.log`). The claim holds for all operand values.*

**Decision rule for a compiler (hand-authored path).** Forward D1 register-direct into GEMM2 if and only if all of these hold: (a) 16x16x16 tiles in one of the five forms above, D1 the fp32 accumulator; (b) the fed operand's element type is that slot's MMA operand type (16-bit: one element-wise `fptrunc` per element,
fp32: none); (c) every source tile of a consumer tile sits in the registers of the same simdgroup; (d) the feed is one of the four in part 3, with its operand arrangement, output relabeling and C2 relabeling applied; (e) operands, products and sums are normal numbers (no fp32 denormals, no bfloat overflow); (f) D1 has at most 4x4 tiles. Otherwise use the memory bridge. *Correction (section 136): condition (e) is removed and replaced by (e') there; the int8 and uint8 requantization chains excluded in part 6 are eligible (section 136 part 5).*

*Addendum (section 133): the rule above is unchanged. Section 133 adds (c'), (f'), (g) and (h). The 4x4 bound in (f) was the extent of what was measured, not a hardware limit; twelve of the 141 flat-order kernels built for this section spilled to per-thread stack memory (exact, but not register-resident), including the 4x4 configurations, and all twelve are spill-free in a streaming schedule.*

**5. The library's narrower consumer** (Apple's cooperative input tensors, two `matmul2d` objects of different descriptors, `xc_map.py` and `xr_map.py`; B1 = identity so D1 = tags; a selector operand reads back what GEMM2's operand actually holds).
A-side chain (GEMM1 M x N1 x N1, GEMM2 M x N2 x N1 with GEMM1's destination as the left input): with N2 >= N1 the operand equals D1 element for element; with N2 < N1 only the FIRST (M/16)*(N2/16) tiles of D1 in row-major tile order (block tr*NT1 + tc) hold D1 data
(identity count exactly M * N2), and the rest read as zero or as another D1 element. 27 of 27 configurations (M, N1, N2 in 16, 32, 64) match, including all nine with N2 < N1. B-side chain (GEMM2 M2 x N1 x M1 with the destination as the right input): with M2 >= M1 exact; with M2 < M1
only the first (M2/16)*(N1/16) blocks are valid and the rest read as zero; 27 of 27 (M1, N1, M2 in 16, 32, 64) match. One rule covers both: the consumer sees the first min(#D1 tiles, #D2 tiles) blocks of D1's row-major block order. `is_compatible_as_*_input` returns TRUE in all of them,
so it is not a guard. A library chain is safe iff the consumer's destination has at least as many tiles as the producer's; a narrower consumer can be padded up (N2 -> N1 with zero columns, or M2 -> M1 with zero rows), which is the same-size case. The library refuses transposed cooperative inputs and any input
cooperative tensor with more than one simdgroup (`static_assert`s, sections 128).

**6. Explicitly ineligible (unmeasured or refused): use the memory bridge.** int8/uint8 forms and any chain that changes the element type through requantization (op10384/10385 chains); a consumer whose source tiles are not all produced by the same simdgroup (registers are per simdgroup; cross-simdgroup data needs the memory
path and the barrier rule of section 119); any tile shape other than 16x16x16 (the legacy 8x8 datapath, op2862); fp32 denormal-range operands; bfloat overflow-range operands; chains through a `threadgroup` or device store in between; 16-bit destinations of `matmul2d` (cooperative destination rounded per call, section 126) mixed with hand-authored feeds;
D1 tiles consumed by a different SIMD lane arrangement than one simdgroup's fragments (gather/shuffle chains); mixed feeds not in the table (for example float x half with a bfloat fed operand); larger grids than 4x4 D1 tiles (register pressure was not explored beyond it here; see section 133, part 5). *Correction (section 136): int8 and uint8 requantization chains, fp32 denormal-range operands and bfloat overflow-range operands are eligible; measured in section 136 parts 2, 4 and 5.*

**7. Corrections recorded in place**: section 128 (the reasoning that a register-direct transposed chain would need cross-lane exchanges is wrong: the hardware transposes the fed operand itself), section 127 (the narrower-consumer failure is explained by part 5), section 129 (the multi-tile GEMM is now built), and section 104 (points here for the current table).

## 133. Register-resident composition beyond one simdgroup and beyond 4x4: across simdgroups the handoff is memory; within one, the grid limit is a measured table, and two rules I proposed for it failed

**Result.** (1) Across simdgroups, D1 cannot stay in registers. For each operand role the cheapest legal handoff is one store and one load in fragment order, with a workgroup barrier between them; a barrier is necessary (with no barrier, and with a simdgroup-only barrier, 0 of 12 trials are correct in every role), and any workgroup barrier
is sufficient on this chip in the tests (part 2). No cross-lane instruction I could reach moves an A, B or C operand between simdgroups: `simd_shuffle` with every index 0..255 in a two-simdgroup threadgroup returned a value from the caller's own simdgroup in 256 of 256 cases. The one cooperative-tensor route that avoids memory is the library's
accumulator reuse, which stays inside one cooperative scope (part 3). (2) Chains that never cross a simdgroup are unaffected by how many simdgroups run: independent register-direct chains in 1 to 32 simdgroups of one threadgroup are bit-identical to the model in 50 of 50 runs, so section 132's locality condition (c) is measured
beyond one simdgroup. (3) The 4x4 bound of section 132 is lifted for a streaming schedule: 79 hardware runs of large grids (up to a 12x8 D1 grid with a 12x12 D2 grid, 1,344 MMAs) are bit-identical to the model, 77 of them with no stack stores, and all twelve flat-schedule configurations that spilled in section 132's set and its extras
are exact and spill-free when streamed (part 5). (4) The spill boundary is not a closed-form function of the shape. I registered two predictions before running them, and both failed (6 of 12, then 82 of 144); the eligible set is therefore the measured tables, and nothing beyond them.
Section 132's rule (a) to (e) is unchanged; (c) and (f) are widened in part 6, and section 132's text carries a pointer to this section.

**1. Question and scope.** Section 132 closed register-resident GEMM to GEMM chaining for one simdgroup and D1 grids up to 4x4 tiles. Two things were open: whether a consumer in another simdgroup can take D1 from registers, and what limits the grid. Everything here is hand-authored AIR
(`simdgroup_matrix_16x16x16_multiply_accumulate`, `air.wg.barrier`, threadgroup and device pointers), the same fragment model as section 132, one chip (H17s), one toolchain. Nothing under `agxforge/` or `tools/` was touched and no compiler change is proposed.

**2. Cross-simdgroup handoff, A, B and C separately** (`xsg_chain.py`, `xsg_matrix.sh`, `xsg_matrix.log`). Two simdgroups (64 threads). Before the producer runs, SG1 fills the handoff buffer with NaN and a full barrier follows, so a consumer that runs early or reads stale data returns the sentinel and the comparison fails. SG0 computes D1 and
stores it in fragment order (one vector per lane per tile); a barrier; SG1 loads the same vectors and uses them. The comparison is bitwise against the fragment model of section 132.

    role  what SG0 hands over                          bytes per tile        how SG1 uses it
    A     D1 converted to half (one fptrunc, as in 132)   16 per lane, 512     A operand of GEMM2   (D2 = conv(D1) B2)
    B     D1 converted to half                              16 per lane, 512     B operand of GEMM2   (D2 = A2 conv(D1), the k relabeling of section 132 part 3)
    C     the fp32 accumulator, no conversion               32 per lane, 1,024   C operand of GEMM2   (split-K across simdgroups, D2 = P + A2 B2)

Synchronization, single tile, 12 random trials per cell, both hand-off memories (threadgroup and device):

    what sits between SG0's stores and SG1's loads        roles A, B, C, hop threadgroup     roles A, B, C, hop device
    nothing                                               0/12                                0/12
    air.simdgroup.barrier only                            0/12                                0/12
    air.wg.barrier(0,1)   (execution only, no memory scope)  12/12                            12/12
    air.wg.barrier(2,1)   (mem_threadgroup)               12/12                               12/12
    air.wg.barrier(1,1)   (mem_device)                    12/12                               12/12
    air.wg.barrier(3,1)   (both)                          12/12                               12/12

Tile grids 2x2x2x2, 3x2x2x3 and 4x4x4x4 (MT1 NT1 KT1 X; threadgroup memory, full barrier) are 12/12 for each of the three roles. Section 118 showed that a barrier is required and used `mem_device`; it did not vary the flags. What this adds: the barrier's memory flags did not matter in any tested cell, and the simdgroup-scope barrier is not enough.
That an execution-only barrier suffices is an observation on this chip, not a guarantee the memory model gives; the flag that matches the hop memory (threadgroup for a threadgroup buffer, device for a device buffer) is the specified form and costs nothing extra in the measurement below.

Cost (`xsg_loop.py`, `xsg_loop_cost.sh`, three runs each, 128 threadgroups): the handoff is placed inside a loop so launch overhead cancels; every iteration SG0 makes a fresh D1 tile, converts it and stores it, a barrier, SG1 loads it and feeds one MMA, a second barrier lets SG0 overwrite. Per iteration, the minimum
over three runs of each of the eight variants (A and B and C, threadgroup and device) is 143 to 299 ns and the medians are 230 to 366 ns; the maxima reach 824 ns. The spread between repeats is larger than any difference between roles, hop memories or barrier flags, so I claim no ordering among them. Both a threadgroup and a device hop are
therefore legal and of the same order, about a quarter of a microsecond per handoff round including two barriers and one MMA on each side; a compiler that wants to keep the traffic on chip should use threadgroup memory, and I found no latency reason to prefer either.

Why registers cannot be handed over. Each simdgroup's MMA reads its operands from the issuing simdgroup's registers and writes its own destination fragment (sections 118 and 132). The cross-lane instructions are confined the same way: in a two-simdgroup threadgroup with thread t holding tag 1000+t, `simd_shuffle(tag, idx)` for all idx 0..255 returned
a tag of the caller's own simdgroup for every thread in 256 of 256 cases and one from the other simdgroup in 0 of 256 (`shx_xsg.metal`, `shx_xsg_run.py`; the source lane equals idx mod 32 for idx below 32 and for the multiples of 4 above it, 88 of 256, and I did not model the lane for the other indices). The library refuses a cooperative input tensor with more than one simdgroup (section 128). Searched: `simd_shuffle` for all 256 indices in a two-simdgroup threadgroup, the MMA operand and destination roles, and the library's cooperative tensors. Not searched: the quad and xor shuffle forms in a two-simdgroup threadgroup, and any cross-simdgroup register instruction that exists in the ISA but that AIR or MSL never emits; I have no evidence for or against either.
The claim is limited to what was searched: an arbitrary-index shuffle does not reach another simdgroup, MMA fragments belong to the issuing simdgroup, and the library refuses cooperative inputs over more than one simdgroup. The memory path is the cheapest legal handoff I found, not a proof that none exists.

**3. Can cooperative-tensor state avoid device memory?** Two library objects, two simdgroups, bitwise against `gemm_ref` (`coop_xsg_lib_run.py`, `coop_xsg_lib.log`).
`coop_c_reuse.metal`: the cooperative destination of a 32x32x32 `multiply_accumulate` GEMM is the accumulator of a second 32x32x16 GEMM with the same M, N and execution scope. 10 of 10 trials are exact, and in the decoded object the six MMA issues (four op5106, two op5107) are contiguous with no other instruction between the first and the last:
register-resident, nothing in memory. It needs the same M, N and scope in both calls (any K), and both simdgroups take part in both calls, so no state crosses a simdgroup boundary; it is split-K inside one cooperative scope, not a cross-simdgroup handoff.
`coop_tg_bridge.metal`: the half destination of GEMM1 is stored to a threadgroup tensor, a `threadgroup_barrier(mem_threadgroup)`, GEMM2 reads that tensor as its left input. 10 of 10 exact (D = half(A1 B1) B2, one rounding to half); in the decoded object the eight MMA issues are separated by 16 conversions and 19 other instructions, unlike the contiguous accumulator-reuse case. This avoids device memory but not memory.
No cooperative-tensor route exists for A or B across simdgroups (part 2, section 128). So for the C role the answer is yes, inside one cooperative scope and without device memory; for A and B it is no, and the threadgroup tensor bridge is the least memory traffic the library allows.

**4. Chains that stay inside their simdgroup, at any simdgroup count** (`xsg_local.py`, `xsg_local_matrix.sh`, `xsg_local_matrix.log`). Each simdgroup runs its own section-132 chain on its own operand and output region; all run concurrently in one threadgroup. Modes A, B, At, Bt (half) at 1, 2, 4, 8, 16 and 32 simdgroups, shapes 1x1x1x1 and 2x2x2x2, plus fp32 and
bfloat mode A at 32 simdgroups: 50 of 50 runs, every simdgroup's D2 and D1 tiles bit-identical to the model on 8 of 8 trials. Condition (c) of the section 132 rule ("every source tile of a consumer tile sits in the registers of the same simdgroup") is therefore measured at up to 32 simdgroups (1,024 threads); it was measured at one before.

**5. Larger grids.**

*5.1 Section 132's flat schedule spilled in some configurations.* The census of all 141 flat chain kernels built for section 132 and its extras (`flat_spill_census.json`, decoded op595 = per-thread stack store) finds 12 with stack stores: `A f 3x3x2x3` (95), `A h 4x4x2x4` (96), `A h 5x5x2x5` (300), `A h 6x6x2x6` (424), `At f 3x2x2x3` (10), `At h 3x4x2x3` (4),
`At h 4x2x1x2` (12), `B h 4x4x2x4` (104), `B hf 3x2x2x3` (5), `Bt fh 3x2x2x3` (45), `Bt h 3x2x2x3` (8), `Bt h 4x3x2x3` (20). Those chains are bit-exact, so no numerical claim of section 132 changes, but the fragment passed through the compiler's per-thread scratch memory, so "register-resident" was not true for them. This includes the 4x4 case that section 132's rule (f) names as its bound.
The same twelve configurations, rebuilt in a streaming schedule, are exact on 12 of 12 random cases and have 0 stack stores in every one (`flat_spill_resolved_by_stream.log`).

*5.2 The streaming schedule.* Unit = one D1 row tile (modes A and Bt) or column tile (B and At). Per unit: produce each D1 tile (K-chain, control store, conversion), then run GEMM2's K-chain for the unit's D2 tiles from the converted tiles, then move to the next unit. All operand loads are volatile and placed at their use. The buffer layout is the flat one, so the same host model applies.
`inner` = converted tiles per unit (NT1 for A and Bt, MT1 for B and At); `outer` = D2 tiles per unit (X).

*5.3 Hardware evidence for larger grids* (`chain_big.sh`, `chain_big_stream.log`; `edge_hw.py`, `edge_hw_exact.json`). 22 configurations chosen for coverage and 57 configurations at the table edges of part 5.4: 79 runs, 12 random chained pairs each, every one bit-identical to the model; 77 of the 79 have no stack stores. Coverage: modes A, B, At, Bt; forms half, bfloat, fp32, float x half,
half x float; accumulate into a non-zero C2 in four modes; hardware transpose bits (`t1010`, `t0101`) with accumulate. Largest: `A h` 12x8 D1 tiles (K = 2 tiles) with 12 D2 column tiles (144 D2 tiles, 1,344 MMAs, 4,807 instructions, 0 stack stores); `A h` 2x1 D1 tiles with 96 D2 column tiles; `Bt h` 6x13 D1 tiles (390 MMAs); `A h` 6x4 D1 with 35 D2 columns; `B b` 12x4.
The two runs with stack stores are exact and are the evidence for part 5.5: `A f 3x3x2x14` (4 stores) and `Bt h 6x9x2x11` (7 stores), both at a unit count other than the two that the edge tables use.

*5.4 The measured spill-free edge* (`chain_spill_edge.py`, `spill_edge_*_kt2.json`; compile only, decoded op595; KT1 = 2, two unrolled units, streaming). Largest `outer` with zero stack stores, and a full scan of every smaller `outer` (none spills below its edge in any row):

    inner        1    2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
    A h        96+   80  50  35  26  20  15  12  10   8   6   4   3   0   0   0
    B h        96+   80  50  35  26  20  15  12  10   8   6   5   3   0   0   0
    At h       96+   80  51  36  27  21  17  14  12  10   8   7   5   0   0   0
    Bt h       96+   80  51  36  27  21  17  14  11   9   8   6   5   0   0   0
    A f         47   25  14   7   2   0   0   0        (inner 6, 7, 8: 16, 48, 64 stack stores at outer 1)

96+ means no stack store up to 96, the largest tried. In the half rows, inner 14, 15 and 16 have 32, 48 and 64 stack stores even at outer 1. Two rows of the fp32 table are not contiguous: for inner 1 the outer counts 48 and 49 spill and 50 does not, and for inner 5 the outer count 3 spills and 4 does not, so only the prefix up to the edge is claimed. All 57 edge configurations
run bit-exact and spill-free on hardware (`edge_hw_exact.log`, 57 of 57). This is a table of what one backend produced, not a limit of the hardware (part 5.7).

*5.5 Two rules I registered before measuring, and both failed.*
(i) After reading the KT1 = 2, two-unit half grid for mode A, I fitted "no stack stores if and only if 2 NT1 + NT2 <= 28, spill at 30 or more" to seven boundary points and registered twelve predictions (`spill_prereg_A_h.json`, sha256 77d8a617...) before building any of them: six kernels predicted free and six predicted to spill, each pair straddling the rule at
NT1 = 6, 5, 4, 3, 2 and 1. Result (`spill_prereg_A_h_result.log`): all twelve have 0 stack stores. The six free predictions were right and the six spill predictions were wrong, so the rule is false; at NT1 = 1 the true edge is beyond 96, not 26. The two-parameter curve (outer + 10) x inner <= 180 reproduces 11 of the 12 finite mode A edges of the table below (inner 2 to 13; it misses inner 12 by one) after the fact and fails at inner 14 to 16 (it gives 2, 2 and 1; measured 0); it was fitted on data I had already seen, was not
registered, and is not claimed.
(ii) I then registered that the edge measured at KT1 = 2 with two units carries over to KT1 = 1, 3, 4 and to three and six units, at inner 4, 9 and 13, for all four modes, predicting the edge outer count free and the next one spilling (144 predictions, `spill_indep_prereg.json`, sha256 ed45e6fe..., `spill_indep.py`). Result (`spill_indep_result.json`): 82 of 144 correct, 22 of 72 for "free at the edge"
and 60 of 72 for "spills one beyond". By KT1: 33 of 48, 25 of 48 and 24 of 48 for KT1 = 1, 3, 4. For modes A and B at KT1 = 1 there are no stack stores at the edge or one beyond it in any of the 24 kernels (the edge is farther out); at KT1 = 3 and 4 most kernels already spill at the KT1 = 2 edge (for example `A` inner 4, six units: 57 stores at KT1 = 3 and 82 at KT1 = 4);
for At and Bt with six units, three of the six cells at KT1 = 1 already spill at the edge. The measured edge depends on KT1 and on the unit count, in a way I cannot state as a rule from these data.
A generator edit made during this work (an added no-op index term) was checked against these runs: all 144 stack-store counts are identical before and after it.

*5.6 Two ways to raise the limit that did not work.* (a) A k-outer order (per unit, one D1 tile at a time and all D2 accumulators of the unit updated with it) is exact but spills much more (`A h 2x6x2x6`, six live accumulators: 160 stack stores against 0 for the streaming order). (b) The same with each operand load made address-dependent on the previous fragment through an opaque
runtime zero (`kdep`): 0 stores at `A h 2x6x2x6`, but 70, 123, 305 and 170 stores at inner 24 (four accumulators), inner 32 (four and eight) and inner 64 (two). The first stack store in the inner-24 kernel, at instruction 40, stores a register derived from the lane index, before any fragment is live in quantity, so at least the onset of spilling there is not fragment pressure. A streaming variant with the same dependency on the converted
tile (`sdep`) spilled 48 stores at a configuration (`A h 2x8x2x12`) that has 0 in the plain streaming order. No claim of an unbounded inner dimension is made; these orders are kept in `chain_mt.py` (`kout`, `kdep`, `sdep`) as negative results.

*Correction (section 134 part 5): the inference above from the first stack store (a register derived from the lane index, "so at least the onset of spilling there is not fragment pressure") is withdrawn. The register file is one pool; the backend spills whichever value it chooses when the pool overflows, whatever class fills it. Spill detection in section 133 counted `op595` only; scalar spills use `op17789`. None of the 2,922 chain kernels contains one (`mtc_spill_recount.json`), so the tables stand, but rule (g) below should read "any stack store (op595 or op17789)".*

*5.7 What the register file does establish* (`gen_acc_live.py`, `acc_live.log`). One lane holds 15 simultaneously live 16x16 fp32 accumulator fragments (120 registers) with no stack store; 16 (128) has 24 and 17 has 28, and the count grows to 92 at 24 accumulators. In the decoded MMAs a half or bfloat operand is a group of four registers and an fp32 operand or accumulator a group of eight (op5106 with `R12..R15`, `R8..R11`
and `R0..R7`). The streaming cliffs are not the register-file arithmetic: 14 converted half tiles are 56 registers and 6 fp32 tiles are 48, far below 128, and a peak-liveness estimate from the decoded def/use sets (`regpressure.py`) is 86 to 95 registers on both sides of the cliff. So the backend allocates more than the minimum live set,
and I did not decode why. The register file is the only limit found; I did not establish that any measured cliff is the hardware minimum. An arithmetic upper bound for a native allocator, (128 - working set) / 4 converted half tiles per unit, would allow more than the 13 measured here, but no measurement supports it.

*Correction (section 134, part 1): the peak-liveness figure of 86 to 95 registers above came from a tool that counted only the first register of each register group. The corrected peaks of these kernels are 118 to 125 registers, so the streaming kernels do fill the 126-register pool, and the sentence "the streaming cliffs are not the register-file arithmetic" is withdrawn: the cliffs are the point where the backend's schedule (all D1 MMAs before the conversions, all operand loads before the K-chain) overflows that pool (section 134 part 5). The "128" here is 126 on this hardware (R126 and R127 hold nothing, section 134 part 2), and 15 accumulator groups is a hardware limit, not only a measured one.*

**6. Decision rule, extended (section 132 (a) to (e) unchanged).**

    (c') every source tile of a consumer tile sits in the registers of the simdgroup that issues the consumer; any number of simdgroups per threadgroup, each with its own chain (measured to 32).
    (f') replaces "D1 has at most 4x4 tiles": forward register-direct in the streaming schedule of part 5.2 if and only if the exact configuration (mode, form, MT1, NT1, KT1, X, unit count) is one of the measured configurations of part 5.3 or 5.4 (bit-exact on hardware, zero op595); the tables of 5.4 apply only at KT1 = 2 with two units.
    (g)  a decoded chain kernel with any stack store (op595) is not register-resident; treat it as the memory bridge even though it is exact. Flat-order kernels in section 132's set are covered by part 5.1.
    (h)  a consumer in another simdgroup takes D1 from memory: fragment-order store, workgroup barrier, fragment-order load (part 2); the C operand alone can stay in registers, and only inside one cooperative scope of the library (part 3).

**7. Explicitly ineligible or unmeasured.** Any register handoff across simdgroups; configurations not in the measured set, including KT1 other than 2 or unit counts other than 2 beyond the runs of 5.3, half chains with 14 or more converted tiles per unit and fp32 chains with 6 or more (every one measured spilled), and everything beyond 96 D2 tiles per unit;
bfloat, float x half and half x float edge tables (measured only at the points of 5.3); tile shapes other than 16x16x16; the k-outer and dependency orders of 5.6; anything a native compiler allocates on its own, since every register claim here is about what Apple's backend produced from my AIR. The int8 and requantization exclusions of section 132 part 6 stand.

**8. Corrections recorded in place.** Section 132: the decision rule's (f) and the last clause of part 6 now point here (the 4x4 bound was the extent of what I had measured, and twelve of the 141 flat kernels built for it spilled, part 5.1). Section 119: the open case "MMA result as a cross-SG A/B operand" is closed by part 2 (memory handoff, barrier required, bit-exact).
Section 128 is unchanged and consistent (the library's refusal of more than one simdgroup for cooperative inputs). My process notes: I fitted the first spill rule on the same grid it then had to predict from and registered predictions only for points that straddled it in one direction of one variable; the second registration varied two factors at once and found both matter.

Artifacts under `results/g17-tensorops-recon-v1/`: `chain_mt.py` (orders `stream`, `kout`, `kdep`, `sdep`; buffer sizing from the tile counts), `chain_big.sh`, `chain_big_stream.log`, `chain_spill_edge.py`, `spill_edge_*_kt2.{json,log}`, `edge_hw.py`, `edge_hw_exact.{json,log}`, `spill_prereg_A_h.json`, `spill_prereg_A_h_result.log`, `spill_indep.py`, `spill_indep_prereg.json`, `spill_indep_result.json`,
`spill_indep_result_first_run.json`, `flat_spill_census.json`, `flat_spill_resolved_by_stream.log`, `gen_acc_live.py`, `acc_live.log`, `regpressure.py`, `xsg_chain.py`, `xsg_matrix.sh`, `xsg_matrix.log`, `xsg_loop.py`, `xsg_loop_run.py`, `xsg_loop_cost.sh`, `xsg_loop_cost*.log`, `xsg_local.py`, `xsg_local_matrix.sh`, `xsg_local_matrix.log`, `shx_xsg.metal`, `shx_xsg_run.py`,
`shx_xsg_run.log`, `shx_xsg_result.json`, `coop_c_reuse.metal`, `coop_tg_bridge.metal`, `coop_xsg_lib_run.py`, `coop_xsg_lib.log`.

## 134. Register pressure and occupancy of TensorOps chains: 15 fp32 accumulators is the hardware's 126 usable registers in groups of eight; the section-133 cliff is the Apple backend's schedule, and my registered models of it failed

**Result.** (1) R126 and R127 do not hold values on this chip. A live 32-bit scalar renamed by patching Apple's own native code (five instructions of four forms) survives in R60, R100, R120, R123, R124 and R125 and comes back all zero in R126 and R127; an fp32 accumulator group (two operand loads, the MMA, two stores) relocated to R88, R96, R104 and R112 is bit-exact,
and at R120..R127 the four registers R124..R127 read back as zero after a load and the MMA leaves R120..R123 unchanged. So, for the scalar, load, store and MMA forms tested, the general register file has 126 usable 32-bit registers, R0..R125, which is exactly the highest register that 25,958 Apple-compiled objects ever use, and an 8-register accumulator group needs 8-aligned registers, so at most
floor(126/8) = 15 accumulator groups exist (bases R0..R112). The boundary "15 live fp32 accumulators fit, 16 spill" is therefore architectural for accumulators in the general register file, not an allocator preference. (2) The 14 accumulators of the loop kernels, the 12 and 13 converted tiles of the section-133 streaming chains and the 6 fp32 tiles are not architectural:
they come from the backend's schedule (part 5), and the two quantitative models I registered for them mostly failed (9 of 12 predictions). (3) Occupancy does not produce any of these cliffs: no dispatch or resident-simdgroup step at 15 or 16 accumulators (the only step is the spill cost of part 7), 1,024-thread threadgroups run at 126 registers per thread, and resident capacity falls only by about 25 to 30% (about 45 to 32 simdgroups per core by atomic peak) between 22 and 126 registers (part 6). (4) Materializing a spill costs +31% (15 accumulators) to +133% (24 accumulators) per MMA at one to four simdgroups per core, and about nothing at 64 per core where the MMA unit is the limit (part 7). (5) Section 133 part 5.7 contained a measurement error of mine (its peak-liveness numbers), corrected in place.

*Correction (section 135): result (1) and part 2 of this section misread the R126 and R127 evidence. Those tests wrote into a zero-filled output buffer, so "reads zero" and "read back as zero" were instructions and stores that did not execute; an instruction that names R126 to R143 is squashed as a whole (section 135), it does not read zero and drop a write. The accumulator field cannot name "a group up to R248": its values 18 to 31 decode to nothing and 15 to 17 (R120..R143) are squashed. The census sentence about operands reaching R127 at most is wrong (ALU fields reach R143). Unchanged: 126 usable registers R0..R125 and 15 accumulator groups, which section 135 explains by the squash.*

**1. Instruments and three traps.** `regfam.py` writes AIR probes in which only the live state of one class changes (N fp32 accumulators, NA/NB half operand fragments, K 32-bit scalars, M half or bfloat fragments, an optional serial chain of dependent loads, and never-updated padding scalars), all loop-carried in a runtime loop so nothing folds or unrolls; `regrun.py` checks every live
value against a host model (0 wrong in every kernel of this section, spilled or not); `regprobe.mm` and `regprobe_multi.mm` load the driver's native archive with `FailOnBinaryArchiveMiss`, report `maxTotalThreadsPerThreadgroup` and time dispatches on the GPU clock; `regprobe_ana.py` decodes the code (registers touched, peak occupancy, stack stores).
*Trap 1, my tool.* The first `regpressure.py` matched only the first register of a group such as `R0_R1_R2_R3` and reported peak liveness of 86 to 95 registers for kernels whose real peak is 118 to 125; section 133 part 5.7 cited those numbers (corrected there). A known-answer control (an accumulator group is eight live registers) would have caught it, and the highest register printed beside it already contradicted it.
*Trap 2, the spill detector.* Fragment spills are `op595` stores, 32-bit scalar spills are `op17789` stores; a scalar-only kernel with 160 live values showed 0 `op595` and 74 `op17789`. A spill is now either. The 2,922 chain kernels of section 133 contain no `op17789` (`mtc_spill_recount.json`), so its tables stand.
*Trap 3, the GPU clock.* The same kernel runs 4.8 times slower after 3 warm-up dispatches than after 400 ms of load (2,318 us against 484 us, with intermediate states at 1,232 to 1,612 us). All timings here come from `regprobe_multi`, which warms until 20 consecutive dispatches agree within 1% and then interleaves the kernels being compared so that they share the clock state (round-to-round spread at most 1.1% except one 6.6%).

**2. The top of the register file (hardware)** (`reg_ceiling_hw.py`, `reg_ceiling_hw.log`, `regpatch2.py`, `regrename.py`). Same-length edits of the code in the driver's native archive, operand fields set through the decoder as oracle (`agxforge.g17.encode.fields_of` and a search for the field value that decodes to the wanted register name, never assumed), run on the GPU.
Scalar loop-carried value in `rf_n0_a1_b1_k1_m0`, R0 renamed inside the loop: R60, R100, R120, R123, R124, R125 correct; R126, R127 all lanes zero. Accumulator group of `rf_n8_a1_b1_k0_m0` (base R64) relocated: R88, R96, R104, R112 exact on all 8 accumulators; R120 wrong on the relocated one. With the MMA left out (load, then store back), R104..R111 and R112..R119 return all eight
registers, R120..R127 return R120..R123 and zeros for R124..R127 (the load group R124..R127 contains R126 and R127). The metadata's register count is not what gates this: with the count set to 82, 126, 128, 136, 144 or 200 the R120 kernel fails identically (the count in the metadata is what the driver reports, and it did not gate any access here). The corpus agrees: 22,037 objects of the repo's Apple-compiler cache (`corpus_maxreg.json`) and 3,921 of mine
(`results_maxreg.json`, hand-patched probes excluded) have a highest register index of 125 (76 and 1,279 objects at 125) and none above. What is not established: R126 and R127 read zero and drop writes here, but I do not know what they are (a zero register, hardware state, or reserved); and the decoder's register table names R0..R143 while the operands I could test reach R127 at most (the load destination field has 7 bits and no value of it prints `R128_R129_R130_R131`; the scalar destination of `op10834` cannot be set to R128; the MMA A and B fields reach R124). The MMA accumulator field alone is 5 bits times 8 (`TMAC_ACC`, `agxforge/g17/asm.py`) and could name a group up to R248, but nothing in this test can load, store or read such a group, so whether the hardware keeps accumulators above R127 is unresolved.

**3. One flat pool with class widths 8, 4 and 1** (`rf_classes.py`, `rf_classes_*.json`, `rf_combo_acc_scal.log`, `rf_combo_acc_half.log`). Loop kernels, one A and one B half fragment, largest live count with no stack store: fp32 accumulators 14, half or bfloat fragments 29 (4 registers each), 32-bit scalars 124 (registers touched 126), a chain over P A/B fragment pairs 14. Accumulators trade against the other classes at exactly their widths:
scalars K against accumulators N gives 8N + K = 115 for N = 2, 4, 6, 8, 10, 12 (K = 99, 83, 67, 51, 35, 19), and half fragments M give 8N + 4M = 108 for N = 2, 6, 10 (M = 23, 15, 7). The constants are the fixed overhead (the A and B operand fragments, the loop counter and bound: 11 registers in the first family, 18 in the second) subtracted from 126. There is no separate accumulator resource visible to the allocator and no
aliasing between classes: an accumulator is eight ordinary registers. Straight-line code (`acc_N`, `gen_acc_live.py`, `acc_live.log`): 15 accumulators touch 122 registers (120 plus 2) with no stack store, 16 need 128 plus addressing, touch 126 and have 24 `op595` stores, 17 have 28, 24 have 92. In a loop the compiler keeps one group less (14 fit, 15 have 32 stores) because the operand fragments and
counters come out of the same pool. Bfloat fragments cost the same 4 registers as half; only the half case has a host model here.

**4. Architectural or compiler-specific.** For accumulators in the general register file the number 15 is architectural: 16 groups of eight would need R0..R127, R126 and R127 are unusable (part 2), and one register at least is needed for the lane and addresses. 14 (loops), 12 to 13 (section 133's streaming chains), 6 (fp32 chains) and every shape-dependent edge of section 133 are compiler-specific: they are the point where the schedule the backend chose
overflows the same 126-register pool. The lower bound on what the hardware sustains, measured, is 126 live registers per thread with correct results (124 live scalars; 14 accumulators plus operands), 15 live accumulator groups, and 1,024-thread threadgroups of such kernels.

**5. The section-133 cliff, explained in part and two models rejected** (`chain_loop.py`, `chain_loop_sweep.py`, `chain_loop_sweep_*.json`, `regpressure.py`, `eager_prereg.json`, `eager_dep_prereg.json`). `chain_loop.py` is the streaming chain with the unit loop and the D2 tile loop as runtime loops (the unit's `inner` converted D1 tiles unrolled, since registers cannot be indexed): the same arithmetic, so the backend cannot hoist across iterations.
It is bit-exact on hardware (6 of 6 random cases in every kernel below) and still has the cliff: mode A half, 2 units, KT1 = 2, X = 4, spill-free up to 12 tiles, 13 has 16 stores. Its code issues all 2 x inner D1 MMAs of a unit, then the conversions, so 8 x inner registers hold fp32 accumulators (peak occupancy 120 at inner 12, 80 of them defined by MMAs). Forcing eager conversion (each D1 chain starts from a runtime zero
that depends on the previous converted tile, values unchanged) interleaves the conversions as intended and the D1 phase shrinks, yet the cliff stays at 12 (13 has 4 stores): the D2 loop then issues all `inner` operand loads before its K-chain (in the code the 12 MMAs are contiguous, instructions 563 to 574, and all 12 loads come earlier, as in the non-eager kernel), so 4 x inner converted registers plus 4 x inner loaded operand registers plus the accumulator and about 5 address registers account for the peak of 109 at inner 12. Making each operand load wait, by a runtime zero, for the previous MMA of the chain
as well moves it to 18 (19 has 2 stores), with 104 registers touched at inner 12 instead of 126 (peak 90 instead of 120) and the K-chain spread over 144 instructions instead of 12; the peak then grows by 5 registers per tile (90, 95, 100, 110 at inner 12, 13, 14, 16).

    loop form, largest spill-free inner (first spilling inner and its stack stores)
    A h                     12   (13: 16)         A f                6  (7: 30)
    A h + eager             12   (13: 4)          A f + dependent loads   7  (8: 3)
    A h + eager + dep       18   (19: 2)          B h + eager + dep      18+  (20: 11; 19 not run)
    B h + eager            13   (14: 5)          At h + eager       12  (13: 5)          Bt h + eager   12  (13: 6)

Registered predictions, both written to disk with a hash before any of these kernels was built. First (`eager_prereg.json`, sha256 8d2558f7...): eager conversion moves the half cliff to 22..26 and the fp32 cliff is 9..11; 1 of 7 predictions held (exactness), the six quantitative ones failed (12; 12 against at least 8 tiles gained; 6; 13; 12; 12). Second (`eager_dep_prereg.json`, sha256 ca593f54..., a pool model `c x inner + 25 <= 126` with c = 8 for
half chains and 16 for fp32): fp32 without intervention 6 held; eager plus dependent loads 20..26 for half failed (18 for A and 18+ for B), fp32 with dependent loads 11..13 failed (7); exactness held. So the direction and the mechanism (two batching phases, each 8 registers per tile) are supported by the intervention, the arithmetic is not: the per-tile cost after both interventions is about 5 registers, not 4, and the fp32 chain spills at 8 tiles with peak occupancy 104 of 126, which
points at 8-register alignment and fragmentation of the pool that I have not resolved. The loop-form edge does not depend on the unit count or on X by more than one tile (KT1 = 2, inner 18: spill-free for all four combinations of 2 or 6 units and X = 4 or 16; inner 19 has 2 stores in three of the four), but it depends on KT1: at inner 18 KT1 = 1 is spill-free (also at 19), KT1 = 2 is spill-free, KT1 = 3 has 2 or 3 stores and KT1 = 4 has 13 to 17 (`chain_loop_indep.log`).

**6. Occupancy and dispatch** (`rf_occ.py`, `rf_occ_pair.py`, `rf_capacity.py`; measured on the M5 Pro with 20 GPU cores). Dispatch never fails: `maxTotalThreadsPerThreadgroup` is 1,024 for every kernel from 20 to 126 registers, spilled or not, and 40 threadgroups of 1,024 threads with a barrier before and after the loop, 126 registers per thread (14 accumulators, and 18 accumulators with 144 stack stores), complete with every value correct and a peak of exactly
640 resident simdgroups, 32 per core. Resident simdgroups counted by atomic arrival (2,048 launched, peak over the GPU, `rf_occ.log`): about 890 to 930 (45 per core) at 22 registers, 760 to 815 at 46, about 640 at 82 to 110, 530 to 660 at 126; with threadgroup barriers (`rf_capacity.log`) up to 1,220 at 22 registers and 640 to 900 at 126. These peaks vary by 20 to 30% from run to run, so I give the trend only: a gradual fall of 25 to 30%, no step, and no difference between 14 accumulators and 15 to 24 spilling ones. Relative capacity from interleaved timing of a load-latency-bound loop
(1 MMA and 4 dependent loads per iteration, 4,096 single-simdgroup threadgroups, only the number of live padding scalars changes): 1.00 at 20 registers, 0.98 at 40, 0.98 at 60, 0.93 at 80, 0.95 at 100 and 0.95 at 120. Static partitioning of the register file by declared registers would give a factor of six, so the hardware is not doing that (Dynamic Caching is the natural reading; I did not test it). A first version of this experiment with sequential timing gave nonsense (a larger launch ran faster) until the clock trap of part 1 was controlled.

**7. Throughput and the cost of materialization** (`rf_tput.py`, `rf_spill_cost.py`, `rf_flat_vs_stream.py`). The MMA unit runs at 4.046 GMMA/s over the 20 cores, one 16x16x16 MMA per core per 4.94 ns (202.3 per microsecond), independent of the number of live accumulators from 1 to 24 and of stack stores, reached with 8 or more independent MMAs per simdgroup at 4 to 8 simdgroups per core (some kernels are erratic at intermediate concurrency, which I attribute to threadgroup placement and did not resolve: `rf_tput.log`). One simdgroup issues at most one MMA per 20.5 ns with 8 independent accumulators and one per 74 ns with a single dependent chain. Correctness never changes before throughput does:
all 24 accumulator counts are exact. Interleaved GPU time per MMA against the 14-accumulator kernel:

    live accumulators        14      15      16      18      24
    stack stores              0      32      48      72     185
    1 simdgroup per core   1.00    1.31    1.47    2.17    2.30
    4 simdgroups per core  1.00    1.32    1.48    2.14    2.33
    64 simdgroups per core 1.00    1.00    1.00    1.10    1.01     (round spread 6.6% at 18)

So a spill costs latency, not saturated throughput. The section-133 schedules at identical arithmetic (flat: Apple's order; stream: mine), time flat / time stream, at 1 to 4 and at 64 simdgroups per core: `A h 4x4x2x4` (96 stores against 0) 1.07 to 1.12 and 1.15; `B h 4x4x2x4` (104) 1.02 to 1.03 and 1.30; `A h 6x6x2x6` (424) 1.23 to 1.26 and 3.92; and with few stores the streaming order is slower: `At h 3x4x2x3` (4 stores) 0.74 to 0.79 and 0.93,
`A h 4x3x2x3` (4) 0.73 to 0.75 and 0.91, `Bt h 4x3x2x3` (20) 0.74 to 0.79 and 0.94, `A f 3x3x2x3` (95) 0.47 to 0.54 and 0.75. Spill-free is not the same as faster: the volatile at-use loads of my streaming order cost more than a few stack stores.

**8. Safe live-fragment domain for fp32 accumulator chains** (measured, Apple backend, one A and one B half fragment live): straight-line code, N <= 15 accumulators (122 registers); loop code, N <= 14; with other live state, 8N + K <= 115 or 8N + 4M <= 108 in the loop kernels. All of it is bit-exact on hardware. For chains D1 -> D2 (section 133) the spill-free domain per unit is the table of part 5 for the loop form and section 133's tables for the unrolled form; neither is a hardware bound.

**9. Labels.**

    hardware, measured   R126 and R127 hold nothing (reads 0, writes dropped); usable general registers R0..R125; at most 15 eight-register accumulator groups; 1,024-thread threadgroups at 126 registers per thread; no occupancy or dispatch step at 15 or 16 accumulators; capacity falls about 25 to 30% (atomic peaks) or 5% (throughput) from 22 to 126 registers; MMA unit 4.94 ns per MMA per core;
                         a spill costs latency only where fewer than about four simdgroups per core are resident; the GPU clock needs over 100 ms of load
    hardware, decoded    MMA accumulator field 5 bits x 8, A and B fields 5 bits x 4 (`TMAC_*`), the load destination field 7 bits; the decoder names R0..R143
    Apple backend        the allocator's 126-register ceiling (equals the hardware fact); the flat pool and its class widths; 14 loop accumulators; the streaming cliffs and their dependence on KT1; the batching of D1 MMAs before conversions and of D2 operand loads before the K-chain; which value is spilled; the spill opcodes (op595, op17789)
    unresolved           what R126 and R127 are; whether MMA accumulators can live above R127; the register file size and allocation granularity (Dynamic Caching untested); why the fp32 chain spills at 8 tiles with peak occupancy 104; the exact per-tile cost of about 5 registers; the source of the 20 to 30% spread of atomic peaks; the TG placement effects at 8 to 32 simdgroups per core seen in `rf_tput.log`

*Correction (section 135): the first hardware row above should read "R126 and R127 (and R128..R143): every instruction that names them is squashed, not executed; reads and writes are not zero and dropped"; the unresolved row keeps only the question of what they are.*

**10. Corrections recorded in place and process.** Section 133 part 5.7 (liveness numbers and the conclusion drawn from them) and part 5.6 (the inference from the first spilled register) now point here. The regex trap, the second spill opcode and the clock were each found because a number contradicted another number printed next to it or an identical kernel gave two answers; the second registration was written after reading the failing runs of the first, so it was a fit followed by a test, and it failed too. What I would not have found without patching the native code: the two-register hole at the top of the file.

Artifacts under `results/g17-tensorops-recon-v1/`: `regfam.py`, `regrun.py`, `regprobe_ana.py`, `regpressure.py`, `regprobe.mm`, `regprobe`, `regprobe_multi.mm`, `regprobe_multi`, `rf_classes.py`, `rf_classes_*.json`, `rf_combo_*.log`, `rf_sweep_acc.py`, `rf_sweep_acc.{log,json}`, `rf_occ.py`, `rf_occ*.{log,json}`, `rf_occ_pair.py`, `rf_occ_pair.json`, `rf_capacity.py`, `rf_capacity.{log,json}`, `rf_tput.py`, `rf_tput.{log,json}`,
`rf_gwsweep.{py,log,json}`, `rf_spill_cost.py`, `rf_spill_cost.{log,json}`, `rf_flat_vs_stream.py`, `rf_flat_vs_stream.{log,json}`, `chain_loop.py`, `chain_loop_sweep.py`, `chain_loop_sweep_*.json`, `chain_loop_*.log`, `eager_prereg.json`, `eager_dep_prereg.json`, `corpus_maxreg.py`, `corpus_maxreg.{log,json}`, `results_maxreg.json`, `mtc_spill_recount.json`, `regpatch.py`, `regpatch2.py`, `regrename.py`, `rf_patch_run.py`,
`reg_ceiling_hw.py`, `reg_ceiling_hw.log`, `gen_acc_live.py`, `acc_live.log`.

## 135. R126 and R127 are not registers to the hardware: every instruction that names R126 to R143 is squashed, on every form and role tested, and the accessible namespace ends at R125

**Result.** (1) The encoding accepts far more than the hardware uses. Apple's decoder prints R126 or above in 1,524 of the 2,219 register operands of the 962 recovered instruction forms, R128 or above in 1,024 and up to R143 in 907, and nothing above R143 (`reg_encoding_census.json`); a 32-bit ALU operand field is an 8-bit register index of which Apple's tables name 0 to 143 (144 to 255 print nothing) and every register name from R120 up has exactly one encoding in each of the 16 operand fields the tests below use (`reg_encoding_dups.log`); Apple's register description defines GPR32 as R0..R143 (144 members, 288 halves), with IR0..IR15 and the 68 special registers of class SIR32, `SR_ZERO` among them, as separate classes; the MMA accumulator field takes the values 0 to 17 for the eighteen 8-register groups R0..R143 and 18 to 31 decode to nothing.
(2) The hardware executes none of it. An instruction that names R126, R127 or any of R128..R143 in any register operand is squashed as a whole: no register write, no memory write, no atomic update, no accelerator time. Measured on ALU sources and destinations, 32-bit, 16-bit and 128-bit store sources, a store's index register, a 128-bit load's destination group, an atomic's destination, the scalar loop of section 134, and the MMA's accumulator, A and B operands. The same forms with R125 or lower execute normally.
(3) So the accessible namespace ends at R125 for every ordinary and tensor form tested: 126 registers, 15 accumulator groups. R126 and R127 are not a hardwired zero (a store from R126 leaves memory untouched, a store from an unwritten valid register writes zeros), not aliases (no encoding duplicates, nothing else changes), not storage, and independent of the metadata register count, the number of simdgroups and the pipeline's thread limit. What they are (reserved, unimplemented on H17s, or meaningful in a pipeline kind I did not build) is unresolved and labeled below. (4) No Neural-Accelerator state exists beyond the GPRs in anything I could reach: MMAs naming R120..R143 take no accelerator time, and accumulators that are spilled to the stack and reloaded between MMAs give bit-exact results (section 134, N = 15 to 24). (5) Section 134 misread part of this evidence; it is corrected in place.

**1. The discriminator.** The experiments of section 134 wrote into a zero-filled output buffer, so a store that did not execute looked like a store of zeros. Every test here fills the output with 0xDEADBEEF first: a slot that still holds it was never written, a slot of zeros was written with zero. The instruments are `rz_carrier.py` (a straight-line kernel with a 32-bit ALU chain, a 16-bit chain and a 128-bit copy, bit-exact against the AIR model), `regpatch3.py` (redirects one chosen operand of one chosen instruction to a chosen register name by searching the operand's own field bits for the value that Apple's decoder prints as that name, other operands verified unchanged, same-length edit of the driver's native archive), and `regprobe`/`regprobe_multi` for launch and timing. Nothing is inserted, no compiler is involved, and only encodings that Apple's own decoder prints as a named register were executed.

**2. What was executed.** Each row: the control at R100 or R125 (a valid register), then R126, R127 and R128..R143 where the form can name them.

    role and form                                  control (R100/R125)             R126, R127                                  R128..R143
    32-bit ALU source, op10282 (t1 = x + 1000)     R8 written with 1000            R8 stays 0: not executed                    same (R128, R143)
    32-bit ALU source, op10279 (v0 = R0 + 1)       result 1 (zero source + 1)      destination keeps its old value             same (R128, R143)
    ALU write then ALU read, op10282 (R60..R143)   value kept (R60 to R125)        t2..t6 lose it (t2 = 0)                     same for the 10 names tried
    32-bit store source, op17229                   writes zeros                    sentinel intact                             not encodable (7-bit field)
    16-bit store source, op17193 (R125L/H)         writes zeros                    sentinel intact (R126L/H, R127L/H)          not encodable
    128-bit store source group, op17256            R120..R123, R122..R125 execute  R124..R127 and R126..R129 sentinel intact   not encodable
    store index register, op17229 operand 5        executes (index 0)              sentinel intact                             not encodable
    128-bit load destination group, op12709        R120..R123, R122..R125 all kept R124..R127: R124, R125 also unwritten    R125..R128: R125 also unwritten
    atomic add destination, op10094                counter 32                      counter 0                                   not encodable (7-bit field)
    scalar loop-carried value (section 134 test)   correct in R60..R125            final store not executed (sentinel)         not encodable
    MMA accumulator group (rf_n8, 8 MMAs/iter)     R112: 4.94 ns per MMA per core  R120..R127: 0.31 ns                         R128..R135, R136..R143: 0.30 ns
    MMA A or B operand group                       R120..R123: 4.95 ns             R124..R127: 0.31 and 0.34 ns                not encodable

A squashed MMA is recognized by time: an executing MMA costs 4.94 ns per core at saturation and about 20 ns per simdgroup, and the R120, R128 and R136 accumulator variants cost 0.30 ns (1 simdgroup per core: 5.7 against 24 to 26 ns), `rz_mma_timing.log` and `rz_mma_ab_timing.log`, interleaved on the GPU clock. The squash is all or nothing: an accumulator group R120..R127 leaves R120..R123 unchanged (loaded and stored back as the initial values, MMA without effect) and a group load or store that includes R126 or R127 does not deliver even R124 and R125.
Independence (`rz_declared.log`, `rz_modes.log`): with the metadata's register count set to 12, 130, 144 and 255 R125 still works and R126, R128 and R143 still fail; 1 simdgroup or 32 simdgroups per threadgroup and a pipeline compiled with `air.max_work_group_size` 32 (reported limit 32 instead of 1,024) change nothing. Reading a register that nothing writes gives zero for R100 and R125 as well, also after a dispatch of a kernel that keeps 124 live scalars (`rz_alu_ro.log`), so a read-only test cannot separate a zero register from an unwritten one; the store test can.

**3. Meanings ruled out and left open.** Hardwired zero: no, see the store rows (a zero source would write zeros, as R100 and R125 do). Alias of a lower register: no register name from R120 up has a second encoding in the fields tested (`reg_encoding_dups.log`), and nothing else changed in any run, but an aliasing rule cannot be observed through forms that are squashed, so I state only that none was seen. Ordinary storage that the compiler leaves unused: no, write then read fails on every form. Selector into another register class (the IR registers, the special registers, uniform registers): the rows show squashing, not a read of another class; a class that exists only in a pipeline kind I did not build (render or tile kernels) would not show here.
Left open: what R126 and R127 are for (a reserved pair, hardware state, or registers implemented on other G17 parts than the H17s of this machine); what R128..R143 are (Apple's table gives them names and the MMA an accumulator group each, R128..R135 and R136..R143, so they look like an extended file that this part does not implement); the encodings R144..R255 of the 8-bit ALU index and the accumulator field values 18 to 31, which Apple's tables do not name. I did not execute those: malformed tensor.mac bytes have hung a GPU and killed WindowServer before (`ledger/g17-hang-poisons-the-run.toml`), and these encodings are outside anything Apple's tables print. Forms not tested: half-precision ALU sources (the only 10-byte half form, op774, has no destination that discriminates and its 4-byte siblings cannot name these registers), the fp32 and mixed-precision MMA forms op5098 to op5105 (only op5106 and op5107 were run), 64-bit address pairs, loads other than the 128-bit tile load, and every form outside `rz_carrier`, `rz_atomic` and `rf_n8`. *(Section 136 part 1 ran op5098 to op5105 and op10384: the same ceiling, R125.)*

**4. Neural-Accelerator state.** The evidence against state beyond the GPR namespace: (a) MMAs whose accumulator names R120..R143 (fields 15, 16, 17) execute nothing, and so do MMAs whose A or B operand names R126 or R127, so no accumulator lives above R119 in any form that runs; (b) in the loop kernels of section 134 with 15 to 24 live accumulators the compiler spills accumulators to the stack and reloads them between MMAs (32 to 185 stack stores) and every value is bit-exact against the host model, so the accumulate state is the GPR contents and there is no shadow copy inside the accelerator that a register save and restore would lose; (c) no instruction I could build reads R126 or above, so state there, if it existed, could not be observed. What (c) leaves open is exactly the set of forms not tested above.

**5. Corrections to section 134, recorded in place.** "R126 and R127 read zero and drop writes" is wrong: instructions naming them are squashed (a zero register would have made the store and the ALU rows above execute). "R124..R127 read back as zero after a load" and "R124..R127 return zeros" came from a store that did not execute into a zero-filled buffer; with a sentinel the group store leaves the sentinel. "The MMA accumulator field ... could name a group up to R248" is wrong: values 18 to 31 decode to nothing and values 15 to 17 (R120..R143) are squashed. "The operands I could test reach R127 at most" is wrong for the census: ALU operand fields reach R143. The conclusion of section 134 that 126 registers are usable and 15 accumulator groups fit stands, and now has a mechanism (the accelerator does not execute a group that includes R126 or R127) and timing evidence. The 14-versus-15 and streaming-cliff conclusions are unchanged.

Artifacts under `results/g17-tensorops-recon-v1/`: `reg_encoding_census.py`, `reg_encoding_census.{json,log}`, `reg_encoding_dups.py`, `reg_encoding_dups.{json,log}`, `rz_carrier.py`, `rz_run.py`, `regpatch3.py`, `rz_alu_rw.py`, `rz_alu_rw.log`, `rz_alu_ro.py`, `rz_alu_ro.log`, `rz_half_tuple.py`, `rz_half_tuple.log`, `rz_forms.py`, `rz_forms.log`, `rz_atomic.py`, `rz_atomic_run.py`, `rz_atomic.log`, `rz_declared.py`, `rz_declared.log`, `rz_modes.py`, `rz_modes.log`, `rz_mma_timing.py`, `rz_mma_timing.log`, `rz_mma_ab_timing.py`, `rz_mma_ab_timing.log`, `regpatch2.py`, `regrename.py`, `reg_ceiling_hw.py`, `reg_ceiling_hw.log`.

## Limits

One chip (H17s), one toolchain, single simdgroup at origin, hand-written AIR (the driver accepted
an intrinsic `metal-opt` rejects; it is the same intrinsic the rtlib uses). Layout labels are
relative to D's classes and consistent across A, B, D; absolute orientation against memory
tensors is fixed by matmul2d's loads, not measured here. The reduction structure was fitted on element (0,0) and then verified on all 256 elements over 20
random tiles.

## 136. The remaining forms and numerical modes: fp32 and mixed MMAs follow the R0-R125 rule at no extra cost, the arithmetic is one exact rule that removes the section-132 range exclusions, int8 chains are register-resident and exact, and Apple's backend lowers a saturating int8 MMA that the library never emits

**Result.** (1) The fp32 and mixed forms (op5098 to op5105) and the int8 form (op10384, op10385) obey the register rule of section 135 in every operand role: an accumulator group R120..R127 and an fp32 A or B group at R120 are squashed, half groups execute up to R120 and are squashed at R124, an int8 pair executes up to R124_R125 in its A slot and R122_R123 in its B slot, and a pair that contains R126 or R127 is squashed. The ceiling is R125 for every form tested. They issue at the rate of the half form (4.94 to 4.98 ns per MMA per core at saturation, int8 2.47), so fp32 and mixed operands cost registers, not time. (2) The arithmetic of all five float forms is one rule (part 2): exact products, one rounding per stage, a fixed summation tree, IEEE special values. It reproduces the hardware bit for bit on 79,500 tiles in 53 (form, stimulus family, mode) cells that reach overflow, subnormals, infinities, NaNs and signed zeros. The two exclusions of section 132 (bfloat overflow, fp32 denormal magnitudes) were errors of my model, which rounded every product to fp32, and are removed. (3) Register-direct chaining is exact for f, fh, hf and b in all four feed modes: 304 of 304 runs, and 152 of 152 runs at the operand scales that section 132 excluded (bfloat overflow, fp32 denormals). (4) int8 and uint8 accumulate modulo 2^32 under repeated MMAs, and register-resident requantized chains are exact for signed and unsigned data, two requantization schemes (an integer shift, and a per-column bias, scale and zero point in float), all four feed modes, flat and streaming order: 272 chain runs, no mismatch. The spill-free streaming grid is 20 converted tiles per unit for the shift scheme against 12 for half chains and 4 for fp32 chains under the same protocol. (5) Apple's backend also lowers `widening_multiply_accumulate_saturate`, an op10384 with one more bit set, whose result is exactly `clip(C + A B, INT32_MIN, INT32_MAX)` (one signed saturation on the exact sum, for all four sign combinations). It costs nothing. The library never emits it. (6) The other spellings the backend names are aliases (float `_saturate` and `fast_`) or conversions around the same instruction (16-bit accumulators); one (`scaled_`) crashes the compile service.

**1. Forms, tuples, ceiling and cost.** `mmaform.py` builds a single-MMA, a multi-tile and a loop kernel for each form from hand-written AIR; opcodes and register widths are read from the decoded code, field alignment from `agxforge.g17.asm.encode_tensor_mac` re-encodings decoded by Apple's decoder (`mmaform_align.log`), the ceiling and rate from patched loop kernels timed interleaved at 1 and 64 simdgroups per core (`mmaform_squash.log`), the register cost from spill-free loop kernels (`mmaform_classes.log`).

    form  operands           opcode acc/no-C  regs A/B/C,D  field alignment A,B,acc  ns/MMA/core  spill-free N (1 A, 1 B) / triples P
    h     half x half        5106/5107        4 / 4 / 8     4, 4, 8                  4.944        14 / 7
    b     bfloat x bfloat    5106/5107        4 / 4 / 8     same as h                not run      not run
    f     fp32 x fp32        5098/5099        8 / 8 / 8     8, 8, 8                  4.982        12 / 4
    fh    fp32 x half        5100/5101        8 / 4 / 8     8, 4, 8                  4.944        13 / 6
    hf    half x fp32        5104/5105        4 / 8 / 8     4, 8, 8                  4.972        13 / 6
    ss su us uu  int8 x int8 10384/10385      2 / 2 / 8     4, 4, 8 (pair +2)        2.474        14 / 10

The int8 pairs sit at a field value that is a multiple of 4 and, in one of the two operands of the compiled instruction, two registers above it (R2, R3 next to R0, R1). Squash timing, executing against squashed, ns per MMA per core at saturation: f acc R112 4.98 and acc R120..R127 0.38, A at R112 4.95 and at R120 0.38, B 4.98 and 0.42; fh acc 4.94 and 0.30, A at R112 4.98 and R120 0.30, B at R120 4.94 and R124 0.34; hf acc 4.97 and 0.38, A at R120 4.98 and R124 0.38, B at R112 4.94 and R120 0.42; int8 acc 2.47 and 0.23, A at R124 2.475 (executes), B at R120 2.47 and R124 0.25. Half (`mmaform_squash.log`, the section-135 result): acc R112 4.94 and R120..R127 0.30, A at R120 4.98 and R124 0.30, B at R120 4.94 and R124 0.34. The fields encode groups that do not execute (accumulator values that name R128 and above, A or B groups that contain R126 or R127); the ceiling is an execution fact, not a decoding one. Register cost follows the widths: an fp32 operand group is eight registers, so with one A and one B fragment live the loop kernels hold 12 fp32-by-fp32 accumulators against 14 for half, and 4 independent (accumulator, A, B) triples against 7. The class of b and of the su, us and uu forms were not run separately (same widths as h and ss); the timing of su, us and uu equals ss (part 5.5).

**2. The arithmetic rule** (`numhw.py`, `numverify.py`, `numverify.log`). For the 16 products p_k = a_k b_k of one output element, taken exactly (no rounding, no overflow, no underflow of a single product): P_i = RNE_fp32(p_2i + p_2i+1), i = 0..7; Q_j = RNE_fp32(P_j + P_j+4), j = 0..3; acc = C, then acc = RNE_fp32(acc + Q_j) for j = 0..3. Each RNE_fp32 rounds the exact sum once, overflows to infinity, underflows gradually, and follows IEEE for infinities, NaNs and the sign of a zero. Operands are exact for half and bfloat and truncated to 10 mantissa bits for fp32 (section 132). The no-C forms (op5099, op5101, op5105, op5107) start at acc = Q_0; C = +0 differs only in the sign of a zero result. Verified over the families ord, over, over2, under, wide (every operand 2^u over the full exponent range), cancel (pairs that cancel exactly or to a tiny residue in stage 1), zeros (signed-zero rules decide) and special (0, 1, max and min normal, max and min subnormal, infinities, NaN) for h, b, f, fh and hf, with and without C: 53 cells of 1,500 tiles, 0 tiles differ (`numverify.log`). The fast float64 model was checked against an exact-rational reference on 2,821 finite tiles of the same families, 0 differ (`numcross.log`). What section 132 got wrong: `run_lowered.mma_np` rounds each product to fp32 before summing. A product that leaves the fp32 range while the pair sum stays inside it (bfloat operands scaled to 1e38, fp32 products below 2^-149) then differs; the first census (`numcorners_first.log`) found 1,077 of 76,800 elements differing at fp32 scale 1e38 and 1,577 at 3e-39, all of them explained by the model, and the section-132 observation that a model fed with the hardware D1 predicted GEMM2 for bfloat overflow but not for fp32 denormals came from the same model; with the rule above both cases are exact at chain level (part 4). There is no hardware deviation from IEEE in range or in the denormal range. A first version of `numverify.py` compared the no-C forms against C = +0 and showed 2 to 10 sign-of-zero tiles per form in family zeros (`numverify.before_noc_fix.log`); the no-C rule above removes them.

**3. The conversions between the MMAs** (`convcorners.py`, `convcorners.log`, `convcorners_denorms.log`). `fptrunc` of 1,510,400 fp32 patterns (5,970 NaNs, 5,912 subnormals, structured and random): to half it is IEEE round to nearest even (0 mismatches, subnormal results included); to bfloat it is round to nearest even except that an fp32 SUBNORMAL input gives a zero with the input's sign (5,879 mismatches, every one an fp32 subnormal input, and the hardware returns zero with the input's sign for all of them; the other 33 of the 5,912 subnormal patterns agree with the reference). Every NaN input gives one output (half 0x7e00, bfloat 0x7fc0). Compiling with denormals enabled and fast math disabled changes nothing. `chain_mt.py` now applies the bfloat flush in `narrow`.

**4. Direct chaining of f, fh, hf and b** (`chain_forms_matrix.sh`, `.log`; `chain_mt.py` with the section-2 rule). Modes A, B, At and Bt of section 132 for each of the four forms: 8 tile grids from 1x1x1x1 to 4x4x2x4, five transpose-flag combinations, accumulate with and without flags, streaming order at 3x3x2x3 and 4x4x2x4: 304 runs, each 12 random chained pairs, all bit-identical to the fragment model. The ranges that section 132 excluded are run at chain level in `chain_range_matrix.sh` (A1 scaled by 2e4, 1e-5, 6e4 for h; 1e38, 1e-38, 3e38, 1e-40 for b; 1e38, 3e-39, 1e-40, 3e38 for f, fh and hf; four modes; 2x2x2x2 and 3x2x2x3 grids; 12 random pairs each): 152 of 152 bit-identical, including the bfloat overflow and fp32 denormal cases and the infinities and NaNs they produce. No range exclusion remains. Special values survive the chain (they are IEEE through the MMA and through the conversions of part 3).

**5. int8 and uint8.**
*5.1 Repeated accumulation* (`int8_repeat.py`, `.log`). Eight accumulators carried through L = 1 to 1,000,000 iterations of `acc = mma(a_j, b_j, acc)` against `C0 + L A B mod 2^32`, extreme, edge and random operands, C0 at INT_MAX - 1000 and INT_MIN + 1000 among them (so L = 1 wraps inside one MMA), one simdgroup and 64 per core: all 608 accumulators exact in the four sign combinations, 429 of them after leaving the int32 range. The accelerator wraps; it does not saturate. The no-C form (`int8_repeat_noc.py`): eight sums `s += mma_noC(a xor c_i, b)`, the operand varied by the iteration counter so the MMA cannot be hoisted, 352 of 352 exact; the compiler keeps eight op10385 and emits no op10384 (it does not fold the running sum into C).
*5.2 Requantized chains* (`chain_i8.py`; matrices `chain_i8_matrix.log`, `chain_i8_float_matrix.log`, `chain_i8_t_matrix.log`). D1 = A1 B1 in int32, an element-wise requantization in the registers, D2 = H B2 (mode A) or A2 H (mode B), or through the transposed feed (At, Bt), with signed or unsigned int8 and int32 accumulators; the model is exact integer and float arithmetic (the integer part is order-free, a B feed needs A2's k index permuted by rotl1 as in section 132, and At and Bt relabel the output rows by rotr1 and columns by rotl1 exactly as for half). Scheme S: `q = clamp((d + 2^(S-1)) >> S, lo, hi)`, S = 0, 1, 4, 8, 12, 16, 24, 31. Scheme F (the zero-point and bias case that section 109 left unmeasured): int32 add of a per-column bias (wraps), int32 to fp32 (round to nearest even), fp32 multiply by a per-column scale, `air.rint` (round half even, verified on 256 values including ties in `rint_probe.log`), clamp to +-1024 in fp32, to int32, integer zero point (0 signed, 128 unsigned), integer clamp, `trunc`; scales include powers of two (many exact ties), zero and negative zero, biases within 1000 of +-2^31. Runs, 8 random chained pairs each, boundary-heavy on alternate trials: S 76 of 76 (modes A and B, 6 grids up to 4x4x4x4, both orders); F 80 of 80 (A and B, 5 grids, both orders, parameters loaded on demand or kept resident); S and F through At and Bt 116 of 116. 272 of 272 exact, uint8 included (the uint8 cell that section 109 refused).
*5.3 Register pressure* (`chain_i8_edge.py`, `chain_i8_float_edge.py`, `chain_i8_float_edge_fine.py`). Largest spill-free converted-tile count per unit in streaming order (units = 2, KT1 = 2, X = 4; first spilling count in parentheses with its stack stores), bit-exact on hardware at each int8 edge (the half and fp32 rows are stack-store counts of compiled kernels; their exactness is section 133's):

    scheme                                mode A       mode B
    S (shift)                             20 (24: 16)  20 (24: 16)
    F, parameters on demand               10 (11: 25)  20 (22: 4)
    F, parameters resident                5 (6: 16)    20 (22: 4)
    half chain, same generator family     12 (13: 5)   12 (13: 1)
    fp32 chain, same generator family     4 (6: 16)    4 (6: 54)

The counts tried were 8, 12, 16, 20, 24, 28, 32, 40, 48 for S; 8, 12, 16, 18, 20, 22, 24, 28, 32 for F (with 9 to 11 and 3 to 7 added for mode A); and 4, 6, 8, 10 to 14, 16, 18, 20, 24 for half and fp32 (`chain_i8_edge.log`, `chain_i8_float_edge*.log`, `chain_h_edge.log`), so an edge is the largest spill-free count among those tried and the first spilling count is the next one tried. An int8 converted tile is two registers, a half tile four and an fp32 tile eight, which is the order of the edges.
Flat order spills at 4x4x2x4 (S: 8 stores; F: 76 to 84 in mode A, 24 to 36 in mode B) and is spill-free up to the 3x2x2x3 and 2x3x1x5 grids tried. In mode A the inner count is the number of D1 column tiles, and each carries a bias and a scale fragment (16 registers); in mode B two column tiles serve any number of rows. That accounts for the shape of the table, but I did not test it directly.
*5.4 Cost* (`chain_i8_float_cost.log`, one interleaved run; `chain_i8_cost.log`; 4x4x2x4 streaming, mode A, signed, 96 MMAs per simdgroup). At 1 simdgroup per core: trunc only 32.2 us (447 instructions), S 40.6 us (959), F on demand 64.7 us (1,714), F resident 57.5 us (1,650), half chain 35.9 us (511). At 64 per core: 45.6, 47.2, 59.1, 53.8 and 47.0 us. S costs 1.26 to 1.27 times the trunc-only chain at one simdgroup per core and 1.04 to 1.05 at saturation (three runs); F on demand costs 1.60 and 1.25 times S. At saturation the half chain, the S chain and the trunc-only chain are within 4 percent of each other, so at that occupancy the int8 chain is not faster than the half chain (both are bound by the loads, stores and conversions, not by the MMA). A first run of the S timing at saturation (`chain_i8_cost.first.log`) gave 219, 211 and 221 us for the trunc-only, S and half kernels, 4.6 times slower with the same ratios (1.04); a rerun gave the numbers above; I did not find the cause (clock state or load on the GPU).
*5.5 Sign combinations.* ss, su, us and uu have the same rate, 2.472 ns per MMA per core at saturation and 11.28 at one simdgroup per core (`int8_sign_timing.log`; half 4.941 and 20.31, so int8 is 2.0 times half at saturation and 1.8 at one simdgroup per core).

**6. The saturating integer MMA** (`variant_probe.py`, `variant_int_survey.py`, `variant_int_groups.py`, `sat_int8.py`, `sat_int8_repeat.py`, `sat_int8_noc.py`, `sat_int8_timing.py`, `sat_classes.py`, `sat_encoder_check.py`). Where the names come from: the offline backend library of the newer generation on this machine (`libapplegpu25-nt.dylib`, Metal toolchain 32023) lists eight prefixes of the 16x16x16 family: `multiply_accumulate`, `multiply_accumulate_saturate`, `fast_multiply_accumulate`, `fast_multiply_accumulate_saturate`, `widening_multiply_accumulate`, `widening_multiply_accumulate_saturate`, `scaled_multiply_accumulate` and `scaled_multiply_accumulate_saturate`; `libapplegpu24-nt.dylib` lists none. The on-device compiler of this part lowers `widening_multiply_accumulate_saturate` to the same op10384 (or op10385 without C), same register widths (2, 2, 8), with byte 8 bit 4 set (0x10; decoder immediate 41 instead of 9). In the float family that bit means "A is fp32" (section 25); in the int8 family (byte 6 bit 0) it is the saturate bit. Every instruction is Apple's own compiler output; no encoding was constructed here.
*Semantics.* 2,400 tiles per sign combination (extreme, edge and random operands; C at INT_MAX - d, INT_MIN + d with d up to 1.5e6, random, zero, and the uint32 limits as bit patterns), hardware D against five models, tiles equal:

    form    wrap    sat_s1 (one signed clip of the exact sum)   sat_s2 (clip at every accumulator step)   sat_u1 (uint32 range)   sat_u2
    ss_sat  1746    2400                                         1975                                       1168                    1150
    su_sat  1759    2400                                         1886                                       1217                    1207
    us_sat  1742    2400                                         1897                                       1189                    1180
    uu_sat  2129    2400                                         2400                                       1845                    1845

The result is `D = clip(C + A B, -2^31, 2^31 - 1)` with A B the exact 16-term sum, for signed and for unsigned operands (uu saturates at the signed limit). My first hypothesis, staged saturation in the order of the float tree (sat_s2), failed on signed data; for uu it cannot be told apart from sat_s1 because every product is non-negative. Under repeated accumulation (`sat_int8_repeat.log`) 608 of 608 accumulators equal `clip(C0 + L A B)` with L to 1,000,000, 429 of them clamped (the wrapping model matches none of those), on one simdgroup and on 64 per core; saturation is not sticky: an accumulator driven to INT_MAX by 20,000 iterations leaves the limit exactly under 4,000 negative-product iterations (ss, su, us). The no-C form (op10385 with the same bit) equals A B on 2,400 of 2,400 tiles. Cost: the rate equals the wrapping form to three places (2.472 ns per MMA per core at saturation, 11.28 at one simdgroup per core, `sat_int8_timing.log`) and the register classes are identical (14 accumulators, 10 triples, `sat_classes.log`).
*Spelling.* The four-letter suffix of the saturating family is `.x.A.B.y`: A's signedness is the second letter, B's the third, and the first and fourth letters change nothing (the 16 spellings produce four distinct code sections, `variant_int_groups.log`). The two-letter suffix is a trap: A is always signed, B is signed exactly when the first letter is `s`, and the second letter is ignored, so `.u.s` and `.u.u` give a signed A and an unsigned B and unsigned A cannot be spelled with two letters. The non-saturating four-letter suffix reads its third and fourth letters as A and B. Use `mmaform.int_name('ss_sat')`, which emits `.s.A.B.s`.
*Provenance.* `libTensorOps.rtlib` (the code behind `matmul2d`) contains 1,149 occurrences of the non-saturating float name and 235 of the non-saturating widening names (81 uu, 81 ss, 37 us, 36 su) and no string containing `saturate`; the Metal Performance Primitives headers do not mention saturation. The saturating MMA is reachable only from hand-written AIR. The existing encoder `mmaenc.py` reproduces 405 of the 413 MMAs of 152 kernels byte for byte from decoded values, the four accumulating saturating forms included; the eight it misses are the no-C saturating forms (bytes 6 and 7, `sat_encoder_check.log`).

**7. Other adjacent forms** (`variant_float_survey.py`, `variant_acc_probe.py`, `variant_acc_census.py`). Float `multiply_accumulate_saturate`, `fast_multiply_accumulate` and `fast_multiply_accumulate_saturate` for h, b, f, fh and hf, with C and without: all 40 kernels compile to a code section byte-identical to the base spelling, so they add no instruction form. A half or bfloat accumulator (`.h.h.v8f16...`, `.b.b.v8bf16...`) lowers to the same op5106 with the 8-register fp32 accumulator, wrapped in conversions (8 op1004 before and 8 op1016 after for half, 52 instructions against 42; bit operations and 8 op1048 for bfloat, 62): the compiler emulates a 16-bit accumulator, the hardware has none, consistent with section 126. `scaled_multiply_accumulate` with half operands crashed the compile service (`XPC_ERROR_CONNECTION_INTERRUPTED`) on 3 of 3 attempts while a control compiled; *Correction (section 137): the crash was an arity error of my five-operand call. The scaled call takes eleven operands and compiles, and the scale is not applied on this target.* I tried no other signature (its operand types are unknown, fp8 and scaled formats are the likely purpose, and the newer backend has 26, 18 and 4 matching strings for e4m3, e5m2 and e2m1 against 4, 0 and 0 in the older one), and `scaled_multiply_accumulate_saturate` was not tried. The `llvm.roundeven` and `llvm.nearbyint` intrinsics crash the compile service in the same way; `air.rint` and `llvm.rint` build and round half to even (`rint_probe.log`).

**Decision rule for a compiler (hand-authored path), replacing (e) of section 132 and the exclusions of its part 6.** (e') No operand-range condition for h, b, f, fh and hf: the arithmetic of part 2 holds for every finite, infinite, NaN, zero and subnormal operand, so a model that follows part 2 predicts D bit for bit, and the conversions between the MMAs follow part 3. (i) int8 and uint8 chains are eligible in modes A, B, At and Bt with an element-wise requantization between the MMAs (schemes S and F of part 5.2 are measured); the grid bound is the spill edge of part 5.3, not a hardware limit: stream the units when the converted tiles per unit exceed the table, and budget 16 registers per D1 column tile for per-column parameters in mode A. (j) Wrapping int32 accumulation is the default and the library's; a compiler that wants saturation emits the saturating spelling of part 6, which costs nothing, and takes the exact clip semantics from the table above. (k) Conditions (a) to (d) and (f) of section 132 and the streaming and pressure rules of sections 133 to 135 are unchanged for the float forms; the R0-R125 ceiling applies to every form.

**Corrections recorded in place.** Section 132 part 4 ("Not exact" paragraph), rule (e) and part 6 (bfloat overflow, fp32 denormals, int8 chains): model error, superseded by parts 2 and 5. Section 131 ("overflow inside one MMA was not tested", "signed-zero results not tested", "not tested: op5100 and op5104"): measured here (parts 2 and 5.1). Section 109 (int8 requantized reuse refused for the same kernel, uint8 cell not run, no zero-point measurement): register-resident chains are exact in hand-authored AIR for both types and both schemes (the refusal concerns `matmul2d`, which runs one `run()` per kernel). Section 135 ("forms not tested: op5098 to op5105"): tested in part 1.

**Labels.** *Hardware (measured on this H17s):* the ceiling, rates and register classes of part 1, the arithmetic and conversion rules of parts 2 and 3, chain exactness and edges of parts 4 and 5, the saturating semantics, cost and non-stickiness of part 6. *Apple backend:* the names in `libapplegpu25-nt.dylib`, which spellings the on-device compiler lowers and to which bytes, the aliasing of the float prefixes, the 16-bit accumulator emulation, the spelling rules of the saturate suffix, the compile-service crash on `scaled_`, the spill edges. *Unresolved (resolved in section 137):* what `scaled_multiply_accumulate` takes and does; the meaning of the first and fourth letters of the saturating suffix (they change no byte); saturation combined with the transpose bits, with chains, and with R120 and above (not executed; the bit is independent of the operand fields but I did not run the combinations); why the first saturation timing of `chain_i8_cost.py` was 4.6 times slower; the direct test of the parameter-fragment explanation of the F edge; the timing and register class of b, su, us and uu individually beyond the rate row.

Artifacts under `results/g17-tensorops-recon-v1/`: `mmaform.py`, `mmaform_squash.py`, `mmaform_squash.{json,log}`, `mmaform_classes.py`, `mmaform_classes.{json,log}`, `mmaform_align.py`, `mmaform_align.log`, `numform.py`, `numhw.py`, `numexact.py`, `numcorners.py`, `numcorners_first.log`, `numverify.py`, `numverify.{json,log}`, `numverify.before_noc_fix.log`, `numcross.py`, `numcross.log`, `convcorners.py`, `convcorners.log`, `convcorners_denorms.log`, `chain_mt.py`, `chain_forms_matrix.sh`, `chain_forms_matrix.log`, `chain_i8.py`, `chain_i8_matrix.sh`, `chain_i8_matrix.log`, `chain_i8_matrix.first.log`, `chain_i8_float_matrix.sh`, `chain_i8_float_matrix.log`, `chain_i8_t_matrix.sh`, `chain_i8_t_matrix.log`, `chain_i8_edge.py`, `chain_i8_edge.log`, `chain_i8_float_edge.py`, `chain_i8_float_edge.log`, `chain_i8_float_edge_fine.py`, `chain_i8_float_edge_fine.log`, `chain_range_matrix.sh`, `chain_range_matrix.log`, `chain_h_edge.py`, `chain_h_edge.log`, `chain_i8_cost.py`, `chain_i8_cost.log`, `chain_i8_cost.first.log`, `chain_i8_float_cost.py`, `chain_i8_float_cost.log`, `int8_repeat.py`, `int8_repeat.{json,log}`, `int8_repeat_noc.py`, `int8_repeat_noc.{json,log}`, `int8_sign_timing.py`, `int8_sign_timing.log`, `rint_probe.py`, `rint_probe.log`, `variant_probe.py`, `variant_probe.log`, `variant_int_survey.py`, `variant_int_survey.log`, `variant_int_groups.py`, `variant_int_groups.log`, `variant_float_survey.py`, `variant_float_survey.log`, `variant_acc_probe.py`, `variant_acc_census.py`, `variant_acc_probe.log`, `sat_int8.py`, `sat_int8.{json,log}`, `sat_int8_repeat.py`, `sat_int8_repeat.{json,log}`, `sat_int8_noc.py`, `sat_int8_noc.log`, `sat_int8_timing.py`, `sat_int8_timing.log`, `sat_classes.py`, `sat_classes.log`, `sat_encoder_check.py`, `sat_encoder_check.log`.

## 137. The scaled TensorOps forms: the high-level API is blocked by the OS in four places, the OS compiler accepts the AIR intrinsic and drops the scale, and what is reachable is an exact fp8 operand class (op17642 unpack plus a bf16 op5106)

**Result.** (1) The scaled form is MX-style block scaling: an fp8 or fp4 operand tensor carries a plane of `ue8m0` scales, one per 32 elements along K for each row of the left operand and each column of the right operand (transposed), the destination has none, and the accumulate is half or float; there is no bias, zero point or per-tensor scale (part 1). (2) On this machine (macOS 26.6.2, H17s) the API path is blocked four times over, each block sufficient by itself: the frontend needs `-std=metal4.1`; the OS library loader refuses a Metal 4.1 language tag and an AIR 2.9 deployment target; the OS runtime library `libTensorOps.rtlib` has no fp8, fp4, scale or `_v2` entry point; the OS tensor library is AIR v28 and lacks the multiplane accessors (part 2). (3) Below the API, the OS compiler (AGXCompilerCore) does implement the AIR intrinsic `air.simdgroup_matrix_16x16x16_scaled_multiply_accumulate`. I read the operand layout out of the compiler's code, called it with that layout, and it compiles (part 3). (4) The result on hardware is that the scale has no numeric effect: 36 configurations of selectors and has-scale constants for fp8 and 4 runs with half and bfloat operands leave the output equal to the unscaled result, and the op5106 the compiler emits is byte-identical whether the scale operands are absent, published or dropped (part 4). The scale is not applied on this target, and none of the 56 MMA forms the decoder tables declare (op5090 to op5143, op10384, op10385) has a scale operand: each has three or four register groups, D, A, B and for the accumulate forms C (`mma_signature_census.log`). (5) What the lowering does provide is an exact fp8 operand class: eight op17642 instructions unpack the fp8 fragments (two fp8 to two bf16 per instruction, format immediate 97 for e4m3fn and 98 for e5m2) and a bf16-typed op5106 accumulates in fp32. It is also reachable through the ordinary five-operand `multiply_accumulate` with fp8 type tokens, which compiles to the same code. Decode is exact for every finite code of both formats, 96 of 96 random tiles and 36 of 36 transposed tiles match the section-136 arithmetic, a register-direct chain into a bf16 MMA is exact (20 of 20), the register cost equals bf16 when the unpack is hoisted, and the rate equals bf16 while at most four operand fragments are unpacked per MMA (part 5). (6) The int8 scaled form with fp32 accumulate aborts the OS compile with `Failed to encode.`, fp4, fp6 and packed tokens and the fp8 pack/unpack (quantize) calls crash the compile service at named sites (part 6). (7) A compiler must not use the scaled intrinsic for scaling here; the rule is in part 8, and the blocking layer for a hardware block-scaled MMA is the instruction set: no declared accelerator form has a scale operand, and the OS compiler emits none.

**1. What the API form is** (`scaled_layers.log`, MPP headers of SDK 27.0). `matmul2d` accepts a tensor with a `tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32>` tag (a rank-2 tag has block sizes 32 and 1). The header's static asserts fix the semantics: the scale element type is `metal_fp8_ue8m0_format`, block size 0 is 32 and block size 1 is 1, the scale tensor has rank 1 or 2, the left tensor must not be transposed and the right tensor must be transposed if they carry scales, and the destination cannot carry scales. Tensor extents are 32-aligned and strides 128-byte aligned. The supported datatype combinations listed in `MPPTensorOpsMatMul2d.h` are half x fp4 e2m1, half x fp8 e4m3 and half x fp8 e5m2 with half or float destination, and fp4 x fp4, e4m3 x e4m3 and e5m2 x e5m2 with half or float destination. So the scale is per row of A and per column of B for each block of 32 K elements (MX granularity), its representation is a power of two 2^(e - 127) in one byte, and the accumulator and destination are half or float. There is no bias or zero-point operand, no rounding or clamp mode for the scaled product (the quantize side, `pack<Format, Rm, Sm>`, has rounding and saturation modes: round to nearest even and saturate by default for fp4 and e4m3, no saturation for e5m2), and no per-tensor scale. The header declares 3,310 `__tensorops_impl_matmul2d_op_run` externals, 360 of them with fp4, e4m3 or e5m2 operands (120 each); scale information travels as extra integer arguments of those externals (`rightScaleDataType`, `rightScaleBlockSize0`, `rightScaleBlockSize1`).

**2. Why the API path stops here** (`scaled_layers.py`, `scaled_layers.log`). (a) Frontend: the installed frontend (`xcrun metal`, 32023.921 from the Metal toolchain cryptex) compiles `sc_fp8_mm.metal` under `-std=metal4.1` (rc 0; the AIR is version 2.9 and calls `__tensorops_impl_matmul2d_op_run_dv_fp8e4m3_dv_fp8e4m3_dv_f32_v2`) and rejects it under `-std=metal4.0` (11 errors, the first `use of undeclared identifier 'tensor_plane_scales'`); `__HAVE_TENSOR_MULTIPLANE__` is defined only inside the `#if __METAL_VERSION__ == 410` block of `metal_config`. (b) OS library loader: `newLibraryWithData` fails with `This library is using language version 4.1 which is not supported on this OS.`; with the language tag rewritten to 4.0 and AIR 2.9 kept it fails with `This library is using a deployment target (0x00020009) that is not supported on this OS.`; with both tags rewritten (Metal 4.0, AIR 2.8) `metallib` refuses to package it (`air version set to 2.8.0 ..., but expecting 2.9`). (c) Runtime library: the OS `libTensorOps.rtlib` (macOS 26.6.2) has 2,950 `metal-nm` lines naming `matmul2d_op_run` and no line of its symbol list contains fp8, fp4, e4m3, e5m2, e2m1, ue8m0, scale, blockwise or `_v2`; the frontend of this toolchain emits `_v2` names. (d) Tensor library: the device tensor library in `AGXMetalG17X.bundle/Contents/Resources/tensor.metallib` is AIR v28 with 137 functions (the same names as the toolchain's applegpu25-nt copy) and lacks `air.get_block_size_*_tensor`, `air.is_plane_valid_*_tensor`, `air.get_type_*`, `air.get_rank_*` and `air.get_stride_bytes_*` that the AIR v29 library (applegpu-nt) has. The toolchain frontend is newer than the OS compiler (32023.921 against 32023.886); everything the toolchain emits for 4.1 needs the newer OS. Earlier sections recorded the language gate for fp8 and fp4 (section 6); the other three layers are new here.

**3. The AIR intrinsic in the OS compiler** (`scaled_probe.py`, `scaled_crash_sites.py`, `re_AGCSimdMatrix_buildMMA.asm`, `re_scale_helper.asm`). The OS compiler image (`AGXCompilerCore`, extracted from the dyld shared cache, same UUID as the crashing service) names eight prefixes of the 16x16x16 family, including `scaled_multiply_accumulate` and `scaled_multiply_accumulate_saturate`, and the type tokens `v4f8`, `v8f8`, `v16f8`, `v128f8` and `v256f8` with `e4m3fn` and `e5m2` (fp8 vectors) and the packed `pn4`, `pn8` and `pn16` forms of e4m3, e5m2, e2m1, e2m3 and e3m2. My first scaled call (section 136, five operands like the ordinary form) crashed the compile service with an MTE tag-check fault in `AGCSimdMatrix::buildSimdMatrixMultiplyAccumulate(llvm::CallInst*)`. Reading that function (radare2 on the extracted image) shows why: it splits the callee name at '.', takes four type tokens (in name order the destination, A, B and C types; with `scaled` it takes tokens n-6, n-4, n-3 and n-1 of the name and skips n-5 and n-2, which sit where the scale types are), and reads call operands by index from the operand array, eleven of them on the scaled path: op[0] scale vector, op[1] a constant selector vector, op[2] a constant integer tested for zero (the has-scale flag), op[3] A, op[4] A's transpose flag, op[5] B, op[6] B's transpose flag, op[7] scale vector, op[8] selector vector, op[9] has-scale flag for B, op[10] C. A helper (`re_scale_helper.asm`) `extractelement`s two elements out of the selector vector, requires them to fold to constants, and either extracts one 16-bit half of the scale vector or, when the has-scale flag is zero, substitutes 0x7f (the ue8m0 code for 2^0). Building the call that way moved the failure exactly as the reading predicts: five operands crash reading operands 5 to 10 past the call (MTE fault); eleven operands with scalar i8 scale operands crash in `ConstantInt::get` from the helper (an `extractelement` of a scalar); eleven operands with `<4 x i8>` scale vectors, a constant `<2 x i32>` selector and constant flags compile. The working call is `air.simdgroup_matrix_16x16x16_scaled_multiply_accumulate.f.f.v8f32.v4i8.v8f8e4m3fn.v8f8e4m3fn.v4i8.v8f32(<4 x i8> sA, <2 x i32> selA, i1 hasA, <8 x i8> A, i1 transA, <8 x i8> B, i1 transB, <4 x i8> sB, <2 x i32> selB, i1 hasB, <8 x float> C)`. The earlier section-136 statement that the compile service crashes on `scaled_multiply_accumulate` is therefore an arity error of my call, not a property of the form.

**4. What the lowering emits and what the scale does** (`scaled_uniform.log`, `scaled_gate.log`, `scaled_half.log`, `scaled_sem_1.log`, `scaled_forms.log`). For fp8 operands the compiler emits (single-MMA kernel, 33 instructions with scale publishes): two 32-bit uniform scale loads, two `op592` instructions that publish the scale words (the instruction the repository documents as the register-to-slot publish of the texture path), two 64-bit fragment loads, eight `op17642` (`GPR32 <- imm, imm, GPR16, imm`: two fp8 in a 16-bit register to two bf16 in a 32-bit register), one `op5106` with A and B typed bfloat (type code 3) and an fp32 accumulator, and the stores. The publishes appear only when the scale words are lane-uniform and the has-scale flag is nonzero: with per-lane scale addresses the scale loads, the publishes and every use disappear silently (50 instructions, the same opcode census as the has-scale-0 kernel), and with has-scale 0 the same. The `op5106` bytes are `270825122202a0020004` in all three variants (uniform, per-lane, has-scale 0), and differ from the plain bf16 MMA only in the wait and tag bits (byte 0 0x27 against 0x2f, byte 1). Numerics on hardware, one simdgroup, sentinel-filled output, A = B = 1.0 and C = 0 (an unscaled result is 16 everywhere) with lane-uniform scale words A = [97, 117, 137, 157] (exponents -30, -10, +10, +30 against the bias 127) and B = 127, and the same with the words on B: for each selector pair (i0, i1) in 0..3 x 0..3 on A and on B (32 kernels) and for has-scale constants 0, 1, 2, 3 (4 kernels) every output element is 16, so the scale had no effect in any configuration. With half and bfloat operands (the scaled call lowers to the plain op5106 with the publishes and no unpack) the same test gives 16 everywhere in 4 of 4 runs. With every scale word 127 and has-scale 0 or 1 the fp8 kernel equals the exact model on 12 of 12 random tiles each. The selector pair and the has-scale constant are therefore not gates. What I did not establish is the meaning of the published words: `op592` writes them to a slot that no consumer in these kernels reads, and I did not execute any instruction the compiler did not emit.

**5. The fp8 operand class** (`scaled_class.py`, `scaled_class.log`, `scaled_loop.py`, `scaled_loop.log`, `scaled_chain.py`, `scaled_chain.log`, `fp8_plain.py`, `fp8_plain.log`). Every instruction is Apple's compiler output, run on one simdgroup with sentinel-filled output. Decode: with the other operand the identity and through both the A path and the B path, all 254 finite e4m3fn codes and all 248 finite e5m2 codes give exactly the ml_dtypes value; each special code alone in a tile behaves as the arithmetic model predicts (e4m3fn 0x7f and 0xff NaN; e5m2 0x7c and 0xfc infinities, 0x7d to 0x7f and 0xfd to 0xff NaN). Arithmetic: 24 random tiles for each of e4m3 x e4m3, e5m2 x e5m2, e4m3 x e5m2 and e5m2 x e4m3, 96 of 96 bit-identical to `numhw.mma_hw` on the exactly decoded operands (fp8 embeds exactly in bf16, so the section-136 rule applies unchanged: exact products, one rounding per stage, the fixed tree). Transposes: the transA, transB and both bits, 12 tiles each, 36 of 36 exact with the fragment conventions of section 132. Chain: D1 from the fp8 class stays in the accumulator registers, eight `op1048` convert it to bfloat, a no-C bf16 `op5107` consumes it as A, no stack stores: D1 20 of 20 and D2 20 of 20 bit-identical to the model with the section-136 conversion rule. Register cost (loop kernels, one A and one B fragment, spill-free = no `op595`, no `op17789`): 14 accumulators and 7 (accumulator, A, B) triples when the fp8 fragments are loop invariant (the compiler hoists the unpack; identical to bf16), 13 and 6 when the unpack is repeated every iteration. Rate, ns per MMA per core, 8 accumulators, interleaved on the GPU clock, at 1 and at 64 simdgroups per core: bf16 20.32 and 4.959; fp8 hoisted 20.32 and 4.960; unpack repeated with one A and one B fragment for 8 MMAs 24.34 and 4.960; 8 distinct A fragments and one B (4 unpacked fragments per MMA on average) 35.39 and 5.325; 8 distinct A and 8 distinct B (8 unpacks per MMA, one stack store) 96.05 and 8.383. At saturation the unpack costs nothing up to about one fragment per MMA, 7 percent at four, and 69 percent at eight (a spill included); at one simdgroup per core it is latency and shows at once. The class does not need the scaled call: the ordinary `air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f8e4m3fn.v8f8e4m3fn.v8f32(<8 x i8> A, i1, <8 x i8> B, i1, <8 x float> C)`, and its `fast_` and `_saturate` prefixes, compile to `op17642` and `op5106` instructions byte-identical to the scaled has-scale-0 lowering, for e4m3 (24 of 24 random tiles run exact) and for e5m2 (format immediate 98; compiled, not run). This corrects the statement of section 2 that fp8 formats are only convertible in AIR around the MMA: on this OS the compiler does the conversion, from hand-written AIR, and the MMA remains the bf16 form; the accelerator still consumes 16-bit or int8 fragments.

**6. Variants that do not reach the hardware** (`scaled_crash_sites.py`, `scaled_crash_sites.log`; the failing site is read from the OS crash report of each rebuilt kernel). The scaled form with int8 operands and fp32 accumulate (`.s.s` and `.u.u`, `v8f32`) aborts the compile with `Failed to encode.` (`llvm::report_fatal_error` from `MCObjectStreamer::emitInstruction`; the message was recovered by matching the sha256 in the crash report against the strings of the OS `libLLVM.dylib`): the selection reaches a machine instruction that this build's encoder cannot emit (by the backend's intrinsic names, the `igemm ... .fp` form, which I did not observe directly), which fits the count of section 27 (132 declared forms, 10 admitted). The same form with int32 accumulate compiles to the ordinary `op10384` with no scale ops (the scale is ignored there too). fp4 and fp6 packed operands (`pn8e2m1`, `pn8e2m3`, `pn8e3m2`, with i32 or i64 operands) crash in `AGCLLVMAirBuiltins::buildConvert` (null dereference) in the scaled form (4 of 4 kernels) and in the ordinary form (6 of 6, which also include the packed fp8 token `pn8e4m3` and `<8 x i4>` and `<8 x i8>` operands); I do not know the operand type the compiler expects for these tokens, and the vector token that works for fp8 (`v8f8...`) exists only for fp8. The quantize side is closed: the calls the Metal 4.1 frontend emits for `pack<Format>()` and `unpack<T>()` (`air.pack.f.pn8e4m3.f.v8f32`, `air.unpack...`) crash the OS compiler in `AGCLLVMAirBuiltins::buildPack` (2 of 2 formats), and the OS-internal spellings `air.quantize_pack` and `air.quantize_saturate_pack` crash `NumericPackUnpackPass` (2 of 2). So an fp8 or fp4 requantized chain has no compiler-emitted quantize instruction on this OS.

**7. Against the scalar requantization path** (sections 106 to 110 and 136). The int8 requantization is ordinary ALU code between the MMAs, exact, register-resident in all four feed modes, with a measured cost (shift scheme 1.27 times the trunc-only chain at one simdgroup per core and 1.05 at saturation; the float-scale scheme with per-column bias and zero point 1.6 and 1.25 times that) and it supports what the scaled API withholds: bias, zero point and per-column scale, in registers. The fp8 class replaces the dequantization half of a quantized matmul by hardware unpack at four `op17642` per operand fragment (int4 operands are unpacked by ordinary ALU instructions, section 6), exact, and needs no scale because it has none; the quantization half has no reachable primitive here. A block-scaled fp8 matmul on this OS therefore has the dequantize unpack in hardware and the scale application, per row and per column, in ALU code around the MMA (a power-of-two scale is an exact multiply, or an exponent add on nonzero normal values); I did not build or time that scaling.

**8. Compiler rule (hand-authored AIR path).** (a) Do not emit `scaled_multiply_accumulate` or `scaled_multiply_accumulate_saturate` to obtain scaling: on this OS the scale operands are not applied (part 4), and per-lane scale operands are dropped without a diagnostic. (b) fp8 operands (e4m3fn, e5m2, independent per operand, fp32 accumulate) are eligible through the five-operand `multiply_accumulate` with `v8f8e4m3fn` or `v8f8e5m2` tokens and `<8 x i8>` operands, with C given (the no-C form of this class was not tested); D = C + sum of the exactly decoded products under the section-136 rule; transposes are eligible; a chain into a bf16 MMA follows the section-132 conversion rules. (c) Charge the class as bf16 plus four `op17642` per unpacked operand fragment, hoist the unpack out of loops (14 accumulators, 7 triples as bf16), and keep unpacked fragments per MMA at four or fewer to stay at the MMA rate at saturation. (d) Not eligible: fp4, fp6 and packed (`pn`) tokens; int8 operands with fp32 accumulate; fp8 or fp4 quantization (`air.pack`, `air.unpack`, `air.quantize_pack`); the `matmul2d` fp8 and scaled paths (OS gates of part 2). (e) MX block scaling, if wanted, is software: the scale words of a scaled call must not be trusted for any target without a numeric test like the one in part 4.

**Labels.** *Hardware (measured on this H17s):* every numeric result of parts 4 and 5. *Apple frontend and OS gates:* the errors of part 2, the header rules of part 1. *Apple OS compiler (read and observed):* the operand layout and lowering of part 3, the crash sites and the `Failed to encode.` abort of part 6. *Unresolved:* what the published scale words are for (a consumer on another G17 part, or a code path for a target that encodes scaled forms); why the compiler keeps the scale only for lane-uniform values; the operand type of the packed fp4, fp6 and `pn` tokens; the spelling of a working quantize call; the no-C fp8 form; scaled forms with transposes and the `_saturate` prefix beyond identical code; a software MX scaling kernel and its cost.

*Correction (section 138 part 10): part 8(d) above says fp8 quantization is not eligible. fp8 quantization is reachable through `air.convert` (op13618, op13620): round to nearest even, no saturation, every NaN to the positive canonical NaN, exact over all 2^32 fp32 patterns. The crash records for `air.pack` and `air.quantize_pack` stand and are not the only spelling; fp4 stays open. Of the Unresolved list, the spelling of a working quantize call is `air.convert`, the no-C fp8 form compiles to op17642 plus the bf16 no-C op5107 and is exact, and a software MX scaling kernel and its cost are section 138.*

Artifacts under `results/g17-tensorops-recon-v1/`: `scaled_layers.py`, `scaled_layers.log`, `sc_fp8_mm.metal`, `sc_fp8_mm.frontend.ll`, `sc_layers_patch.py`, `scp2_fp8_mm.ll`, `scp_fp8_mm.ll`, `mma_signature_census.py`, `mma_signature_census.log`, `scaled_probe.py`, `scaled_probe.log`, `scaled_crash_sites.py`, `scaled_crash_sites.log`, `re_AGXCompilerCore.macho`, `re_AGCSimdMatrix_buildMMA.asm`, `re_scale_helper.asm`, `scaled_run.py`, `scaled_uniform.py`, `scaled_uniform.log`, `scaled_sem.py`, `scaled_sem_1.log`, `scaled_map.py`, `scaled_map_sel00.log`, `scaled_gate.py`, `scaled_gate.log`, `scaled_half.py`, `scaled_half.log`, `scaled_forms.py`, `scaled_forms.log`, `scaled_class.py`, `scaled_class.log`, `scaled_loop.py`, `scaled_loop.log`, `scaled_chain.py`, `scaled_chain.log`, `fp8_plain.py`, `fp8_plain.log`, `fp4_plain.py`, `fp4_plain.log`, `fp8_pack.py`, `fp8_pack.log`, `fp8_pack_names.py`, `fp8_pack_names.log`.

## 138. Software MX scaling and the quantize direction for the fp8 class: post-MMA fp32 scaling is the cheapest correct placement, `air.convert` supplies the fp8 pack, chains are exact, and INT8 is faster and more accurate wherever its range suffices

**Result.** (1) `air.convert` converts in both directions between fp8 (e4m3fn, e5m2) and fp32, fp16 and bf16, on eight-element vectors and on scalars; the OS compiler emits op17634, op17642 and op17632 for the unpack and op13618 and op13620 for the pack. Dequantization is exact for all 256 codes of both formats. Quantization was compared with a model over all 2^32 fp32 patterns and all 65,536 bf16 and fp16 patterns with no mismatch: round to nearest even, no saturation (e4m3fn overflow gives NaN, e5m2 overflow gives infinity), every NaN input gives the positive canonical NaN (0x7f and 0x7e), signed zeros are kept. This retires the section-137 statement that fp8 quantization is not reachable (part 10). (2) Eight placements of a ue8m0 scale per 32 K-elements (per A row and B column) were built in hand-authored AIR (four of them also with predecoded fp32 scale tables), and every one has an exact composed model: 477 kernel-and-case comparisons of the byte-table kernels (both fp8 formats and the mixed pairs, both transposes, K = 16, 32, 40 and 100, one to four accumulators, powers of two, overflow and underflow, NaN and Inf codes, scale codes 0 and 255) and 128 of the predecoded kernels were bit-identical to the model. (3) The cheapest correct placement is post-MMA fp32 scaling: per 32-element block the two MMAs run into a zero accumulator (the compiler emits the no-C form op5107 for the first one), the block sum is multiplied by the product of the two factors and added to an fp32 accumulator with one `fma`. It costs 8 instructions per MMA (12 with separate multiply and add), plus 16.5 instructions per scale vector per block when the scales are decoded from bytes, and at saturation it runs at 1.03 to 1.08 times the unscaled fp8 MMA time on grids of 2x2 to 4x2 output tiles (unscaled 5.0 ns per MMA per core, which equals bf16 and is set by the MMA). It equals the exactly rounded MX value for a single block (K = 16 and 32: 14,336 of 14,336 elements) and differs from it by one fp32 rounding per block across blocks (K = 40 and 100: 85.6 percent of elements identical, worst 5.8e-6 relative). Its limits are the fp32 ALU's: subnormal inputs and results flush to zero, so scale code 0 and block sums below 2^-126 vanish. (4) Operand-side scaling (in fp32 or bf16 after the unpack, or by integer exponent addition on bf16 or on the fp8 byte) is more expensive per MMA at saturation (5.7 ns, 10.1 ns and 7.4 ns at 2x2), less accurate (74 percent of elements identical to the exact MX value for one block and 61 percent for several, worst 7.6e-5) or wrong outside a narrow exponent range, and the bf16-multiply form aborts the OS compiler at 3x3 tiles. (5) The `fma` placement is spill-free up to 4x2 output tiles (4 row tiles, 2 column tiles, peak live 115 of the 126 registers R0..R125; the unscaled 2x2 kernel has 60 live, the scaled one 86), not at 2x4 (27 stores), and every placement that scales in fp32 or bf16 spills at 3x3 (bf16 multiply does not compile there). (6) The quantize direction is a software sequence around the hardware pack: block amax by integer max and two `simd_shuffle_xor` steps, scale code from the exponent field, an fp32 multiply by the power-of-two inverse, and `air.convert`. Two scale rules were built: the floor rule (code = exponent of amax minus 8 for e4m3fn or 15 for e5m2, as I recall the OCP MX rule; not re-read this session), which needs a software clamp because the pack does not saturate (202 of 1,024 blocks of gaussian data would otherwise produce NaN in e4m3fn), and a ceil rule (one integer add before the shift) that never needs the clamp. Both are exact against a reference model on 112 adversarial cases each (3.67 million elements in all, scale codes and bytes, 0 mismatches), for fp32, bf16 and fp16 sources; the ceil rule is 127 instructions and 15.0 ns per 16x32 block against 157 and 19.4 for the floor rule, and it is more accurate. (7) Chained GEMM, quantize, GEMM runs are exact at every stage in 20 of 20 configurations, with the quantizer reading the first GEMM's output buffer and the second GEMM reading the quantizer's output buffer as its A operand, with no host-side relayout: the D fragment layout equals the untransposed A layout. (8) Against INT8: the int8 MMA runs at 2.5 ns per MMA per core, twice the fp8 route, and a block-scaled int8 GEMM (the same post kernel plus a `sitofp`) at 3.7 to 4.6 ns, 1.2 to 1.4 times faster than block-scaled fp8; INT8 block-32 with an fp32 scale is 3 to 5 times more accurate than e4m3fn on well-scaled tensors (relative error 0.0076 against 0.0376 for a 64x512x64 gaussian product), but per-tensor INT8 collapses on outlier channels (0.0885) where fp8 without any scale does not (0.039), and fp8 without a scale fails outside about 2^-9 to 448 (relative error 1.0 for values of 1e-4) where every scaled scheme is unaffected.

**1. Conversion, both directions** (`fp8_convert_probe.py`, `fp8cv.py`, `fp8_convert_probe.log`, `fp8cv_dq.log`, `fp8cv_q.log`, `fp8cv_q16.log`, `fp8cv_qsample.log`). The probe declares `air.convert.f.<dst>.f.<src>` for every pair of {e4m3fn, e5m2} with {fp32, fp16, bf16} in both directions on `<8 x i8>` and on scalars; the OS compiler accepts all 14 variants that were tried. Eight elements convert with four packed instructions (op17634 to fp32, op17642 to bf16, op13618 from fp32) and one element with a scalar instruction (op17632, op13620). Dequantization: 256 of 256 codes equal `ml_dtypes` for both formats to fp32, bf16 and fp16. Quantization: a GPU kernel generates every fp32 bit pattern (2^32, in 256 chunks of 2^24) and compares the returned byte with the model `ml_dtypes` round to nearest even, except that every NaN gives the positive canonical NaN; the model differs from `ml_dtypes` only in NaN sign (hardware 0x7e or 0x7f, `ml_dtypes` keeps the input sign). 0 mismatches for both formats, and for all 65,536 bf16 and all 65,536 fp16 patterns as sources. Overflow is not saturated: an fp32 value above 464 converts to e4m3fn NaN (0x7f; 464 itself rounds to 448 by the tie rule) and a value of 61440 or more to e5m2 infinity (0x7c) *(sign correction, Set A 2026-09-24, MM section 25.105: the sign is kept, so a negative overflow gives 0xff and 0xfc. The canonical positive NaN applies to NaN INPUTS only. On hardware, every negative e4m3fn overflow of the compiler's op13618 pack returned 0xff, 471 of 471, `results/g17-tensor-lowprec-v1`. This matches this part's own model, `ml_dtypes` apart from NaN inputs, and not the parenthetical)*, so a software clamp is required whenever the scale rule can place a value above the format's largest finite value (448 and 57344).

**2. The placements and their models** (`mxgen.py`, `mxhost.py`, `mxrun.py`, `mxsuite.py`, `mxsuite_S1.log`, `mxsuite_S2.log`, `mxsuite_S3.log`). Kernels are runtime loops over K blocks of 32 (two k-tiles), one simdgroup, mt x nt output tiles, four distinct blocks cycled so nothing is hoisted, scale tables in device memory as one byte per row (or column) per block, or as fp32 for the predecoded forms (suffix p). `post`: block sum P by two MMAs (first into a zero accumulator), F = fa x fb, acc += P x F with fmul, fmul, fadd; `postfma`: the same with one fma; `postsep`: (P x fa) x fb; `f32op`: unpack to fp32, multiply by the row or column factor, fp32-operand MMA (op5098); `bf16op`: as `f32op` then `fptrunc` to bf16 and a bf16 MMA; `bf16mul`: unpack to bf16 and multiply in bf16; `bf16exp`: integer addition on the bf16 exponent with zero, infinity and NaN guards; `fp8exp`: integer addition on the fp8 byte before the unpack. The scale table for a post variant is in the D layout (row of D, column of D) whatever the operand transposes, and for an operand variant it is in the operand layout (an A row lane holds two rows, a B lane four columns, transposed operands the reverse). The composed models use the section-136 MMA arithmetic (`numhw.mma_hw`, `mma_hw_noC`) and, for the surrounding fp32 operations, the rule measured here (part 5). Suites: S1 (9 input kinds by 9 variants, e4m3 x e4m3), S2 (6 kinds by 6 variants by 3 other format pairs), S3 (6 variants by 4 transpose pairs by 3 grids by 4 K values); 81, 108 and 288 comparisons, all model-identical. The predecoded variants (`postp`, `postfmap`, `f32opp`, `bf16opp`) are checked separately in `mxpre_check.py` (two shapes with a K remainder, four transpose pairs, e4m3 x e4m3 and e5m2 x e4m3, random and special scale codes): 128 of 128 identical to the models of their byte-table counterparts. The first version of the post variants failed 72 of 288 transposed runs because their tables were in operand order; D-order tables fixed all of them (part 11).

**3. Cost** (`mxcost.py`, `mxcost.log`, `mxcost_i8.log`, `mxcost_1x1.log`, `mxcost_grids.log`, `mxcost_grids2.log`). Instruction counts are decoded from the built kernels, registers and spill stores come from the section-134 analyser (stores with op595 or op17789), time is interleaved on the GPU clock (median of 9 rounds) at 1 and at 64 simdgroups per core, in ns per MMA per core. Extra instructions per MMA are against the unscaled fp8 kernel of the same grid (113 instructions at 2x2, 173 at 4x2 with 8 and 16 MMAs per block).

    variant     extra instr/MMA   ns/MMA/core    peak live   |  extra instr/MMA   ns/MMA/core   stack stores
                     2x2             64 SG        2x2 regs   |       4x2             64 SG          4x2
    none              0               5.05           60      |        0              5.36              0
    post             20.8             5.27           88      |       22.3            6.59             13
    postp            12.5             5.23           91      |       15.9            6.10              9
    postfma          16.8             5.34           86      |       14.0            5.50              0
    postfmap          8.5             5.25           92      |        8.5            5.86              0
    postsep          20.8             5.38           84      |       20.0            6.21              1
    f32op            16.8             5.74           85      |       23.2            6.68             65
    f32opp            8.5             5.55           87      |       13.9            6.28             58
    bf16op           16.8             5.75           70      |       12.0            5.93              0
    bf16opp           8.5             5.42           72      |        6.5            5.95              0
    bf16mul          21.4             5.50           71      |       15.1            5.99              0
    bf16exp          62.4            10.09           90      |       46.3            8.56              0
    fp8exp           43.8             7.42           65      |       32.5            7.08              0

A second sweep of `none`, `post` and `postfma` (`mxcost_i8.log`) gave 4.95, 5.20 and 5.08 at 2x2 and 5.00, 6.14 and 5.16 at 4x2, so differences below about 0.4 ns are within run-to-run variation and the ordering `postfma` before `post` at 4x2 holds in both. At one simdgroup per core the scaling is latency, not hidden: at 2x2 the unscaled kernel takes 27.1 ns per MMA and `postfmap`, `postfma` and `post` take 33.0, 36.5 and 38.7. Larger grids: at 3x2 and 2x3 (48 accumulator registers) `postfma` and `postfmap` are spill-free (peak 109 and 111), `post` stores 3, `f32op` stores 17 at 2x3; at 4x2 `postfma`, `postfmap`, `bf16op` and `bf16opp` are spill-free and at 2x4 they are not (27, 25, 3 and 1 stores); at 3x3 every byte-table variant stores (post 34, postfma 24, postsep 3, f32op 89, bf16op 11, bf16exp 1) except `fp8exp` (peak 115); at 4x3 the unscaled kernel itself stores 36. `bf16mul` builds up to 2x4 and aborts the OS compiler at 3x3 and 4x3 (`Failed to encode.`, `mxcost.log`). The compile-service crashes are probes and are read from the crash reports (`scaled_crash_sites.py`).

**4. Reuse of scale values.** A row-tile scale vector is decoded once per block and used by every column tile, a column-tile vector by every row tile; nothing carries across blocks because each block has its own scale. Measured from the pairs (`post`, `postp`) and (`postfma`, `postfmap`): a byte-decoded scale vector costs 16.5 instructions per block (33 at 1x1, 66 over four vectors at 2x2, 80 over five at 3x2), so the byte-decode cost per MMA is 16.5 (mt + nt) / (2 mt nt): 16.5 at 1x1, 8.3 at 2x2, 6.4 at 4x2, 4.1 at 4x4. The epilogue per tile and block is independent of the grid: 24 instructions (12 per MMA) for fmul, fmul, fadd, 16 (8 per MMA) for the `fma` form once F = fa x fb is formed (8 fmul). Predecoded fp32 tables remove the 16.5 at four bytes instead of one per row and block (33 against 36 bytes per 32 elements of one row of fp8 data); at saturation they gain 0 to 0.4 ns and reduce register pressure only at 4x2, and at 3x2 and above the fp32 tables are loaded per block and add live registers (`postp` stores 1 at 3x2, `postfmap` none). The compiler's own hoisting was not exploited beyond what the loop structure gives; constant scales across blocks (per-tensor) would hoist the decode entirely and were not measured.

**5. Numerical semantics of the placements** (`alu_flush.py`, `alu_flush.log`, `mxprobe_extreme.py`, `mxprobe_extreme_{default,denorms,strict,fastdn}.log`). The fp32 ALU: `fmul`, `fadd`, `fsub` and `fma` flush subnormal inputs and subnormal results to zero with the sign kept; normal results are IEEE round to nearest even (8,454,144 subnormal inputs per operation and 69,632 subnormal-result cases: 0 IEEE-gradual results, all flushes with sign). The four compile modes (denorms enabled or disabled, fast math on or off) give the same results, so the flush is not a compile option. The MMA keeps gradual underflow (section 136). Consequences for scaling: scale code 0 (2^-127) is a subnormal factor and is flushed, so a product scale that involves it multiplies to zero (`over_ops`, row scale code 254 with column scale code 0: `post` 0 of 256 identical to the exact value, which is finite) and a block sum whose scaled value is subnormal gives zero (`tiny_prod`: `post` 215 of 256 identical); an overflow of the scaled block sum gives the infinity that the exact value rounds to (`over_prod`, 256 of 256). NaN scale code 255 gives NaN for that row or column block in the post variants (256 of 256 in S1 `special`, 255 of 256 with zero relative error in S2), and Inf and NaN elements follow IEEE. Agreement with the exactly rounded MX value (Fractions), S1 with one block: the post family is identical in 7 of 9 input kinds (rand, pow2, boundary, over_prod, cancel, under, special: 256 of 256 each) and differs in `over_ops` and `tiny_prod` for the reasons above. S3 (48 runs per variant): the post family 14,336 of 14,336 elements for one block (K = 16, 32) and 12,272 of 14,336 (85.6 percent, worst 5.84e-6 relative, from cancellation) for K = 40 and 100, which accumulate one fp32 rounding per block; the operand-side variants (`f32op`, `bf16op`, `bf16mul`, `bf16exp`) 10,602 of 14,336 (74.0 percent) for one block and 8,716 of 14,336 (60.8 percent) for several, worst 7.6e-5. The composed models reproduce the operand-side results bit for bit; the source of their difference from the exact value, even for one block, was not isolated (candidates: the operand conversion and flush rules and the staged rounding of the MMA, section 136). `bf16exp` and `fp8exp` are wrong on underflow (`under`, `over_ops`), and on `rand` data `fp8exp` matches 73 of 256 elements with worst relative error 1.2e6: the sum of the scale and the fp8 exponent leaves the 4 or 5 bit exponent field and carries into the mantissa or the sign (mechanism read from the code, not isolated on hardware).

**6. Quantization** (`mxquant.py`, `mxquant_suite.py`, `mxquant_suite.log`, `mxquant_suite_ceil.log`, `mxquant_suite*.json`, `mxquant_suite_hw*.npz`, `mxquant_controls.py`, `mxquant_controls.log`, `mxquant_cost.py`, `mxquant_cost.log`). Layout: fp32 (or bf16 or fp16, converted exactly) fragments in the D layout in, fp8 fragments in the D layout and a `<2 x i8>` scale byte pair per lane out. A lane holds two rows and four columns of each 16x16 tile, so a block row is spread over four lanes: the kernel takes the integer maximum of the absolute bit patterns of its 2 x 4 elements over both k-tiles (after flushing fp32 subnormals to zero, `icmp ult 0x00800000`), then two `air.simd_shuffle_xor.i32` steps with masks 1 and 8 with integer max. Scale: code = min(max(E - emax, 0), 254) with E the biased exponent field of the amax pattern, zero amax gives 0, any NaN or Inf in the block gives 255 (an amax pattern at or above 0x7f800000). Ceil rule: E is taken from the pattern plus 0x1fffff (e4m3fn, e5m2) or 0x1ffff (int8), which carries into the exponent exactly when the mantissa of amax exceeds that of the format's largest finite value (1.75 for 448 and 57344, 1.984375 for 127). Elements: fp32 multiply by 2^(127 - code) built from the integer (254 - code) << 23 (factor 1 in code-255 blocks), then the floor rule clamps to plus or minus the maximum (`fcmp`, `select`), the ceil rule does not, then `air.convert` (op13618). Under the ceil rule the scaled amax is at most the maximum, so no element can overflow the pack; the coverage counts (f32-source cases, two rounds each) confirm 0 clamped elements in all 14 ceil case types against up to 2,042 in the floor cases (`mxquant_controls.log`).
Validation: 14 case types by 2 formats by (2 rounds of fp32 sources, one bf16, one fp16) = 112 cases per rule, 64 rows by 256 columns each: gaussian activations, sigma 0.02 weights, post-ReLU, outlier channels, heavy tails, per-row scale spread 2^-60 to 2^60, dynamic range 2^-126 to 2^127 per element, amax placed at 256 to 511.99 (e4m3fn) and 32768 to 65535 (e5m2) with the ties and one-ulp neighbours of the clamp and carry thresholds, every midpoint between adjacent fp8 values and its two fp32 neighbours in both signs (7,958 exact ties in e4m3fn and 7,936 in e5m2 in the ties case, two rounds), a ladder through the fp8 subnormals (4,158 subnormal outputs and 18,160 underflows to zero in e4m3fn), NaN and Inf in all positions with several payloads and signs (768 code-255 blocks of 1,024), zeros, negative zeros and fp32 subnormals (17,585 subnormal inputs), fp32 extremes around both ends of the code clamp, random bit patterns. Scale codes and bytes equal the reference in 112 of 112 cases for each rule, including the element bytes of code-255 blocks (they are not part of the contract; the kernel's bytes there equal the reference's). Controls (`mxquant_controls.py`): seven wrong models of the floor rule (no clamp, no flush of subnormals, code off by one, no code 255, amax from the first k-tile only, emax swapped between formats, factor a power of two low) are each rejected by 12 to 108 of the 112 cases, and nine of the ceil rule (the floor rule with and without clamp, the carry threshold one mantissa step low and high, and the five errors above) by 4 to 108. The controls found a gap: no case placed the block maximum one ulp above the carry threshold, so a rule with the threshold one mantissa step high was accepted by 0 of 112 cases; the boundary case now includes the threshold and its neighbours and rejects it in 4. Cost (one simdgroup, 4 row tiles, per 16x32 block, 512 elements, ns per block per core at 64 simdgroups per core): e4m3fn and e5m2 floor 157 instructions and 19.4 ns (26.4 elements per ns per core), ceil 127 and 15.0 (34.1); int8 floor 189 and 23.1, ceil 159 and 18.5; bf16 sources add about 10 instructions (ceil 136, 15.8 ns) and fp16 sources about 1 (ceil 128, 13.4 ns); 32 to 35 registers, no stores. At one simdgroup per core the block takes 141 ns (ceil, e4m3fn). Amortization: quantizing the M x K activation matrix takes M K / 512 blocks of 15.0 ns and a GEMM against N output columns takes M K N / 4096 MMAs of 5.0 ns, a ratio of 24 / N (38 percent at N = 64, 2.3 percent at N = 1024, int8 59 / N); weights are quantized once.

**7. Chaining** (`mxchain.py`, `mxchain.log`, `mxchain.json`). Y = Q(Q(X) W1) W2 with X quantized on the GPU, GEMM1 the post kernel, D1 quantized on the GPU straight from GEMM1's output buffer (the quantizer indexes the GEMM's tile order, `gemm_nt`, address arithmetic only), GEMM2 the post kernel reading the quantizer's output buffer as its A buffer (fragments at ((blk 2 + kk) mt + m) 32 + lane and scale bytes at the A table, as written). Only the static weights are quantized and packed on the host (B fragments have their own map). Five configurations by two seeds by two rules: e4m3fn everywhere (32x64x64x32), e5m2 everywhere, mixed formats with K = 80 (a half-filled last block, `mixed_kremainder`), e4m3fn with 3x4 tiles in GEMM1 (48x96x64x32), and mixed formats with K = 48 and N1 = 96. Each stage is compared bit for bit with the composed model (quantizer against `quantize_ref`, GEMMs against `mxrun.model` on the stage's actual input): 20 of 20 chains exact at all four stages, 0 sentinels left, 0 NaN outputs. End-to-end relative error against the fp64 unquantized product (D1 after the first GEMM, D2 after the second): all-e4m3fn floor rule D1 0.042 to 0.055 and D2 0.062 to 0.071, ceil rule 0.033 to 0.039 and 0.052 to 0.058; all-e5m2 floor rule 0.078 to 0.080 and 0.107 to 0.116, ceil rule 0.075 and 0.102 to 0.112; the mixed configurations lie between 0.042 and 0.077 (D1) and 0.075 and 0.115 (D2). The fp8 fragments come out in the D layout, which equals the untransposed A layout, so an output can feed the next GEMM's left operand as it is; as the right operand it needs the B map (the two lanes-by-slots maps differ), not built here. The chain is memory-linked across three kernel launches, not register-resident: fusing the quantizer into the GEMM kernel (the accumulator `<8 x float>` values are already the quantizer's input) is not built.

**8. Against the INT8 route** (`mxint8.py`, `mxint8.log`, `mxacc.py`, `mxacc.log`, `mxacc.json`, `mxcost_i8.log`, `mxquant_cost.log`). Kernels: `i8none` (int32 accumulation through the K loop, the per-tensor or per-channel dequantization once after the loop, op10384: 80 instructions at 2x2 against 113 for fp8, no unpack), `i8post` and `i8postfma` (per block the int8 MMAs into a zero accumulator, op10385 for the first, then `sitofp`, the post scaling of part 2), and predecoded forms; bit-exact against the integer model in 87 of 87 cases (random, all -128, 127 times -128, sparse, small, scale codes 0, 1, 127, 254, 255; K = 64, 80, 96). Rate at saturation (ns per MMA per core, 2x2 / 3x2 / 4x2): `none` fp8 4.95 / 5.04 / 5.00; `i8none` 2.48 / 2.50 / 2.50 (twice, as section 31 measured for the int8 MMA); `post` fp8 5.20 / 5.61 / 6.14; `i8post` 3.79 / 4.02 / 4.56; `postfma` fp8 5.08 / 5.44 / 5.16; `i8postfma` 3.67 / 4.12 / 4.17 (all from `mxcost_i8.log`). The block-scaled int8 GEMM is 1.2 to 1.4 times faster than the block-scaled fp8 GEMM (ratios 1.24 to 1.40 over the six pairs), and its scaling costs more relative to its MMA (`i8post` is 1.5 times `i8none` at 2x2) because the 8 to 12 ALU instructions per MMA that hide under the fp8 MMA no longer do. The instruction totals equal the fp8 kernels' (279 for `post` and 247 for `postfma` at 2x2): the 32 unpack instructions of fp8 are replaced by 32 `sitofp`. Accuracy (`mxacc.py`): GEMM 64x512x64, relative Frobenius error of the product against the fp64 product of the unquantized operands, mean of 3 seeds, all schemes quantize A per row and B per column along K:

    scheme                                   gauss   llm_outlier  heavy_tail  relu    row_range  small_wts  tiny(1e-4)  huge(1e4)
    fp8 e4m3fn block-32, floor rule          0.0427   0.0555      0.0446    0.0453    0.0456     0.0417     0.0440      0.0415
    fp8 e4m3fn block-32, ceil rule           0.0376   0.0384      0.0372    0.0387    0.0390     0.0377     0.0372      0.0378
    fp8 e5m2 block-32, ceil rule             0.0757   0.0755      0.0736    0.0766    0.0749     0.0740     0.0748      0.0750
    fp8 e4m3fn, no scale                     0.0380   0.0392      0.0376    0.0391    0.0394     0.0427     1.0000      0.9880
    int8 per tensor                          0.0135   0.0885      0.0660    0.0141    0.0264     0.0135     0.0135      0.0135
    int8 per row / column                    0.0105   0.0433      0.0247    0.0106    0.0105     0.0105     0.0105      0.0105
    int8 block-32, fp32 scale                0.0076   0.0142      0.0109    0.0072    0.0075     0.0076     0.0076      0.0076
    int8 block-32, power-of-two, ceil rule   0.0112   0.0209      0.0165    0.0106    0.0109     0.0112     0.0114      0.0114

The two-layer chain Y = Q(Q(X) Q(W1)) Q(W2) gives 0.053 to 0.057 (e4m3fn ceil), 0.101 to 0.109 (e5m2 ceil), 0.010 to 0.016 (int8 block-32 fp32 scale) and 0.088 for per-tensor int8 on outlier channels. The fp32 accumulation of the post kernel adds 1.4e-8 to 5.1e-8 relative to the fp64 product of the same dequantized operands, against a quantization error of 0.04 to 0.08, so accumulation is not the accuracy limit. The fp8 formats have a fixed relative precision (3 or 2 mantissa bits), so scaling does not improve them on data inside about 2^-9 to 448; it only avoids underflow and overflow and, with the ceil rule, clamping. INT8 has no exponent range, so its accuracy depends on the scale granularity: the SQNR of A is 45.5 dB for block-32 with an fp32 scale, 40.5 dB per tensor on gaussian data and 21.5 dB per tensor on outlier channels, against 31.5 dB for e4m3fn with the ceil rule and 25.5 dB for e5m2.

**9. Decision table** (hardware-backed unless marked).

    | Question | e4m3fn | e5m2 | Evidence |
    | Dequantize for an MMA | op17642 (fp8x2 to bf16x2), 4 per fragment, exact 256/256, hoist out of loops, bf16 MMA at the bf16 rate | same, format immediate 98 | 137 part 5, part 1 |
    | Dequantize to fp32 | op17634, exact | same | part 1 |
    | Quantize | `air.convert` op13618 (RNE, no saturation, NaN to positive canonical NaN) | same, overflow gives infinity | part 1 |
    | Scale placement | post-MMA fp32, `postfma` (byte tables) or predecoded factors | same | parts 2 to 5 |
    | Scale table | ue8m0 byte per row or column per block in D order (post), 16.5 instr per vector per block; or fp32 | same | part 4 |
    | Usable scale codes | 1..254, code 255 gives NaN, code 0 flushed to zero by the fp32 ALU | same | part 5 |
    | Cost of scaling | 8 instr/MMA plus decode; 1.03 to 1.08 x MMA time at saturation, 2x2 to 4x2 | same | part 3 |
    | Register budget | `fma` placement spill-free to 4x2 tiles (peak 115 of 126), not at 2x4 or 3x3 | same | part 3 |
    | Transposes, K remainder, mixed pair | exact (S3: 4 transposes, K = 16, 32, 40, 100; S2: all pairs); K remainder by zero padding | same | part 2 |
    | Quantize rule | ceil, 127 instr / 15.0 ns per 16x32 block, no clamp | same | part 6 |
    | Quantize rule (OCP floor, from memory) | 157 instr / 19.4 ns, clamp to 448 needed | clamp to 57344 | part 6 |
    | Chaining | exact, quantizer output = next A operand as written; B needs the B map | same | part 7 |
    | Accuracy, 64x512x64 gaussian | 0.038 (ceil), 0.043 (floor) | 0.076 | part 8 |
    | Against int8 | int8 MMA 2x rate; block-scaled int8 1.2 to 1.4x faster than block-scaled fp8; int8 block-32 3 to 5x more accurate on well-scaled data; fp8 without a scale is unaffected by outlier channels, int8 per tensor is not | | part 8 |

**10. Corrections to section 137, recorded in place.** *Part 8(d) said fp8 quantization is not eligible (`air.pack`, `air.unpack`, `air.quantize_pack`); fp8 quantization is reachable through `air.convert` (op13618, op13620) with the semantics of part 1. The crash records for `air.pack` and `air.quantize_pack` stand; they are not the only spelling. fp4 quantization is still open. The Unresolved list named the spelling of a working quantize call, the no-C fp8 form and a software MX scaling kernel and its cost: the first is `air.convert`, the second compiles to op17642 plus the bf16 no-C op5107 (a zero accumulator constant, in every post kernel; exact in the S3 runs), the third is this section.* Part 5 of section 137 measured the fp8 class at the bf16 MMA rate with the unpack hoisted; the fp8 `none` kernels of this section confirm it (4.95 to 5.36 ns per MMA per core at 2x2 to 4x2, 32 op17642 and 8 op5106 at 2x2).

**11. Failed hypotheses, kept.** (a) Adding the scale to the fp8 exponent byte before the unpack (`fp8exp`) works only while the sum stays in the format's 4 or 5 exponent bits; on `rand` data it matches 73 of 256 elements. (b) Adding to the bf16 exponent (`bf16exp`) fails on underflow and costs 62 instructions per MMA and twice the MMA time. (c) Scale code 0 through an fp32 multiply: the factor 2^-127 is subnormal and the ALU flushes it. (d) The first composed ALU model assumed IEEE gradual underflow and failed the `over_ops` and `tiny_prod` cases; the hardware flushes inputs and results, and no compile flag changes that. (e) Post tables in operand order failed for transposed operands (72 of 288 S3 runs); they must be in D order. (f) I first wrote that post scaling is exactly rounded; it is for one block only, and across blocks it carries one fp32 rounding per block (part 5). (g) I assumed the OCP floor rule with a clamp was the accuracy-optimal rule; the ceil rule is 12 to 31 percent more accurate on e4m3fn (0.0555 to 0.0384 on outlier channels) and cheaper. (h) I assumed the fp8 pack saturates; it does not. (i) I assumed `ml_dtypes` NaN signs match the hardware; the hardware gives the positive NaN. (j) A first suite accepted a wrong carry threshold (part 6); the control run exposed it. (k) I attributed the operand-side variants' difference from the exact MX value to extra rounding from the scaling; the composed models reproduce them, but the cause was not isolated (part 5), and it appears already for one block. (l) I expected the fp8 scaling overhead to be ALU-bound and visible at saturation; it is hidden under the bf16-rate MMA and visible only at one simdgroup per core and in the register budget, and it becomes visible against the int8 MMA.

**12. Compiler rules (hand-authored AIR path).** (a) Dequantize with `air.convert` to bf16 and keep the unpack out of loops (section 137). (b) Apply an MX scale after the MMA, in fp32: two MMAs per 32-element block into a zero accumulator, F = fa x fb, `fma(P, F, acc)`; keep the tables in D order; treat scale code 255 as NaN and do not use code 0 or block products below 2^-126. (c) Keep the grid at 3x2 or 2x3 tiles with byte tables and 4x2 with predecoded factors to stay spill-free; do not use `bf16mul` at 3x3 or above. (d) Quantize with the ceil rule (amax pattern plus 0x1fffff, shift, minus 8 or 15, clamp to 0..254, 255 for NaN or Inf, zero for a zero block), multiply by the integer-built factor, `air.convert`; no clamp is needed; flush fp32 subnormals before the maximum. (e) A chain feeds the quantizer's output to the next GEMM's left operand unchanged; the right operand is packed offline. (f) Choose int8 for accuracy and speed where a scale per 32 elements or per channel covers the range; choose fp8 where outlier channels or a wide range rule out a per-tensor int8 scale and one byte per element is wanted.

**Labels.** *Hardware (measured on this H17s):* every number in parts 1 to 8. *Model (checked against hardware bit for bit):* the composed models of part 2, the quantizer reference of part 6, the integer model of part 8. *Model only:* the accuracy numbers of the scale-per-tensor, no-scale, fp32-scale int8 and per-row schemes in part 8 (the quantizers of the fp8 block schemes and the power-of-two int8 scheme are hardware-validated; the rest are numpy formulas), the amortization ratios, the read of the OCP rule (from memory). *Left open:* fp4 and fp6 quantization; the semantics of the native scale words (unchanged from section 137); a register-resident fused GEMM, quantize, GEMM kernel; a D-to-B relayout for the right operand; constant per-tensor scales hoisted out of the loop; data dependence of the timing (random gaussian bytes only); block sizes other than 32; sources other than fp32, bf16 and fp16; the int8 quantizer's own numerics beyond the suite of part 6 (10 case types and the integer-tie cases, exact); the `air.pack` and `air.quantize_pack` crash sites were not retried.

*Correction, recorded in section 140: two of the items above are resolved. Part 7's statement that the right operand "needs the B map (the two lanes-by-slots maps differ), not built here", the decision-table row "B needs the B map" and the Labels items "a register-resident fused GEMM, quantize, GEMM kernel" and "a D-to-B relayout for the right operand" are superseded: the maps differ only by the row rotation of section 129, which the library's loads and D labelling already apply, and the fused kernel is built and bit-exact for the left and right operand in both transpose states (section 140).*

Artifacts under `results/g17-tensorops-recon-v1/`: `fp8_convert_probe.py`, `fp8_convert_probe.log`, `fp8cv.py`, `fp8cv_dq.log`, `fp8cv_qsample.log`, `fp8cv_q.log`, `fp8cv_q16.log`, `mxhost.py`, `mxgen.py`, `mxrun.py`, `mxsuite.py`, `mxsuite_S1.log`, `mxsuite_S1.json`, `mxsuite_S2.log`, `mxsuite_S2.json`, `mxsuite_S3.log`, `mxsuite_S3.json`, `alu_flush.py`, `alu_flush.log`, `mxprobe_extreme.py`, `mxprobe_extreme_default.log`, `mxprobe_extreme_denorms.log`, `mxprobe_extreme_strict.log`, `mxprobe_extreme_fastdn.log`, `mxcost.py`, `mxcost.log`, `mxcost_1x1.log`, `mxcost_grids.log`, `mxcost_grids2.log`, `mxcost_i8.log`, `mxint8.py`, `mxint8.log`, `mxpre_check.py`, `mxpre_check.log`, `mxquant.py`, `mxquant_suite.py`, `mxquant_suite.log`, `mxquant_suite.json`, `mxquant_suite_hw.npz`, `mxquant_suite_ceil.log`, `mxquant_suite_ceil.json`, `mxquant_suite_hw_ceil.npz`, `mxquant_controls.py`, `mxquant_controls.log`, `mxquant_controls.json`, `mxquant_cost.py`, `mxquant_cost.log`, `mxquant_cost.json`, `mxchain.py`, `mxchain.log`, `mxchain.json`, `mxacc.py`, `mxacc.log`, `mxacc.json`.

## 139. The positional B-offset failure is two errors of the reference, not a hardware or stream-position rule: the branch's failing bundles execute exactly, and both reported values are reproduced to the last digit from the references

**Result.** (1) The four bundles that the six-layer session left in `/tmp` (pure tensor with body 4 at 4096, at 12288, and with the layer-1 tuple (12288, 16384, 20480); the two-layer graph with scalar rows and the same tuple), replayed with that session's own inputs (`a.f16`, `b.f16`, `c.f32`) through the same worker, give hardware outputs that equal an instruction-level model bit for bit in the three pure-tensor cases (0 of 512 elements differ) and to 4.77e-7 in the mixed case, the level of the no-offset control. (2) The pure-tensor value the branch reports, max_abs 0.00048828125 (2^-11, one ulp at magnitudes 4,096 to 8,191), is reproduced to the last digit in all three bundles against the runtime tool's reference, which runs the MMA chain of an accumulate body from C ("C-first"); the hardware, `tlower` and Apple's compiler all run it from zero and add C afterwards ("C-last"). (3) The mixed value, max_abs 0.0017757415771484375 with 512 of 512 elements differing, is reproduced to the last digit against `_transformer_layer_weight_offset_reference`, whose first GEMM omits the measured fp32-A truncation that layer 1's first body (its A operand is the fp32 LayerNorm output of layer 0) performs; with the truncation the same hardware output differs by at most 4.77e-7. The same offset-layer reference, called with offsets (0, 0, 0) on the NO-OFFSET production control program (image ba4b1984ebb76f00) with the same inputs, gives 1.37e-3 with 511 of 512 elements differing and fails the 2e-5 bound, while the no-offset reference passes at 4.77e-7: the failure follows the reference function, not the offsets. Body 4 is the first fp32-A body of a second layer, which is why every offset-bearing case that failed began at body 4 and why the first three-body layer (half A) passed. (4) Independent of those bundles, the identical offset-bearing body is exact at every position 1 to 12, after 0 to 4,096 bytes of NOP padding, after five different histories, in streams of 6, 9, 12 and 18 bodies (17 offset-bearing bodies, 14,722 bytes), at unaligned and displacement-limit offsets (2 to 40,002 bytes), with exact-size and 64 KiB payloads, deterministically (233 dispatches) and under concurrent GPU load (193 dispatches); a reference-free relocation differential is bit-identical at every position tested; and Apple's own compiler emits byte-identical offset-bearing loads at bodies 1 to 16. (5) No stream position, reset or setup sequence was found and none is needed; the compiler-facing rule is part 10. No file under `agxforge/` or `tools/` of any checkout was edited.

**1. The images and the overlay** (`posoff_prod.py`, `posoff_mixed.py`, `posoff_bundle_replay.py`). The six-layer commit 09c9c04b was extracted with `git archive` to `/tmp` (the working tree of that branch has uncommitted edits by its own session and was neither run nor modified; I read its diff). `cc.compile_function` is the shell (six `tensor_matmul` ops with the production `_transformer2` shapes, no offsets, plus the two SR-reading rows of the measured `(SR130, SR156)` class; or the production two-layer graph with GELU, residual and LayerNorm rows). For a compile, `agxforge.g17.tensor.emit_gemm` is replaced by a function returning the bodies the experiment assigns to that call, each from `tensorgemm.lower_gemm` with exactly the arguments `cc` passes (registers 126, R0..R15 reserved, `end=False`); a call may carry several bodies or NOP padding. `scanlink.author` builds the image and the common worker (rebuilt from the snapshot source with a larger B payload, read from `POSOFF_BBYTES`) is run in a fresh process per dispatch under the runtime's GPU lock and event-counter check. The program sha256 prefixes equal the ones the branch records: 150c6af7587d0ae5 (body 4 at 4096, 4,058 bytes), 3b1a3cea6a7a955e (at 12288), 57dfe6fc32762819 (tuple, 4,074 bytes), ab07b69af35d4f9d (mixed tuple, 157,290 bytes), ba4b1984ebb76f00 (mixed no-offset control, 157,266 bytes); the pure-tensor no-offset control is 615c6632e90b6cb7 (4,050 bytes). `posoff_setup.sh` recreates the snapshot and applies the one overlay edit. The branch's own bundles are copied unmodified to `posoff_bundles/` (their authored archives, `program.bin`, manifests and inputs) and replayed directly.

**2. The reported values and their causes** (`posoff_bundle_replay.py`, `posoff_bundle_replay.log`, `posoff_bundle_ref.py`, `posoff_bundle_ref.log`). Hardware output of the branch's bundle, its inputs, against the two references:

    bundle (program sha)             hardware vs C-last   vs runtime tool (C-first)     reported by the branch
    one-low: body 4 at 4096 (150c6af7)     0 of 512 differ    4.8828125e-4 (299 differ)     4.8828125e-4
    one-high: body 4 at 12288 (3b1a3cea)   0 of 512           4.8828125e-4 (299)            4.8828125e-4
    pure tuple 12288/16384/20480 (57dfe6fc) 0 of 512          4.8828125e-4 (316)            4.8828125e-4
    mixed tuple, scalar rows (ab07b69a)    4.77e-7 (165)      1.7757415771484375e-3 (512), fp32-A truncation omitted   1.7757415771484375e-3

For the mixed bundle the four reference variants against the same hardware output: C-first and no fp32-A truncation in layer 1's first GEMM (the committed `_transformer_layer_weight_offset_reference` with the tool's C-first `_gemm_mma`): 1.7757415771484375e-3, 512 differ; C-last without the truncation 1.7757117748260498e-3; either order with the truncation 4.76837158203125e-7 (318 and 165 differ). The reported 0.00048828125 is one ulp at 4,096 to 8,191, and 2e-5 is 0.04 ulp there, so any accumulate body at that magnitude fails the runtime tool's bound against a reference of the wrong order. The truncation is the measured operand rule of sections 5 and 33 (the low 13 bits of an fp32 A operand are cleared): `_transformer_layer_reference(inp, b, 32)`, used for the no-offset control, applies it to layer 1's first GEMM, `_transformer_layer_weight_offset_reference`, used for every offset-bearing case, does not (the uncommitted working-tree edit of that function in the branch's checkout adds `truncate_a` for fp32 input; that edit is the fix for this half). Reference swap on the control (`posoff_ref_swap.py`, `posoff_ref_swap.log`): no-offset reference 4.77e-7, offset reference with zero offsets 1.37e-3 (511 of 512 differ, fails 2e-5), same hardware output, same inputs.

**3. What fixes the hardware's accumulate order** (`posoff_spec.py`, `posoff_refcheck.py`, `posoff_cfirst.py`, `posoff_cfirst.log`, `tlower_cf.py`). C-last runs the MMA chain over K from zero (op5107 first, op5106 after) and then adds the old C (`fadd`); C-first starts the chain from the old C. My C-first model equals the runtime tool's `_gemm_mma` in 6 of 6 cases and its `_transformer_layer_weight_offset_reference` in 3 of 3 (`posoff_refcheck.log`). `tlower` emits C-last and is bit-identical to the C-last model in every sweep here; `tlower_cf.py`, a results-only variant that loads C into the accumulators and starts the chain from them (op5106 at k = 0, waiting on slot 6), is bit-identical to the C-first model and differs from the C-last one in 265 to 305 of 512 elements, for the no-offset stream, body 4 at 4096 and the layer-1 tuple, 3 seeds each, 18 of 18 runs; Apple-compiled `multiply_accumulate` streams (part 7) equal C-last in 6 of 6 runs and C-first in 0 of 6. The pure-tensor signature therefore follows the reference and never the offset: against C-first the no-offset control differs in 24 of 24 seeds (`posoff_E1.log`; max_abs 4.9e-4 to 1.46), as do the five offset streams (120 of 120), 144 of 144 seed runs in all; against C-last none differs. The branch did not run a pure-tensor no-offset control; this is it.

**4. Position, history, address, payload, determinism and load** (`posoff_sweep.py`, `posoff_seeds.py`, `posoff_det.py`, `gpu_stress.py`; logs `posoff_E1.log`, `posoff_E2.log`, `posoff_E3.log`, `posoff_E5.log`, `posoff_E7.log`, `posoff_E9.log`, `posoff_E10.log`, `posoff_E11.log`, `posoff_bundle_det.log`, `posoff_bundle_det_load.log`). Test body: 16x32x32 float/half no-accumulate, B offset 4096, ranks (2,1,2); predecessors identical without the offset; data uniform, weights scaled 0.3 for the long streams. Preregistered in `posoff_prereg.md` before E2 to E8 were run, with the deviations recorded in its addendum.

    experiment                                          streams   dispatches   differing elements vs C-last
    the five reported or control images, 24 seeds each, plus runtime-tool data   6    144    0
    position p = 1..12                                      12        96           0
    NOP padding 2..4096 B before the body at p = 4 and 2    8         48           0
    predecessors: accumulate, wide 16x32x64, half A, mixed  4         24           0
    every body carries the offset (6 and 12 bodies)         2         12           0
    long: 6, 9, 12, 18 bodies, distinct offsets per body    4         24           0
    B offsets 2, 6, 4098, 4352, 4354, 12290, 32766, 32768, 32770, 40002 at p = 4 and 8    20    80    0
    exact-size B payload (6,144 and 22,528 bytes)           3         20           0
    established three-body class (0, 4096, 8192)            1         6            0
    same program, same input, 100 dispatches, twice         2         200          one distinct output each
    same, with a concurrent fp8 MMA loop on all cores       3         130          0 (17 heavy dispatches over 157 s)
    the mixed bundle: 30 + 60 dispatches and one 3-query worker each (12 heavy dispatches over 110 s for the second)   1   96 outputs   one distinct output

The 18-body stream (first body 16x32x64 half/half, then the production `_transformer2` pattern) uses the base-register form of `tlower.based` for the offsets above 32,767 (34,816, 36,864); offsets 32,766 to 32,770 straddle that limit; 12-byte and 16-byte load forms are both present. All exact.

**5. Reference-free relocation differential** (`posoff_reloc.py`, `posoff_E4.log`). Program A reads each offset-bearing body's B region at offsets O; program B is a different program (offsets O', bodies at offset 0 unchanged) whose payload holds the same weights at O' and junk where they were; same X and initial C. If addressing is right at every position the outputs are bit-identical, whatever the arithmetic or the scalar rows between. Six seeds each, all identical (6 of 6): body 4 at 4096, the layer-1 tuple, offset at body 4 (`pos:4`), offset at body 12 (relocated to 47,104), every-body offset (6 bodies), and the mixed graph with scalar rows and the layer-1 tuple. Sensitivity: adding 0.5 to one weight inside the test body's region changes the output in 6 of 6; the element just after the region in 0 of 6; the element just before it in 0 of 6 except for the body-4 4096 stream, where it lies in body 1's own region (halves 0 to 2047) and is read legitimately.

**6. The mixed graph over many seeds** (`posoff_mixed.py`, `posoff_E6.py`, `posoff_E6.log`; 60 seeds per program). Against the C-first, truncating reference of the no-offset tool path the no-offset control exceeds 2e-5 on 9 seeds (median 4.8e-7, max 1.0e-3), the body-4 4096 program on 7 (max 1.5e-3) and the layer-1 tuple on 8 (max 1.07e-3); against C-last with truncation all three have median 2.4e-7 and share one worst seed (seed 20: 5.9e-4, 9.5e-4, 4.9e-4), which comes from the compiler-owned scalar rows (GELU exp2 and reciprocal, LayerNorm reductions) that the reference models only to its preregistered tolerance, again with or without offsets.

**7. Apple-compiled streams** (`posoff_apple.py`, `posoff_apple_bytes.py`, `posoff_E8.log`). A Metal kernel of NB sequential `matmul2d` bodies, each C = A(C) x B(slice), A the previous body's fp32 C in place, B half tensors at pointer offsets in the source, `simdgroup_barrier(mem_flags::mem_device)` between bodies, built by the OS compiler: 8 bodies at 4096 bytes, 8 bodies at 4096 + 2048 i, 16 bodies at 4096 (each exact in 6 of 6 runs), and 8 and 12 accumulate bodies at 4096 and 4096 + 2048 i (exact against C-last in 6 of 6, against C-first in 0 of 6). Each body is 8 op12674 tensor loads (16-byte form; displacements 4096, 4128, 4608, 4640, 5120, 5152, 5632, 5664 for the 4096 stream), 4 mixed MMAs (op5100, op5101), 4 op17257 stores and 4 op12710; the loads of every body are byte-identical, only their addresses differ, and `memenc` reproduces 64 of 64, 64 of 64, 128 of 128, 64 of 64 and 96 of 96 of them (`tload16`, and `tload16_end` for the group-end form). Between bodies there is no descriptor setup, barrier instruction or other setup row: Apple's late-body offset loads are the ones tlower emits, and Apple's accumulate order is C-last.

**8. Bytes and metadata that vary with the stream** (`posoff_meta.py`, `posoff_meta2.py`, `posoff_meta.log`, `posoff_meta2.log`). The emitted body for a spec is the same wherever it sits. The authored GPU metadata is 488 bytes for all eleven streams and differs in at most two bytes: byte 208 is the highest register named plus one (84 for the one- to twelve-body streams, 96 for the six-body and 18-body streams) and byte 205 is 1 for code up to 3,368 bytes and 2 from 4,050 bytes on (also at 14,722); the no-offset control and both offset images share the same metadata, and images on either side of the byte-205 change (2,892 and 4,320 bytes) both run exactly, so it is not a cause.

**9. Hypotheses eliminated.** Hidden descriptor or index state set up by the first three bodies: the same offset works at position 1 and at 12, after padding and other histories, and Apple's stream carries no setup between bodies. Base-address state: every body carries its own offset (12 bodies, 17 in the long stream) and the relocation differential is identical. Fragment or address register reuse: tlower re-derives its index registers per body, Apple's stream reuses them, both exact. Scoreboard or lifetime effects: each body's first loads wait on slot 0 and the chain runs through in-place stores and loads of one C for 18 bodies. Immediate-field ownership: the displacement is the same 16-byte form at every position and the base-register form beyond 32,767 is exact. A reset at END or a barrier or a per-stream setup: none exists between bodies of Apple's stream; END occurs once, at the end, in every stream. Payload size and code-size metadata: exact on both sides. Accumulator lifetime: accumulate bodies at positions 2, 3, 5, 6, 8, 9, ... are exact. Timing or concurrency: identical outputs in 233 dispatches and in 193 under load. An environment condition of the reporting overlay: the bundles it left replay exactly here.

**10. Compiler-facing rule (measured domain; production not changed).** Eligible, exact in every test here: a tensor body of the measured bodies (16x32x64 half/half, 16x32x32 float/half with and without accumulate; one simdgroup; ranks (0,1,2) for the first body and (2,1,2) after it; registers R16..R125 with R0..R15 reserved; public bindings 1/2/3 and the `(SR130, SR156)` class) may carry a B byte offset in rank 1 at any body position from 1 to 18 in one program of up to 14,722 bytes, at offsets that are multiples of 2 from 2 to 47,104 bytes in either displacement form, distinct per body, with no setup, padding or reset sequence between bodies, and with the B payload covering offset + K N 2 bytes. Verification rule: a body that accumulates is checked against the C-last model (or against a relocated-weights twin, bit-identical); every fp32 A operand is truncated in the reference, including layer 1's first GEMM; an absolute tolerance is stated relative to the magnitude of the compared values. The runtime tool's `_gemm_mma(..., c=accumulator)` chain (C-first) and `_transformer_layer_weight_offset_reference` (no truncation on the first GEMM) are wrong for this hardware; the second is corrected in the branch's uncommitted working tree, the first is not. Still refused (not measured, or outside the question): A and C offsets at any position; odd byte offsets; regions beyond the payload; other shapes, types, transposes or simdgroup counts; more than 18 bodies; scalar rows other than the measured GELU, residual and LayerNorm rows, whose compiler-owned code differs from its reference by up to 1e-3 on rare inputs with or without offsets. The refusal in `cc._prepare_tensor_composition` (offsets only at body index 1 or the tuple (0, 4096, 8192)) has no hardware basis inside this domain; lifting it is the compiler owner's, after the runtime tool's references are corrected. The established three-body class (0, 4096, 8192) is exact here and untouched.

**11. Corrections.** Recorded here because the files are another session's and I did not edit them. docs/archive/g17-six-layer-minilm-offset-blocker.md (branch `codex/g17-six-layer-minilm`): the statements that a pure-tensor six-body stream with an offset-bearing body 4 "fails" and that the failure "is positional in the longer stream", that "one high-offset body after the preceding three tensor bodies is sufficient" to fail, and that the mixed six-layer stream fails with max_abs about 2e-3, describe reference errors (C-first accumulate order in the pure-tensor comparison, missing fp32-A truncation in the offset-layer reference in the mixed one); the hardware executes those images exactly. docs/archive/g17-transformer-layer-weight-offset.md: the same for its pure six-body diagnostic ("max_abs=4.8828125e-04") and for the 2.01e-3 mismatch of the focused six-region graph (reproduced here only for the mixed bundle with layer 0 at zero offsets; the 2.01e-3 image was not among the bundles I could read).

**12. Failed hypotheses, kept.** (a) I expected a real positional or address effect and built the harness to isolate it; it did not appear in the first harness (an Apple-compiled template entry), in the production overlay, in Apple's compiler or under load, and the reported values then followed from the references. (b) My first attempts to reproduce the mixed value with plausible input generators (seed 1729, uniform (-1, 1) and (-2, 2), zero and tiled payloads, 10 conventions) all gave at most 4.8e-7 because the discriminating variable was the reference, not the data; the value appeared only once the branch's own inputs were replayed with the truncation-free reference. (c) The payload size and the code-size metadata byte were candidate causes, eliminated by exact runs on both sides. (d) Two bugs of mine, found by the checks: my reference used the first K columns of X for a body with lda = K where the body reads contiguous rows (fixed in `posoff_spec.reference`), and a shadowed parameter in the driver; neither reached a reported number.

**Labels.** *Hardware (measured on this H17s):* every exactness, determinism, load, relocation, replay and causal result above; the Apple-compiled streams and their accumulate order. *Apple OS compiler (observed):* the bytes of part 7. *Model:* the C-last and C-first references and the truncation rule (each checked against a hardware stream of its own order, C-first also against the runtime tool's functions). *Read from another session's files:* the bundles (copied unmodified), the overlay scripts in `/tmp`, and the uncommitted diff of its working tree. *Left open:* A and C offsets, other shapes and simdgroup counts, more than 18 bodies, the source of the scalar-row tail (seed 20), the meaning of metadata byte 205 (a size threshold between 3,368 and 4,050 bytes with no effect on results), the 2.01e-3 image.

Artifacts under `results/g17-tensorops-recon-v1/`: `posoff_prereg.md`, `posoff_spec.py`, `posoff.py`, `posoff_prod.py`, `posoff_sweep.py`, `posoff_seeds.py`, `posoff_replay.py`, `posoff_mixed.py`, `posoff_replay_mixed.py`, `posoff_replay_mixed2.py`, `posoff_E6.py`, `posoff_reloc.py`, `posoff_cfirst.py`, `tlower_cf.py`, `posoff_apple.py`, `posoff_apple_bytes.py`, `posoff_meta.py`, `posoff_meta2.py`, `posoff_refcheck.py`, `posoff_det.py`, `posoff_bundle_replay.py`, `posoff_bundle_ref.py`, `posoff_bundle_det.py`, `posoff_ref_swap.py`, `gpu_stress.py`, `posoff_logs.sh`, `posoff_setup.sh`, `posoff_bundles/README.txt`, `posoff_E1.log`, `posoff_replay.log`, `posoff_E2.log`, `posoff_E3.log`, `posoff_E4.log`, `posoff_E5.log`, `posoff_E6.log`, `posoff_E7.log`, `posoff_E8.log`, `posoff_E9.log`, `posoff_E10.log`, `posoff_E11.log`, `posoff_cfirst.log`, `posoff_meta.log`, `posoff_meta2.log`, `posoff_refcheck.log`, `posoff_replay_mixed.log`, `posoff_replay_mixed2.log`, `posoff_bundle_replay.log`, `posoff_bundle_ref.log`, `posoff_bundle_det.log`, `posoff_bundle_det_load.log`, `posoff_ref_swap.log`, `gpu_stress.log`, `gpu_stress2.log`.

## 140. The register-resident GEMM -> scale/quantize -> GEMM chain exists for all four consumer classes, D to B included: the fp8 pack feeds the unpack with no repacking, the row relabeling is the production loads' own convention, and the only lane exchange beyond the amax reductions is a scale-vector transpose in the two transposed modes

**Result.** (1) GEMM1's fp32 accumulators can be block-scaled, quantized and consumed by a second fp8 MMA without leaving registers, whichever way the consumer reads D1: as the left operand (A), as the left operand with the transpose bit (At), as the right operand (B), or as the right operand with the transpose bit (Bt). Hand-authored AIR, one simdgroup, e4m3fn and e5m2 as the intermediate independently, weights in either fp8 format, both scale rules: 48 kernels (2 formats by 4 modes by 6 tile configurations: one block, 2x2, two K2 blocks, a K2 remainder of 48, four and three K1 blocks) by 7 data variants (normal, small and large spans, wide scales, overflow to Inf, partial overflow, code-0 blocks) are bit-exact at D1, at the quantized bytes and scale codes as the registers hold them, and at the final output: 336 of 336 runs for each of four generator variants (register hand-off with the ceil rule and with the floor rule, an in-kernel memory bridge, the other chain order), 1,344 of 1,344 in all; with an unscaled fp8 producer, 48 of 48. (2) The register layout: `op13618` packs two fp32 registers into one 16-bit register half (vector elements 2p and 2p+1 into half p), `op17642` unpacks one half into a 32-bit register of two bf16; in a fused kernel each unpack of a quantized tile reads the half a pack wrote (16 of 16), and the fp8 operand fragment is the D fragment byte for byte. (3) From the measured fragment tables alone, D element (r, c) (hardware row and column) is, in each consumer reading, A: (m, k) = (r, c); At: (m2, k) = (rotl1(c), rotr1(r)); B: (k, n) = (rotr1(r), c); Bt: (k, n) = (c, rotr1(r)), at the identical lane and slot (256 of 256 each; rot is the 4-bit rotation of section 129). The row rotation is the convention the production lowerer already uses: its per-lane row `m = 4 (lane >> 4) + ((lane >> 1) & 3)` with rows m and m + 8 equals `rotr1` of the hardware row for 64 of 64 lane and slot-group pairs. With stage 1 fed by the production loads and D2 labelled as the library labels D, the four chains therefore hold in natural coordinates with no relabeling step (D2 rows m2 = rotl1(c) in At are the D1 columns c once labelled by rotr1); the suite decodes D2 in hardware coordinates, so its natural-coordinate reference relabels the A1 rows in B, At and Bt and the B1 columns in At, and the natural-coordinate chain (the memory bridge's) equals the fused chain bit for bit in 48 of 48 kernels per variant. (4) The two transposed modes need the block scales in the other lane layout: 8 general lane shuffles (`air.simd_shuffle`) per free tile and block, followed by 4 selects in Bt and 6 in At, for both formats, verified against the tables for 0 of 192 wrong lane and slot pairs and on hardware. Nothing else crosses lanes except the amax reductions (row-wise xor masks 1 and 8, column-wise 2, 4 and 16). (5) Negative controls: with stage-1 loads not relabeled, B, At and Bt differ from the natural-coordinate chain in 896 to 1,024 of 1,024 elements and A does not differ; a wrong consumer table (896 to 1,024 of 1,024), scale vectors not transposed (824 to 916), and the wrong block axis (941 to 950 quantized bytes of 1,024) each break exactness. (6) Cost (part 6; e4m3fn, four modes, six grids from 4 to 16 D1 tiles, ns per chain per core; an e5m2 spot check at two grids per mode has identical instruction, register and spill counts and times within 4.1 percent). The fused streaming kernel is spill-free at the (contraction, free) tile grids (2,2), (2,4) and (4,2) (985 to 2,261 instructions, 83 to 118 registers) and spills 4 to 38 stores at (4,4), (6,2) and (8,2), where the stage order spills 11 to 132 and the in-kernel byte bridge 57 to 104. At 64 simdgroups per core it matches the in-kernel byte bridge within 5 percent at 4 D1 tiles and is 1.15 to 1.56 times faster at 8 to 16 tiles; against the sum of the three-kernel bridge it is 1.90 to 2.04 times faster at 4 tiles, 1.14 to 1.61 at 8 tiles and 0.91 to 1.17 at 12 and 16 tiles, where it spills. At one simdgroup per core it is faster than the in-kernel bridge in all 24 rows (1.05 to 2.15 times) and than the three kernels at 4 tiles (1.08 to 1.16), but at 8 tiles it is 1.9 to 2.2 times slower than the three-kernel sum in seven of eight rows (mode B (4,2) is 1.15 times faster), and the stage order is within 1.01 to 1.13 times of the three-kernel sum in five of those seven rows and 2.03 to 2.15 times in the two At rows. (7) The eligibility table and compiler rules are parts 8 and 9. No production file was edited, and no memory bridge is required in any of the four modes.

**1. The construction** (`mxfuse.py`, `mxfuse_host.py`). Stage 1 is the post-scaled fp8 MX GEMM of section 138 (`postfma` arithmetic, byte scale tables, statically unrolled K1 blocks) into D1 = mt1 x nt1 fp32 tiles. The quantizer is the section-138 sequence on the accumulator registers: integer maximum of the absolute bit patterns after flushing fp32 subnormals, `simd_shuffle_xor` reductions, scale code by the ceil rule (or the floor rule with clamp), multiplication by the integer-built factor, `air.convert` (op13618). Stage 2 reads the quantized tiles as its operand registers in one of the four modes (A, At: weights are the right operand from buffer 1; B, Bt: weights are the left operand from buffer 0) and applies the post-scaled epilogue with the consumer-side factor taken from the quantizer's codes in registers and the weights' factor from their byte tables. Two chain orders: `stage` (all D1 tiles first, then all quantization, then stage 2) and `stream` (the free tile index outermost: the D1 tiles of one free index, their quantization, then the stage-2 tiles that depend on them; a stage-2 tile depends on one D1 row of tiles in the row-wise modes A and Bt and on one column in B and At). Variants of the same arithmetic: `bridge='mem'` (quantized tiles and codes stored, `air.wg.barrier`, reloaded inside the kernel), and three kernels (`p1`: stage 1 with D1 stored as fp32; `p2`: D1 loaded, quantized, bytes and codes stored; `p3`: bytes and codes loaded, stage 2) that pass the output buffer along, which is the section-138 memory bridge in this harness (it equals the fused kernel and the reference in 16 of 16 pairs). Debug stores write D1, the quantized bytes and the codes from the same registers that feed stage 2. Buffers: 0 = A-side [stage-1 A1 | weights when A-side], 1 = B-side [B1 | weights when B-side], 2 = out, 3 = iteration counter.

**2. The register layout of the quantize hand-off** (`mxfuse_layout.py`). `op13618`: destination one 16-bit half (for example R36L), sources two 32-bit fp32 registers, format immediate 97 (e4m3fn) or 98 (e5m2); it writes bytes 0 and 1 of the half from the first and second source; four of them convert a `<8 x float>` tile to `<8 x i8>` held in two registers, half p carrying vector elements 2p and 2p+1. `op13620` is the scalar form (destination half, one fp32 source). `op17642`: destination a 32-bit register (two bf16), source one 16-bit half; `op17632` the scalar unpack (fp32 destination, half source); four `op17642` per fp8 fragment build the four registers of the bf16 MMA operand. In the fused kernel (mode B, 2x2 grid, the debug-store build of 999 instructions with 16 packs and 144 unpacks) every quantized fragment's unpacks read the halves that the packs wrote: 16 of 16, with no register move in between, so the data conversion of the hand-off is these packs and unpacks (4 packs and 4 unpacks per quantized tile). The byte in slot j of the D fragment is the byte in slot j of the consumer's fragment (verified by comparing the stored bytes with the reference in D layout: 0 of the stored bytes differ in any run).

**3. The fragment relations** (`mxfuse_maps.py`, `mxfuse_maps.log`). D and A share `pos(r, c)` (lane bits [4] = r3, [3] = c3, [2:1] = (r2, r1), [0] = c2; slot bits [2] = r0, [1:0] = (c1, c0)); B is `pos_b(k, c) = pos(rotl1(k), c)`; a transposed A is `pos_at(k, m)` (lane [4] = k2, [3] = m0, [2:1] = (k1, k0), [0] = m3; slot [2] = k3, [1:0] = (m2, m1)). Inverting each table at the D element's position gives the four readings of the Result, each verified for all 256 elements, and the row rotation in B, At and Bt is a permutation inside each 16-row tile, so the K blocks of 32 (two tiles) contain the same set of rows whatever the labeling. Consumer scales follow the same maps: A carries one code per D1 row (the consumer's row scale, in the D2 row layout as it stands), B one code per D1 column (D2 column layout as it stands), At needs the D1 column codes as D2 row codes (m2 = rotl1(c)) and Bt the D1 row codes as D2 column codes (n = rotr1(r)). The lane sequences that do this were simulated against the tables (0 wrong of 128 and 0 of 64) before and after the hardware runs: Bt takes, for each of its four columns, the source lane `((L & 1) << 4) | (2 j)`, shuffles both slot-group codes and selects by lane bit 3; At shuffles the four column codes from lane `((L >> 4) & 1) | (g << 3)` for each of its two row groups g and selects among them by lane bits 2 and 1 (8 `simd_shuffle` per free tile and block in either mode: Bt shuffles two codes for each of four columns, At four codes for each of two row groups; the selects that follow number 4 and 6). The lowerer's convention (`indexgen.prologue`, `tlower.lower`): `m = 4 (lane >> 4) + ((lane >> 1) & 3)`, `col = (lane & 8) + ((lane & 1) << 2)`, A rows m and m + 8, B rows k = m and m + 8; its logical row of every (lane, slot group) equals rotr1 of the hardware row (64 of 64) and its column the hardware column (128 of 128).

**4. Exactness** (`mxfuse_suite.py`, `mxfuse_suite.log`, `mxfuse_suite_floor.log`, `mxfuse_suite_mem.log`, `mxfuse_suite_stage.log`, `mxfuse_suite_*.json`, `mxfuse_producer.py`, `mxfuse_producer.log`). Reference: fragment positions only (`reference` in `mxfuse_host.py`): D1 from the stage-1 model (`mxrun.model`), the quantizer reference (`mxquant.quantize_ref`) on D1 in hardware coordinates, the consumer's logical operand read out of the quantized fragment through the consumer's own table, stage 2 from the composed model; every comparison bitwise. Tile configurations (mt1, nt1, K1 blocks, weight tiles), row-wise / column-wise modes: (1,2,1,1), (2,2,2,2) with weights in the other fp8 format, (2,4,1,1) or (4,2,1,1) with two K2 blocks, (2,3,1,1) or (3,2,1,1) with a K2 remainder of one 16-tile, (2,2,4,1) and (2,3,3,3) or (3,2,3,3). Data variants by the largest D1 they reach: normal (5e6), small span (5e2), large span (4.6e9), wide scales 96..158 (2.1e19), overflow to Inf (all 32 scale codes 255, all D2 NaN), partial overflow (18 to 23 of 32 blocks code 255, 576 to 736 of 1,024 D2 elements NaN), code-0 blocks (10 of 32 blocks, 540 of 1,024 D1 elements zero: a code-0 scale is a subnormal factor that the fp32 multiply flushes). Result: 336 of 336 exact for each of `ceil`/`reg`/`stream`, `floor`/`reg`/`stream`, `ceil`/`mem`/`stream` and `ceil`/`reg`/`stage`; natural-coordinate equivalence 48 of 48 for each (with the A1 rows loaded in the production convention, and for At the B1 columns relabeled, the fused D2 decoded in hardware coordinates equals the chain computed in natural coordinates); the unscaled producer 48 of 48 (D1 from the fp8 MMA chain straight into fp32, no block scales).

**5. Negative controls** (`mxfuse_negative.py`, `mxfuse_negative.log`). One condition of the eligible class is removed at a time (2x2 grids, both formats): N1, stage-1 loads in natural order where the mode needs the production row convention: the fused D2 differs from the natural-coordinate chain in 1,024 of 1,024 elements for B and At and in 896 of 1,024 for Bt, and 896 of 1,024 for At when only the rows are relabeled; A, which needs no relabeling, differs in 0. N2, the consumer read through the identity table instead of the measured `pos_b` or `pos_at` reading: the hardware differs from that model in 1,024 (B, At) and 896 (Bt) of 1,024 and from the right model in 0. N3, scale vectors not transposed in At and Bt (kernel mutation `notrans`): 832 and 908 of 1,024 elements differ for e4m3fn, 824 and 916 for e5m2. N4, the other block axis on the same D1: 941 to 950 of 1,024 quantized bytes differ, so the axis must be the consumer's contraction axis.

**6. Cost** (`mxfuse_cost.py`, `mxfuse_cost.log`, `mxfuse_table.py`, `mxfuse_cost_B_A_At_Bt_e4m3.json`, `mxfuse_cost_e5m2.log`). Kernels without debug stores, four input sets cycled by the loop counter so nothing is hoisted, two K1 blocks (kb1 = 2) and two weight tiles (nw = 2); instructions decoded from the built kernel, registers and spill stores from the section-134 analyser, time interleaved on the GPU clock (median of 9 rounds) in ns per chain per core (a chain is one execution of stage 1, quantization and stage 2 for one simdgroup) at 1 and at 64 simdgroups per core. Grids are (contraction tiles of the consumer, free tiles); the first two blocks give mode B, e4m3fn, in full, and the third gives the ratios over all four modes (24 rows, `mxfuse_cost_B_A_At_Bt_e4m3.json`); `reg` is the fused streaming kernel, `stage` the fused stage-order kernel, `mem` the in-kernel byte bridge (bytes and codes stored, barrier, reloaded), `3k` the sum of the three kernels `p1 + p2 + p3`. The e5m2 spot check (`mxfuse_cost_e5m2.log`, `mxfuse_cost_B_A_At_Bt_e5m2.json`) covers the grids (2,2) and (4,2) in all four modes.

    mode B, e4m3fn; grid = (contraction tiles, free tiles); instructions / spill stores
    grid   D1 tiles  MMAs      regs   instructions: reg  stage   mem    3k       spill stores: reg  stage  mem
    (2,2)      4     16+8       83                  985    897   1038   1289                  0     0     0
    (2,4)      8     32+16      98                 1927   1726   2024   2467                  0    23     0
    (4,2)      8     32+16     105                 1769   1668   1868   2442                  0    16     2
    (4,4)     16     64+32     126                 3544   3305   3858   4959                 10    59    73
    (6,2)     12     48+24     126                 2580   2369   2829   3654                  7    11    57
    (8,2)     16     64+32     126                 3358   3127   3744   4939                  4    19    92

    ns per chain per core (median of 9 rounds)
    grid     64 simdgroups per core: reg   stage    mem     3k       1 simdgroup per core: reg   stage    mem     3k
    (2,2)                           217.1   218.2   219.7   441.9                            1329    1531    1495    1478
    (2,4)                           351.7   307.4   448.9   488.7                            5928    3414    6890    3173
    (4,2)                           297.9   314.1   368.5   479.7                            2644    3095    5673    3036
    (4,4)                           836.4  1016.3  1131.2   891.6                           12586   12170   13882   12696
    (6,2)                           568.0   592.3   663.4   660.2                            8660    8493   10134    8063
    (8,2)                           777.5   946.8  1206.8   912.7                           11749   11706   13507   12768

    ratio over the four modes (min to max); a value above 1 means the fused streaming kernel is faster
    grid    D1 tiles   64 simdgroups: mem/reg      3k/reg        stage/reg     1 simdgroup: mem/reg     3k/reg        stage/reg
    (2,2)      4      0.95 to 1.01     1.90 to 2.04   0.95 to 1.01     1.12 to 1.27     1.08 to 1.16   1.05 to 1.15  
    (2,4)      8      1.15 to 1.28     1.14 to 1.39   0.85 to 1.03     1.13 to 1.16     0.45 to 0.54   0.53 to 0.96  
    (4,2)      8      1.15 to 1.24     1.18 to 1.61   0.85 to 1.09     1.10 to 2.15     0.46 to 1.15   0.53 to 1.17  
    (4,4)     16      1.35 to 1.45     0.99 to 1.09   1.19 to 1.33     1.05 to 1.16     0.87 to 1.01   0.91 to 0.97  
    (6,2)     12      1.17 to 1.39     0.96 to 1.16   1.02 to 1.35     1.12 to 1.17     0.80 to 0.93   0.91 to 0.98  
    (8,2)     16      1.39 to 1.56     0.91 to 1.17   1.17 to 1.79     1.07 to 1.15     0.91 to 1.09   0.91 to 1.01  

**Reading the tables.** Instructions: at 2x2 the in-kernel byte bridge adds 3 to 5 percent (mode B 1,038 against 985, A 1,034 against 993, At 1,193 against 1,152, Bt 1,135 against 1,100) and 3 to 12 percent over all 24 rows; the three kernels together add 22 to 31 percent at 2x2 and 20 to 47 percent over all rows; the streaming order is 4 to 20 percent longer than the stage order, a difference I did not trace. The hand-off itself is 4 `op13618` packs per D1 tile (16 at 2x2, 32 at (4,2)). Registers and spills: streaming is spill-free through 8 D1 tiles with 83 to 118 registers, where the stage order already spills 10 to 26 stores and the in-kernel bridge 0 to 15. From 12 tiles every variant spills, streaming least (4 to 38 stores against 11 to 132 for the stage order and 57 to 104 for the in-kernel bridge); in mode B the three-kernel bridge's stage-1 kernel never spills while its quantize kernel spills 30 to 46 stores and its stage-2 kernel 8 to 42 at 12 and 16 tiles. Time at 64 simdgroups per core (latency hidden by occupancy): the stage order is up to 15 percent faster than streaming at 8 tiles although it spills (stage over reg 0.85 to 1.09) and 2 to 79 percent slower at 12 and 16 tiles (1.02 to 1.79). Time at one simdgroup per core: the stage order is the faster fused order in seven of the eight 8-tile rows (all but mode B (4,2)), and neither fused order is uniformly ahead of the three kernels. Two 1-simdgroup results are not explained: the 2.0-fold spread among kernels of the same size (mode B (4,2), 2,644 ns, against 5,296 ns for mode A with the same 8 D1 tiles and MMA counts, and 5,928 ns for B (2,4)) and the mode At rows, which are 2.0 to 2.2 times the three-kernel sum in both orders; the instruction counts of the kernels involved differ by less than 20 percent, so the spread is not instruction count, and I did not trace it further. **The cheapest memory-mediated form**, for grids outside the spill-free set or a producer that cannot fuse: at 64 simdgroups per core the in-kernel byte bridge costs nothing beyond 5 percent at 4 D1 tiles and 15 to 28 percent at 8 tiles (mem over reg 1.15 to 1.28), and beyond the spill-free set the three-kernel form is the cheaper memory form (three kernels over fused 0.91 to 1.17, in-kernel bridge over fused 1.17 to 1.56); at one simdgroup per core the in-kernel bridge is 1.87 to 2.56 times slower than the three kernels at 8 tiles, so the three-kernel form is the cheaper memory form there in all eight rows, and the two are within 15 percent at 4 tiles (in-kernel bridge over three kernels 1.00 to 1.15).

**7. What crosses lanes and what does not.** Operand data: nothing, in any mode (the quantized bytes stay in the lane and slot that the accumulator gave them). Scales: for the amax, every mode reduces across lanes (row-wise 2 row groups x 2 xor masks = 4 shuffles per free tile and block; column-wise 4 columns x 3 masks = 12, the in-lane part folding the slots and the tile pair first); after it A and B use the codes where they are, At and Bt exchange 8 general shuffles per free tile and block (the generator computes the transposed vectors once per free tile and block and reuses them for every output tile; the instruction, register and spill counts of the `reg` and `stage` kernels equal those of the earlier generator that recomputed them per output tile in 24 of 24 rows, `mxfuse_cost_v1_uncached.log`, so the compiler had already removed the duplicates). Instructions at the 2x2 grid (contraction 2, free 2, 16 + 8 MMAs): A 993, B 985, At 1,152, Bt 1,100 fused; the transposed modes cost 11 to 17 percent more instructions than the plain ones, the memory bridges 3 to 5 percent (in-kernel) and 22 to 31 percent (three kernels) more.

**8. Eligibility table** (producer: fp32 accumulators D1 in the D layout, from the post-scaled or the unscaled fp8 MMA chain; the layout is the same for every measured MMA form, sections 129 and 132). Every row was run on hardware (parts 4 and 5), positive and negative.

    consumer reading of D1 (contraction)     element map (D1 hardware r, c)      block axis / amax reduction        scale vector for stage 2                           relabel in the suite's natural-coordinate check (library supplies it)   extra lane exchange
    A   left operand (D1 columns)            (m, k) = (r, c)                     row-wise; xor masks 1, 8           row codes as they are (D2 row layout)              none                                           none
    At  left operand, transpose bit (rows)   (m2, k) = (rotl1(c), rotr1(r))      column-wise; xor masks 2, 4, 16    column codes -> D2 row codes                       A1 rows rotr1, B1 columns rotl1                   8 shuffles per free tile and block
    B   right operand (D1 rows)              (k, n) = (rotr1(r), c)              column-wise; xor masks 2, 4, 16    column codes as they are (D2 column layout)        A1 rows rotr1 (production A-row order)          none
    Bt  right operand, transpose bit (cols)  (k, n) = (c, rotr1(r))              row-wise; xor masks 1, 8           row codes -> D2 column codes                       A1 rows rotr1                                  8 shuffles per free tile and block
    common conditions: both fp8 formats for the intermediate and for the weights independently; block scale over 32 elements of the contraction axis (one or two tiles; a last single tile is a remainder block); ceil or floor scale rule; scale code 255 (NaN or Inf in a block) and code 0 propagate as in section 138;
    a free tile is independent of the others, so the free axis can stream; the fused kernel is spill-free at the measured grids (2,2), (2,4) and (4,2) (contraction, free) and spills 4 to 38 stores at (4,4), (6,2) and (8,2), part 6. Note on At: the B1 column relabel listed for At is what the library's D labelling (logical row = rotr1 of the hardware row) already does to D2's rows m2 = rotl1(c); the suite decodes D2 in hardware coordinates and therefore applies it to the inputs instead.

**9. Compiler-facing rules (hand-authored AIR path; production not changed).** (a) The quantize hand-off between two fp8 MMAs is register-resident: quantize the accumulator registers with the section-138 sequence and pass the `air.convert` results to the next `multiply_accumulate` as its fp8 operand; the OS compiler keeps pack and unpack adjacent. (b) The block axis of the quantization is the consumer's contraction axis; take the amax with xor shuffles over the lanes that hold that axis (masks 1 and 8 for rows, 2, 4 and 16 for columns). (c) Load stage-1 rows in the library's order (logical row m at hardware row rotl1(m), as the production lowerer does) and label D2 as the library labels D; then D to A, At, B and Bt need no data movement, and the only exchange is the scale-vector transpose for At and Bt (8 shuffles per free tile and block). (d) Stream the free tile axis (one row of D1 tiles for the row-wise modes, one column for the column-wise ones) so that D1 does not have to be resident: this lifts the spill threshold from 4 to 8 D1 tiles. (e) At one simdgroup per core and 8 D1 tiles the streaming chain is slower than the three-kernel sum in seven of eight rows and the stage order recovers five of them; fuse there only when several simdgroups share the core, or use the three kernels. (f) Refuse or bridge through memory: grids beyond the measured spill-free set (the measured (4,4), (6,2) and (8,2) spill), an int8 producer (its int32 D1 needs the requantization sequence, not this hand-off), a GEMM tile shared by more than one simdgroup (the cross-simdgroup hand-off is memory, section 133), and fp4 or fp6. Not measured here: a stage 1 emitted by `tlower` (only its convention is checked against the tables), a runtime-loop stage 1, and bf16 or fp16 producers (same D layout).

**10. Correction to section 138, recorded in place.** *Part 7 says that as the right operand the chain "needs the B map (the two lanes-by-slots maps differ), not built here", and the decision table and the Labels list "a register-resident fused GEMM, quantize, GEMM kernel" and "a D-to-B relayout for the right operand" as open. The maps differ only by the row rotation of section 129, which the production loads (and, for At, the D labelling) already apply; the fused kernel is built and exact in all four consumer classes; no relayout exists.*

**11. Failed hypotheses, kept.** (a) I expected D to B to need a lane exchange or a K-permuted weight matrix; the tables show a row-bit rotation at the same lane and slot, and the library's own row order is that rotation. (b) I expected the fused chain to beat the bridges everywhere; at one simdgroup per core the streaming order is slower than the three-kernel sum for eight-tile grids (up to 2.2 times slower than the sum at (2,4) and (4,2)), and once the fused kernel spills its advantage over the three kernels disappears (0.91 to 1.17 times at 12 or more D1 tiles at 64 simdgroups per core). (c) My first generator computed all D1 tiles first and spilled at eight tiles; streaming the free axis removed that. (d) A first checker of the column-to-row scale transpose mislabeled a lane bit and reported 32 wrong pairs; the hardware was exact all along and the simulation of the kernel's own shuffle sequence reports 0. (e) Caching the transposed scale vectors per free tile changed no instruction, register or spill count (24 of 24 rows): the compiler had already removed the duplicates.

**Labels.** *Hardware (measured on this H17s):* every exactness, control and cost result. *Model (tables, checked on hardware):* the fragment relations, the composed references, the row convention of the production lowerer (64 of 64 against the tables; no `tlower`-emitted stage 1 was run). *Left open:* a GEMM tile shared by several simdgroups (the 64-simdgroup timings run independent chains), a runtime-loop stage 1, int8 producers, bf16 and fp16 producers on hardware, fp4 and fp6, a `tlower`-emitted stage 1 feeding the fused hand-off, and scheduling that interleaves free tiles (which might recover the low-occupancy speed of the stage order within the streaming register budget; not tried).

*Update, sections 141 to 144: a runtime-loop stage 1 (section 141), int8, bf16 and fp16 producers on hardware in the A reading (section 141) and a tile shared by several simdgroups (section 143) are no longer open; a `tlower`-emitted stage 1, the At, B and Bt readings for the classes other than fp8, and the hidden-chunk loop in the transposed readings remain open.*

Artifacts under `results/g17-tensorops-recon-v1/`: `mxfuse.py`, `mxfuse_host.py`, `mxfuse_suite.py`, `mxfuse_negative.py`, `mxfuse_producer.py`, `mxfuse_cost.py`, `mxfuse_table.py`, `mxfuse_maps.py`, `mxfuse_layout.py`, `mxfuse_suite.log`, `mxfuse_suite_floor.log`, `mxfuse_suite_mem.log`, `mxfuse_suite_stage.log`, `mxfuse_suite_ceil_reg_stream.json`, `mxfuse_suite_floor_reg_stream.json`, `mxfuse_suite_ceil_mem_stream.json`, `mxfuse_suite_ceil_reg_stage.json`, `mxfuse_negative.log`, `mxfuse_negative.json`, `mxfuse_producer.log`, `mxfuse_cost.log`, `mxfuse_cost_v1_uncached.log`, `mxfuse_cost_B_A_At_Bt_e4m3.json`, `mxfuse_cost_e5m2.log`, `mxfuse_cost_B_A_At_Bt_e5m2.json`, `mxfuse_maps.log`, `mxfuse_maps.json`.

## 141. The fused GEMM, activation, quantize, GEMM pipeline runs as a runtime-loop microkernel that reproduces the Section 140 identities at every boundary: nested runtime loops with allocator-chosen registers are bit-exact in all four consumer readings, in fp8, int8, bf16 and fp16, for a chunked FFN, a gated FFN, a three-GEMM chain and an attention flow, and the runtime K loop is also the only form of stage 1 that scales

**Result.** (1) Section 140 was a statically unrolled chain of at most a few K blocks. Here the same arithmetic is a reusable microkernel (`mxpipe.py`, generator; `mxpipe_host.py`, packing, references, comparison): a runtime loop over K1 blocks of 32 with loop-carried fp32 accumulators, inside a runtime loop over hidden chunks that carries the second GEMM's accumulators, inside the timing loop; its trip counts are control words, so one compiled kernel serves K1 from 32 to 4,112 and 1 to 16 chunks. Per chunk it does: GEMM1 with block scaling (`postfma`), bias and activation in fp32, block-32 amax and scale code, `air.convert` to fp8, and the consumer GEMM with the quantized registers as its operand (D to A). The other forms are the same code with one part changed: gating (two stage-1 accumulator sets, `act(G) * U`), a three-GEMM chain without a chunk loop (K loop, activation, quantize, GEMM, activation, quantize, GEMM), and an attention flow (S = Q K_j^T per key block, online softmax with hardware `exp2`, P quantized as the left operand of P V_j, O rescaled by the running-max factor). (2) Every boundary is compared bitwise with the composed reference on hardware: D1 per chunk, the post-activation values, the quantized bytes and the scale codes as the registers hold them, the middle stage of the chain, D2 and the final output. 1,258 of 1,258 runs are exact (the first version of this section reported 1,289 of 1,289; 99 of its 460 FFN-family runs compared a reference that was mostly NaN with the hardware's NaN, which proves nothing, so the suites were rerun on data inside the finite domain of the hardware tanh, part 10 (f)): 429 in the FFN-family suite (61 kernels: loop counts, tile grids, K remainders, the four fp8 format pairs, five activations with and without bias, gating, residual and RMS-norm epilogues, the three-GEMM chain, int8, bf16 and fp16), 54 for the attention flow (11 kernels), 216 for the multi-simdgroup strategies (108 kernels, section 143), 416 for the four consumer readings A, At, B and Bt with a runtime K loop plus 112 repeat runs that alternate two input sets, 7 at long K (2,048, 4,096 and 4,112 elements), and 24 repeat runs of whole pipelines over two input sets holding different data (12 kernels: the three-GEMM chain in fp8, int8 and bf16 with residual, norm and K remainder, the gated FFN, the FFN with gelu and norm, attention in fp8 and bf16, and the multi-simdgroup kernels with threadgroup and device memory). The suites compare 585,728 D1 elements, 698,880 quantized bytes, 69,440 scale codes and 398,848 D2 elements in the FFN-family suite alone, and 424 of its 429 runs have a reference whose D2, Y and Z are at least 90 percent finite (the other five are deliberate overflow or wide-scale variants, part 10 (f)). (3) The four consumer readings of section 140 hold in a runtime loop: 416 of 416 (e4m3fn and e5m2, one to 32 K1 blocks, K2 remainders, all seven data variants including overflow to Inf, partial overflow and code-0 blocks) and 112 of 112 repeat runs, where the timing loop runs 2 or 3 iterations over two input sets that hold different data, so a stale accumulator or a register carried across iterations would show in the last iteration's output. (4) Compiled loop structure (part 6): in every single-simdgroup kernel each loop-carried fp32 accumulator is updated by an fma whose destination is its own source register (16 of 16 in the K loop, 32 of 32 and 64 of 64 in the chunk loop, 48 of 48 with gating), and in every single-simdgroup chunk loop each unpack of a quantized tile reads the half a pack wrote (8 of 8, 16 of 16 with two token tiles). Register numbers are the compiler's; the identities did not depend on them in any of the 1,258 runs. (5) The runtime K loop is not only equivalent to the section-140 unrolled stage 1, it is the only form that scales: at 8 blocks it is 1.3 to 1.8 times faster at 64 simdgroups per core and 2.1 to 2.6 times at one, at 32 blocks 5.9 to 7.2 times and 3.0 to 3.4 times, and the unrolled kernels spill (24 to 62 stores at 8 blocks, 786 to 923 at 32) where the loops do not. (6) Hardware facts found on the way (part 8): `air.fast_tanh` returns NaN for arguments of 44.36 and above (a tanh-based SiLU is NaN from x = 88.73, a tanh-based GELU from x = 10.06), `llvm.maxnum` with a NaN operand returns the other operand (relu of NaN is 0), `air.fast_divide` and `air.erf` do not lower on this toolchain, and exp2, log2 and rsqrt are single instructions (`air.fast_sqrt` is two: a second form of the reciprocal square root and a multiply). (7) Negative controls: other data, a missing K-remainder block, one block or one chunk fewer than the data, and the other scale rule each fail at 203 to 1,536 elements (part 9). Production sources were not edited.

**1. The construction** (`mxpipe.py`). One generator, `gen_ffn(cfg)`, emits hand-authored AIR with a small structured-loop emitter (`Kern.loop`: do-while loops whose phi nodes are patched after the body is known, so nested loops and values that leave a loop are ordinary Python). Buffers as in section 140: 0 = A-side [activation X (block-major fragments and byte scale tables) | bias tables, residual tiles, gamma], 1 = B-side [W1 chunks | up-projection chunks when gated | W2 chunks], 2 = out (D2, Y, then per-chunk debug regions D1, Z, quantized bytes, codes), 3 = control words ([2] timing iterations, [4] K1 blocks, [5] chunks). Every region is replicated `sets` times and the timing loop selects `it & (sets - 1)`, so no load can be hoisted out of the timing loop; the packing is `mxrun.pack` with a block capacity `nb` fixed at compile time and a trip count set at run time (table offsets use the capacity, addresses use the block index). Operand classes: fp8 (e4m3fn and e5m2 for the intermediate and for the weights independently, block-scaled by `postfma`: two MMAs into a zero accumulator, then `acc = fma(P, fa * fb, acc)`), int8 (`widening_multiply_accumulate.s.s`, int32 block sum, `sitofp`, the same fma; quantizer `fmul`, `rint`, `fptosi`), bf16 and fp16 (fp32 accumulation chained through the MMA, no scales, `fptrunc` at the boundary). The K remainder is one peeled block after the loop with a single MMA (K = 32 kb + 16); a hidden width that is an odd number of tiles gives a K2 remainder block in the consumer's contraction. Debug stores can be removed (`dbg=False`) and the analyser and timing runs use those builds.

**2. What is compared and against what** (`mxpipe_host.py`). The reference is numpy on fragment positions only: the stage-1 model of section 138 (`mxrun.model`, or the int8 and 16-bit chains built on `numhw.mma_hw`), IEEE fp32 with the flush rule for the surrounding operations, the section-138 quantizer (`mxquant.quantize_ref`) on the post-activation values in hardware coordinates, and stage 2 as the composed model with the previous accumulator as its initial value (chunk c adds to the D2 that chunk c - 1 left, in the kernel's order). Elementwise stages that use only IEEE operations (relu, relu2, hardswish, the residual add, the norm's fma and add trees in the kernel's lane order) have exact numpy models. Stages that use a hardware transcendental (silu and gelu through `air.fast_tanh`, `air.fast_rsqrt` in the norm, `exp2` and a division in attention) take the isolated-hardware result of the same emitter: `mxpipe.gen_unary` runs the identical instruction sequence on the exact fp32 inputs of the stage (lane-wise, in the fragment order of the fused kernel), so the fused kernel is compared with the hardware's own transcendental, not with a library value. Everything downstream of such a stage (the quantizer, the second GEMM, the accumulation) is exact numpy.

**3. Suite results** (`mxpipe_suite.py`, `mxpipe_suite.log`, `mxpipe_suite_loops_tail_formats_acts_gated_epi_mlp3_dtypes.json`; four data variants per case: normal, small span, wide scales 96..150, large span; a case with silu or gelu uses three variants scaled into the finite domain of the hardware tanh instead, `auto` with a pre-activation standard deviation of 2 (0.4 for gelu), `auto_small` (0.25) and `auto_large` (6, gelu 0.8), and every row records the finite fraction of the reference at D1, Z, D2 and Y). Runs exact by group: loop counts and tile grids 48 of 48 (mt 1 and 2, 1 to 4 hidden tiles per chunk, 1 to 3 output tiles, K1 32 to 1,024, 1 to 4 chunks), K remainder with the tail block 24 of 24 (K1 = 48, 80, 176, 208, 400), format pairs 32 of 32 (e4m3/e4m3, e5m2/e5m2, e4m3 with e5m2 weights, e5m2 with e4m3 weights), activations 60 of 60 (relu, relu2, hardswish, silu, gelu; each with and without a per-column bias), gated 34 of 34, epilogues 25 of 25 (residual; residual and RMS norm across up to 4 output tiles), three-GEMM chain 56 of 56 (both boundaries in registers; mt 1 and 2, first hidden width 2 to 5 tiles, second 3 to 4, output 2 to 3, activations at both boundaries, K remainder, mixed formats), int8, bf16 and fp16 150 of 150 (FFN with three activations, K remainder, gating, norm, and the chain). Attention (`mxattn_suite.py`, `mxattn_suite.log`): 54 of 54, reference finite fraction 1.0 in every run, over d = 64 to 144, 2 to 8 key blocks per run of 32 to 64 keys, mt 1 and 2, e4m3, e5m2 with e4m3 values, bf16, fp16, int8, a K remainder in the head dimension, and three softmax scales. Long K (`mxpipe_longk.py`, `mxpipe_longk.log`): K1 = 2,048 and 4,096 (64 and 128 blocks) in e4m3 with silu, K1 = 4,096 in e5m2, bf16 and int8, and K1 = 4,112 and 2,064 with the remainder tile, 7 of 7 (finite fraction 1.0 at D2 and Y in every run).

**4. The consumer readings in runtime loops** (`mxfuse.py` with `kloop=True`, `mxloop_modes_suite.py`, `mxloop_modes_suite.log`). The section-140 generator gained a runtime K loop for stage 1 (all tiles of one free index carried together, loads addressed by the block index, trip count from a control word) and is otherwise unchanged: quantizer, scale-vector transposes, stage 2 and the streaming order are the section-140 code. Configurations: the six of section 140 (one block, 2x2 with weights in the other format, two K2 blocks, K2 remainder 48, four K1 blocks, three K1 blocks with a K2 remainder), 8 K1 blocks and 32 K1 blocks, for A, At, B and Bt, e4m3fn and e5m2, seven data variants (32-block kernels: three): 416 of 416 exact at D1, quantized bytes (499,712 compared), scale codes (108,032) and D2 (314,368 elements). Repeat test: a kernel with two input sets holding different data, run for 3 iterations (sets 0, 1, 0) and for 2 iterations (sets 0, 1); the output equals the reference of the last iteration's set in 112 of 112 runs (7 configurations by 8 format-mode pairs by 2). The same repeat test on whole pipelines (`mxpipe_repeat.py`, `mxpipe_repeat.log`; a kernel with two input sets of different data, 3 iterations over sets 0, 1, 0 and 2 iterations over sets 0, 1, every boundary of the last iteration compared with the reference of that set) is exact in 24 of 24 runs (finite fraction of D2 and Y at least 0.999 in 23 of them and 0.875 in the chain with silu, norm and a K remainder) for the three-GEMM chain (fp8 with silu, norm and a K remainder; fp8 relu with bias; bf16 silu; int8 relu with residual), the gated FFN, the FFN with gelu, norm and a K remainder, bf16 FFN, attention in fp8 and bf16, and the four-simdgroup and two-simdgroup shared-buffer kernels, so no accumulator, loop counter or shared-buffer content leaks across iterations of the timing loop. The readings that need the scale-vector transpose (At and Bt, 8 shuffles per free tile and block) and the row-rotated stage-1 loads (B, At, Bt) therefore need nothing that depends on the loop or on the registers the allocator picked.

**5. Long K, remainders and multiple tiles.** The K loop runs 1 to 128 blocks and the chunk loop 1 to 16 (multiple of the simdgroup count in the multi-simdgroup kernels); a remainder of 16 in K uses the peeled single-MMA block (data packed with the second k-tile zero, the reference pads with zeros, both give the same bits); hidden widths of 16 to 80 (odd tile counts are K2 remainders in stage 2); mt = 1 and 2 token tiles (a kernel with mt = 2, two hidden and two output tiles and a silu activation already touches 126 registers and spills 15 stores, the results stay exact); 1 to 4 output tiles per simdgroup in the exactness suites (up to 8 in the cost runs of section 142); and three chained GEMMs with first hidden widths of 2 to 5 tiles. The cost of these shapes is section 142.

**6. The compiled loops** (`mxpipe_regs.py`, `mxpipe_regs_all.py`, `mxpipe_regs.log`, `mxpipe_regs.json`). Loops are found from the back-edge branches (decoder facts used: `op458` is a branch by a signed byte displacement whose target is an instruction boundary, `op578` the conditional, `op10370` the compare into FLAG0; the fp32 fma is `op2190` with sources (product, factor, accumulator), the MMA `op5106` with and `op5107` without an accumulator source, the fp8 pack `op13618`, the unpack `op17642`, `op590` a 16-bit register move). For the fp8 FFN with silu (mt 1, 2 hidden tiles, 2 output tiles): K loop 165 instructions, 4 MMAs, 16 fp32 fma of which 16 have destination equal to the accumulator source; chunk loop 573 instructions, 8 MMAs, 32 fma, 32 in place, 8 packs and 48 unpacks, 8 of the 8 pack halves read directly by an unpack; the timing loop encloses both. (The loop finder is checked over every kernel compiled in this lane, 2,109 kernels: all 4,940 back-edge displacements land on an instruction boundary.) The same counts hold for e5m2 (8 of 8), mt = 2 (64 of 64 fma in place, 16 of 16 pack halves read directly), gating (48 of 48, 8 of 8), the attention flow (33 of 34 in place, 8 of 8) and int8 (32 of 32; the int8 boundary has no pack, the quantizer produces the bytes directly), and for bf16 and fp16, whose loops hold only chained MMAs (no fp32 fma; the MMA accumulator register is its own destination). The loops contain 9 to 14 16-bit register moves (`op590`) per chunk iteration in the fp8 kernels and none in bf16 and fp16; I did not assign them to a role. In a kernel that applies the stage 2 of several chunks in sequence (the shared rounds of section 143) the destinations rotate among the accumulator registers (37 of 80 fma in place in the threadgroup-memory kernel, 36 of 80 in the device-memory kernel) and 4 of 8 pack halves (threadgroup memory) or 8 of 8 (device memory) are read directly by an unpack, the others reaching the unpack through the shared buffer. In the K-remainder kernels 39 of 48 fma in the chunk loop are in place; the other nine are the peeled tail block's updates of D1, which is not loop-carried. The unpack of a quantized tile is emitted once per tile and reused by every output tile that reads it, as in section 140.

**7. Runtime loop against the unrolled generator** (`mxloop_modes_cost.py`, `mxloop_modes_cost.log`; e4m3fn, no debug stores, four input sets, ns per chain per core). The unrolled kernels are the section-140 generator (tile by tile, loads repeated per tile), not an optimised unroll. (mt1, nt1, K1 blocks, weight tiles): (2,2,8,2): unrolled 2,507 to 2,763 instructions, 126 registers, 24 to 43 spill stores, 664 to 717 ns (64 SG) and 9,085 to 9,807 ns (1 SG); runtime loop 832 to 1,023 instructions, 68 to 80 registers, no spills, 484 to 534 ns and 4,210 to 4,518 ns (0.70 to 0.75 and 0.45 to 0.47 of the unrolled time). (2,2,32,2): unrolled 10,847 to 11,444 instructions with 786 to 923 spill stores, 11,574 to 13,150 ns and 59,628 to 62,055 ns; runtime loop 859 to 1,050 instructions, 70 to 81 registers, no spills, 1,796 to 1,951 ns and 17,725 to 19,926 ns (0.14 to 0.17 and 0.30 to 0.33). (2,4,8,1): 4,640 to 5,131 instructions and 52 to 62 spills against 1,300 to 1,739 and none, 0.55 to 0.58 and 0.39 to 0.47 of the time. In all twelve loop kernels the K loop's fma are 100 percent in place (16 of 16, or 32 of 32 when two free tiles share one loop).

**8. What the math units do** (`mathprobe.py`, `mathprobe.log`, `mxpipe_tanh_domain.log`). One lane-wise kernel per candidate, decoded: `llvm.exp2`, `air.exp2` and `air.fast_exp2` compile to one instruction (`op1272`), `llvm.log2` to one (`op2570`), `air.fast_rsqrt` to one (`op3850`), `air.fast_sqrt` to two (`op3978`, which computes the reciprocal square root like `op3850`, followed by a multiply: sqrt(x) = x rsqrt(x); a peer session's hand-decoded pairs give op3978(4) = 0.5, and the same session measured why two opcodes carry the name: at a flushed-to-zero input op3850 returns +inf, the IEEE reciprocal square root of 0, and op3978 returns 1.0, which is what lets `x * rsqrt(x)` give sqrt(0) = 0 instead of 0 * inf = NaN; a user's rsqrt lowers to op3850 alone and writing the product by hand also gets op3850, so the choice belongs to the lowering; the sqrt witnesses are that session's, commits ede9116c and e84f48a9 on `claude/g17-isa-cartography`, and this lane's own evidence is only the compiled instruction count), `air.fast_tanh` to five, `llvm.exp` to two (a multiply and `op1272`); kernel totals (a lane-wise load, the operation, a store; 34 for a single-instruction operation) are 35 for `llvm.exp`, 39 for `air.fast_tanh`, 43 for `fdiv`, 46 for `air.rsqrt`, 47 for `air.sqrt` and 84 for `air.tanh`; `air.fast_divide` and `air.erf` fail in the native compiler ("Encountered unlowered function call"). `air.fast_tanh(t)` is NaN for t >= 44.364 and finite for every smaller argument and for every negative one (-1 in the limit); this is the exp overflow of a tanh built from `exp(2 t)` (2 t > 88.72 = ln of the largest fp32), so a tanh-based silu is NaN from x = 88.7277 (finite at 88.7228) and a tanh-based gelu from x = 10.0623; a compiler that lowers silu or gelu through `air.fast_tanh` needs a clamp of the argument (or `air.tanh`, 45 instructions longer than `air.fast_tanh`). `llvm.maxnum(x, 0)` returns 0 for a NaN x, so relu built on it maps NaN to 0 where numpy's `maximum` returns NaN (the reference uses `fmax`). An int8 quantizer that converts with `fptosi` and `trunc` gives unspecified bytes for the elements of a block whose scale code is 255 (a NaN or Inf in the block): 4,992 elements of the suite are in such blocks and are excluded from the byte comparison, the codes and everything else are compared.

**9. Negative controls** (`mxpipe_negative.py`, `mxpipe_negative.log`, e4m3fn relu chain, K1 = 256, 3 chunks, mismatching elements of D1 / Z / quantized bytes / codes / D2 / Y): the same data compared with the reference of other data 1,536 / 1,133 / 1,127 / 152 / 512 / 512; a kernel without the tail block on data with a 16-wide K remainder 921 / 468 / 92 / 0 / 460 / 460; one K block fewer than the data 1,475 / 716 / 220 / 16 / 508 / 508; one chunk fewer than the data 512 / 512 / 512 / 64 / 512 / 512; a reference with the floor rule against a kernel with the ceil rule 0 / 0 / 203 / 52 / 255 / 255. The matching cases are 0 in every column.

**10. Failed hypotheses and corrections, kept.** (a) I expected the pack to unpack forwarding of section 140 to depend on the schedule and to break inside loops; it holds in every single-simdgroup loop body (8 of 8, 16 of 16); it is partial only where the quantized tiles are also stored to and reloaded from the shared buffer (4 of 8 in the threadgroup-memory kernel). (b) My first int8 gated runs failed at 11 to 137 quantized bytes with equal codes and equal Z; the cause was the hardware tanh returning NaN for large arguments (the int8 data span +-127 makes the pre-activation large) and an int8 conversion that is unspecified for code-255 blocks, not an arithmetic error; the int8 data were rescaled by 2^-18 and the special blocks excluded, which is recorded in `mxpipe_host.py`. (c) The first relu reference used `np.maximum` and failed on hardware for NaN inputs (1,024 of 1,024 elements at the second boundary in bf16 and fp16 large-span runs); `llvm.maxnum` semantics fixed it. (d) The attention kernel name did not include the softmax scale, so three configurations shared one directory; the suite rebuilds each configuration immediately before running its cases, so no result mixed kernels, and the name now carries the scale (the suite was rerun after the rename). (e) I first expected the fused stage 1 to need unrolling by the block count for scheduling; the unrolled form is the slow and spilling one (part 7). (f) The first version of the suites counted equality of NaN with NaN as agreement. The four fixed data variants put the pre-activation of silu and gelu kernels outside the finite domain of `air.fast_tanh` (part 8) for most fp8 data, so the reference and the hardware were both NaN over most of the tensor. Found while building the layer-scale runs of section 145, where a first run was exact only because both sides were NaN. The audit added a finite-fraction column to every suite and reran the first suite's original variants (`MXSUITE_OLD=1 python3 mxpipe_suite.py`, `mxpipe_suite_..._olddata.json`): of its 460 runs 99 had a reference at least 10 percent non-finite at D2, Y or Z (74 below 10 percent finite, 62 with none at one of the three; 95 of the 99 in silu or gelu kernels, 62 in e4m3 fp8, the rest e5m2, int8, bf16 and fp16; all 99 compared equal). The corrected suite scales those cases into the finite domain and reports 429 of 429 exact, 424 of them with a reference at least 90 percent finite; the five others are a silu-gelu chain with bias and norm at finite fraction 0.625, a hardswish/relu2 chain at wide scales (0.586 and 0.266) and two fp16 large-span runs whose overflow to Inf is the point of the variant (0.0). The fp8 evidence for silu and gelu in the first version therefore stands only where the rerun reproduces it, and the timing rows of sections 142 and 143 are unaffected because no kernel branches on data: over the 1,643 compiled kernels of this lane the only branch instructions the decoder finds are `op458` back-edges (4,273) and one `op578` conditional per loop (4,273), and no forward branch.

**Labels.** *Hardware (measured on this H17s):* every exactness count, the compiled-loop analysis, the loop-against-unrolled costs, the math-unit facts. *Model (numpy, checked on hardware bit for bit):* the composed references; the transcendental stages use the isolated hardware result of the same emitter and are checked in the fused kernel by equality with it, not with an independent formula. *Not tested:* stage-1 loops emitted by `tlower`, an int8 block containing NaN or Inf (unspecified), fp4 and fp6, the At, B and Bt readings inside the hidden-chunk loop (they are tested with the K loop and the repeated timing loop; the chunk loop itself is mode A), bias and gating in the transposed readings.

Artifacts under `results/g17-tensorops-recon-v1/`: `mxpipe.py`, `mxpipe_host.py`, `mxpipe_suite.py`, `mxpipe_suite.log`, `mxpipe_suite_loops_tail_formats_acts_gated_epi_mlp3_dtypes.json`, `mxpipe_negative.py`, `mxpipe_negative.log`, `mxpipe_negative.json`, `mxpipe_longk.py`, `mxpipe_longk.log`, `mxpipe_longk.json`, `mxpipe_repeat.py`, `mxpipe_repeat.log`, `mxpipe_repeat.json`, `mxattn_suite.py`, `mxattn_suite.log`, `mxattn_suite.json`, `mxpipe_sg.py`, `mxpipe_sg_exact.log`, `mxpipe_sg_exact.json`, `mxfuse.py` (runtime-loop option), `mxloop_modes_suite.py`, `mxloop_modes_suite.log`, `mxloop_modes_suite.json`, `mxloop_modes_cost.py`, `mxloop_modes_cost.log`, `mxpipe_regs.py`, `mxpipe_regs_all.py`, `mxpipe_regs.log`, `mxpipe_regs.json`, `mathprobe.py`, `mathprobe.log`, `mxpipe_tanh_domain.log`.

## 142. Fusion depth and the break points: the chain stays register-resident for two boundaries and three GEMMs, it is the fastest form (5 to 20 percent at high occupancy) only while the fused kernel does not spill, block scaling is a quarter of the fp8 time while the amax, the activation and the softmax exponential are nearly free, and the memory cut with the best measured cost is the quantize boundary

**Result.** (1) Depth. Register-resident chains of two consecutive tensor boundaries run and are bit-exact (section 141): GEMM, activation, quantize, GEMM, activation, quantize, GEMM (three tensor regions, two register-resident boundaries), the gated FFN (gate and up GEMMs, `act(G) * U`, quantize, down GEMM, in a runtime chunk loop with the down GEMM accumulating across chunks), and the attention flow (QK^T, online softmax, quantize P, PV). The fused three-GEMM kernel with one token tile is spill-free for first and second hidden widths up to 4 tiles with 2 to 4 output tiles per simdgroup, and for a 2-tile first width with a 4-tile second width and 8 output tiles (73 to 126 registers); a 6-tile second width spills 5 (bf16), 30 (e4m3fn) and 63 (int8) stores and two token tiles with 4-tile widths spill 27 to 62 (part 6). (2) Cost of the boundaries against memory-mediated execution, e4m3fn FFN with silu, ns per chain per core, medians of interleaved rounds, ratios within a run repeat to 1.7 percentage points (part 1): when the fused kernel is spill-free (four shapes) it is faster than the in-kernel byte bridge by 0 to 8 percent, than a cut after the quantize (kernel 1 = stage 1, activation, quantize; kernel 2 = stage 2) by 5 to 10 percent, than a cut before it (D1 stored as fp32) by 6 to 8 percent and than three kernels by 8 to 20 percent at 64 simdgroups per core; when it spills 15 to 76 stores (four shapes) the ratios are 0.98 to 1.07, 0.95 to 1.00, 0.95 to 0.98 and 0.99 to 1.06, that is the two cuts need 0 to 5 percent less time; at one simdgroup per core the cuts need 1 to 17 percent less time than the fused kernel in the spilling shapes and 1 to 7 percent less at long K (1,024 blocks or more) even when it does not spill. (3) Where the time goes (ablations of the same kernel, part 3): the MMA loop alone runs at 5.3 to 6.4 ns per MMA per core in fp8, bf16 and fp16 and 2.7 to 3.7 in int8; block scaling (the table loads, decode, fmul and fma per K block) is 25 percent of the fused fp8 time (1.9 ns per MMA, 27 percent at 64-element K) and 44 to 49 percent of the fused int8 time; the block amax and scale code cost 0 to 5 percent in fp8 and 2 to 10 percent in int8, the silu activation 0 to 8 percent in fp8 and 2 to 11 percent in int8 (both largest at K1 = 64), and removing the hardware exp2 of the attention flow changes the time by less than 3 percent in every class; in bf16 and fp16 the fused attention kernel is within 1 to 5 percent of the MMA-only kernel: the scalar work overlaps the MMAs. (4) Traffic: the device bytes requested per chain grow from 43 KiB (fused) to 84 KiB (three kernels) at K1 = 64 and by 2 to 9 percent at K1 >= 1,024, which coincides with 16 to 20 percent more time for three kernels at K1 = 64 and 8 to 10 percent at K1 >= 1,024 in these cache-resident spill-free shapes (section 143 shows where the working set leaves the cache). (5) Break-point rules (part 7): fuse the whole chain when the fused kernel is spill-free; otherwise cut after the quantize boundary (fp8 bytes and codes cross memory, 1 byte per element instead of 4) or before it, the two cuts are within 5 percent of each other; never unroll K (section 141); fuse the gated FFN's two stage-1 accumulations. (6) Attention: the fused flow costs 7.9 to 9.1 ns per MMA in fp8 (7.2 to 8.1 in int8) and 5.8 to 6.3 in bf16 and fp16; bridging P through device memory inside the loop adds 3 to 8 percent at 64 simdgroups per core and 1 to 17 percent at one. (7) Register budgets (part 6, compile only): the fused FFN with silu is spill-free up to 6 output tiles per simdgroup for fp8 with 2 hidden tiles per chunk, 8 for bf16 and 4 for int8, and only 1 to 3 when the chunk is 4 hidden tiles wide or two token tiles are carried; the stage-1-plus-quantize cut never spills in the tested grids (56 to 109 registers) and the stage-2-only cut is spill-free up to 3 to 10 output tiles depending on class and chunk shape. The spill counts are the OS compiler's schedule for the IR that I emit; the register domain of the hardware is R0..R125 (sections 134 and 135). No production file was edited.

**1. Method and noise** (`mxpipe_cost.py`, the `mxpipe_cost_*.log` files). Kernels without debug stores, four input sets cycled by the timing iteration, instructions decoded from the built kernel, registers touched and spill stores from the section-134 analyser, time on the GPU clock with interleaved kernels (`regprobe_multi`, median of 9 rounds) at 1 and at 64 simdgroups per core, reported as ns per chain per core (a chain is one execution of the whole pipeline for one simdgroup). Five repeats of two shapes (`mxpipe_cost_noise_1.log` to `mxpipe_cost_noise_5.log`): the fused kernel's absolute time varies by 3.1 percent (4,081 to 3,955 ns and 1,476 to 1,431 ns), and the ratios of the other variants to it vary by at most 1.7 percentage points at 64 simdgroups (0.7 at one); between separate sweeps at different times the fused kernel's absolute time moved by up to 10 percent (8.2 and 7.7 ns per MMA for the same shape), so ratios are taken inside one run and differences below 3 percent are not claimed. Variants of the same arithmetic (exact against each other in the suite): `fused`; `mem` (the quantized tiles and codes stored, a workgroup barrier and reloaded inside the loop); `p12_p3` (kernel 1 = stage 1, activation, quantize, bytes and codes stored; kernel 2 = stage 2 and epilogue); `p1_p23` (kernel 1 = stage 1 with D1 stored fp32; kernel 2 = activation, quantize, stage 2); `p1_p2_p3`; for the chain `mem1`, `mem2`, `mem12`, `ab_c`, `a_bc`, `a_b_c`. Ablations change the arithmetic and are timed only: `no_scale` (no table loads, fmul or fma: the MMAs chain into the accumulator), `no_quant` (unit factor instead of the amax and scale code), `no_act` (identity), `mma_only` (all three), `no_exp` (attention without the exponential).

**2. The fused chain against its cuts** (e4m3fn, silu, (mt, hidden tiles per chunk, output tiles, K1, chunks); `mxpipe_cost_ffn_e4m3_silu.log`; ratios are the variant's time over the fused time, above 1 means the fused kernel is faster).

    shape             MMAs  fused: instr  regs spills   ns   ns/MMA | 64 SG: mem  p12_p3 p1_p23 p1_p2_p3 | 1 SG: mem p12_p3 p1_p23 p1_p2_p3
    (1, 2, 2, 64, 8)     96     643    89     0    893    9.3 |      1.06   1.10   1.08    1.20 |      1.08   1.05   1.05    1.11
    (1, 2, 4, 64, 8)    128     776   104     0   1087    8.5 |      1.08   1.09   1.06    1.16 |      1.08   1.02   1.01    1.07
    (1, 4, 4, 64, 4)    128    1418   126    22   1095    8.6 |      1.07   0.98   0.96    1.06 |      1.09   0.90   0.88    0.93
    (2, 2, 2, 256, 4)   288    1080   126    15   2087    7.2 |      0.98   1.00   0.95    1.01 |      0.89   0.92   0.92    0.93
    (1, 2, 8, 128, 8)   256    1095   126    23   2338    9.1 |      1.02   0.95   0.96    0.99 |      1.02   0.83   0.87    0.85
    (2, 2, 4, 128, 4)   192    1378   126    76   1525    7.9 |      1.02   1.00   0.98    1.04 |      1.04   0.95   0.99    0.98
    (1, 2, 2, 1024, 4)  528     646    90     0   4341    8.2 |      1.00   1.05   1.06    1.10 |      1.01   0.97   0.93    0.94
    (1, 2, 2, 4096, 2) 1032     648    89     0   8170    7.9 |      1.03   1.06   1.07    1.08 |      1.03   0.99   0.94    0.94

Gated FFN, silu, e4m3fn (`mxpipe_cost_ffn_e4m3_silu_gated.log`; three tensor regions per chunk, two of them accumulating in the same K loop): (1,2,2,64,8) fused 851 instructions, 108 registers, 0 spills, 1,284 ns (8.0 ns per MMA), `p12_p3` 1.06, three kernels 1.30; (1,2,4,64,8) 124 registers, 0 spills, 1.03 and 1.21; (1,2,2,1024,4) 108 registers, 7.1 ns per MMA, 1.02 and 1.05; the spilling shapes (1,4,4,64,4) 57 stores, (2,2,2,256,4) 52 stores, (1,2,8,128,8) 43 stores give `p12_p3` 0.97, 1.01 and 0.94.

**3. Ablations: what block scaling, the amax, the activation and the exponential cost** (`mxpipe_cost_ffn_e4m3_silu_abl.log`, `mxpipe_cost_ffn_i8_silu_abl.log`, `mxpipe_cost_ffn_*_silu.log`, `mxpipe_cost_attn_*.log`; ns per MMA at 64 simdgroups per core).

    class shape            fused  no_scale no_quant no_act mma_only
    e4m3  (1, 2, 2, 1024, 4)     7.7      5.8      7.7      7.7      5.8
    e4m3  (1, 2, 4, 128, 8)      8.5      6.4      8.2      8.1      6.0
    e4m3  (1, 2, 2, 64, 8)       9.3      6.8      8.8      8.6      6.0
    i8    (1, 2, 2, 1024, 4)     6.4      3.4      6.3      6.3      3.3
    i8    (1, 2, 4, 128, 8)      7.0      3.6      6.6      6.5      3.2
    i8    (1, 2, 2, 64, 8)       8.1      4.5      7.3      7.2      3.3
    e5m2  (1, 2, 2, 64, 8)       8.6        -        -      -     5.5
    e5m2  (1, 2, 4, 64, 8)       8.6        -        -      -     5.9
    e5m2  (2, 2, 2, 256, 4)      7.4        -        -      -     6.0
    e5m2  (1, 2, 8, 128, 8)      9.3        -        -      -     6.3
    e5m2  (1, 2, 2, 1024, 4)     8.5        -        -      -     6.4
    e5m2  (1, 2, 2, 4096, 2)     7.9        -        -      -     6.3
    bf16  (1, 2, 2, 64, 8)       6.2        -        -      -     5.8
    bf16  (1, 2, 4, 64, 8)       6.1        -        -      -     6.0
    bf16  (2, 2, 2, 256, 4)      5.5        -        -      -     5.5
    bf16  (1, 2, 8, 128, 8)      6.2        -        -      -     5.9
    bf16  (1, 2, 2, 1024, 4)     5.4        -        -      -     5.5
    bf16  (1, 2, 2, 4096, 2)     5.5        -        -      -     5.5
    f16   (1, 2, 2, 64, 8)       6.5        -        -      -     6.2
    f16   (1, 2, 4, 64, 8)       6.0        -        -      -     5.9
    f16   (2, 2, 2, 256, 4)      5.4        -        -      -     5.3
    f16   (1, 2, 8, 128, 8)      6.7        -        -      -     6.4
    f16   (1, 2, 2, 1024, 4)     5.6        -        -      -     5.7
    f16   (1, 2, 2, 4096, 2)     5.5        -        -      -     5.6
    i8    (1, 2, 2, 64, 8)       8.0        -        -      -     3.4
    i8    (1, 2, 4, 64, 8)       7.3        -        -      -     3.1
    i8    (2, 2, 2, 256, 4)      5.7        -        -      -     2.7
    i8    (1, 2, 8, 128, 8)      9.0        -        -      -     3.7
    i8    (1, 2, 2, 1024, 4)     6.4        -        -      -     3.3
    i8    (1, 2, 2, 4096, 2)     6.3        -        -      -     3.3

The MMA-only loop is 5.5 to 6.4 ns per MMA per core for fp8 (which unpacks each fp8 fragment into four bf16 registers before the bf16 MMA), bf16 and fp16 alike, and 3.1 to 3.7 for int8: the fp8 unpack costs nothing over bf16 at this rate, and the int8 MMA runs at about twice the rate of the others (section 138). Block scaling adds 1.9 to 2.5 ns per MMA in fp8 (25 to 27 percent of the fused time) and 3.0 to 3.6 in int8 (44 to 49 percent): the scale epilogue is the same fp32 work in both, so the faster int8 MMA leaves it as the larger share. Accelerator utilization, taken as the MMA-only time over the fused time at 64 simdgroups per core: fp8 0.65 to 0.75 in the longer K shapes, int8 0.41 to 0.52, bf16 and fp16 0.95 to 1.00. The amax, the scale code and the scaled convert together cost 0 to 5 percent (`no_quant`). A silu built on the hardware tanh costs 0.7 ns per MMA (8 percent) at K1 = 64 and nothing at K1 = 1,024 or more; in the attention flow removing the hardware exp2 changes the time by -3 to +3 percent (fp8) and 1 to 3 percent (bf16, fp16), with the row-max shuffles and the O rescale still in place, and the kernel is within 2 to 5 percent of the MMA-only kernel in bf16 and fp16: the exponential and the softmax bookkeeping are hidden under the MMAs in the 64-simdgroup regime. Scalar and ALU work against the MMA loop, as static instruction shares of the whole kernel (per-chunk and per-block work weighted like per-iteration work): at (1,2,2,1024,4) the scale epilogue is 213 of 502 instructions (42 percent) and 25 percent of the time, the amax and scale code 112 (22 percent) and 0 percent of the time, the silu activation 144 of 646 (22 percent) and 0 percent; in the bf16 attention flow at (1,2,2,64,16) the exponential and its bookkeeping are 18 of 350 instructions (5 percent) and 2 percent of the time. At long K the K loop dominates the dynamic instruction count (128 iterations of 165 instructions against 4 chunk bodies), so these shares overstate the per-chunk components; what the measurements support is the time each component adds, not an overlap fraction.

**4. The three-GEMM chain: two register-resident boundaries** (`mxpipe_cost_mlp3_e4m3_silu.log`, `mxpipe_cost_mlp3_e4m3_id.log`; e4m3fn; (mt, first hidden tiles, second hidden tiles, output tiles, K1); silu at both boundaries or identity).

    silu  (1, 4, 4, 2, 256) MMAs   88 fused  2091i 116r 0s   781 ns (8.9/MMA) | 64 SG: mem1=0.99 mem2=1.01 mem12=1.00 ab_c=1.17 a_bc=0.93 a_b_c=1.06 no_scale=0.70 mma_only=0.67 | 1 SG: mem12=1.05 ab_c=0.65 a_bc=0.61 a_b_c=0.60
    silu  (1, 4, 4, 2, 1024) MMAs  280 fused  2107i 114r 0s  2264 ns (8.1/MMA) | 64 SG: mem1=1.02 mem2=1.01 mem12=1.03 ab_c=1.05 a_bc=0.98 a_b_c=1.02 no_scale=0.73 mma_only=0.69 | 1 SG: mem12=1.09 ab_c=0.85 a_bc=0.95 a_b_c=0.94
    silu  (1, 2, 4, 2, 64) MMAs   24 fused  1491i 100r 0s   220 ns (9.2/MMA) | 64 SG: mem1=1.05 mem2=1.09 mem12=1.11 ab_c=1.76 a_bc=1.35 a_b_c=1.64 no_scale=1.05 mma_only=1.05 | 1 SG: mem12=1.03 ab_c=0.94 a_bc=0.94 a_b_c=0.90
    silu  (2, 2, 2, 2, 256) MMAs   80 fused  1689i 107r 0s   551 ns (6.9/MMA) | 64 SG: mem1=1.02 mem2=1.06 mem12=1.04 ab_c=1.35 a_bc=1.18 a_b_c=1.32 no_scale=0.83 mma_only=0.80 | 1 SG: mem12=1.02 ab_c=1.02 a_bc=1.01 a_b_c=1.01
    silu  (1, 4, 4, 4, 256) MMAs   96 fused  2390i 126r 0s   867 ns (9.0/MMA) | 64 SG: mem1=1.01 mem2=1.01 mem12=1.04 ab_c=0.96 a_bc=0.78 a_b_c=0.88 no_scale=0.65 mma_only=0.61 | 1 SG: mem12=1.00 ab_c=0.57 a_bc=0.60 a_b_c=0.54
    id    (1, 4, 4, 2, 256) MMAs   88 fused  1515i 116r 0s   625 ns (7.1/MMA) | 64 SG: mem1=1.02 mem2=1.04 mem12=1.03 ab_c=1.30 a_bc=1.05 a_b_c=1.18 mma_only=0.75 | 1 SG: mem12=1.05 ab_c=1.01 a_bc=0.96 a_b_c=0.92
    id    (1, 4, 4, 2, 1024) MMAs  280 fused  1532i 122r 0s  2090 ns (7.5/MMA) | 64 SG: mem1=1.05 mem2=1.04 mem12=1.06 ab_c=1.12 a_bc=1.04 a_b_c=1.08 mma_only=0.77 | 1 SG: mem12=1.12 ab_c=0.99 a_bc=1.10 a_b_c=1.09
    id    (1, 2, 4, 2, 64) MMAs   24 fused  1061i  99r 0s   251 ns (10.5/MMA) | 64 SG: mem1=0.98 mem2=0.98 mem12=0.98 ab_c=1.33 a_bc=1.14 a_b_c=1.29 mma_only=0.90 | 1 SG: mem12=1.10 ab_c=0.99 a_bc=0.97 a_b_c=0.94
    id    (2, 2, 2, 2, 256) MMAs   80 fused  1112i 109r 0s   492 ns (6.2/MMA) | 64 SG: mem1=1.03 mem2=1.06 mem12=1.05 ab_c=1.42 a_bc=1.27 a_b_c=1.39 mma_only=0.84 | 1 SG: mem12=1.02 ab_c=1.02 a_bc=0.99 a_b_c=1.00

Reading: the in-kernel byte bridges (a store, a barrier and a reload of the quantized tiles and codes at one or both boundaries) cost -2 to 11 percent at 64 simdgroups per core and 0 to 12 percent at one; the cuts into two or three kernels cost 14 to 76 percent at 64 simdgroups for the K1 = 64 chain, -22 to 42 percent for the K1 = 256 chains and -2 to 12 percent at K1 = 1,024; at 64 simdgroups the fused kernel is not the fastest form in one case, the (1,4,4,4,256) chain that reaches 126 registers without spilling: `a_bc` (kernel 1 = stage 1 and the first boundary, kernel 2 = the second and third GEMMs) is 22 percent faster and `a_b_c` 12 percent faster, which is consistent with section 134's fall of resident capacity by 25 to 30 percent between 22 and 126 registers but which I did not isolate. At one simdgroup per core the two silu chains at K1 = 256 take 0.54 to 0.65 of the fused time when cut (0.65, 0.61 and 0.60 at (1,4,4,2,256); 0.57, 0.60 and 0.54 at (1,4,4,4,256)), that is the fused chain is 1.5 to 1.85 times slower than the sum of its cut kernels with no spills, while the identity-activation chains take 0.92 to 1.01 of the fused time when cut and the silu chains at K1 = 1,024 and at mt = 2 take 0.85 to 1.02; the cut silu chain costs about as much as the fused identity chain (5,444 against 5,383 ns at (1,4,4,2,256)), so the extra time of the fused silu chain at one simdgroup per core sits in its activation stages; the instruction counts and spills do not explain it and I did not trace it.

**5. Attention** (`mxpipe_cost_attn_*.log`; (mt, key tiles per block, output tiles, d, key blocks); the S GEMM is the section-141 stage 1 with the head dimension as K, PV is the consumer GEMM with the V blocks as weights; hardware exp2, running max, per-lane partial row sums, O rescaled by exp2(m_old - m_new)).

    e4m3  (1, 2, 2, 64, 16) MMAs  192 fused  660i  93r  0s  1626 ns (8.5/MMA) | mem=1.07 no_scale=0.77 no_quant=0.94 no_exp=1.00 no_exp_quant=0.92 mma_only=0.70 | 1 SG mem=1.16
    e4m3  (1, 4, 4, 64, 8) MMAs  256 fused 1420i 126r 28s  2080 ns (8.1/MMA) | mem=1.03 no_scale=0.79 no_quant=0.97 no_exp=1.02 no_exp_quant=0.98 mma_only=0.76 | 1 SG mem=1.06
    e4m3  (2, 2, 2, 64, 8) MMAs  192 fused 1128i 126r 24s  1614 ns (8.4/MMA) | mem=1.06 no_scale=0.75 no_quant=0.93 no_exp=0.99 no_exp_quant=0.91 mma_only=0.69 | 1 SG mem=1.14
    e4m3  (1, 2, 4, 128, 16) MMAs  384 fused  823i 121r  0s  3506 ns (9.1/MMA) | mem=1.03 no_scale=0.72 no_quant=0.93 no_exp=1.03 no_exp_quant=0.95 mma_only=0.71 | 1 SG mem=1.05
    e4m3  (1, 2, 2, 64, 64) MMAs  768 fused  659i  93r  0s  7012 ns (9.1/MMA) | mem=1.08 no_scale=0.75 no_quant=0.93 no_exp=0.99 no_exp_quant=0.91 mma_only=0.70 | 1 SG mem=1.17
    e5m2  (1, 2, 2, 64, 16) MMAs  192 fused  660i  93r  0s  1747 ns (9.1/MMA) | mem=1.07 no_exp=0.99 mma_only=0.68 | 1 SG mem=1.16
    e5m2  (1, 4, 4, 64, 8) MMAs  256 fused 1420i 126r 28s  2026 ns (7.9/MMA) | mem=1.04 no_exp=1.00 mma_only=0.75 | 1 SG mem=1.06
    e5m2  (2, 2, 2, 64, 8) MMAs  192 fused 1128i 126r 24s  1519 ns (7.9/MMA) | mem=1.08 no_exp=1.00 mma_only=0.70 | 1 SG mem=1.14
    e5m2  (1, 2, 2, 64, 64) MMAs  768 fused  659i  93r  0s  6483 ns (8.4/MMA) | mem=1.07 no_exp=1.01 mma_only=0.70 | 1 SG mem=1.17
    bf16  (1, 2, 2, 64, 16) MMAs  192 fused  350i  80r  0s  1208 ns (6.3/MMA) | mem=1.07 no_exp=0.98 mma_only=0.98 | 1 SG mem=1.16
    bf16  (1, 4, 4, 64, 8) MMAs  256 fused  621i 126r  2s  1491 ns (5.8/MMA) | mem=1.05 no_exp=0.99 mma_only=0.99 | 1 SG mem=1.06
    bf16  (2, 2, 2, 64, 8) MMAs  192 fused  588i 124r  0s  1140 ns (5.9/MMA) | mem=1.07 no_exp=0.99 mma_only=0.99 | 1 SG mem=1.13
    bf16  (1, 2, 2, 64, 64) MMAs  768 fused  350i  80r  0s  4725 ns (6.2/MMA) | mem=1.07 no_exp=0.98 mma_only=0.98 | 1 SG mem=1.16
    f16   (1, 2, 2, 64, 16) MMAs  192 fused  340i  80r  0s  1156 ns (6.0/MMA) | mem=1.06 no_exp=0.98 mma_only=0.97 | 1 SG mem=1.15
    f16   (1, 4, 4, 64, 8) MMAs  256 fused  595i 126r  2s  1494 ns (5.8/MMA) | mem=1.04 no_exp=0.99 mma_only=0.99 | 1 SG mem=1.06
    f16   (2, 2, 2, 64, 8) MMAs  192 fused  576i 124r  0s  1121 ns (5.8/MMA) | mem=1.07 no_exp=0.99 mma_only=0.99 | 1 SG mem=1.15
    f16   (1, 2, 2, 64, 64) MMAs  768 fused  340i  80r  0s  4790 ns (6.2/MMA) | mem=1.06 no_exp=0.97 mma_only=0.95 | 1 SG mem=1.16
    i8    (1, 2, 2, 64, 16) MMAs  192 fused  675i  94r  0s  1556 ns (8.1/MMA) | mem=1.06 no_exp=0.98 mma_only=0.50 | 1 SG mem=1.11
    i8    (1, 4, 4, 64, 8) MMAs  256 fused 1450i 126r 21s  2001 ns (7.8/MMA) | mem=1.04 no_exp=0.98 mma_only=0.48 | 1 SG mem=1.01
    i8    (2, 2, 2, 64, 8) MMAs  192 fused 1184i 126r 20s  1391 ns (7.2/MMA) | mem=1.07 no_exp=0.99 mma_only=0.51 | 1 SG mem=1.07
    i8    (1, 2, 2, 64, 64) MMAs  768 fused  674i  90r  0s  6228 ns (8.1/MMA) | mem=1.06 no_exp=0.98 mma_only=0.51 | 1 SG mem=1.11

**6. Register budgets** (`mxpipe_budget.py`, `mxpipe_budget.log`; compile only, section-134 analyser; registers touched / spill stores).

    FFN (act silu): compiler register footprint (regs) and spill stores by output tiles per simdgroup (nto), fused kernel; frontier = largest nto with 0 spill stores
    class  mt ch   nto: 1       2       3       4       6       8       10      12     | frontier fused (regs) | stage-2-only kernel frontier | stage-1 + quantize kernel
    e4m3   1  2         81/0    89/0    97/0   104/0   124/0   126/23  126/48  126/65  | nto <=  6 (124 r)      | nto <=  8 (118 r)          | 80 regs, 0 spills at every nto
    e4m3   1  4        116/0   126/8   126/13  126/25  126/43  126/70  126/106 126/123 | nto <=  1 (116 r)      | nto <=  6 (115 r)          | 107 regs, 0 spills at every nto
    e4m3   2  2        124/0   126/15  126/44  126/76  126/166 126/272 126/367 126/460 | nto <=  1 (124 r)      | nto <=  3 (125 r)          | 109 regs, 0 spills at every nto
    bf16   1  2         68/0    81/0    97/0   108/0   109/0   126/0   126/29  126/69  | nto <=  8 (126 r)      | nto <= 10 (116 r)          | 56 regs, 0 spills at every nto
    bf16   1  4        102/0   112/0   120/0   126/24  124/32  124/56  126/70  126/95  | nto <=  3 (120 r)      | nto <=  8 (114 r)          | 88 regs, 0 spills at every nto
    bf16   2  2        102/0   120/0   126/8   126/44  126/111 126/207 126/311 126/407 | nto <=  2 (120 r)      | nto <=  4 (126 r)          | 86 regs, 0 spills at every nto
    i8     1  2         77/0    85/0    95/0   110/0   126/4   126/8   126/34  126/62  | nto <=  4 (110 r)      | nto <=  6 (122 r)          | 67 regs, 0 spills at every nto
    i8     1  4        112/0   119/0   126/4   126/12  126/28  126/54  126/93  126/170 | nto <=  2 (119 r)      | nto <=  8 (114 r)          | 103 regs, 0 spills at every nto
    i8     2  2        117/0   126/8   126/25  126/54  126/123 126/215 126/294 126/370 | nto <=  1 (117 r)      | nto <=  3 (126 r)          | 99 regs, 0 spills at every nto

    three-GEMM chain (act silu at both boundaries): spill stores (registers) of the fused kernel and of its cuts a = stage 1 + boundary 1, ab, bc, c
    class  mt ch n2t nto | fused       a          ab         bc         c
    e4m3   1  2  2   2   |   0( 73)     0( 64)     0( 72)     0( 65)     0( 44)  
    e4m3   1  4  4   2   |   0(116)     0(102)     0(112)     0(104)     0( 85)  
    e4m3   1  4  4   4   |   0(126)     0(102)     0(112)     0(111)     0( 98)  
    e4m3   1  4  6   4   |  30(126)     0(102)     0(120)    12(126)     0(110)  
    e4m3   1  2  4   8   |   0(123)     0( 64)     0( 93)     5(126)     8(126)  
    e4m3   2  2  2   2   |   0(107)     0(103)     0(103)     0( 99)     0( 79)  
    e4m3   2  4  4   2   |  62(126)    31(126)    38(126)    18(126)     8(126)  
    bf16   1  2  2   2   |   0( 80)     0( 52)     0( 61)     0( 66)     0( 40)  
    bf16   1  4  4   2   |   0(120)     0( 88)     0(115)     0(101)     0( 68)  
    bf16   1  4  4   4   |   0(116)     0( 88)     0(113)     0(105)     0(100)  
    bf16   1  4  6   4   |   5(126)     0( 88)     0(120)    15(126)     0(106)  
    bf16   1  2  4   8   |   0(112)     0( 52)     0( 97)     0(117)     0(116)  
    bf16   2  2  2   2   |   0( 98)     0( 76)     0( 95)     0( 91)     0( 64)  
    bf16   2  4  4   2   |  27(126)     0(126)    16(126)    32(126)     0(108)  
    i8     1  2  2   2   |   0( 72)     0( 63)     0( 76)     0( 62)     0( 44)  
    i8     1  4  4   2   |   0(110)     0( 97)     0(103)     0( 99)     0( 73)  
    i8     1  4  4   4   |   0(126)     0( 97)     0(103)     0(123)     0( 91)  
    i8     1  4  6   4   |  63(126)     0( 98)     0(121)    15(126)     0( 96)  
    i8     1  2  4   8   |   0(115)     0( 63)     0( 94)     0(115)     0(123)  
    i8     2  2  2   2   |   0(101)     0( 95)     0( 98)     0( 92)     0( 78)  
    i8     2  4  4   2   |  46(126)    34(126)    40(126)    18(126)     0(114)  

The fused FFN's live state is D1 (mt x hidden tiles x 8 registers) plus D2 (mt x output tiles x 8) plus the operand fragments in flight (an fp8 fragment unpacks to four bf16 registers, a bf16 fragment is four registers) and the scale vectors, and the compiler's schedule keeps many fragments in flight: the frontier is 8 state tiles (64 registers of accumulators) for fp8 with 2 hidden tiles (124 registers touched), 10 for bf16, 6 for int8, and 5 to 8 state tiles when the chunk has 4 hidden tiles or two token tiles are carried, so the transient share is 60 to 76 registers in fp8. Peak live registers from the analyser, over the 498 kernels of this grid (`mxpipe_budget.json`): 28 to 123 in the 268 spill-free kernels and 101 to 126 in the 230 spilling ones. These are compiler results for the emitted IR, not hardware limits: the hardware register domain is R0..R125 (sections 134 and 135), 126 registers touched is the compiler using the whole domain, and a 15-tile accumulator bound (8-aligned groups) is the architectural one.

**7. Break-point rules from these measurements.**

    situation                                                       cheapest measured form                                                     measured margin
    fused kernel spill-free, many simdgroups per core (>= 32)        fully fused chain                                                          5 to 10 percent over any cut, 8 to 20 over three kernels
    fused kernel spills 15 or more stores                            cut after the quantize (p12|p3) or before it (p1|p23)                      0 to 5 percent less time at 64 SG, 1 to 17 at 1 SG; the two cuts within 5 percent of each other
    one simdgroup per core, long K (1,024 blocks or more)            cuts, even when the fused kernel is spill-free                             1 to 7 percent less time
    three-GEMM chain, K1 <= 64                                       fully fused                                                                1.14 to 1.76 over the cuts
    three-GEMM chain, fused kernel at 126 registers, no spill        cut after stage 1 (a|bc)                                                   22 percent (one chain; occupancy consistent, not isolated)
    K loop of any length                                             runtime loop, never unrolled                                               1.3 to 7.2 times (section 141)
    attention                                                        fused; bridge P only if registers force it                                 bridging costs 3 to 8 percent at 64 SG

**8. Failed hypotheses, kept.** (a) I expected the fused chain to win everywhere; once it spills it needs 0 to 5 percent more time than its cuts at 64 simdgroups per core and up to 20 percent more at one, and it needs more time than its cuts at one simdgroup per core at long K even without spills. (b) I expected the quantize boundary to be the cheapest cut because it moves one byte per element instead of four; the two cuts are within 5 percent in time (the traffic is 123 against 146 KiB, but these runs are cache-resident; section 143). (c) I expected the activation and the softmax exponential to add visible time; at 64 simdgroups per core they cost 0 to 8 percent and 0 to 3 percent. (d) I attributed the fused chain's slowness at one simdgroup per core to spills; the spill-free silu chains are the slowest relative to their cuts (part 4). (e) A first version of the cost tables was overwritten by later sweeps that wrote a JSON of the same name; the logs are the record and the ambiguous JSON files were removed.

**Labels.** *Hardware (measured on this H17s):* every time, ratio and instruction count; the register and spill counts are the OS compiler's for the emitted IR. *Model:* the analytic traffic (fragments 256 B per fp8 tile, 512 B per 16-bit tile, block-scale tables 2 or 4 B per lane and block, fp32 tiles 1,024 B) counts device bytes requested, not DRAM bytes. *Not measured:* a fused chain against a production three-kernel schedule with its own tiling, other tile counts per simdgroup than listed, the cause of the one-simdgroup slowness of the fused chain and of the 126-register chain at 64 simdgroups.

Artifacts under `results/g17-tensorops-recon-v1/`: `mxpipe_cost.py`, `mxpipe_tables.py`, `mxpipe_tables.txt`, `mxpipe_budget.py`, `mxpipe_budget.log`, `mxpipe_budget.json`, `mxpipe_cost_ffn_e4m3_id.log`, `mxpipe_cost_ffn_e4m3_silu.log`, `mxpipe_cost_ffn_e4m3_silu_gated.log`, `mxpipe_cost_ffn_e4m3_silu_abl.log`, `mxpipe_cost_ffn_i8_silu_abl.log`, `mxpipe_cost_ffn_e5m2_silu.log`, `mxpipe_cost_ffn_bf16_silu.log`, `mxpipe_cost_ffn_f16_silu.log`, `mxpipe_cost_ffn_i8_silu.log`, `mxpipe_cost_mlp3_e4m3_silu.log`, `mxpipe_cost_mlp3_e4m3_id.log`, `mxpipe_cost_mlp3_e5m2_silu.log`, `mxpipe_cost_mlp3_bf16_silu.log`, `mxpipe_cost_mlp3_f16_silu.log`, `mxpipe_cost_mlp3_i8_silu.log`, `mxpipe_cost_attn_e4m3.log`, `mxpipe_cost_attn_e5m2.log`, `mxpipe_cost_attn_bf16.log`, `mxpipe_cost_attn_f16.log`, `mxpipe_cost_attn_i8.log`, `mxpipe_cost_noise_1.log` to `mxpipe_cost_noise_5.log`.

## 143. Operand classes and simdgroup scaling: a bf16 or fp16 chain runs at the MMA rate, block-scaled fp8 and int8 pay 25 and 45 percent of their time in the scale epilogue and win only when the working set leaves the cache, and across simdgroups a fragment-order shared buffer beats recomputation in every measured case and beats one large simdgroup only at low occupancy or when that simdgroup spills heavily

**Result.** (1) Same pipeline, five operand classes (fused FFN, silu, ns per chain per core at 64 simdgroups per core, six shapes): bf16 5.4 to 6.2 ns per MMA and fp16 5.4 to 6.7, e4m3fn 7.2 to 9.3 and e5m2 7.4 to 9.3, int8 with block scales 5.7 to 9.0. The MMA loop alone runs at 5.3 to 6.4 ns per MMA in the four float classes and 2.7 to 3.7 in int8, so the bf16 and fp16 chains are at the MMA rate (MMA-only time over fused time 0.93 to 1.02) and the conversion and scaling costs are in the fp8 (0.65 to 0.81) and int8 (0.41 to 0.52) classes: e4m3fn needs 1.31 to 1.52 times the bf16 time, int8 with block scales 0.78 to 0.99 of the e4m3fn time. An int8 chain without the per-block epilogue (timing only, unvalidated arithmetic) runs at 3.4 to 4.5 ns per MMA, 1.4 to 1.6 times faster than bf16 at the two shapes measured in both classes. e4m3fn and e5m2 have identical instruction, register and spill counts and times within noise. Register and instruction cost: the bf16 and fp16 fused kernels are about half as long (320 to 533 against 643 to 1,095 instructions) and, with 2 hidden tiles per chunk, spill-free to 8 output tiles per simdgroup where fp8 is to 6 and int8 to 4 (section 142). Traffic: bf16 requests 1.54 to 1.65 times the bytes of fp8 and int8. (2) Working set: with 4 input sets (1.7 to 2.6 MiB) bf16 is the fastest class; as the set count grows bf16 slows from 5.5 to 9.5 ns per MMA while e4m3fn stays at 8.4 to 9.1 and int8 rises from 6.8 to 8.1: int8 overtakes bf16 between working sets of 21 and 42 MiB (bf16 bytes) and e4m3fn overtakes bf16 between 84 and 169 MiB, so the byte halving of fp8 and int8 pays only outside the cache and the scale epilogue makes fp8 the slowest class inside it. (3) Across simdgroups, three strategies for one threadgroup of 2, 4 or 8 simdgroups that produces nsg x nto output tiles from all hidden chunks (216 of 216 runs exact, section 141): recompute (every simdgroup runs the whole chunk loop for its slice), fragment-order threadgroup memory (each simdgroup computes chunk r x nsg + s once, stores the quantized fragments and codes in fragment order, workgroup barrier, every simdgroup applies all nsg chunks to its slice, barrier) and the same through a per-threadgroup device region (device barrier), against a single simdgroup that holds all nsg x nto tiles. Recomputation is never the fastest: it needs 1.16 to 6.66 times the shared-buffer time at 64 threadgroups per core and 0.97 to 6.81 times at one (level with sharing in one row only, part 4). Threadgroup memory and device memory are within 0.94 to 1.08 of each other. Against one large simdgroup: at one threadgroup per core sharing is 1.66 to 1.89 (nsg = 2), 2.98 to 5.93 (4) and 8.62 to 10.16 (8) times faster; at 64 threadgroups per core the single simdgroup needs 0.82 to 0.97 of the shared time in 11 of the 21 rows (7 fp8 rows, two of which spill 29 and 32 stores, and 4 bf16 rows; sharing costs 3 to 22 percent there) and 1.08 to 1.15 in the six fp8 rows where it holds 16 output tiles and spills 134 to 138 stores. (4) Rules (part 6): share through fragment-order threadgroup or device memory, never recompute; use sharing when occupancy is low (one to a few resident simdgroups per core) or when a single simdgroup's state would spill more than about a hundred stores; otherwise keep the output tiles in one simdgroup. No production file was edited.

**1. The classes** (`mxpipe.py`, kinds and classes as in section 141; `mxpipe_cost_ffn_*_silu.log`; fused kernel, no debug stores, four input sets, silu, 64 simdgroups per core; shapes (mt, hidden tiles, output tiles, K1, chunks); ns per chain (ns per MMA), registers, spill stores, device bytes requested).

    shape                                e4m3 |                   e5m2 |                   bf16 |                    f16 |                     i8
    (1, 2, 2, 64, 8)   |    893 ( 9.3)  89r  0s   43K |    821 ( 8.6)  89r  0s   43K |    592 ( 6.2)  80r  0s   68K |    623 ( 6.5)  80r  0s   68K |    771 ( 8.0)  85r  0s   43K
    (1, 2, 4, 64, 8)   |   1087 ( 8.5) 104r  0s   57K |   1107 ( 8.6) 104r  0s   57K |    778 ( 6.1) 108r  0s   88K |    765 ( 6.0) 108r  0s   88K |    932 ( 7.3) 110r  0s   57K
    (2, 2, 2, 256, 4)  |   2087 ( 7.2) 126r 15s   89K |   2139 ( 7.4) 126r 15s   89K |   1585 ( 5.5) 120r  0s  144K |   1552 ( 5.4) 121r  0s  144K |   1642 ( 5.7) 126r  8s   89K
    (1, 2, 8, 128, 8)  |   2338 ( 9.1) 126r 23s  114K |   2376 ( 9.3) 126r 23s  114K |   1577 ( 6.2) 126r  0s  176K |   1723 ( 6.7) 126r  0s  176K |   2299 ( 9.0) 126r  7s  114K
    (1, 2, 2, 1024, 4) |   4341 ( 8.2)  90r  0s  241K |   4498 ( 8.5)  90r  0s  241K |   2869 ( 5.4)  80r  0s  396K |   2940 ( 5.6)  80r  0s  396K |   3377 ( 6.4)  87r  0s  241K
    (1, 2, 2, 4096, 2) |   8170 ( 7.9)  89r  0s  470K |   8184 ( 7.9)  89r  0s  470K |   5716 ( 5.5)  80r  0s  776K |   5681 ( 5.5)  80r  0s  776K |   6499 ( 6.3)  86r  0s  470K

The MMA-only times per MMA (section 142 part 3) are 5.3 to 6.4 ns for e4m3fn, e5m2, bf16 and fp16 and 2.7 to 3.7 for int8. fp8 operands unpack to bf16 before the MMA (four `op17642` per fragment) and the MMA is the bf16 MMA, which is why the fp8 MMA-only loop is not faster than the bf16 one. What separates the classes is the work around the MMA: block scaling (25 to 27 percent of the fused fp8 time, 44 to 49 percent for int8, whose MMA is twice as fast), the amax and scale code, the fp8 pack and unpack or the int8 `rint` and `fptosi`, and, for bf16 and fp16, one `fptrunc` per element. The bf16 and fp16 boundary is one convert per tile and costs nothing measurable at 64 simdgroups per core (MMA-only time over fused time 0.93 to 1.02). fp16 has a range limit (65,504) that bf16 does not; the fp16 data of the suites lie inside it.

**2. Where the byte halving pays** (`mxpipe_cost_ffn_*_id_sets*.log`; the same kernels with 4 to 256 input sets, the timing iteration i reading set i mod sets, so the working set is sets times the bytes of one set: 0.418 MiB per set for fp8 and int8, 0.660 MiB for bf16).

    ns per MMA at 64 simdgroups per core, FFN (mt 1, 2 hidden tiles, 2 output tiles, K1 = 4,096, 2 chunks, act id); timing iteration i reads input set i mod sets
    input sets    working set (MiB): fp8/int8  bf16  |   e4m3    bf16    i8
         4                             1.7      2.6 |   8.43    5.54   6.75
        16                             6.7     10.6 |   8.37    6.58   6.99
        32                            13.4     21.1 |   9.06    7.19   7.39
        64                            26.8     42.2 |   9.06    8.27   7.59
       128                            53.5     84.5 |   9.07    8.85   8.06
       256                           107.0    169.0 |   9.07    9.52   8.12

Inside the cache (4 sets) bf16 is 34 percent faster than e4m3fn; at 256 sets bf16 is 5 percent slower than e4m3fn and 17 percent slower than int8. The working sets here are large only because the timing loop cycles sets; the crossover is a statement about the bytes that a pass touches, not about a particular model. I did not measure a bandwidth-bound regime with DRAM-resident weights streaming from a fixed single pass, so the working-set numbers are the measured evidence and DRAM bandwidth is not inferred from them.

**3. Multi-simdgroup strategies** (`mxpipe_sg.py`, `mxpipe_sg_exact.log`, `mxpipe_sg_time_e4m3_id.log`, `mxpipe_sg_time_e4m3_id_short.log`, `mxpipe_sg_time_bf16_silu.log`). The kernels are the section-141 generator with `nsg` and `share`: the output slice of simdgroup s is tiles s x nto to (s + 1) x nto of the wide output (the weights are packed for all nsg x nto tiles and each simdgroup adds its tile offset to the fragment and table addresses); the shared buffer holds nsg slots of the quantized tiles (`<8 x i8>` per lane per tile) and codes (`<2 x i8>` per lane per block) in the layout the consumer reads, so a consumer's load returns the same bits in the same lane and slot as the producer's registers held; the barrier flags are 2 (threadgroup memory) and 1 (device memory), placed before the loads (visibility) and after them (before the next round overwrites). Exactness (`mxpipe_sg_exact.log`): recompute, threadgroup and device strategies, nsg = 2, 4 and 8, e4m3fn and bf16, identity and silu, three shapes, two data variants: 216 of 216, every boundary (D1, Z, quantized bytes, codes per chunk, and the wide D2). Timing (ns per chain per core; a chain is one threadgroup's whole pipeline; 1 threadgroup per core is 20 threadgroups on the 20 cores, 64 per core is 1,280):

    strategies for one threadgroup of nsg simdgroups producing nsg x nto output tiles from nch hidden chunks (ch = 2 hidden tiles, mt = 1); times are ns per chain per core; ratios are over the tg (threadgroup memory) time
    class act   nto  K1  nch nsg | single: regs spills | tg: regs spills | 1 threadgroup per core: tg ns  single/tg recompute/tg dev/tg | 64 per core: tg ns  single/tg recompute/tg dev/tg
    bf16  silu   2    64  16  4 |   126      0    |  126     3    |     6446      4.48      1.95      1.06   |    2961     0.84      1.59      1.06
    bf16  silu   2   256  16  4 |   126      0    |  126     3    |    13275      4.00      2.71      1.02   |    5392     0.85      2.48      1.05
    bf16  silu   2  1024  16  4 |   126      0    |  126     3    |    51245      3.93      3.28      0.99   |   16437     0.93      3.16      1.02
    bf16  silu   4   256  16  4 |   126    113    |  126    94    |    20077      4.01      1.98      1.00   |    8371     0.92      1.69      1.02
    e4m3  id     2    32   2  2 |   112      0    |   96     0    |      932      1.66      1.43      0.98   |     227     1.06      1.16      1.08
    e4m3  id     2    32  16  2 |   112      0    |   98     0    |     7602      1.67      1.27      1.05   |    1695     0.87      1.29      1.01
    e4m3  id     2    64  16  2 |   112      0    |  100     0    |     9871      1.75      1.48      1.07   |    1996     0.91      1.50      1.06
    e4m3  id     2   256  16  2 |   112      0    |  100     0    |    24038      1.89      1.80      1.05   |    5032     0.97      1.79      1.02
    e4m3  id     1    32   4  4 |   112      0    |   89     0    |     1004      2.98      2.10      0.97   |     467     0.82      1.85      1.01
    e4m3  id     1   256  16  4 |   112      0    |   91     0    |    12561      3.67      3.21      1.00   |    5133     0.88      3.33      1.01
    e4m3  id     2    32   4  4 |   126     32    |  107     0    |     1317      4.70      2.05      0.94   |     686     0.97      1.53      1.01
    e4m3  id     2    32  16  4 |   126     29    |  106     0    |     5951      5.93      1.66      1.05   |    2377     1.07      1.72      1.03
    e4m3  id     2    64  16  4 |   126     29    |  106     0    |     7146      5.60      2.02      1.05   |    3047     1.03      2.06      0.99
    e4m3  id     2   256  16  4 |   126     29    |  106     0    |    14473      4.81      2.88      1.03   |    6005     0.96      2.97      0.94
    e4m3  id     2  1024  16  4 |   126     28    |  107     0    |    44740      4.57      3.57      1.01   |   17132     1.07      3.80      1.06
    e4m3  id     4    32   8  4 |   126    138    |  120     0    |     7680      4.12      0.97      1.05   |    2562     1.08      1.22      1.02
    e4m3  id     4   256  16  4 |   126    134    |  120     0    |    23765      4.16      1.93      1.03   |    8158     1.09      2.54      1.01
    e4m3  id     2    32   8  8 |   126    138    |  105     0    |     3413      9.28      1.68      0.95   |    2549     1.13      1.77      0.97
    e4m3  id     2    64  16  8 |   126    134    |  107     0    |     6727     10.16      2.44      0.98   |    5091     1.09      2.41      1.01
    e4m3  id     2   256  16  8 |   126    134    |  107     0    |    11018      9.29      4.41      0.98   |    7899     1.15      4.61      1.00
    e4m3  id     2  1024  16  8 |   126    134    |  107     0    |    29042      8.62      6.81      1.00   |   19786     1.12      6.66      1.00

**4. Reading the table.** Recompute repeats stage 1 nsg times: at one threadgroup per core it takes 1.27 to 1.80 times the shared time at nsg = 2, 0.97 to 3.57 at 4 and 1.68 to 6.81 at 8, and at 64 threadgroups per core, where the machine is saturated and redundant MMAs are lost throughput, 1.16 to 1.79, 1.22 to 3.80 and 1.77 to 6.66; the one row where it is level with sharing (0.97 at one threadgroup, nto = 4, K1 = 32, 8 chunks, nsg = 4) is the row with the smallest stage-1 share, and even there it is 1.22 times slower at 64 per core. A single simdgroup holding all output tiles serializes the work at low occupancy (1.66 to 10.16 times the shared time) and costs nothing extra at high occupancy when its state fits: it takes 0.82 to 0.97 of the shared time in 11 of the 21 rows (7 fp8 rows and 4 bf16 rows; the bf16 single kernels spill 0 to 113 stores) and the shared strategies pay 3 to 22 percent for the barriers, the stores and the loads there; when the single simdgroup holds 16 fp8 output tiles it spills 134 to 138 stores and the shared strategies are 8 to 15 percent faster (nsg = 8: 1.09 to 1.15; nsg = 4 with nto = 4: 1.08 to 1.09); at 8 fp8 tiles (28 to 32 stores) the two are level (0.96 to 1.07). bf16 kept its single-simdgroup advantage even at 16 output tiles with 113 spill stores (0.92). The threadgroup and device versions are level (0.94 to 1.08, median 1.01); in fp8 the threadgroup kernels have no spill stores (the device kernel with 4 output tiles spills 8), in bf16 the threadgroup kernels spill 3 stores (94 at 4 output tiles) and the device kernels 11 (95). I found no measured cooperative mechanism that avoids memory: a cross-simdgroup `simd_shuffle` returned the caller's own simdgroup's value in 256 of 256 cases and the cooperative-tensor accumulator reuse stays inside one scope (section 133).

**5. Failed hypotheses, kept.** (a) I expected threadgroup memory to be faster than device memory by a margin; the two are within 8 percent and the device version is faster in 3 of the 21 rows at 64 threadgroups per core and in 8 at one. (b) I expected recomputation to win at the shortest K where the shared hand-off's barriers cost more than one block of stage 1; it lost at K1 = 32 (one block) as well, 1.16 to 1.85 times the shared time at 64 per core. (c) I expected the single large simdgroup to lose to sharing whenever it spills; in bf16 it needs 7 to 16 percent less time up to 113 spill stores and in fp8 it is level at 28 to 32 stores. (d) I expected fp8 to be the fastest class because the operands are half the bytes; inside the cache it is the slowest of the four float classes (part 1).

**6. Rules.**

    situation                                                            measured choice
    stage 1 shared by several simdgroups of one threadgroup             quantized fragments and codes in fragment order through threadgroup or device memory, workgroup barrier before and after; never recompute stage 1 per simdgroup
    one threadgroup per core (low occupancy)                             share: 1.7 to 10 times faster than one simdgroup holding all tiles (nsg = 2 to 8)
    64 threadgroups per core, fp8 with up to 4 output tiles (no spill), or bf16 up to 16 tiles   keep the output tiles in one simdgroup: sharing costs 3 to 22 percent
    64 threadgroups per core, fp8 with 8 output tiles (28 to 32 spill stores)                       level (0.96 to 1.07)
    64 threadgroups per core, fp8 with 16 output tiles (134 to 138 spill stores)                    share: 8 to 15 percent less time
    class, working set inside the cache                                   bf16 or fp16 (5.4 to 6.7 ns per MMA); int8 block-scaled 5.7 to 9.0; fp8 7.2 to 9.3
    class, working set beyond about 20 to 40 MiB (bf16 bytes)              int8 (7.6 to 8.1), then e4m3fn or e5m2 (9.1), bf16 slower than fp8 beyond 84 to 169 MiB
    int8 with per-block scales                                             1 to 22 percent faster than e4m3fn; per-tensor or per-channel scaling (no block epilogue) would run at 3.4 to 4.5 ns per MMA (timing only)

**Labels.** *Hardware (measured on this H17s):* the timings, counts and exactness runs; the working-set result is for kernels whose simdgroups read successive input sets. *Model:* none new. *Not measured:* DRAM-streaming bandwidth of a single pass over weights, more than 8 simdgroups per threadgroup, three or more threadgroups sharing through device memory (each threadgroup has its own region), int8 per-tensor scaling with validated arithmetic, fp16 with data near its range limit.

Artifacts under `results/g17-tensorops-recon-v1/`: `mxpipe_sg.py`, `mxpipe_sg_exact.log`, `mxpipe_sg_exact.json`, `mxpipe_sg_time_e4m3_id.log`, `mxpipe_sg_time_e4m3_id_short.log`, `mxpipe_sg_time_bf16_silu.log`, `mxpipe_sg_time_e4m3_id_short.json`, `mxpipe_sg_time_bf16_silu.json` (the JSON of the first sweep was overwritten by the second; its log is the record), `mxpipe_cost_ffn_e4m3_id_sets4.log`, `mxpipe_cost_ffn_e4m3_id_sets16.log`, `mxpipe_cost_ffn_e4m3_id_sets32.log`, `mxpipe_cost_ffn_e4m3_id_sets64.log`, `mxpipe_cost_ffn_e4m3_id_sets128.log`, `mxpipe_cost_ffn_e4m3_id_sets256.log`, and the same for `bf16` and `i8`.

## 144. Execution model for a fused TensorOps pipeline on H17s, as measured: canonical layouts, legal producer to consumer mappings, scale and quantize placement, the streaming schedule, register budgets, fusion and cut rules, the simdgroup boundary, operand-class rules and the numeric domains, each with the section that measured it

**Result.** This section collects sections 129 to 143 into the rules that a compiler needs to emit a fused GEMM, activation, quantize, GEMM (and further GEMM) pipeline without architectural guesswork. Nothing here is new measurement; every rule names its evidence, and a rule with no hardware evidence is listed as open in part 12. The acceptance subgraphs that the rules were derived from ran on hardware and are exact at every intermediate boundary (section 141): (a) K loop (runtime), bias and activation, block-32 quantize, GEMM, activation, block-32 quantize, GEMM, residual, RMS norm, store: three tensor regions, two consecutive register-resident boundaries, 60 of 60 runs in fp8 and the same chain in int8, bf16 and fp16 within the 168 of 168 runs of the class suite; (b) a hidden-chunk loop around a K loop, gate and up GEMMs, `act(G) * U`, quantize, down GEMM accumulating across chunks: three tensor regions per chunk, 40 of 40; (c) an attention flow with a runtime loop over key blocks: 54 of 54. Against memory-mediated execution the chain (a) costs -2 to 11 percent for in-kernel byte bridges and -22 to 76 percent for cuts into kernels (section 142). No production file was edited.

**1. Tile, fragment and byte layouts** (sections 129, 132, 140; `transpose.py`, `mxfuse_maps.log`). A tile is 16 x 16 in 32 lanes; a lane holds 8 elements. Hardware row r = (r3 r2 r1 r0), column c = (c3 c2 c1 c0).

    D (fp32 accumulator, `<8 x float>`), A (`<8 x i8>` fp8 or int8, `<8 x half>`, `<8 x bfloat>`)    lane = 16 r3 + 8 c3 + 2 (r2 r1) + c2,   slot = 4 r0 + (c1 c0)
    B (right operand)                       pos_b(k, c) = pos(rotl1(k), c)
    A with the transpose bit (At)           lane = 16 k2 + 8 m0 + 2 (k1 k0) + m3,   slot = 4 k3 + (m2 m1)
    a lane's D and A elements               row group g = slot >> 2 (rows r0 = g), four consecutive columns 8 c3 + 4 c2 + (slot & 3) per row group
    library logical row m                   m = 4 (lane >> 4) + ((lane >> 1) & 3), rows m and m + 8; equals rotr1 of the hardware row (64 of 64 lane and slot-group pairs), column = hardware column (128 of 128)

Memory layout of a packed operand (block-major, `mxrun.pack`): fragment (block b, k-tile kk in 0..1, tile t of nt) at index ((b x 2 + kk) x nt + t) x 32 + lane in units of the fragment size (8 bytes for fp8 and int8, 16 for the 16-bit classes); the block-scale tables follow the fragment region: for a block b and tile t, 32 lanes x 2 bytes (A side: the codes of the lane's two row groups) or 32 x 4 bytes (B side: the codes of the lane's four columns) at ((b x nt + t) x 32 + lane); the block capacity fixes the table base, the trip count does not. Scale factor of code e: 2^(e - 127); code 0 is a subnormal factor that the fp32 multiply flushes (the block is exactly zero), code 255 is NaN. A scale vector is expanded to a tile with the lane shuffle `[0,0,0,0,1,1,1,1]` (two row codes) or `[0,1,2,3,0,1,2,3]` (four column codes).

**2. Legal direct producer to consumer mappings** (section 140, runtime loops in section 141). Producer: fp32 accumulators D1 in the D layout. The consumer reads the same lane and slot:

    consumer reading of D1                 element of D1 at hardware (r, c) is the consumer's          block axis of the quantizer         scale vector for the consumer                    lane exchange
    A   left operand                       (m, k) = (r, c)                                              row-wise, xor masks 1 and 8         row codes as they are (D2 row layout)            none
    At  left operand, transpose bit        (m2, k) = (rotl1(c), rotr1(r))                                column-wise, xor masks 2, 4, 16     column codes -> D2 row codes                     8 shuffles per free tile and block
    B   right operand                      (k, n) = (rotr1(r), c)                                        column-wise, xor masks 2, 4, 16     column codes as they are (D2 column layout)     none
    Bt  right operand, transpose bit       (k, n) = (c, rotr1(r))                                        row-wise, xor masks 1 and 8         row codes -> D2 column codes                     8 shuffles per free tile and block

With the library's row order (logical row m at hardware row rotl1(m)) the four readings hold in natural coordinates and nothing is relabeled; the fp8 or int8 byte in slot j of the D fragment is the byte in slot j of the consumer's fragment; a 16-bit operand is the element-wise `fptrunc` of the D fragment in place. The quantize hand-off at register level: 4 `op13618` packs per D tile, each pack a 16-bit half of two fp32 registers, and 4 `op17642` unpacks per operand fragment reading those halves (8 of 8 and 16 of 16 forwarded directly in every single-simdgroup loop body, section 141 part 6). Evidence: 416 of 416 runs with a runtime K loop in all four readings and 112 of 112 repeat runs over two input sets (section 141 part 4); the int8, bf16 and fp16 classes were run in the A reading only.

**3. Producer arithmetic and scale placement** (sections 138, 141). Block scaling is applied after the MMA: per K block of 32, `P = mma(a0, b0, 0); P = mma(a1, b1, P); acc = fma(P, fa x fb, acc)` with the fp32 fma (`op2190`), the factors decoded from the byte tables (`fac_bits`: `(code << 23)`, code 0 to 2^-127, code 255 to NaN); the first MMA has no accumulator source (`op5107`). int8 uses the int32 MMA, `sitofp`, the same fma; bf16 and fp16 chain the MMAs into the fp32 accumulator with no scales. A K remainder of 16 is a peeled block with one MMA. Cost: block scaling is 25 to 27 percent of the fused fp8 time and 44 to 49 percent of the int8 time (section 142 part 3).

**4. Boundary: activation, block amax, scale code, conversion** (sections 138, 141). In fp32, in this order: optional per-column bias (a `<4 x float>` table per lane expanded `[0,1,2,3,0,1,2,3]`), the activation on the D tile, gating by an elementwise product; then per row-wise block of 32 (two tiles of a tile row) or column-wise block: (1) integer maximum of the absolute bit patterns after flushing fp32 subnormals to zero, (2) the cross-lane maximum by `simd_shuffle_xor` with masks 1 and 8 (row-wise: 4 shuffles per free tile and block) or 2, 4 and 16 (column-wise: 12), (3) the scale code by the ceil rule `((amax_bits + BIAS) >> 23) - EMAX` with BIAS = 0x1fffff and EMAX = 8 (e4m3fn) or 15 (e5m2), 0x1ffff and 6 for int8, which needs no clamp (the floor rule needs a software clamp to the format's largest finite value), code 255 for a block with NaN or Inf, code 0 for zero, (4) multiply by the factor 2^(127 - code) built from the code by integer ops, (5) convert: `air.convert` to fp8 (round to nearest even, no saturation: values beyond 464 for e4m3fn become NaN, the ceil rule keeps them in range), int8 by `rint`, `fptosi`, `trunc`, bf16 and fp16 by `fptrunc`. The codes are stored, or used from registers by the consumer's epilogue (the consumer's factor is the quantizer's code, transposed by shuffles in At and Bt). Cost: 0 to 5 percent in fp8, 2 to 10 percent in int8; the activation 0 to 8 percent (fp8), 2 to 11 (int8); hardware exp2 and the softmax bookkeeping of attention under 3 percent (section 142).

**5. Streaming schedule** (section 141). The loop nest of the FFN family: an outer runtime loop over hidden chunks of `ch` tiles that carries the down-GEMM accumulators D2; inside it a runtime loop over K1 blocks of 32 that carries the chunk's D1 tiles (all tiles of the chunk in one loop so that A fragments are loaded once per m and B fragments once per n); after the K loop the peeled remainder block, then boundary and stage 2 for the chunk, then the loop-carried D2 update. Trip counts are control words; block capacity and table base are compile-time. A three-GEMM chain has no chunk loop: K loop, boundary, static stage 2, boundary, static stage 3, epilogue. Attention: a runtime loop over key blocks that carries the running row max and partial row sum (two floats per lane per token tile) and O; per block S, hardware exp2, per-lane partial sums (tile-ordered vector sum, four slots in order, fma with alpha), P quantized as the left operand, O rescaled by exp2(m_old - m_new) and accumulated; the row sum is reduced across the four lane groups of a row once, at the end (xor 1, then xor 8). Rules: never unroll the K loop (the unrolled kernels need 1.3 to 1.8 times the runtime-loop time at 8 blocks and 5.9 to 7.2 times at 32, with 24 to 923 spill stores, section 141 part 7); keep every loop-carried accumulator updated by an fp32 fma that writes the register it reads (the compiler did in 100 percent of the single-simdgroup loops); in a static chain stream the free tile axis (section 140).

**6. Register and accumulator budgets** (sections 134, 135, 142). The hardware register domain is R0..R125 (126 registers, 15 accumulator groups of eight); R126 and above are squashed. The OS compiler's spill decisions are not the hardware limit. For the emitted IR the measured spill-free frontier of the fused FFN chunk kernel (silu), output tiles per simdgroup: fp8 6 (2 hidden tiles per chunk, one token tile; 124 registers), 1 (4 hidden tiles or two token tiles), bf16 8, 3, 2, int8 4, 2, 1; the stage-2-only kernel 8, 6, 3 (fp8), 10, 8, 4 (bf16), 6, 8, 3 (int8); the stage-1-plus-quantize kernel never spilled (56 to 109 registers); the fused three-GEMM chain with one token tile is spill-free to first and second widths of 4 tiles with 2 to 4 output tiles and to a 2-tile first width with 8 output tiles. Accounting: accumulator state 8 registers per fp32 tile, plus operand fragments in flight (an fp8 fragment unpacks to 4 bf16 registers, a bf16 or fp16 fragment is 4) and scale vectors: 60 to 76 registers of transient state in fp8.

**7. Fusion and cut rules** (section 142). (1) Fuse the whole chain when the fused kernel is spill-free: 5 to 10 percent faster than any cut and 8 to 20 percent faster than three kernels at 64 simdgroups per core (differences below 3 percent are not claimed; noise 1.7 points on ratios). (2) When the fused kernel spills 15 or more stores, cut after the quantize boundary or before it (0 to 5 percent less time at 64 simdgroups, 1 to 17 at one; the two cuts within 5 percent of each other); the quantize cut moves 1 byte per element plus the codes instead of 4 bytes. (3) In-kernel byte bridges (store, workgroup barrier, reload of the quantized tiles and codes) cost -2 to 11 percent and do not relieve register pressure (their kernels spilled as many or more stores than the fused ones: 26 against 22 at (1,4,4,64,4), 78 against 69 at (2,2,4,128,4)): they are dominated by the cuts. (4) At one simdgroup per core and long K the cuts take 0.93 to 0.99 of the fused time. (5) Fuse the two stage-1 accumulations of a gated FFN (fused over three kernels 1.03 to 1.30). (6) A three-GEMM chain: fuse both boundaries when spill-free; if the fused kernel reaches 126 registers, cut after stage 1 (one chain: 22 percent). (7) Attention: fuse, and bridge P through memory only when registers force it (3 to 8 percent).

**8. Across simdgroups** (sections 133, 143). Registers never forward across simdgroups (a cross-simdgroup `simd_shuffle` returns the caller's own value, 256 of 256; the cooperative accumulator reuse stays inside one scope). Hand off the quantized fragments and their codes in fragment order, one slot per producing simdgroup, through threadgroup memory (`air.wg.barrier(i32 2, i32 1)`) or a per-threadgroup device region (`air.wg.barrier(i32 1, i32 1)`): a store by the producer, a barrier, loads by every consumer, a barrier before the next round overwrites. A consumer's load returns the producer's register bits in the same lane and slot, so no relayout exists. Rules: never recompute stage 1 per simdgroup (1.16 to 6.66 times the shared time); share at low occupancy (1.7 to 10 times faster than one simdgroup holding all tiles) and when one simdgroup would hold 16 fp8 output tiles (134 to 138 spill stores: 8 to 15 percent less time); keep the tiles in one simdgroup otherwise (sharing costs 3 to 22 percent when the single simdgroup fits, bf16 up to 16 tiles); threadgroup and device memory are within 8 percent.

**9. Operand-class rules** (sections 138, 142, 143). Inside the cache: bf16 or fp16 (5.4 to 6.7 ns per MMA per core, at the MMA rate), then int8 with block scales (5.7 to 9.0), then fp8 (7.2 to 9.3). Beyond a working set of about 20 to 40 MiB of bf16 bytes: int8 (7.6 to 8.1 ns), then fp8 (9.1); bf16 is slower than fp8 beyond 84 to 169 MiB. The fp8 MMA is the bf16 MMA on unpacked operands (MMA-only 5.3 to 6.4 ns in fp8, bf16, fp16; 2.7 to 3.7 in int8). e4m3fn and e5m2 cost the same. int8 with a coarse scale (no per-block epilogue) would run at 3.4 to 4.5 ns per MMA (timing only). Choose fp8 for capacity and bandwidth, not for compute speed inside the cache.

**10. Optimal memory-break points** (sections 142, 143). In order of preference when a break is required: (1) after stage 1's accumulate but keeping the activation and the quantize with stage 2 (`p1|p23`, D1 as fp32) or (2) after the quantize (`p12|p3`, fp8 bytes and codes): equal within 5 percent, (2) moves 4 times fewer bytes; (3) the in-kernel bridge is dominated (no register relief, -2 to 11 percent); (4) never between the K loop and its epilogue and never inside the K loop; across simdgroups, the fragment-order shared buffer of part 8. A three-kernel schedule (`p1|p2|p3`) costs -1 to 30 percent over fused (the largest at K1 = 64) and is the worst form except at one simdgroup per core.

**11. Numeric domains found** (sections 138, 141). fp32 ALU operations flush subnormal inputs and results to zero; the MMA keeps gradual underflow. `air.fast_tanh(t)` is NaN for t >= 44.364: a tanh-based silu is NaN from x = 88.7277 and a tanh-based gelu from x = 10.0623 (clamp the argument or use `air.tanh`, 45 instructions longer); `llvm.maxnum` with a NaN operand returns the other operand (relu(NaN) = 0); `air.fast_divide` and `air.erf` do not lower; exp2, log2 and rsqrt are single instructions (`op1272`, `op2570`, `op3850`; `air.fast_sqrt` is `op3978`, another reciprocal-square-root form whose zero returns 1.0, plus a multiply; the zero behavior is the cartography lane's measurement, section 141 part 8), `fdiv` is 9 instructions longer than a single-instruction operation; an int8 conversion of a block with a NaN or Inf gives unspecified bytes; the fp8 pack does not saturate (section 138), so the scale rule must keep values in range; an fp16 chain overflows at 65,504.

**12. Open, and what a compiler must not assume.** Not measured: the D to At, B and Bt readings inside the hidden-chunk loop (measured with the K loop and the repeated timing loop), int8, bf16 and fp16 readings other than A in loops, a stage 1 emitted by `tlower`, tiles shared by more than 8 simdgroups, a DRAM-streaming single pass, fp4 and fp6, and the causes of two timing results (the spill-free fused silu chain at one simdgroup per core, 1.5 to 1.85 times its cuts, and the 126-register chain at 64 simdgroups, 22 percent). Do not treat the OS compiler's spill counts as hardware limits (the compiler used 126 registers for the whole domain in every spilling kernel, which is R0..R125), and do not extend any table beyond the measured grids.

**Labels.** *Hardware (measured on this H17s):* every rule's evidence in sections 138 to 143. *Model:* the fragment relations (tables, checked on hardware, section 140). *Assembled here, not new:* the rules as a single specification. Artifacts: those of sections 138 to 143.

## 145. Streaming, hand-off and instruction-fetch constants of H17s for the layer scheduling model: an SLC plateau at 520 to 528 GB/s up to 20 MiB, DRAM at 281 to 294 GB/s from 40 MiB, a 32 KiB threadgroup memory limit, hand-off rounds of 38 to 160 ns, and a sixfold per-instruction fetch penalty for hot code beyond 26 KB

**Result.** (1) Sustained read bandwidth against footprint (`mxcal.py stream`; every simdgroup reads its own contiguous range of buffer 1 in 512-byte fragments of 16 bytes per lane inside the timing loop; footprint is the bytes read per iteration by all simdgroups; 20 GPU cores). Footprints of 10 and 20 MiB run at 520.3 to 527.8 GB/s (six measurements), footprints of 40 MiB and above at 280.7 to 293.7 GB/s (seven measurements, 40 to 160 MiB), and footprints up to 5 MiB at 1.3 to 4.8 TB/s (on-core caches; the timing iteration lasts 0.2 to 1.7 us there, so the figure includes loop overhead and is a lower bound). The same plateaus hold for 1, 8 and 32 simdgroups per threadgroup (320, 320 and 640 simdgroups in flight), so at these counts bandwidth does not depend on the threadgroup shape. The SLC to DRAM step lies between 20 and 40 MiB in this access pattern (one simdgroup per threadgroup: 521.7 GB/s at 20 MiB, 280.9 at 40 MiB). (2) Threadgroup memory: kernels declaring 16 and 32 KiB compile and run; 48, 60 and 64 KiB are refused by the native compiler with "Threadgroup memory size (N) exceeds the maximum threadgroup memory allowed (32768)"; 96, 128 and 256 KiB end in a dropped compiler connection (`XPC_ERROR_CONNECTION_INTERRUPTED`), which says only that they do not compile. The limit is 32,768 bytes. (3) A hand-off round (every simdgroup stores a value, all execute `air.wg.barrier`, each loads a neighbour's value) costs, through threadgroup memory, 38.0 ns at one simdgroup per threadgroup with one threadgroup per core, 56.6 to 59.6 ns at 2 to 8 simdgroups, 65.3 ns at 16 and 81.9 ns at 32; with 64 simdgroups per core the same shapes cost 99.6 to 131.1 ns. Through a per-threadgroup region of device memory (barrier flag 1) the round costs 57.6 to 83.0 ns at one threadgroup per core and 150.8 to 161.0 ns at 4 to 32 simdgroups per threadgroup with 64 simdgroups per core. At 32 simdgroups per threadgroup the device round is 1 percent above the threadgroup round at one threadgroup per core (83.0 against 81.9 ns) and 17 percent above at 64 simdgroups per core (153.0 against 131.1 ns). (4) Straight-line code is fetched at a rate that falls with code size (`mxcal.py icache`; sixteen independent fp32 fma chains, no memory, repeated by the timing loop; ns per instruction at one and at 64 simdgroups per core):

```
instructions  code bytes   1 SG/core   64 SG/core
     117         1,084       0.411       0.095
     251         2,318       0.522       0.097
     386         3,426       0.774       0.128
     644         5,498       0.896       0.139
   1,160         9,606       1.390       0.144
   2,189        17,844       1.933       0.160
   3,213        26,040       2.492       0.175
   4,237        34,232       2.461       0.176
```

The per-instruction time at one simdgroup per core grows sixfold from 1 KB to 26 KB of code and is flat beyond; at 64 simdgroups per core it grows 1.85 times over the same range. The code is repeated by the timing loop, so the table is the cost of a loop body that does not fit the instruction cache, and 26 KB (about 2,800 instructions of about 9 bytes) is where the plateau starts.

**Method and errors kept.** The kernels are hand-authored AIR (`mxcal.py`: `gen_stream`, `gen_barrier`, `gen_icache`, `gen_tgmem`; harness `regprobe_multi`; logs `mxcal_stream_sg1.log`, `mxcal_stream_sg8.log`, `mxcal_stream_sg32.log`, `mxcal_barrier_tg.log`, `mxcal_barrier_dev.log`, `mxcal_icache.log`, `mxcal_tgmem.log`). (a) My first barrier kernel issued eight barriers back to back and the compiler merged them into one, so its per-barrier times were eight times too small; the store, barrier, load round is the rewrite, and the instruction count of each kernel is in the log (75 to 87). (b) The first device hand-off kernel hung for more than 14 minutes: its scratch offset used `nsg * 32 * 4` elements, past the end of the region; the offset is `nsg * 32` and the run completed. (c) The 1 and 2 simdgroup device rows at 1,280 and 640 threadgroups (246.6 and 379.9 ns) are outliers against the 150.8 to 161.0 ns of the other shapes and were not investigated.

**Consequences used by the model.** Three bandwidth levels (about 3 TB/s on core to 5 MiB, 0.52 TB/s to 20 MiB, 0.28 TB/s beyond 40 MiB), a threadgroup memory budget of 32,768 bytes (the per-kernel formula is in section 146), a hand-off round that costs 0.04 to 0.16 us (the fused FFN of section 146 runs 8 rounds of two barriers per pass at Hh = 8192, 16 barriers, under 3 us of a 1.8 ms pass, so the synchronization itself is not what separates its schedules), and a hot-loop size guideline of 26 KB.

**Labels.** *Hardware (measured on this H17s):* every number above. *Not measured:* mixed read and write traffic, access patterns other than contiguous 512-byte fragments, page-table effects at footprints above 300 MiB, threadgroup memory bandwidth (only the hand-off round), and instruction-cache size as such (the table gives the cost curve of one code shape).

## 146. The gated FFN at layer scale (d = 2048, Hh = 8192, up to 1,280 tokens) is bit-exact through fused, two-kernel and three-kernel schedules in fp8, int8 and bf16; in fp8 and int8 the fused and two-kernel schedules tie at one wave and the fused kernel loses 7 to 11 percent at two and four waves, a spilling bf16 fused kernel loses by about 1.5 times, one wave is 20 threadgroups, and the threadgroup memory budget bounds the fused form (the device-shared timing rows are withdrawn: part 7)

**Result.** (1) Problem: gated FFN (gate and up projections, silu, product, quantize, down projection), residual and RMS norm across the width d = 2048; hidden width Hh = 8192 in 256 chunks of 32; one threadgroup per block of 16 tokens (mt = 1), 32 simdgroups per threadgroup each holding 4 output tiles of 16 columns (16 in bf16 at 16 simdgroups: 8 tiles); weights (gate, up, down) 3 x 2048 x 8192 = 48 MiB in fp8 and int8 and 96 MiB in bf16. The generator is `mxpipe.py` (section 141) with one task per threadgroup: tokens by threadgroup, weights shared by all threadgroups. The three schedules differ only in where the hidden activation lives: fused (one kernel; chunk rounds of 32 chunks, one per simdgroup, exchanged through threadgroup memory or a device region, then every simdgroup applies all 32), two kernels (`p12` = stage 1, activation, quantize, store the quantized chunks; `p3` = down projection, residual, norm), three kernels (`p1` stores fp32 D1; `p2` activation and quantize; `p3`). (2) Exactness, layer scale, finite data (`mxlayer_host.py`, `mxlayer_ffn_exp.py exact`, logs `mxlayer_ffn_exact_e4m3.log`, `mxlayer_ffn_exact_i8.log`, `mxlayer_ffn_exact_bf16_sg16.log`, `mxlayer_ffn_exact_bf16_sg32dev.log`): 4 threadgroups (64 tokens), the first two compared with the composed reference (batched, `mxfast.py`, the hardware silu and rsqrt through the isolated kernels, section 141 part 2): D2 and Y, 32,768 elements each per threadgroup, 0 mismatches for fused, two-kernel and three-kernel schedules in e4m3 (32 simdgroups, threadgroup memory), int8 (32, threadgroup memory), bf16 (16 simdgroups, threadgroup memory) and bf16 (32 simdgroups, device memory): 4 configurations x 3 schedules x 2 threadgroups = 24 comparisons, 24 exact, and the reference is finite over 100 percent of D1, the activation output, D2 and Y in every one (the section 141 correction, part 10 (f), applies here: a first attempt at this scale compared NaN with NaN and is not counted; `mxlayer_ffn_big.py`). The kernels are 9,389 to 9,568 instructions (fused fp8), 745 and 693 (two-kernel fp8) and 480, 374 and 693 (three-kernel fp8).

(3) Timing (`mxlayer_ffn_exp.py time`, log `mxlayer_ffn_time1.log`; ms per pass of the whole FFN, sum of the kernels of the schedule, median of 7 interleaved rounds; `sets` copies of the weights and activations of which the timing loop touches copy `it & (sets - 1)`, so sets = 3 alternates between copies 0 and 2 (three are allocated, two are touched; the drivers now assert a power of two); allocated footprint 65 to 115 MiB at sets = 1 and 196 to 345 MiB at sets = 3, of which two thirds is touched at sets = 3; TFLOP/s = 2 x tokens x 3 x d x Hh over the fastest schedule; spill stores are the compiler's stack stores in the kernels of the schedule). Sets = 1:

```
fmt   SG hand tokens ntg |  fused ms (spill stores) | two ms (spills) | three ms (spills) | best TFLOP/s
e4m3  32 tg    160  10 |   1.780 (  26) |   1.757 (  0) |   1.806 (  0) |   9.2
i8    32 tg    160  10 |   1.626 ( 26*) |   1.621 (  0) |   1.651 (  0) |   9.9
bf16  32 dev+  160  10 |   3.697 (1040) |   1.649 (  0) |   1.677 (  0) |   9.8
bf16  16 tg    160  10 |   2.868 ( 525) |   1.841 ( 42) |   1.877 ( 42) |   8.8
e4m3  32 tg    320  20 |   1.879 (  26) |   1.839 (  0) |   1.905 (  0) |  17.5
i8    32 tg    320  20 |   1.703 ( 26*) |   1.722 (  0) |   1.777 (  0) |  18.9
bf16  32 dev+  320  20 |   7.095 (1040) |   1.828 (  0) |   1.919 (  0) |  17.6
bf16  16 tg    320  20 |   3.056 ( 525) |   2.057 ( 42) |   2.180 ( 42) |  15.7
e4m3  32 tg    640  40 |   3.837 (  26) |   3.621 (  0) |   3.818 (  0) |  17.8
i8    32 tg    640  40 |   3.393 ( 26*) |   3.181 (  0) |   3.356 (  0) |  20.2
bf16  32 dev+  640  40 |  13.684 (1040) |   3.555 (  0) |   3.753 (  0) |  18.1
bf16  16 tg    640  40 |   6.492 ( 525) |   4.251 ( 42) |   4.501 ( 42) |  15.2
e4m3  32 tg   1280  80 |   8.842 (  27) |   8.067 (  0) |   8.488 (  0) |  16.0
i8    32 tg   1280  80 |   7.713 (  26) |   7.138 (  0) |   7.497 (  0) |  18.1
bf16  32 dev+ 1280  80 |  29.590 (1040) |   7.789 (  0) |   7.986 (  0) |  16.5
bf16  16 tg   1280  80 |  12.418 ( 525) |   8.332 ( 42) |   8.839 ( 42) |  15.5
```

Rows marked dev+ (fused with device-memory sharing at 32 simdgroups) are withdrawn, part 7: the fused kernel ran out of bounds in the timing driver, and the two-kernel and three-kernel times of those rows were measured in the same process. Sets = 3 (allocated footprint 196 to 345 MiB, two thirds touched) gives the same picture for the other rows; its 16 rows are in the log. (4) One wave is 20 threadgroups. The two-kernel time relative to the 20-threadgroup time is 0.96, 1.00, 1.97 and 4.39 at 10, 20, 40 and 80 threadgroups in e4m3 (0.94, 1.00, 1.85, 4.15 in int8): flat below the 20 cores (one 1,024-thread threadgroup per core), then linear in waves. One wave costs 1.8 ms in fp8, 1.7 ms in int8 and 2.1 ms in bf16 at 16 simdgroups, that is 0.79 to 0.95 TFLOP/s per core (15.7 to 18.9 TFLOP/s at 320 tokens) against 1.66 TFLOP/s per core for one MMA per 4.94 ns (section 136). bf16 moves twice the weight bytes of fp8 and int8 in 1.1 times the time (100.7 MB against 50.3 MB per threadgroup pass, 2.0 GB against 1.0 GB per 20-threadgroup wave, 0.98 TB/s against 0.55 TB/s of logical weight traffic), which is above the 0.52 TB/s SLC plateau of section 145: the twenty threadgroups walk the same weights together, so the bytes come from shared caches, and at this shape the run is limited by MMA and ALU issue and not by the weight stream. (5) Schedules. Fused against two-kernel time in fp8 and int8 over the 16 rows of the sweep: 0.95 to 1.14 (median 1.03); three-kernel against two-kernel: 1.02 to 1.06 (median 1.04). Repeated in fresh processes (`mxlayer_ffn_repeat.log`, `mxlayer_ffn_time3.log`; ratio fused / two): 10 threadgroups 1.013 (e4m3); 20 threadgroups 1.028 (e4m3) and 0.999 (int8); 40 threadgroups 1.076 (e4m3) and 1.069 (int8); 80 threadgroups 1.111 and 1.108 (e4m3, two runs) and 1.103 and 1.091 (int8, two runs) against 1.096 and 1.081 in the sweep; three-kernel / two 1.019 to 1.059 in every run. The ratio reproduces to about 1.5 percent and is a real effect that grows with the number of waves: a tie at one wave (10 and 20 threadgroups), the two-kernel form 7 to 11 percent faster at two and four waves in fp8 and int8 (the sets = 3 rows agree in e4m3, 1.07 and 1.14, and show 1.00 in int8). The fp8 fused kernels carry 26 to 46 spilled stores and the int8 fused kernels 26 to 39 (126 registers; the three rows marked * in the table were decoded only up to the first instruction that `agx3dis` reports as bad, 461 of about 9,000 instructions, so their 26 is not independently established, although the 80-threadgroup row of the same kernel decodes in full and also shows 26), and the cut kernels none; the three-kernel form is 2 to 6 percent slower than the two-kernel form in every row of the two classes. In bf16 at 16 simdgroups with threadgroup sharing the fused kernel loses clearly: 1.45 to 1.56 times the two-kernel time (525 to 543 spilled stores), and the two-kernel form is 1.02 to 1.06 times faster than the three-kernel form. Whether 32 simdgroups (the cut kernels `p12` 401 instructions and `p3` 473, no spills, 80 and 112 registers, against 8 output tiles and 42 spilled stores in the `p3` at 16 simdgroups) is faster for bf16 is not established: the 32-simdgroup rows are withdrawn. Run-to-run spread, measured (`mxclock.py`, `mxclock_s2048.json`; a fused attention kernel of 0.32 ms per pass, 40 recorded rounds after the harness's warm-up, 2, 8, 32 and 128 iterations per launch): the per-round time has a coefficient of variation of 0.4 to 1.3 percent and max / min of 1.018 to 1.063, with two weakly separated modes (Ashman D 2.6 to 9.1) 1 to 5 percent apart, so there is no clock-state bimodality of the size Rigel reports for the M4 Max (4.65 against 6.5 TFLOP/s at 512^3, arXiv 2606.12765) inside a run. Between runs: the same e4m3 configuration (20 threadgroups, sets = 3) measured in two separate processes gave 1.878, 1.773 and 1.844 ms and 2.056, 1.997 and 2.077 ms (9 to 13 percent apart); sets = 1 rows re-measured in a fresh process later the same day agree within 4 percent at 10, 20 and 40 threadgroups and moved by 18 percent at 80 threadgroups (four waves: the two-kernel time was 8.07 ms in the sweep and 6.64 ms twice later). Absolute times of many-wave runs and of sets = 3 runs therefore carry 10 to 18 percent between processes, and ratios between schedules timed together in one process reproduce to about 1.5 percent. Between sets = 1 and sets = 3 the two-kernel time moves by 0.89 to 1.21 times in fp8 and int8 and by 0.97 to 1.01 times in bf16 at 16 simdgroups, so the extra footprint (up to 230 MiB touched) does not change these runs; the 12.08 ms row that this paragraph first called an unexplained sets-3 exception was a withdrawn device-sharing row (part 7).

(6) Threadgroup memory bounds the fused form. The shared buffer holds, per simdgroup and token tile, the quantized chunk (512 bytes in fp8 and int8, 1,024 in bf16 and fp16), the block codes (64 bytes, absent in bf16 and fp16) and, with the norm epilogue, the row-sum partials (64 bytes), so the fused kernel needs nsg x mt x 640 bytes in fp8 and int8 and nsg x mt x 1,088 in bf16 and fp16 (the unused code array is removed by the compiler). The native compiler refuses more than 32,768 bytes: e4m3 with mt = 2 and 32 simdgroups asks for 40,960 bytes and fails ("Threadgroup memory size (40960) exceeds the maximum threadgroup memory allowed (32768)"), bf16 with mt = 1 and 32 simdgroups asks for 34,816 and fails, while e4m3 with mt = 1 and 32 simdgroups (20,480 bytes) and bf16 with 16 simdgroups (17,408) compile. The fused form with threadgroup sharing therefore needs nsg x mt <= 51 in fp8 and int8 and <= 30 in bf16 and fp16; the device-memory form has no such bound (its exactness at bf16 and 32 simdgroups is in part 2; its timing is withdrawn) and costs 0 to 17 percent more per hand-off round (section 145).

(7) Errors kept and rows withdrawn. (a) The timing driver sized its shared out buffer from the three-kernel layout (`scratch_state`), but the fused kernel with device-memory sharing keeps its shared quantized chunks in a region that comes last in its layout, so in every dev+ row that region started at the end of the buffer and lay 0.69 MB (bf16, 20 threadgroups) to 2.78 MB (bf16, 80 threadgroups) beyond it (1.39 MB for the bf16, mt = 2, 20-threadgroup configuration and 0.36 MB for the e4m3 configuration that was queued). Those kernels read and wrote outside the buffer they were given. The exactness runs of part 2 sized the buffer from the fused kernel's own layout and are not affected. The driver now sizes the buffer for the layouts of all its kernels (`out_bytes_of`) and asserts it (`assert_fits`); the dev+ rows are withdrawn and were not repeated. (b) The device region was also reserved at 2,048 slots per threadgroup (`reg('S', 2048 * slot)` times the threadgroup count), so the device-shared runs handed the harness out buffers of 1.4 to 5.7 GB; it is one slot per threadgroup now (an 82 MiB buffer for the mt = 2, 20-threadgroup case). (c) `sets` has to be a power of two and was not checked (part 3). (d) The last configuration of the first sweep (bf16, mt = 2, 32 simdgroups, device sharing, 20 threadgroups, sets = 3) was in flight when that session was interrupted and the machine rebooted (boot at 16:23:36 on 2026-09-19, as a peer session measured); I ran it again after the reboot for about 55 seconds and stopped it; the GPU was then reported at 100 percent utilization with no process holding a client, and uptime at 16:56 showed a second boot at about 16:53. A peer session (the cartography lane, relayed to me) reports that a crash report attributes the 16:23 reboot to a page fault (BIF0) raised by `regprobe_multi` on this configuration, which is consistent with the out-of-bounds region of (a), and that the GPU stayed busy afterwards because a SIGTERM on the host process does not cancel a command buffer that is already submitted; I have not read the crash report myself. [Corrected 2026-09-23, from Set C's reading of the crash report and system log; I have not re-read them myself: the BIF0 read fault (`gpuEvent-regprobe_multi-2026-09-19-161533.ips`, `is_read: true`) came from a `regprobe_multi` tensor kernel whose configuration was never recorded. The configuration named here was a DIFFERENT process (pid 76676), which HUNG the GPU for about 320 s, starting about 71 s after the fault, and tripped the WindowServer watchdog at 16:22. That hang left no crash report. So this configuration is known to hang, not to fault, and the read fault is not attributed to any recorded configuration. An ordinary shader load out of range returns 0 and does not fault; only the accelerator's own reads have faulted.] The second boot was the user's.

**Labels.** *Hardware:* every time, spill count and exactness count above except the withdrawn dev+ rows; the instruction, register and spill counts are from the decoded code (`regprobe_ana.py`; `agxforge.g17.model.decode` frames instructions with `tools/agx3dis` and parses its text). Checked over the 1,924 kernels compiled in this lane: no exception (the base-10 `int()` on hexadecimal operand tokens that a peer found in `decode` did not fire), decoded bytes equal code bytes in 1,921, and in 3 fused int8 layer kernels the decode stops after 4,364 of 84,412 bytes because `agx3dis` reports a bad instruction at offset 0x110c (bytes `a700a518220ea182c00a2302870030a0`, previous instruction op10279 of length 12; resuming after it decodes the remaining 8,500 instructions at several even offsets, so the length of that form is not settled). Those three kernels are the three starred rows; the kernels run and are bit-exact on hardware. *Model:* the composed reference for exactness. *Not measured in this section:* mt = 2 timings (the fp8 fused form does not compile at 32 simdgroups), fewer simdgroups per threadgroup at fp8, K unrolling, and the small-token regime below one wave (16 to 160 tokens use 1 to 10 cores; the weight-stationary schedule that would use all 20 is not built here).

## 147. The attention subgraph (QK^T, online softmax, AV, output projection, residual and RMS norm) runs as independent simdgroup tasks and is bit-exact through fused, P-materialized, S-materialized and fully cut schedules, with K and V in the transposed-B mode, in fp8, int8, bf16 and fp16; the memory-mediated schedules reproduce the fused bits exactly

**Result.** (1) Construction (`mxpipe.py`, `mxattn_host.py`, `mxattn_exp.py`; the section 141 attention flow with five additions). (a) One task per simdgroup: task = threadgroup x `tgs` + simdgroup x `sgs` (cfg keys), the query tile of task t at `t x XSTEP`, the K and V chunks of head `t / nq` (`wdiv`), outputs in per-task regions; `same_head` puts consecutive query tiles of one head in one threadgroup (`tgs = nsg`, `sgs = 1`), `cross_head` gives a threadgroup tasks that are `tasks / nsg` apart (different heads); no barrier and no threadgroup memory are used. Each simdgroup holds the whole head width (dh / 16 output tiles) and the weight layout is that of one simdgroup (`sgtask` sets the layout's simdgroup count to 1). (b) Cut points inside the attention loop, named as in the FFN: `p1` computes S = Q K_j^T and stores it in fp32 D layout; `p2` loads S, runs the online softmax (row max by xor shuffles 1 and 8, running max, hardware `exp2`, per-lane row-sum partials by fma with the rescale factor), quantizes P and stores the quantized fragments, the block codes, the rescale vector alpha (one `<2 x float>` per lane and chunk) and at the end the row-sum partials; `p3` loads them, rescales the running output, accumulates P V_j and normalizes; `p12` and `p23` are the two fusions of neighbours and `all` is the fused kernel. Schedules: fused = `all`; P materialized = `p12` then `p3`; S materialized = `p1` then `p23`; both = `p1`, `p2`, `p3`. (c) The normalized output can be quantized per 32 columns straight into the chunk layout that the FFN `p3` kernel reads (`oq`, offsets given by the consumer's layout, `out_shift` keeping the attention's own regions clear of it); the output projection, residual and RMS norm across the model width are then that `p3` kernel with the concatenated head outputs (nh x dh / 32 chunks) as its hidden dimension (`Oproj` in `mxattn_host.py`), one threadgroup of 32 simdgroups per query tile. (d) `tb1` and `tb2` select the transposed-B MMA flag for stage 1 (K given row-major) and stage 2 (V stored transposed), with the host packing `Bt` fragments (section 140); the reference is unchanged, so the bits must not change. (e) A subset of the scratch regions can be allocated (`regions`); kernel names carry a checksum of the layout keys.

(2) Reference (`Attn.reference`): S by the composed MMA model over the head width, scaling by the fp32 constant, per-chunk row maximum and the running maximum as an exact cumulative maximum, alpha and P through the isolated hardware `exp2` kernel (two batched calls per task), the lane-partial row sums in the kernel's order (tile-ordered vector sum, four slots in order, fma with alpha), P quantized by the section 138 quantizer, the output accumulated chunk by chunk (rescale by alpha, then the post-scaled model of P V_j), the final row sums reduced by xor 1 then xor 8 and inverted by the hardware reciprocal. Every store of a cut is compared as well: S fragments, quantized bytes, block codes, alpha and the row-sum partials.

(3) Small matrix (`attn_exact_small.sh`, `mxattn_exact_small.log`; 2 heads, 4 query tiles, tasks 0, 1, 4 and 7 compared: the first two tasks of head 0, the first task of head 1 and the last task): nine configurations, each run through the four schedules, four tasks each, 144 task comparisons, 0 mismatches at O, Y and at every stored boundary; the four schedules give identical Y bits in all nine; the reference is 100 percent finite at S, P, O and Y in every task. The configurations: e4m3 (dh 64, 256 keys) at 1 and at 4 simdgroups per threadgroup; e4m3 (dh 128) at 2 with both transposed flags; e4m3 (dh 64, 320 keys) at 4 with two token tiles per task and the `cross_head` map; bf16 at 4; bf16 at 2 with both flags; int8 at 4; fp16 (192 keys) at 4 with two token tiles and `tb2`; e5m2 at 2 with `tb1`.

(4) Layer-scale chain (`attn_chain_layer.sh`, `mxattn_chain_layer.log`; 8 heads of width 128, 8 query tiles of 16 queries per head, model width 2048, output projection of 8 x 128 / 32 = 32 chunks; the attention kernels run first and the projection kernel next on the same out buffer, with the debug stores on): (i) e4m3, 4,096 keys per head (128 key blocks of 32 per task), 8 simdgroups per threadgroup, 64 tasks; (ii) bf16, 2,048 keys, 4 simdgroups, `tb1` and `tb2`; (iii) int8, 2,048 keys, 8 simdgroups, `cross_head`; (iv) e4m3, 2,048 keys, 8 simdgroups, `tb1`. For query tiles 0 and 7 and each of the four schedules: the output of all 8 heads (2,048 elements each) against the reference, and the projection's D2 and Y (32,768 elements each) against the composed projection, residual and norm reference: 4 x 4 x 2 = 32 tile comparisons, all 0 mismatches, the schedules identical in all four, the references 100 percent finite. Kernel sizes for (i): fused attention 1,925 instructions; `p12` 519 and `p3` 1,474; `p1` 263 and `p23` 1,671; `p1`, `p2`, `p3` 263, 314 and 1,474; projection 696.

(5) Errors kept. (a) My first chain run gave wrong output for every task of head 0 in the two schedules that read stored P, and right output in the other two. The cause was mine: the chain mode had turned the debug stores on and also restricted the scratch regions to a subset, so the debug stores of S and P (which write every region) ran past 256-byte regions into their neighbours. Isolating it (the same failure without the projection, `p12` alone storing 1,662 wrong quantized bytes) found it; chain mode now allocates every region, and `layout` asserts that a region subset is used only without debug stores. (b) My first selection of the tasks to compare was truncated to the first three tasks, all of head 0, so the second head's data path was not exercised; the selection now takes the first task of head 0, the second, the first of head 1 and the last.

**Labels.** *Hardware:* every count above. *Model:* the composed reference; the hardware `exp2`, `rcp` and `rsqrt` of the reference are the isolated-kernel results of the same emitters (section 141 part 2). *Scope:* the layer-scale KV footprint is 8 heads x 4,096 keys x 128 x 2 (K and V) = 8 MiB of fp8 values, about 18 MiB with the layout's padding, so it is cache-resident (section 145); exactness at a footprint above 40 MiB, with mt = 2 at layer scale, and with the K and V producers (projections that write K and V in B-operand layout) was not run, and no attention timing exists yet in this document.

## 148. Attention at layer scale: the fused kernel is the fastest schedule in every measured shape, cutting it costs 3 to 30 percent (P), 7 to 42 percent (S) and 14 to 103 percent (both), one simdgroup's dependent chain costs 5.35 us per 32-key block so about 250 simdgroups saturate the GPU, and splitting the KV range by 8 turns the few-task decode case into a DRAM-bound one

**Result.** (1) Kernels and harness: the section 147 attention kernels timed with `mxattn_exp.py time` (`attn_batch.sh`; logs `mxattn_time_b1.log` to `mxattn_time_b7.log`); each row is the sum of the kernels of the schedule, median of 7 interleaved rounds, `sets` copies of the KV and queries of which the loop touches copy `it & (sets - 1)`, the out buffer sized from the schedule's own layout and asserted. Unless stated, e4m3, head width 128, one 16-query tile per task (mt = 1), one task per simdgroup, the KV (K and V, block-scaled fragments and scale tables) 1.25 bytes per element, `same_head` mapping, 8 simdgroups per threadgroup. FLOPs are 2 x tasks x 16 x 2 x dh x keys. (2) Schedules (fused; P materialized = `p12` then `p3`; S materialized = `p1` then `p23`; both = `p1`, `p2`, `p3`; section 147); ratio to fused in parentheses:

```
shape (format, heads x query tiles, keys)               tasks | fused ms (TFLOP/s, spill stores) | P cut ms (x) | S cut ms (x) | both cuts ms (x)
e4m3 16 x 16, 8192 keys, 8 SG, sets 2             256 |  1.455 (11.8,  42) |  1.635 (1.12) |  1.856 (1.28) |  2.523 (1.73)
e4m3 16 x 16, 2048 keys, 8 SG, sets 1             256 |  0.330 (13.0,  32) |  0.429 (1.30) |  0.464 (1.41) |  0.670 (2.03)
e4m3 16 x 16, 8192 keys, 8 SG, sets 1             256 |  1.495 (11.5,  36) |  1.680 (1.12) |  1.849 (1.24) |  2.454 (1.64)
e4m3 16 x 32, 8192 keys, 8 SG, sets 2             512 |  2.840 (12.1,  42) |  3.150 (1.11) |  3.869 (1.36) |  4.775 (1.68)
e4m3 8 x 32, 32768 keys, 8 SG, sets 2             256 |  5.844 (11.8,  42) |  6.447 (1.10) |  7.383 (1.26) |  9.882 (1.69)
bf16 16 x 16, 8192 keys, 8 SG, sets 2             256 |  1.280 (13.4,  12) |  1.546 (1.21) |  1.785 (1.39) |  2.439 (1.91)
bf16 8 x 32, 32768 keys, 8 SG, sets 2             256 |  4.988 (13.8,  12) |  6.076 (1.22) |  6.791 (1.36) | 10.098 (2.02)
i8 16 x 16, 8192 keys, 8 SG, sets 2               256 |  1.409 (12.2,  27) |  1.590 (1.13) |  1.815 (1.29) |  2.394 (1.70)
e4m3 16 x 16, 8192 keys, tb1 tb2                  256 |  1.474 (11.7,  42) |  1.663 (1.13) |  1.878 (1.27) |  2.478 (1.68)
e4m3 32 x 1, 16384 keys, 1 SG, sets 1 (decode)     32 |  2.739 ( 1.6,  32) |  3.042 (1.11) |  2.948 (1.08) |  3.250 (1.19)
```

Over all 27 mt = 1 rows of the seven logs: P cut 1.03 to 1.30 times the fused time (median 1.13), S cut 1.07 to 1.42 (1.28), both 1.14 to 2.03 (1.65); no row has a cut faster than the fused kernel. The fused kernel spills (126 registers, 42 stores in fp8, 12 in bf16) and still wins; the P cut kernels do not spill in bf16 and lose by 21 to 22 percent. The working sets run from 8 MiB of KV (2,048 keys, 16 heads, sets 1) to 64 MiB per set in fp8 and 128 MiB per set in bf16 (32,768 keys, 8 heads, two sets alternating, 298 and 394 MiB allocated), which crosses the SLC to DRAM step of section 145, and the fused throughput stays at 11.5 to 13.8 TFLOP/s throughout (13.0 at 8 MiB, 11.8 at 64 MiB per set in fp8; 13.4 and 13.8 in bf16): the kernel is limited by the issue rate of its dependent chain, not by the KV stream (logical KV traffic reaches 861 GB/s in bf16, above both plateaus, because the 32 tasks of a head walk the same K and V together). (3) Choices that make no difference: simdgroups per threadgroup 1, 2, 4, 8 and 16 give 1.459, 1.396, 1.449, 1.455 and 1.544 ms (256 tasks; 256, 128, 64, 32 and 16 threadgroups, so 16 threadgroups leave 4 of 20 cores without a threadgroup and take 6 percent longer) and 32 gives 2.920 against 2.840 ms for 8 at 512 tasks; `cross_head` gives 1.425 ms against 1.455 for `same_head`, so putting the tasks of one head in one threadgroup gives no sharing benefit and the caches serve the shared K and V; the transposed-B flags `tb1` and `tb2` cost 1.3 percent (1.474 ms); the K-loop unroll costs nothing and gains nothing (2 and 4: 1.464 and 1.448 ms, with 98 and 110 spilled stores against 42); the dtype changes the fused time little (bf16 1.280, int8 1.409, e4m3 1.455 ms: bf16 is 12 percent and int8 3 percent faster than e4m3, with the same number of KV blocks and twice the bf16 KV bytes). One choice matters: two token tiles per task (mt = 2, 32 queries) with the same 4,096 queries takes 2.917 ms at 5.9 TFLOP/s (354 spilled stores), twice the mt = 1 time.

(4) Occupancy. The fused loop is a dependent chain (QK^T MMAs, row maximum with shuffles, `exp2`, row-sum fma, quantize, PV MMAs, rescale). With 32 or 64 tasks (one simdgroup each, 1.6 to 3.2 simdgroups per core) the time is the same, 0.682 ms at 4,096 keys and 2.739 to 2.741 ms at 16,384 keys, which is 5.35 us per key block of 32 and per simdgroup whatever the number of tasks up to 64; at 256 tasks (12.8 per core) the time per key block is 5.7 us, and at 512 tasks it doubles (2.840 ms, 433 ns per block and core at saturation, 12.1 TFLOP/s). The knee is therefore 5.35 us / 433 ns = 12.4 simdgroups per core, about 250 simdgroups on 20 cores. (5) Splitting the KV range. The decode-like case (32 heads, one query tile each, 16,384 keys) has 32 tasks and takes 2.739 ms. Its bytes cut into 2, 4, 8 and 16 independent slices per head (`mxattn_time_b6.log`; the slices are given as separate heads of 8,192, 4,096, 2,048 and 1,024 keys, which has the addresses and the instruction stream of a split kernel and no merge): 1.409, 0.788, 0.613 and 0.649 ms. At 8 slices (256 tasks) the run reads 128 MiB of K and V elements (160 MiB with the scale tables) in 0.613 ms, 274 GB/s, the DRAM plateau of section 145, and 16 slices give nothing more. (5b) The dtype under memory pressure (`mxattn_time_b7.log`; 128 heads of 2,048 keys, one 16-query tile each, 128 tasks with a KV of their own, fused kernels, 0.34 ms latency floor): e4m3 0.410 ms, int8 0.398, bf16 0.490. The bf16 run touches 134 MB of K and V in 0.490 ms (274 GB/s, the DRAM plateau) and is memory-bound; the fp8 run touches 84 MB in 0.410 ms (205 GB/s) and is not, so bf16 is 20 percent slower and the model puts bf16 at 0.479 ms (-2.2 percent) and fp8 at 0.342 (-16.5 percent, latency bound). The merge of the per-slice maxima, sums and outputs (8 slices x 16 x 128 fp32 per head) is not built and not included; it is one exchange per task through threadgroup memory (one 0.04 to 0.16 us round, section 145) when the slices are the simdgroups of one threadgroup, or a small second kernel. (6) Exactness at this scale (`mxattn_exact_dram.log`, `mxattn_chain_dram.log`): 16 heads, 8,192 keys per head, head width 128 (72 MiB, 75.5 MB, of KV allocated, 40 MiB touched, above the 20 MiB SLC step), 4 simdgroups per threadgroup: the four schedules for tasks 0, 1, 4 and 63 (heads 0, 0, 1 and 15) have 0 mismatches at O, Y and every stored boundary (S, quantized bytes, codes, alpha, row sums), the schedules give identical Y bits, and the reference is finite over 100 percent; the chain with the output projection (model width 2,048, 64 chunks) and residual and norm has 0 mismatches at the output of all 16 heads and at D2 and Y (32,768 elements each) for query tiles 0 and 3 in each of the four schedules.

**Model check** (`mxcost_model.py`, section 150): the two constants of (4) (5.35 us and 433 ns, fp8) and the bf16 pair (5.0 us and 381 ns) predict the 23 measured fused rows that were not used to fit them to a median 2.7 percent, 20 of 23 within 10 percent, worst -16.5 percent (128 heads of 2,048 keys in fp8; -14.0 in int8 and -13.1 for 4 slices, all latency-bound rows the model under-predicts).

**Labels.** *Hardware:* every time and every exactness count. *Model:* the composed reference. *Not measured:* the KV-split merge, decode at bf16 or int8, head width other than 128, mt = 2 at other head widths, attention with a causal mask or a KV cache update, and the Q, K and V projections (they are GEMMs of the FFN stage-1 form, section 146, and were not built as producers of the attention operand layouts).

## 149. The weight-stationary gated FFN for small token counts is bit-exact and 3.7 to 4.4 times faster than the token-parallel schedule at 16 and 32 tokens; its stage 1 reaches the DRAM floor and its stage 2 is a 1.1 us per chunk latency chain that unrolling makes slower

**Result.** (1) Problem: token-parallel FFN (section 146) gives each block of 16 tokens a threadgroup that streams all weights, so 16 tokens use one core and cost the 1.7 to 1.8 ms of a wave. The weight-stationary schedule streams the weights once over the whole GPU (`mxlayer_ws.py`; `mxpipe.py` keys `wstat`, `hg`, `cs`, `vsg`): kernel A (`p12`) interleaves the 256 hidden chunks over hg threadgroups x nsgA simdgroups (chunk (r hg + g) nsgA + s of threadgroup g, simdgroup s), each running stage 1, silu and the product, quantize and storing the quantized chunk to a region shared by all threadgroups; kernel B (`p3`) splits the 2,048 output columns over cs threadgroups x nsgB simdgroups (32 virtual simdgroups of 4 tiles) and every simdgroup accumulates over all 256 chunks in order; kernel C (`nrm`) is one threadgroup of 32 simdgroups doing the residual and the RMS norm. No threadgroup memory is used except the norm's 2 KiB. The accumulation order of every output tile is the token-parallel order, so the bits must equal those schedules. (2) Exactness (`mxlayer_ws.py exact`): d = 2048, Hh = 8192, one token block, against the composed reference and against the token-parallel two-kernel schedule on one threadgroup: e4m3 with 16 tokens (A 32 x 8, B 8 x 4): D2 and Y 0 of 32,768 mismatches, equal to the token-parallel bits; e4m3 with 32 tokens (mt = 2): 0 of 65,536, equal; bf16 with 16 tokens: 0 of 32,768, equal; the reference is finite over 100 percent in all three. (The fused token-parallel kernel does not compile in the last two cases: 40,960 and 34,816 bytes of threadgroup memory, section 146; the weight-stationary schedule has no such bound.) (3) Timing (`mxlayer_ws.py time`, sets 2, weights 48 MiB in fp8 and 96 MiB in bf16, ms; the best decomposition of the measured grid):

```
format tokens | A (stage 1..quantize) | B (down projection) | C (norm) | weight-stationary total | token-parallel, one threadgroup (fused / two / three)
e4m3     16   |  0.148 (64 x 4 SG)   |  0.287 (8 x 4)      |  0.007   |  0.442 ms (3.65 TFLOP/s) | 1.792 / 1.719 / 1.717 ms    (3.9x vs two)
e4m3     32   |  0.153 (32 x 8)      |  0.665 (32 x 1)     |  0.023   |  0.841 ms (3.83 TFLOP/s) | not compiled / 3.149 / 3.379 ms  (3.7x)
bf16     16   |  0.234 (32 x 4)      |  0.147 (32 x 1)     |  0.007   |  0.388 ms (4.16 TFLOP/s) | not compiled / 1.714 / 1.735 ms  (4.4x)
```

(4) Stage 1 (kernel A) is chunk-parallel latency until the weight stream limits it. In e4m3 with 16 tokens the time is 59 to 63 us per chunk per simdgroup (0.949, 0.479, 0.251 and 0.158 ms for 16, 8, 4 and 2 chunks per simdgroup at 16 threadgroups) and stops falling at 0.148 ms from about 128 to 256 simdgroups: the gate and up projections are 32 MiB (33.6 MB) of fp8 elements, 42 MB with the scale tables, and 42 MB in 0.148 ms is 283 GB/s, the DRAM plateau. In bf16 it stops at 0.234 ms, which is 67 MB at 287 GB/s: the bf16 stage 1 costs 1.58 times the fp8 stage 1, about the byte ratio (1.6). (5) Stage 2 (kernel B) is independent of the decomposition: 0.287 to 0.291 ms for 4, 8, 16 and 32 threadgroups in fp8 (0.321 ms in the later run of part 6), 1.1 to 1.25 us per chunk step, because every simdgroup owns whole output tiles and walks all 256 chunks; it reads 21 MB of weights (16.8 MB of fp8 elements and the scale tables) in 0.32 ms (66 GB/s), so it is neither at the DRAM plateau nor at the MMA rate. bf16 has no block-scale chain and takes 0.147 ms; mt = 2 doubles it in fp8 (0.665 ms, 34 spilled stores). (6) The unroll of the chunk loop (`cunroll`, the operand loads of 2, 4 or 8 chunks issued before their MMAs, the accumulation order unchanged, exact) makes kernel B slower: fp8 0.321, 0.344, 0.384 and 0.549 ms for 1, 2, 4 and 8 (446 to 2,993 instructions, 7 spilled stores at 8), bf16 0.150, 0.181 and 0.333 ms for 1, 2 and 4 (85 spilled stores at 4), fp8 with mt = 2 0.675, 0.743 and 0.810 ms. The rule is an unroll of 1, as for the K loop of the attention stage 1 (section 148). A split of the chunk range across simdgroups with an ordered reduction would cut kernel B roughly by the split factor and would change the fp32 accumulation order; it was not built. (7) Crossover with the token-parallel schedule: from the constants of part 3 (`mxcost_model.py`) the weight-stationary schedule wins up to 64 tokens in fp8 (two blocks of 32: 1.68 ms against the 1.84 ms wave of 20 threadgroups) and up to about 80 tokens in bf16 (5 blocks of 16 at 0.388 ms against the 2.06 ms wave), and loses from 128 tokens; the 64-token point was not measured.

**Errors kept.** (a) The exactness driver first compared with the token-parallel fused kernel, which does not compile at mt = 2 and in bf16 (threadgroup memory); it compares with the token-parallel two-kernel schedule now. (b) A first run of the mt = 2 and bf16 batch lost its output to a shell redirection that covered only the last command, and the rerun then failed at the compile of the fused baseline (a); the batch was run a third time.

**Labels.** *Hardware:* every time and exactness count. *Not measured:* 64 tokens, token counts between 32 and 128, int8, the split-chunk stage 2, mixed formats (fp8 stage 1 with a bf16 stage 2 would cost about 0.148 + 0.147 + 0.007 = 0.30 ms by these rows, not built), and weight-stationary at other d and Hh.

## 150. The compiler cost and scheduling model for a transformer layer on H17s: constants, a decision procedure for attention and for the gated FFN, and its validation against the measurements of sections 145 to 149

**Result.** (1) Form (`mxcost_model.py`, which also calibrates and validates when run). A kernel's time is the largest of three terms: T_latency = steps per simdgroup x L_step (a simdgroup's dependent chain, independent of the number of simdgroups until the core saturates), T_throughput = total steps x C_step / 20 cores (the per-core cost of a step at saturation) and T_memory = unique bytes / BW(footprint). A step is a key block of 32 (attention) or a hidden chunk of 32 (FFN). The number of simdgroups that saturates the GPU is 20 x L_step / C_step. (2) Constants, each a measurement of this document:

```
constant                                         value                                                                                  source
GPU cores                                        20; one wave = 20 threadgroups of up to 1,024 threads                                  145, 146
read bandwidth by footprint (unique bytes)       3 TB/s to 5 MiB (lower bound), 0.52 TB/s to 20 MiB (SLC), 0.28 TB/s from 40 MiB         145
threadgroup memory                               32,768 bytes; the fused FFN needs nsg x mt x 640 (fp8, int8) or x 1,088 (bf16, f16)     145, 146
hand-off round (store, barrier, load)            38 to 82 ns threadgroup, 58 to 83 ns device (1 threadgroup per core); 100 to 161 ns at 64 simdgroups per core     145
MMA at saturation                                4.94 ns (half, bf16), 2.47 ns (int8) per MMA per core; fitted cost in real chains at 64 simdgroups per core: 5.8 (bf16 class) and 7.0 (fp8, int8) ns per MMA plus 0.21 and 0.34 ns per other instruction (321 timings of section 142)    136, 142
dependent MMA step at one simdgroup per core     63 (bf16 class) and 68 (fp8, int8) ns per MMA of a dependent chain (accumulate, scale, quantize, MMA); independent MMAs are much cheaper     142 (fit)
attention step, fp8 and int8                     L_step 5.35 us per key block and simdgroup; C_step 433 ns per key block and core; 247 simdgroups saturate    148
attention step, bf16                             L_step 5.0 us; C_step 381 ns                                                           148
FFN wave, token-parallel two-kernel (d 2048, Hh 8192)    1.84 ms e4m3, 1.72 int8, 2.06 bf16 (16 simdgroups) per 20 blocks of 16 tokens   146
weight-stationary (16 tokens)                    A 59 us per chunk and simdgroup down to a DRAM floor of 0.148 (fp8) and 0.234 ms (bf16); B 0.287 (fp8), 0.147 ms (bf16); C 0.007 ms    149
```

(3) Decision procedure, attention (inputs: heads, queries, KV length S, head width dh, dtype, producer layouts of K and V). 1. Emit one fused kernel: QK^T, scale, online softmax with the hardware `exp2` and the row-sum partials, quantize P, PV, rescale, all in one runtime loop over key blocks of 32; never materialize S or P (the P cut costs 1.03 to 1.30 times, the S cut 1.07 to 1.42, both 1.14 to 2.03; 27 rows). 2. Tile: 16 queries per task (mt = 1) and the whole head width per simdgroup (dh / 16 fp32 accumulators, 8 at dh = 128); mt = 2 spills 354 stores and takes twice as long at dh = 128. 3. Tasks = heads x ceil(queries / 16); N* = 20 x L_step / C_step = 247 simdgroups. 4. If tasks >= N*, run one task per simdgroup with no split. Otherwise split each task's key range into the smallest power of two s with tasks x s >= N* (8 for 32 tasks) and merge the s partial (m, l, O) with an ordered reduction (not built; one hand-off round through threadgroup memory when the slices are the simdgroups of one threadgroup); stop splitting when T_memory is the largest term (16 slices bought nothing after 8). 5. Simdgroups per threadgroup: any from 1 to 16 (measured 1.40 to 1.54 ms); keep at least 20 threadgroups so that every core has one (nsg <= tasks / 20); the task-to-threadgroup mapping is free. 6. Operand layouts: K row-major uses `tb1`, V transposed uses `tb2`, at no measured cost; K-loop unroll 1. 7. dtype: while the loop is issue-bound (T_latency or T_throughput largest) bf16, int8 and fp8 are within 12 percent (bf16 fastest); when T_memory is largest (unique KV per task, as in decode) choose the dtype with fewer bytes: fp8 and int8 read 1.25 bytes per element against 2 for bf16 (measured: bf16 is 20 percent slower at 128 tasks with unique KV, 0.490 against 0.410 ms, section 148 part 5b). 8. Memory breakpoint: T_memory uses the unique bytes of K and V touched per pass (tasks that share a head share through the caches and count once) against the bandwidth of the footprint; the 32 MiB KV of a 16-head, 8,192-key layer is still issue-bound, and the 128 MiB unique KV of the split decode case is DRAM-bound.

(4) Decision procedure, gated FFN (inputs: tokens T, d, Hh, dtype). 1. Cut at the quantized hidden chunk: two kernels, stage 1 with activation, product and quantize (`p12`), then the down projection with residual and norm (`p3`); the register-fused form ties at one wave (10 and 20 blocks) and is 7 to 11 percent slower at 40 and 80 blocks in fp8 and int8 (ratios reproduced in fresh processes to 1.5 percent) and 1.45 to 1.56 times slower in bf16, and the three-kernel form is 2 to 6 percent slower. 2. If T is at most about 64 (fp8, int8) or 80 (bf16), run weight-stationary: kernel A chunk-parallel over at least 128 to 256 simdgroups (for example 64 threadgroups of 4 simdgroups or 32 of 8), kernel B with the output columns over 32 virtual simdgroups (8 threadgroups of 4), kernel C one threadgroup of 32 simdgroups; mt = 1 for up to 16 tokens and mt = 2 for 17 to 32, repeated for more blocks; time = blocks x (A + B + C) from the constants above (0.442 ms for 16 and 0.841 ms for 32 tokens in fp8, 0.388 ms for 16 in bf16). 3. Otherwise run token-parallel: one threadgroup of 32 simdgroups per 16 tokens (mt = 1), 4 output tiles per simdgroup at d = 2048, waves = ceil(blocks / 20), time = waves x wave time; below one wave the time does not fall. 4. Loop structure: runtime K loop and runtime chunk loop, unroll 1 for both (unrolling was equal or slower in every measurement). 5. dtype: token-parallel and attention are issue-bound and the dtypes are within 10 percent, so choose by accuracy and footprint; in the weight-stationary schedule stage 1 is DRAM-bound (fp8 and int8 are 1.58 times faster than bf16, the byte ratio) and stage 2 is a latency chain (bf16 is 1.95 times faster than fp8), so the predicted best mix is fp8 stage 1 with bf16 stage 2 (about 0.30 ms for 16 tokens; not built). 6. Threadgroup memory: fused sharing only when nsg x mt x 640 (fp8, int8) or 1,088 (bf16, f16) is at most 32,768; device-memory sharing has no bound and costs at most 17 percent more per round; the cut schedules need none but the norm's 2 KiB. 7. A fused kernel's spill count is not predictable from shapes; the compiler should count it after the compile and take the cut when the fused kernel spills more than about 50 stores (fp8, int8) or any measurable amount in bf16; attention is the exception, where the fused kernel wins with 42 spilled stores.

(5) Validation (`mxcost_model.py`, `mxcost_model_validation.json`). Attention: 27 measured fused rows; constants fitted from four (the unsplit decode row, the 512-task row, and the two bf16 rows); the other 23 have a median error of 2.7 percent, 20 of them within 10 percent, the worst -16.5 percent (128 heads of 2,048 keys, latency bound). FFN token-parallel two-kernel: constants from the three 20-threadgroup rows of the fresh session (`mxlayer_ffn_time3.log`); the other 16 sets = 1 rows (10, 40 and 80 threadgroups, this session and the sweep) have a median error of 3.3 percent, 15 of 16 within 10 percent and a worst of -10.8 percent (e4m3 at 80 threadgroups, four waves, the row whose absolute time differs by 18 percent between sessions, part 6 of section 146); without the four-wave rows the median is 3.0 percent and the worst 9.9 percent. Weight-stationary stage 1: 10 configurations not used for the step constant, median 5.1 percent, worst 6.2 percent. The planners return `plan_attention('fp8', 32, 16, 16384, 128)` = 8 slices, 0.599 ms (measured 0.613 for the proxy); and for the FFN the weight-stationary schedule up to 64 tokens and the token-parallel one from 128 (the 64-token crossover point is a model prediction). Mt = 2 attention rows and the withdrawn device-sharing FFN rows are outside the model.

(6) The timing observations at one simdgroup per core (sections 140 and 142) and what explains them. I checked whether the fetch penalty of section 145 explains the differences between fused and cut chains at 1 and at 64 simdgroups per core. A least-squares fit of the 321 kernel timings of the section 142 logs (chain time = a x MMAs + b x instructions + c x spilled stores; `mxpipe_cost_*.log`) gives, in the bf16 class (80 rows, R^2 0.99, median error 5.4 percent), 62.6 ns per MMA at 1 simdgroup per core and 5.8 at 64, and in fp8 and int8 (241 rows, R^2 0.94, median error 12.3 percent), 68.2 ns and 7.0 ns per MMA with 1.6 and 0.34 ns per other instruction. The per-instruction terms are small against the per-MMA term, so the 1-simdgroup differences between fused and cut kernels come from the length of the dependent MMA chain and the instruction count, not from instruction fetch. The fit leaves 34 percent of the fp8 and int8 rows more than 20 percent off; the six worst (over-predicted by 109 to 176 percent) are all `mma_only` ablations whose MMAs are independent, so the 63 to 68 ns is the cost of an MMA inside a dependent chain and not of an MMA as such. The attention constants have a decoded counterpart (section 152): the fused chunk loop is 972 instructions whose 170-instruction K loop runs 4 times, so a key block executes about 1,480 instructions for 32 MMAs (46 per MMA); at the 0.24 ns per instruction per core measured in section 152 that is 356 ns of the 433 ns per block at saturation, and the 5.35 us per block of one simdgroup is the same stream at 3.6 ns per instruction. Neither observation changes a rule of this section: the layer kernels run at 12 or more simdgroups per core, or are split to reach it, and at 1.6 simdgroups per core the fused attention kernel still beats every cut (rows 10 of part 2 of section 148). I could not identify, from this document, the two anomalies of the mission statement by name; the ones above are the 1-simdgroup observations of sections 140 and 142.

**Not measured, and stated as such.** The merge of split attention, a producer of K and V in the operand layouts, the Q, K and V projections, other d, Hh, head widths and mt, token counts between 32 and 128 for the FFN, int8 weight-stationary, a split-chunk stage 2, mixed-format schedules, and, of the dual-engine question that a peer relayed after this work started, the layout transposition between the 16x16 tensor fragments and the 8x8 legacy fragments (the engines themselves are measured in section 152). Public prior art for the layouts and forms used here: none was found by a peer session in dougallj/applegpu (G13, register-pair matrix operands only), Mesa `AGX2.xml` (2025, no matrix instruction) or Turner's M1/M2 benchmarks (throughput only); that is a statement about what those sources contain, not about the hardware.

**Labels.** *Hardware:* every measurement that the constants and the validation cite (sections 145 to 149). *Model:* `mxcost_model.py`; its constants are fitted to the rows named in the validation and its decisions extrapolate only where marked. *Decoder:* instruction counts as in section 146.


## 151. Accumulator chains and the accumulator-cache hypothesis (Apple patent WO2025071810): a dependent MMA chain costs no more than independent MMAs, an fp32 fma inside the dependent path costs about 27 ns per step at low occupancy and nothing at saturation, and the compiler regroups and hoists

**Result.** (1) Hypothesis. A public benchmark of the M5 GPU neural accelerators (tzakharko, A19/M5) cites two Apple patents for the datapath. WO2025071810, "Matrix multiplier caching" (abstract and claim 1 as shown by Google Patents through a summarizing fetch tool; I have not read the claims myself): an integrated circuit with a dot product accumulate circuit (a dot product circuit and an adder) and an accumulator cache coupled to the input and the output of the adder, so that the output of one dot product accumulate serves as the accumulation input of the next without a register-file round trip; the compiler passes a cache hint to the scheduler, which schedules the dependent instructions consecutively; each lane is a 4-way dot product. This fits the exact 2-2-4 accumulation tree of the campaign (sections 4 to 6: four products summed in a tree, then added to the accumulator in four steps for K = 16) and the compiler-set `0x20` bit of the first instruction byte that section 21 records on MMAs. The question here is whether the register-file round trip is visible in timing. Only compiler-generated variants were used; no encoding was patched. (2) What the compiler does (`mxaccchain.py`, first version, 12 variants of 16 bf16 MMAs: 1, 2, 4 or 8 accumulators, chains emitted grouped or interleaved, chained or block-scaled): interleaved and grouped sources compile to identical code (`chain4i` and `chain4g` have the same 16 MMAs in the same consecutive runs of four per accumulator register group); the `0x20` bit is set on 15 of the 16 MMAs in every chain variant (all but the first, including the first MMA of each independent chain), so it does not mark where a chain begins and I draw no conclusion about its meaning; and operands that do not change between iterations are hoisted, so the block-scaled variants compiled to 2 MMAs instead of 16. A non-consecutive dependent form therefore cannot be produced from the IR order. (3) Second version: eight bf16 fragments loaded per iteration from one of four sets selected by the iteration number (nothing is invariant or mergeable), four steps of two MMAs per iteration (8 MMAs), distinct operand pairs in every step. `chain{n}`: acc = mma(a, b, acc), then mma(a', b', acc). `post{n}`: p = mma(a, b, 0), p = mma(a', b', p), acc = fma(p, s, acc), the block-scaled form of the fp8 and int8 kernels. `cfma{n}`: the chained form followed by acc = fma(acc, s, c), an fp32 fma inside the dependent path. n = 1 (one accumulator, 4 steps) or 4 (four accumulators, 1 step each). All six compile to 8 MMAs (op5106, with op5107 for the first MMA of the block-scaled form), the `0x20` bit on 8 of 8, 0, 32 and 32 fp32 fma, 48 to 112 registers. (4) Timing (`mxaccchain_time.log`, ns per timing iteration of one simdgroup, 20, 80 and 320 one-simdgroup threadgroups for 1, 4 and 16 simdgroups per core; median of 9 rounds):

```
variant   1 SG/core   4 SG/core   16 SG/core
chain1       218.0       260.6       634.4
chain4       218.0       260.0       635.6
post1        250.8       300.9       634.8
post4        241.0       288.7       636.0
cfma1        324.2       386.8       634.8
cfma4        244.1       292.3       635.9
```

(5) Reading. (a) A dependent chain of 8 MMAs (`chain1`) and four independent chains of 2 (`chain4`) take exactly the same time at 1 simdgroup per core (218.0 ns, 27.3 ns per MMA) and at 4 (260.6 and 260.0 ns): the dependence of a chained accumulate exposes no latency. This is what an accumulator cache would give; it is also what an issue limit of one MMA per 27 ns per simdgroup would give, and the measurement does not separate the two. (b) An fp32 fma in the dependent path costs 26.5 ns per step in a single chain (`cfma1` 324.2 against 218.0 ns, +106 ns over four steps) and about 26 ns per iteration with four chains (`cfma4` 244.1 against 218.0): the register-file round trip of an MMA result that feeds the next MMA is a latency of about 27 ns that other independent chains hide. (c) The block-scaled form costs +8 ns per step, +15 percent (`post1` 250.8 against 218.0), because its MMAs do not depend on the accumulator: the result returns through the register file to the fma but the MMAs of the next step do not wait for it; this is the mechanism behind the cheap post-MMA scaling of section 138. (d) At 16 simdgroups per core all six kernels take 4.96 to 4.97 ns per MMA per core (8,192 FLOP in 8.0 cycles at the 1.62 GHz clock that a peer session read from powermetrics: 1,024 FLOP per cycle and core, the figure the public M5 benchmark estimates for FP16, "Estimated 1024 FLOPS per GPU core per cycle"; the int8 rate of 2.47 ns per MMA, section 136, is 2,048 per cycle against its "<~2048"), so none of this is visible at saturation. One simdgroup issues one MMA per 27 ns with its fragment loads, and 4 simdgroups give 8.1 ns per MMA and core; about 5.5 simdgroups per core saturate.

(6) Rules for the model. Keep the accumulator out of the MMA-dependent path (scale after the MMA pair into a zero accumulator, as in section 138, not between chained MMAs); at low occupancy (few tasks, the decode case of section 148) split the K range into independent accumulator chains, which recovers the 26 ns per step of a dependent fma; at 12 or more simdgroups per core the forms are equal. The 63 to 68 ns per dependent MMA step of the section 142 fits (section 150 part 6) exceeds the 27 ns measured here because the real chains add operand loads, scale tables and the quantizer. (7) Not tested: consecutive against non-consecutive dependent MMAs (the compiler regroups; a non-consecutive form needs an encoding change, which the campaign does not execute), the fp8 and int8 operand forms (unpack plus op5106), and the meaning of the `0x20` bit.

**Errors kept.** (a) The first version of `mxaccchain.py` was written to the name `mxacc.py`, which already held the section 138 accuracy script (`mxacc.py`, `mxacc.log`, `mxacc.json`); the file tool reported an update instead of a creation and I overwrote the script before noticing. The original was recovered from a local Time Machine snapshot taken 5 minutes earlier (mounted read-only) and its SHA-256 matches the manifest entry (`9c5a3924...`); the probe lives in `mxaccchain.py`. (b) The first block-scaled variants compiled to 2 MMAs because their operands were loop-invariant, and the interleaved and grouped variants compiled to the same code; the second version removes both problems and the first is described in part 2.

**Labels.** *Hardware:* every time. *Compiler output:* the decoded MMA counts and bits (`agxforge.g17.model.decode`, section 146 labels). *Source of the hypothesis:* a patent as summarized by a tool, cited by a third-party benchmark; the timings are consistent with the mechanism and do not prove it.


## 152. The tensor MMA, the legacy simdgroup MMA and the ALU on H17s: the two matrix engines overlap completely, the f32-accumulate legacy forms (`op2862`, `op2842`, `op2902`) share their budget with fp32 fma (overlap 0.37 or less) where the f16-accumulate form (`op838`) mostly does not (0.66 to 0.95), beside a tensor stream that carries block-scaling work one legacy MMA per eight tensor MMAs is free, and the layer kernels gain nothing from a second matrix engine

**Result.** (1) Verdict. A kernel can issue tensor MMAs and legacy simdgroup MMAs at the same time: at 8 or more simdgroups per core the legacy stream is hidden under the tensor stream (overlap 0.91 to 1.00, exactly 1.00 at 16 and 32) in all four legacy forms measured, and a dependent hand-off between the engines costs nothing at 8 or more. The legacy forms a transformer kernel would use accumulate in fp32 (`op2862` fp16 inputs, `op2842` fp32 inputs, `op2902` bf16 inputs, schedclass 118), and these take their time from the same budget as fp32 fma: they overlap fma by 0.37 or less (0.07 to 0.37 for `op2862` and `op2902`, -0.14 to 0.36 for `op2842`), so the two nearly add. What is free beside the tensor MMA therefore shrinks with the ALU work of the kernel. For 8 bf16 tensor MMAs per iteration the free legacy issues are 8 with no fma, 4 with 4 fma instructions per tensor MMA and 1 with 8 per MMA (the block-scaling rate of the fp8 and int8 kernels of sections 138 to 143); past the budget each legacy issue costs about 5 percent (8 tensor MMAs with 64 fma and 1, 2, 3, 4 legacy issues: +0.8, +5.6, +10.6, +16.1 percent at 32 simdgroups per core). A legacy MMA is 512 MAC and a tensor MMA 4,096, so the arithmetic gained is at most 12.5 percent with no fma, 6.3 percent with 4 fma per MMA and 1.6 percent with 8, against a cost that is positive from the second legacy issue at 8 fma per MMA. The layer kernels sit far beyond that budget (46 instructions per MMA in the attention loop, part 11). Rule for the model: do not schedule legacy simdgroup MMAs in the layer kernels; the one regime that pays is a tensor-bound bf16 or fp16 loop with at most 4 fma per MMA (part 11 gives the price). This corrects the first version of this section, which measured only `op838` and generalized it (errors kept, part c).

(2) Question. A peer session relayed Spencer's observation that M5 added a second matrix family (tensor `op5106` and `op5107`, length 10) beside the inherited simdgroup family (`op838` and `op839`, `op2842` to `op2905`, length 14 in the peer's table) and asked whether a kernel can schedule work onto both engines and the ALU at once, and how the three contend. Two hypotheses from public work were to be tested: Turner (M1 and M2) reports that `simdgroup_matrix` runs on the existing FP32 pipelines, and Rigel (M4 Max, arXiv 2606.12765) that `matmul2d` runs on the `simdgroup_matrix` and FP32 path; if the legacy MMA were the FP32 pipe, legacy plus fma would serialize. The campaign's earlier test (hand-encoded bodies, one simdgroup, section 32: `op2862` with `op5106` 31.8 ns per pair against 22 ns alone and 38.5 ns summed, overlap 0.41) is extended here to compiler-generated kernels at saturating occupancy.

(3) Construction (`mxdual.py` for `op838`, `mxdualf.py` for the f32-accumulate forms; compiler-generated AIR only, no encoding patched). The accumulator type of the legacy intrinsic selects the opcode: `air.simdgroup_matrix_8x8_multiply_accumulate.v64f16.v64f16.v64f16.v64f16` (fp16 accumulate) compiles to `op838` (schedclass 43, one 32-bit register per accumulator), and the same intrinsic with a `v64f32` accumulator compiles to `op2862` (fp16 inputs), `op2842` (fp32 inputs) or `op2902` (bf16 inputs), all schedclass 118 with a two-register tuple accumulator (`isa/g17-contract.jsonl`; a peer session pointed out the class difference). Units per timing iteration: T = tensor MMA (bf16 16x16x16, `op5106`, 4.95 ns per issue when saturated), L = legacy MMA, F = `<8 x float>` fma (8 `op2190` per unit). Every MMA accumulates into one of at most 8 accumulators (4 in the triple sets, extra units chain onto them; F: 8, or 4 in the add-on sweep) so nothing can be hoisted or merged; operands are loaded once before the loop. Occupancy is 20 x k one-simdgroup threadgroups (k simdgroups per core). All 45 `op838` kernels and all 180 f32-accumulate kernels (60 per form) decode over their full code bytes, have exactly the intended counts of `op5106`/`op5107`, the legacy opcode and `op2190`, no other legacy opcode, and no spilled store (checked by script over the 180; with the 6 probes of section 151, 231 of 231 decode in full).

(4) Rates, ns per instruction per core at 32 simdgroups per core: tensor 4.955 (4 per iteration) and 4.948 (8), which is 1,022 FLOP per cycle at the 1.62 GHz a peer read from powermetrics; legacy 2.79 to 2.80 for the 64-issue kernels of all four forms and 2.48 to 2.58 at 8 and 16 simdgroups per core in all four forms, but bimodal at 32 per core for the shorter solo kernels, 2.50 or 2.80 per issue depending on the kernel and the run (`op2862` with 8 issues 649.1 or 728.0 ns per iteration; `op2842` and `op2902` with 12 issues 961.2 against `op2862` 1,079.5; `op2902` with 16 issues 1,281.2 and 1,278.0 in two runs against 1,433.8 to 1,439.9 for `op2862` and `op2842`), a mode the tensor (634.2 in every run) and fma (268.7 to 272.2 for 32 fma) rates do not show and I did not explain; that is 226 to 253 FLOP per cycle for 1,024 FLOP per issue, 4.0 to 4.5 times less per FLOP than the tensor MMA; fma 0.263 (32 per iteration) and 0.241 (64) per instruction. The legacy MMA does 183 to 205 MAC per ns per core against 122 to 133 for the fma stream (32 MAC per 0.263 or 0.241 ns), so it is not the fma instruction stream running at the fma rate, whatever it shares with it (part 10a).

(5) `op838` (schedclass 43), 32 simdgroups per core, ns per iteration of one simdgroup (all simdgroups concurrent; `mxdual_run2.log`; the rerun `mxdual_run5.log` reproduces each row to within 1.3 ns):

```
kernel(s)                 solo A   solo B    sum     max     mix   overlap of the smaller
T4 + L4                    634.2    366.3   1000.5   634.2   634.6   1.00
T4 + F4 (32 fma)           634.2    268.8    903.0   634.2   634.8   1.00
T4 + L4 + F2               634.2      -        -       -     634.8   (all hidden behind the tensor MMAs)
T4 + L8                    634.2    723.4   1357.6   723.4   772.9   0.92
L4 + F4                    366.3    268.8    635.1   366.3   403.2   0.86
L8 + F8 (64 fma)           723.4    493.7   1217.1   723.4   817.2   0.81
T4 + L8 + F2                 -        -        -       -     821.8
```

Legacy plus fma overlap 0.86 and 0.81 at 32 simdgroups per core, 0.66 and 0.95 at 16 and 0.77 and 0.87 at 8 (L4 + F4, L8 + F8; the 0.95 uses a solo L8 of 407 ns that both runs measure at 16 per core against 330 for the other forms, with 330 it is 0.64; the L4 + F4 cell at 16 was 0.38 in the first run and 0.66 in the rerun).

(6) The f32-accumulate forms, `op2862` at 32 simdgroups per core (minimum over two runs, `mxdualf_run1.log` and `mxdualf_run2.log`; ns per iteration):

```
kernel(s)                 solo A   solo B    sum     max     mix   overlap of the smaller
T4 + L4                    634.2    326.5    960.7   634.2   634.6   1.00
T4 + F4 (32 fma)           634.2    271.9    906.1   634.2   634.6   1.00
T8 + L8                   1266.8    649.1   1915.9  1266.8  1267.3   1.00
L4 + F4                    326.5    271.9    598.4   326.5   534.5   0.24
L8 + F8 (64 fma)           649.1    493.1   1142.2   649.1  1105.2   0.08
```

Overlap of the shorter stream by form and occupancy (`python3 mxdualf_report.py`, output in `mxdualf_report.txt`; simdgroups per core 32, 16, 8, 2):

```
                 L4 + F4                  L8 + F8                  T8 + L8                  T4 + F4
op838   (43)      0.86 0.66 0.77   -       0.81 0.95 0.87   -       1.00 1.00 0.99   -       1.00 1.00 0.96   -
op2862  (118)     0.24 0.25 0.18 0.37      0.08 0.07 0.10 0.09      1.00 1.00 1.00 0.72      1.00 1.00 0.96 0.52
op2842  (118)     0.34 0.19 0.08 0.36      0.18 0.01 -0.14 -0.04    1.00 1.00 1.00 0.70      1.00 1.00 0.95 0.50
op2902  (118)     0.25 0.25 0.24 0.36      0.08 0.07 0.11 0.09      1.00 1.00 1.00 0.72      1.00 1.00 0.95 0.52
```

The solo L8 at 32 per core is bimodal between runs (649.1 and 728.0 ns for `op2862`, part 4), which moves the L8 + F8 cell at 32 between 0.08 and 0.23; the 16 and 8 per core cells, whose solos agree to 1 to 2 percent, give 0.07 to 0.11 (0.01 and -0.14 for `op2842`, where the pair is slower than the sum). The three f32-accumulate forms agree within the noise, including the bf16-input form, and at 16 and 32 per core they cost the tensor stream nothing (T8 + L8 1.00). The 2 per core column is the in-order issue of a simdgroup, not a resource: the section 32 result above (overlap 0.41 at one simdgroup) is this regime.

(7) Budget under a tensor stream that also carries fma work (`mxdualf_run4.log`, `mxdualf_run5.log`, and the first two runs for the F0 row; 4 accumulators per engine, extra units chained; no spill, full decode). Extra time over 8 tensor MMAs alone (1,266.2 ns at 32 simdgroups per core, 634.8 at 16, 318.1 at 8), percent at 32 simdgroups per core with 16 in parentheses, for n legacy issues per 8 tensor MMAs and F0 = no fma, F4 = 32 fma instructions (4 per tensor MMA), F8 = 64 (8 per tensor MMA); `op2862`:

```
        n=1        2          3          4          5          6          8          12         16
F0        -         -          -          -          -          -       0.1 (0.1) 13.6 (8.0) 28.4 (31.2)
F4     0.1 (0.1)  0.1 (0.1)  0.2 (0.1)  0.1 (0.0)  1.5 (1.6)  2.2 (2.2) 10.8 (8.7) 23.7 (26.9) 44.1 (49.8)
F8     0.8 (0.9)  5.6 (6.0) 10.6 (11.6) 16.1 (17.6) 19.0 (20.6) 21.2 (23.0) 32.2 (29.4) 46.0 (50.1)     -
```

`op2842` and `op2902` agree with `op2862` to within 7 points at n up to 6 (mostly within 3; the largest is `op2842` at F8 and n = 6, 27.7 against 21.2 at 32 per core), and the 32 per core cells at n = 8 differ by up to 13 points (F4: 10.8, 23.9 and 14.1), the many-wave drift. Tensor plus fma alone is free up to 8 fma per MMA (T8 + F8: 1,266.8) and tensor plus legacy alone up to 8 issues, but the two together are free only up to about 64 fma-equivalents per 8 tensor MMAs, one legacy MMA counting as about 8 fma instructions: F0 + L8, F4 + L4 and F8 + L1 all sit at the budget, and the three combinations at 96 (L12 + F0, L8 + F4, L4 + F8) cost 13.6, 10.8 and 16.1 percent at 32 per core (8.0, 8.7 and 17.6 at 16). At 8 simdgroups per core the rows are latency-limited and the budget is smaller (F8 already +12 percent at n = 1; F4 free through n = 4, +3 percent at n = 5 and +21 percent at n = 6).

(8) Add-on and split sweeps at 32 simdgroups per core. `op838` (`mxdual_run4.log`, `mxdual_run3.log`): tensor 8 alone 1,266.7 with 4, 8, 12, 14, 16 and 20 legacy issues 1,268.1, 1,267.4, 1,335.2, 1,469.7, 1,710.0 and 1,879.0; tensor 4 alone 634.2 with 4, 8, 10 and 12 legacy issues 634.6, 774.3, 906.8 and 1,026.1; tensor 4 with 2, 4, 8, 12 and 16 fma units (16, 32, 64, 96 and 128 fma instructions, 4, 8, 16, 24 and 32 per MMA) 634.9, 634.9, 861.4, 940.5 and 1,098.6, against fma alone 484.7 (64), 586.4 (96) and 721.6 (128). Equal arithmetic split between the engines (one tensor issue = 8 legacy issues = 8,192 FLOP): tensor 8 alone 1,266.6; 7 tensor + 8 legacy 1,109.5 (0.876 of the time, 1.142 times the throughput); 6 + 16 1,436.6; 4 + 32 2,911.6; 2 + 48 4,321.1; 0 + 64 5,741.6; 14 + 16 2,215.3, 13 + 24 2,712.6 and 12 + 32 3,186.3 at 16 tensor equivalents. `op2862` (`mxdualf_run1.log`, `mxdualf_run2.log`): tensor 8 with 4, 8, 12 and 16 legacy issues 1,267.1, 1,267.3, 1,437.8 and 1,626.2; tensor 4 with 8 and 12 legacy issues 765.6 and 1,103.2 to 1,167.8; the split rows 7 + 8, 6 + 16, 14 + 16, 13 + 24, 4 + 32 and 2 + 48 give 1,109.6 to 1,130.2, 1,442.5, 2,214.8 to 2,221.8, 2,683.5 to 2,775.9, 2,826.1 to 2,939.4 and 3,981.4 to 4,324.1 (the last three drift 3 to 9 percent between the two runs). When one pipe is clearly critical the mix takes the time of that pipe (7 + 8: 1,109.5 against 7 x 158.3 = 1,108; 14 + 16: 2,215.3 against 2,216); near balance it takes more than the maximum (13 + 24: 2,712.6 against 2,143). The split gains up to 14 percent in the no-fma case and nothing in the presence of fma (part 7).

(9) Dependent workloads. A tensor result feeding a legacy operand (`TL`), a legacy result feeding a tensor operand (`LT`), and controls doing the same conversions from a loop-varying independent value (`TLc`, `LTc`), four links per iteration (`op838`, `op2862`, `op2842`, `op2902`). At 32 simdgroups per core all four give 634.2 to 634.7 ns (independent 634.4 to 634.6; two controls 644.9 and 646.1), at 8 per core 161 to 170 against independent 163 to 168, and at 2 per core `TL` costs +15.0, +14.9, +10.8 and +13.0 ns over `TLc` (+9.7 to +13.5 percent; `TLc` 111.2 to 113.2 ns) and `LT` +1.1, +5.6, +6.2 and +5.6 ns over `LTc` (120.5 to 121.8 ns). A tensor result into fp32 fma: 634.6 (equal to T4); a legacy result into fma: 382.0 against 366.3 for L4 (+4 percent, `op838`). The conversions are element conversions between a vector and the other engine's operand type; the layout transposition between the 16x16 tensor fragments and the 8x8 legacy fragments that a real kernel needs was not built.

(10) Reading. (a) The tensor MMA is independent of both other resources: it overlaps the legacy MMA and fma completely at 16 and 32 simdgroups per core (part 6). The legacy MMA is not one thing. The f16-accumulate `op838` overlaps fma by 0.66 to 0.95, so it uses mostly different resources; the f32-accumulate forms overlap fma by 0.37 or less, so on H17s they draw on the same budget as fp32 fma (at 16 and 32 per core the pair takes 89 to 97 percent of the sum). This is consistent with Turner's report for M1 and M2 (`simdgroup_matrix` on the FP32 pipelines) for the f32-accumulate class and is not consistent with it for `op838`. What is shared is not identified: a legacy MMA does 1.4 to 1.7 times the MAC rate of the fma stream (part 4), so it is not the fma instruction stream at the fma rate, and about 8 fma instructions of budget per legacy issue (part 7) is a measured exchange rate, not a mechanism. Rigel's statement about `matmul2d` on M4 Max was not tested. (b) Tensor plus one other resource is free while the other resource stays under about half of the tensor time; all three at once are free only under the joint budget of part 7. (c) A second engine pays only in a tensor-bound loop that has spare ALU budget: at one legacy issue per tensor issue with no fma it adds 12.5 percent arithmetic at no time cost (T8 + L8: 1,267.3 against 1,266.8 ns for `op2862`, 1,267.4 against 1,266.7 for `op838`), and the split of part 8 gives 1.14 times the throughput; the ceiling from the rates (226 / 1,022 = 22 percent) is not reached because the pipes contend near balance. (d) Dependence between the engines costs nothing at 8 or more simdgroups per core and 0 to 14 percent at 2.

(11) The layer kernels are issue-bound and ALU-heavy, not tensor-bound, so a second matrix engine has nothing to accelerate. The fused attention loop reaches 35 to 42 percent of the tensor MMA's 33.1 TFLOP/s (section 148) and the FFN 47 to 57 percent (section 146); decoded, the attention chunk loop is 972 instructions whose 170-instruction K loop runs 4 times, about 1,480 instructions for 32 MMAs (46 per MMA), and 1,480 instructions at 0.24 ns per instruction is 356 ns of the 433 ns per key block and core (section 148), with the tensor pipe busy 158 ns of it. At 8 non-MMA instructions per tensor MMA (the block-scaling rate) the free legacy budget is already one issue per eight tensor MMAs (part 7), so a legacy MMA in these loops adds ALU budget and register pressure, not capacity. Rule for the model: do not schedule legacy simdgroup MMAs in the layer kernels. The one regime that pays is a tensor-bound bf16 or fp16 loop (a GEMM tile loop) with at most 4 fma per MMA, for at most 6.3 percent more arithmetic at 4 fma per MMA and 12.5 percent at none (about 4 and 8 legacy issues per 8 tensor issues), at the price of a second operand layout (8x8 fragments), a different accumulation arithmetic (a sequential fma chain in the legacy family against the 2-2-4 tree, sections 4 to 6 and 31, so the two engines' results differ bit for bit and each output tile must stay on one engine) and more accumulators (the triple kernels spilled at 8 accumulators per engine). The int8 tensor MMA runs twice as fast (2.47 ns), which halves the tensor time and with it the free budget; that was not measured with a legacy stream.

**Errors kept.** (a) The first version (`mxdual_v1.py.bak`, `mxdual_run1.log`) loaded four legacy and eight tensor fragments per iteration by iteration number; the legacy kernels with 4 and 8 issues then took 653 and 694 ns against 361 and 722 ns from the slope of the 16 and 64 issue kernels, because the loads and their address arithmetic set the time. Accumulating MMAs cannot be hoisted, so the final kernels load once; the first-version numbers are not used. (b) Variants with 16 fma accumulators or with 8 tensor fragments in registers spilled (56 to 248 stores) and were replaced; the first triple set (8 tensor accumulators with 4 or 8 fma units and 4 to 12 legacy issues) spilled 69 to 156 stores and was replaced by the 4-accumulator set of part 7; the final sets have no spill. (c) I first measured `op838` only, called it the legacy MMA, and concluded that the legacy MMA is not the FP32 pipe and overlaps fma by 81 to 86 percent, and that up to 12.5 percent free arithmetic and 8 fma per MMA are free beside the tensor MMA. The `op838` numbers stand for `op838`. A peer session pointed out from the contract file that `op838` is schedclass 43 with one 32-bit accumulator while `op2842`, `op2862` and `op2902` are schedclass 118 with tuple accumulators and unmeasured; measuring them changed the fma conclusion (0.07 to 0.37 overlap) and left the tensor overlap unchanged. The earlier statement that the Turner and Rigel results were not transferable was too strong for Turner's; it holds for `op838` and fails for the f32-accumulate class. (d) I expected the legacy intrinsic to compile to `op2862` and it compiled to `op838`: the source declared an fp16 accumulator, and the accumulator type selects the opcode (part 3). (e) The `LF` kernel of the first set compiles without `op2190` (the fma of half-converted values took another opcode), so its ALU work is not the same instruction as in the other kernels. (f) The 16-simdgroup L4 + F4 cell of `op838` was 0.38 in the first run and 0.66 in the rerun, and the 32-simdgroup solo legacy times are bimodal (part 4); the many-wave absolute times drift 4 to 18 percent, so the tables use minima over runs, and the conclusions rest on cells that reproduce (the 16 and 8 per core cells, where the legacy rate is 2.5 ns in every run, agree with the 32 per core cells to within the noise).

**Labels.** *Hardware:* every time (`mxdual_run2.log` to `mxdual_run5.log`, `mxdual_time_*.json` for `op838`; `mxdualf_run1.log` to `mxdualf_run5.log`, `mxdualf_time_*.json` for the f32-accumulate forms; the tables of parts 6 and 7 are reproduced by `mxdualf_report.py`, output `mxdualf_report.txt`; another session's unit-test run, CPU-bound, shared the machine during `mxdualf_run1.log` to `mxdualf_run5.log` and `mxdual_run5.log`; six GPU utilization samples read 0 to 7 percent and two single readings 50 and 90 percent, so contamination cannot be excluded as one source of the drift above). *Compiler output:* instruction counts (`agxforge.g17.model.decode`, section 146 labels; 225 kernels decoded in full). *Contract file:* the schedclass of each opcode. *Not measured:* mixed engines inside one real GEMM or attention kernel, the layout transposition between the two fragment layouts, a legacy stream beside the int8 and fp8 tensor MMAs, power and clock, `op839` (`simdgroup.mul.f16`), the meaning of the `0x20` bit, and occupancies between 2 and 8 simdgroups per core other than the dependent variants.


## 153. A real mixed GEMM, both directions: the tensor MMA and the legacy MMA can contribute to the same output tile, bridged through each engine's own hardware-validated memory convention, bit-exact against the composed accumulation model in 32 of 32 trials

**Result.** (1) Question. Section 152 measured the two engines side by side (independent streams, a dependent register hand-off with an element conversion) but never built the thing a real mixed kernel needs: one output tile whose value comes from both engines, through the layout each one actually reads and writes. This closes that gap. (2) Construction. `D[16x16] = A[16x24] @ B[24x16]`, split by K: 16 columns through the tensor MMA (`op5106`, 16x16x16, one issue) and 8 through the legacy MMA (`op2862`, 8x8x8, four issues, one per quadrant of D). Two directions: tensor-then-legacy (`mixgemm1.py`, K = 0..15 then 16..23) and legacy-then-tensor (`mixgemm_rev.py`, K = 0..7 then 8..23, using `new_f16f16_nn`'s already-validated `C + A.B` form to add the tensor contribution on top). The bridge is a real host-mediated round trip through ordinary memory, not an in-kernel cross-lane permute: the tensor engine's fragment converts to and from a real 16x16 row-major matrix via `transpose.py`'s `to_frag`/`from_frag` (`pos()`/`pos_b()`, validated since section 3); the legacy engine reads and writes real row-major memory directly, through the compiler's own `air.simdgroup_matrix_8x8_load`/`_store` intrinsics (found in `old_msl_8x8/input.ll`, the AIR that real MSL `simdgroup_load`/`simdgroup_store` compiles to), with a quadrant selected by the `origin` operand. Both engines' own kernels (`new_f16f16_nn`, `old_msl_8x8`) are reused unmodified from earlier sections; nothing new was compiled for the main result. Two operand regimes: small integers (exact in fp32 regardless of accumulation order) and realistic random fp16 magnitudes, checked against the fully composed model (`tensorops_model.mma16`'s 2-2-4 tree for the tensor step, an 8-term sequential RNE32 chain for the legacy step, from sections 4 and 5). (3) Result: 8 of 8 trials exact in both directions and both regimes (32 of 32 total), including the fp16 regime checked against the composed model rather than plain fp64 matmul: the real bridge introduces no error beyond each engine's own already-characterized rounding.

(4) The `origin`/`strides` operand order of `air.simdgroup_matrix_8x8_load`/`_store`, established first (`quadtest.py`, 3 dispatches, load then store back to a different quadrant of a numbered 16x16 buffer, value = row*16+col so row and col are distinguishable): both operands are `<2 x i64>` in **(column axis, row axis)** order. For a buffer of row-width W, `strides = {1, W}`; `origin = {col_offset, row_offset}`; the address is `base + (origin0 + dim0)*strides0 + (origin1 + dim1)*strides1`. Confirmed by two independent placements (origin `(8,0)` moves the top-left block to columns 8-15 of the same rows; origin `(0,8)` moves it to rows 8-15 of the same columns) and cross-checked with swapped strides. This is the fact the mixed kernel's quadrant addressing runs on (`origin = (8*qc, 8*qr)`, `strides = (1, 16)` for a 16-wide buffer).

(5) Errors kept. The first version of `quadtest.py` decoded to 32 instructions, all `ret`/padding, and the load/store round trip had no effect on any of the three variants tried. Cause: the declaration lines were copied verbatim from `old_msl_8x8/input.ll`, including their `local_unnamed_addr #1` suffix; attribute group numbers are per-module, and in `template.ll` (the module `quadtest.py` builds into, via `gen.py`'s own head/tail split) `#1` is `readnone`, not `readonly` as in `old_msl_8x8`'s module. A `readnone` load whose result feeds only a `readnone`, unused-result store call is dead code by definition, and the optimizer removed the whole body down to a bare return. Fixed by defining fresh attribute groups (`readonly` for the load, plain `nosync nounwind willreturn writeonly` for the store, no `readnone` on either) rather than reusing a borrowed number. The same risk applies to every hand-authored declaration this campaign copies from one compiled module into another; gen.py's own kernels never hit it because their MMA result always feeds a real `store` instruction (not a second external call), so the borrowed `#1` (also `readnone` there) never had a chance to matter.

(6) Scope. This validates a **memory-mediated** bridge, the same shape section 101 already validated between two tensor-engine forms, now between the two *different* engines. It deliberately does not touch the legacy engine's internal per-lane register layout for its `<64 x T>` SIMD-group-collective operand (how the physical 32 lanes divide the 64 conceptual elements): the load/store intrinsics handle that translation inside the compiler, the same way `simdgroup_load`/`simdgroup_store` always have, and nothing here needed to open that box. A single-dispatch, in-register version (writing the tensor engine's fragment output directly to a row-major scratch address computed per lane, skipping the host round trip) is possible in principle, using `pos()`'s closed form inverted the same way `pos_b()`'s inverse was derived and checked by brute force while building this section, but was not attempted; the two-dispatch form already answers the question section 152 left open, at lower risk, and the cost of the extra dispatch was not something this experiment measured.

**Labels.** *Hardware:* every trial (`mixgemm1.py`, `mixgemm_rev.py`, 8 trials x 2 directions x 2 regimes = 32; `quadtest.py`, 3 dispatches). *Compiler output:* `old_msl_8x8/input.ll` (the real `air.simdgroup_matrix_8x8_load`/`_store` intrinsics), decoded instruction counts for `quadtest.py`'s variants (41, 41, 56, all full-length). *Model:* `tensorops_model.mma16` and `rne32` (already validated, sections 4, 12) composed with the section 5 legacy sequential-chain model. *Not measured:* the legacy engine's own per-lane physical layout, a single-dispatch in-register bridge, the cost of the bridge at any occupancy, K splits other than 16+8 and 8+16, shapes other than 16x16 output, and int8 or fp8 operands through this specific bridge.

## 154. The KV-split merge, built and measured: the merge is bit-exact against a split-aware model in 6 of 6 configurations, splitting the key axis is not value-preserving against the sequential chain, and the split pays up to about four ways before its own cost reverses the gain

Every attention measurement to section 153 ran one simdgroup over the whole key axis. The KV-split merge -- the
step that combines partial softmax states from simdgroups that each covered part of the keys -- was carried in
the ledger as item M1, "not built, and therefore excluded from every reported split time". It is now built.

**What was added.** `mxpipe.py` takes a new key `kvs = n`. The attention kernel launches `n` simdgroups per
query tile; simdgroup `s` runs the online softmax over key blocks `[s * NCH / n, (s + 1) * NCH / n)` and leaves
a partial state in its own device slot: the running max `m_s`, the per-lane row-sum partial `l_s`, and the
unnormalized `O_s` rebased on `m_s`. A device-scoped workgroup barrier then lets every simdgroup read all `n`
slots and combine them:

    M = max_s m_s,   f_s = exp2(m_s - M),   O = sum_s f_s O_s,   l = sum_s f_s l_s,   Y = O / rowsum(l)

with `s = 0` folded in by `fmul` and the rest by `fma`, and the cross-lane row sum (xor masks 1 and 8) taken
after the merge rather than before it. The partial `O` needs a region of its own (`OP`); `D2` and `Y` keep a
single slot, which the merge writes. Harness `results/g17-tensorops-recon-v1/mxattn_kvs.py`.

**One generator bug worth recording**, because it is the kind a KV split invites. `tdyn_of()` adds a
per-simdgroup offset into the B-side weights -- correct for the output-parallel schedules, where simdgroup `s`
owns its own output tiles and therefore its own weight columns. A KV split is not an output split: every
simdgroup consumes the *same* V tiles. Left in place, the partial `O` of split 0 was exact and every later
split read the wrong V, which the per-split `m` and `l` checks localized immediately (they were exact for both
splits while `O` was wrong for one). The two axes a simdgroup index can mean are not interchangeable.

**The merge is exact.** Against a reference that mirrors the split association, 6 of 6 configurations, every
element, both `O` and `Y`: zero mismatches, all elements finite.

| fmt | dh | S | heads x tiles | kvs | mismatches vs split reference |
| --- | --- | --- | --- | --- | --- |
| e4m3 | 64 | 128 | 1 x 1 | 2 | 0 of 2,048 |
| e4m3 | 64 | 128 | 1 x 1 | 4 | 0 of 2,048 |
| e4m3 | 64 | 256 | 2 x 2 | 4 | 0 of 8,192 |
| bf16 | 64 | 128 | 1 x 1 | 2 | 0 of 2,048 |
| bf16 | 128 | 256 | 1 x 2 | 4 | 0 of 8,192 |
| e4m3 | 128 | 256 | 1 x 1 | 8 | 0 of 4,096 |

**Splitting the key axis is not value-preserving**, and the gap is far larger than rounding. Against the
unsplit sequential chain no configuration is bit-exact, and the difference reaches 0.4 to 4 percent of the
row's own maximum (p99 0.07 to 1.6 percent):

| fmt / kvs | mismatching Y elements | diff / row max, p50 | p99 | max |
| --- | --- | --- | --- | --- |
| e4m3, 2 | 810 of 1,024 | 4.4e-06 | 2.7e-03 | 6.6e-03 |
| e4m3, 4 | 913 of 1,024 | 3.2e-05 | 1.5e-03 | 4.0e-03 |
| e4m3, 4 (2 heads) | 3,612 of 4,096 | 1.8e-06 | 1.6e-02 | 4.2e-02 |
| bf16, 2 | 891 of 1,024 | 1.9e-05 | 8.8e-04 | 1.8e-03 |
| bf16, 4 | 4,095 of 4,096 | 9.4e-05 | 6.7e-04 | 1.1e-03 |
| e4m3, 8 | 1,606 of 2,048 | 6.0e-06 | 5.1e-03 | 1.2e-02 |

The cause is not the association of the sums. It is that the softmax numerator is **rounded to the operand
format at a rebased exponent**: the kernel computes `exp2(s - m)` and rounds the result to e4m3 or bfloat
before the second MMA, and `m` is the running max of whichever key range the simdgroup covered. A different
`m` is not a power-of-two rescaling of `exp2(s - m)`, so the mantissas differ, and the difference survives the
merge no matter how exactly the merge is done. This holds for bf16 as well as fp8, so it is not an artifact of
MX quantization. The practical consequence is a validation rule: a KV-parallel attention cannot be checked bit
for bit against a sequential reference, and a reference that splits the same way is not optional.

**Cost.** One query tile and a long key axis -- the regime a KV split exists for -- with the grid fixed at one
threadgroup per query tile so the total attention work is identical across splits. e4m3, dh 128, median of 9
rounds, ns per launch:

| kvs | S = 512 (nch 16) | speedup | S = 1024 (nch 32) | speedup |
| --- | --- | --- | --- | --- |
| 1 | 44,343 | 1.00 | 105,175 | 1.00 |
| 2 | 27,742 | 1.60 | 57,469 | 1.83 |
| 4 | 17,594 | **2.52** | 31,663 | **3.32** |
| 8 | 27,828 | 1.59 | 35,047 | 3.00 |
| 16 | 82,747 | 0.54 | 87,279 | 1.21 |

The split pays, and then it stops paying. The turn is not subtle: at nch 16 the sixteen-way split is **1.9
times slower than not splitting at all**. The shape follows from how this merge is built -- every simdgroup
reads all `kvs` partials and merges them redundantly, so per-simdgroup time goes as `nch / kvs + a * kvs`
while the *total* merge work goes as `kvs^2`. That model puts the optimum near `sqrt(nch)`, which is 4 at
nch 16 and 5.7 at nch 32. The measurement is consistent in direction without confirming the exponent: the
argmax stayed at 4 in both, but the gap between 4 and 8 narrowed from 1.58x to 1.11x as nch doubled, which is
the movement the law implies. Distinguishing `sqrt(nch)` from a fixed optimum of 4 needs a longer key axis
than these runs cover.

The redundant merge is a deliberate simplification -- it avoids a divergent branch -- and it is also exactly
what rule 29 warns against. A merge in which one simdgroup combines and the others wait would trade the
`kvs^2` term for a barrier, and is the obvious next variant; it is not built here, so the crossover above
should be read as a property of *this* merge and not of KV splitting in general.

**What this closes and what it does not.** Ledger item M1 moves from "not built" to built, exact, and costed.
The reported split times of sections 145 to 150 remain as they were: they measured schedules that did not
split the key axis, and this section does not change them. What it adds is that a KV split was available all
along, worth up to 3.3x at one query tile, and carrying a value change that no amount of care in the merge
removes.

## 155. Attention with a causal mask: the mask is eight integer compares and a select per score tile from the inverse of the fragment layout, costing 0 to 3 percent and bit-exact in 6 of 6 configurations; truncating the loop at the diagonal is provably and measurably a no-op on the values, and its speedup is set by occupancy, rising from 1.06x to 1.37x as the machine fills

Ledger row M3 read "attention with a causal mask, or a KV-cache update -- not built". The mask is built here.
The KV-cache update is not, so the row narrows rather than closes.

**The mask needs no table.** Masking a score tile means knowing, for each lane and each element of its
`<8 x float>`, which `(row, column)` of the 16x16 tile it holds. That is the inverse of section 129's `pos()`:
element `(r, c)` lives at lane `16*(r>>3) + 8*(c>>3) + 2*((r>>1)&3) + ((c>>2)&1)`, slot `4*(r&1) + (c&3)`, so
element `j` of lane `L` is

    row = 8*(L>>4) + 2*((L>>1)&3) + (j>>2),        column = 8*((L>>3)&1) + 4*(L&1) + (j&3)

The lane-dependent parts are two integer expressions computed once; the element-dependent parts are the
compile-time vectors `<0,0,0,0,1,1,1,1>` and `<0,1,2,3,0,1,2,3>`. With the task's first global query row in a
register, the mask is one splat, two vector adds, one `icmp sgt` and one `select` per score tile. No lookup
table, no memory traffic, no divergence. Harness `results/g17-tensorops-recon-v1/mxattn_causal.py`.

The mask is applied to the *scaled* scores and writes `-1e30`, the same value the running max is initialized
to, which keeps the masked entries out of the max without introducing an infinity for the `maxnum` rule of
section 126 to swallow.

**Exact in 6 of 6 configurations**, against a reference with the mask applied at the same point, `O` and `Y`
both, every element. The last column is the check that the mask is doing something: a mask that silently did
nothing would still match a masked reference only if the reference were also inert.

| fmt | dh | S | heads x tiles | mt | visible fraction | O and Y mismatches | elements differing from the *unmasked* reference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e4m3 | 64 | 128 | 1 x 2 | 1 | 0.129 | 0 | 1,024 |
| e4m3 | 64 | 128 | 1 x 8 | 1 | 0.504 | 0 | 821 |
| bf16 | 64 | 128 | 1 x 8 | 1 | 0.504 | 0 | 960 |
| e4m3 | 128 | 256 | 2 x 16 | 1 | 0.502 | 0 | 1,371 |
| bf16 | 128 | 256 | 1 x 16 | 1 | 0.502 | 0 | 1,920 |
| e4m3 | 64 | 256 | 1 x 8 | 2 | 0.502 | 0 | 1,472 |

**Truncating the loop at the diagonal changes nothing.** Chunks entirely above the diagonal need not be
visited, and the argument that skipping them is exact is not statistical. For a fully masked chunk the chunk
max is `-1e30`, so the new running max equals the old one and `alpha = exp2(m - m)` is exactly 1; every
probability is `exp2(-1e30 - m)`, which underflows to exactly zero, so the row sum gains nothing and `O` is
multiplied by one. The measurement agrees: with the loop stopped at `min(NCH, ((qb + 16*mt - 1) >> 5) + 1)`,
**all 6 configurations remain bit-exact** against the full-loop causal reference.

**The mask is nearly free; the skip is worth what occupancy lets it be.** e4m3, dh 128, median of 7 to 9
rounds, one threadgroup per query tile:

| | S = 512, nch 16 | S = 1024, nch 32 |
| --- | --- | --- |
| unmasked | 52,026 ns | 109,956 ns |
| causal, full loop | 51,642 ns (0.99x) | 113,678 ns (1.03x) |
| causal + diagonal skip | 49,536 ns | 104,773 ns |
| skip against unmasked | **1.05x** | **1.05x** |

A causal mask removes about half the key-block work, so 1.05x is almost none of it. The reason is that the
query tiles run concurrently and the *last* tile still reads the whole key axis: with one threadgroup per
query tile and the machine not full, the kernel time is set by the longest task and not by the mean. That is
a prediction, and it was tested by filling the machine -- the same shape at 1, 8 and 32 heads, so 32, 256 and
1,024 tasks over the same key length:

| tasks | unmasked | causal + skip | speedup |
| --- | --- | --- | --- |
| 32 | 52,188 ns | 49,317 ns | 1.06x |
| 256 | 76,206 ns | 63,102 ns | 1.21x |
| 1,024 | 239,400 ns | 174,350 ns | **1.37x** |

Monotone, and still climbing at 1,024 tasks against a work ratio near 1.9x. So the skip is a **throughput**
optimization: it pays in proportion to how much of the machine is already busy, and a schedule that hands one
query tile to one threadgroup converts most of it into idle lanes waiting on the tail. Recovering the rest
needs the tiles load-balanced -- pairing tile `i` with tile `nq - 1 - i` is the usual device -- which is not
built here.

**What remains in M3.** The KV-cache update: appending a key block to a cache and attending over a prefix
whose length is not a multiple of the chunk. The mask machinery above is what such a kernel would need for
the ragged tail, so the remaining work is addressing and dispatch, not layout.

### 155.1 The causal workload is the first non-uniform one in this record, and it finds a defect in the section-150 cost model: `max(latency, throughput)` is a makespan lower bound that is tight only when every simdgroup does the same work

The three occupancy points above are a case the section-150 model was never fitted on, so they are a test of
it. The model is `T = max(T_latency, T_throughput, T_memory)` with `T_latency = steps per simdgroup * L_step`
and `T_throughput = total steps * C_step / 20 cores` (`mxcost_model.py`, constants calibrated from sections
145 to 149: `L_step` 5.35 us, `C_step` 433 ns).

A causal skip does exactly one thing to that model: it divides *total* steps by the work ratio and leaves the
*longest* simdgroup's step count alone, because the last query tile still walks the whole key axis. At nq 32
and nch 16 the kernel's own trip counts are 1, 1, 2, 2, ... 16, 16, totalling 272 against 512, a work ratio of
**1.882**, with the maximum over tiles equal to nch. So the prediction is
`max(lat, thr) / max(lat, thr / 1.882)`, which is 1.0 while latency dominates and clamps at 1.882.

**It does not reproduce the measurement**, and the failure survives refitting.

| tasks | predicted, calibrated constants | predicted, constants refitted to my own uniform runs | measured |
| --- | --- | --- | --- |
| 32 | 1.000 | 1.000 | 1.058 |
| 256 | 1.037 | 1.460 | 1.208 |
| 1,024 | 1.882 | 1.882 | 1.373 |

The calibrated constants do not transfer to these kernels -- the model's `L_step` and `C_step` come from rows
with more work per step, and it over-predicts the uniform time at 1,024 tasks by 48 percent -- so the first
column conflates two errors. The second column removes that: fitting `L_step` and `C_step` to my own three
*uniform* points gives the best the `max()` form can do, 2,868 ns and 327 ns at a worst error of 12.1 percent,
and the causal prediction is then **optimistic by 20.9 and 37.1 percent**. The defect is in the form, not the
constants.

Two things are wrong, and they are separable.

**The two-regime shape is too sharp even for the uniform workload.** The form says time is flat in task count
until throughput crosses latency and linear after. Measured, an eightfold increase in tasks (32 to 256)
multiplies the time by 1.46, and the next fourfold (256 to 1,024) multiplies it by 3.14. Neither flat nor
linear: the machine is partly saturated across the whole range, and the best `max()` fit leaves plus or minus
12 percent.

**Applied to a non-uniform workload it is optimistic by construction.** `max(longest task, total work / cores)`
is the standard makespan lower bound. It is *tight* when every task is identical, which is why the model has
been accurate on this record's workloads so far -- all of them uniform. A triangular distribution breaks the
tightness: a list schedule of tasks of length 1, 1, 2, 2, ... 16, 16 leaves cores idle while the long tiles
finish, so the true makespan exceeds the bound, and the model claims a benefit the schedule does not deliver.

This sharpens rather than contradicts the form. `T_latency` must be the **maximum** over simdgroups and not a
common steps-per-simdgroup value -- with a common value the prediction would be worse still -- but making it
the maximum is necessary and not sufficient. The honest statement of the model's domain is: **it is a lower
bound on time, exact to within its calibration where every simdgroup does equal work, and optimistic by up to
about 40 percent where they do not.** Until a scheduling term is added, a non-uniform workload's predicted
speedup should be read as a ceiling.

That also answers what the causal skip is worth at scale without another measurement: the 1.37x at 1,024 tasks
is below the 1.882x work ratio not only because occupancy is still climbing but because a triangular schedule
cannot reach the ratio at all without load balancing. Pairing tile `i` with tile `nq - 1 - i` makes every pair
cost `nch + 1` steps and restores uniformity, which is the change that would let the bound bind.

## 156. U1 and U2 identified: there is no mode effect, no transpose effect and no MMA-count discontinuity -- there is an instruction-footprint cliff at about 16 KiB that costs 2.3 times the per-instruction rate at one simdgroup per core and vanishes at 64

Ledger rows U1 ("~2-fold mode-B/mode-A spread") and U2 ("A-transposed rows 2 to 2.2x slower") were narrowed
in section 25.9 to a single "scaling discontinuity" between 24 and 48 MMAs, with one cell, `B/4x2`,
unexplained on the lower branch. That framing is also wrong, and the real variable was in the objects the
whole time. **No new dispatches were made; this is a re-reading of `mxfuse_cost*.log` plus the code size of
the twenty-four kernels still on disk.**

**The rate is bimodal, and MMA count is not the variable.** Dividing each cell's one-simdgroup fused time by
its own instruction count separates the twenty-four cells into two groups with nothing between them:

| regime | cells | ns per instruction at 1 SG | MMA counts present |
| --- | --- | --- | --- |
| fast | 5 | 1.25 to 1.50 | 24 **and 48** |
| slow | 19 | 2.85 to 3.56 | **48**, 72, 96 |

48 MMAs appears in both, so the MMA count cannot be the cause. Registers cannot be either: the fast cells
span 83 to 105 and the slow cells start at 93, and `A/4x2` at 93 registers is slow while `B/4x2` at 105 is
fast. Spill stores are zero in every 48-MMA cell of both regimes.

**The variable is code size, and the threshold is about 16 KiB.** These kernels average 9.1 bytes per
instruction (G17 instructions are variable length, so bytes and not instruction count is the physical
quantity). Sorting all twenty-four by the size of their decoded `code.bin`:

| | bytes | ns per instruction at 1 SG |
| --- | --- | --- |
| the five fast cells | 8,828 / 9,120 / 10,188 / 10,594 / **16,066** | 1.35 / 1.27 / 1.31 / 1.25 / **1.50** |
| the nineteen slow cells | **17,158** to 38,932 | 2.85 to 3.56 |

**Max fast 16,066 bytes, min slow 17,158 bytes, and 16,384 lies between them.** The separation is perfect
over twenty-four cells with no exception. The largest fast kernel sits 318 bytes under 16 KiB and the
smallest slow one 774 bytes over it.

**The control that names the mechanism.** The same twenty-four kernels at **64 simdgroups per core** show
**no cliff at all**: 0.164 to 0.228 ns per instruction below the threshold against 0.181 to 0.249 above,
two overlapping ranges. A cause that were extra work per instruction -- a longer dependent chain, more
scale arithmetic, worse scheduling -- would appear at both occupancies. One that is **instruction fetch**
appears only where there is nothing to hide it behind, which is exactly one simdgroup per core. That is the
discriminator, and it is why this is an identification rather than a correlation.

**What it explains.** `B/4x2` escapes because it is the only 48-MMA kernel under the threshold, at 16,066
bytes; nothing about mode B or grid 4x2 matters except that this combination emits the least code. The
"48-MMA band spread" is the band that happens to straddle 16 KiB. The "A-transposed 2 to 2.2x" holds
because the transposed modes emit the most code (`At/2x4` is 21,074 bytes against `A/2x4`'s 17,158), not
because transposition is expensive. The "one step, linear either side" of section 25.9 is a capacity
boundary: a footprint either fits or it does not, and once it does not, the per-instruction rate is flat at
3.2 to 3.5 across a 2.3-fold range of kernel sizes.

**Standing.** The threshold is *bracketed*, not measured: it lies in (16,066, 17,158] bytes and 16 KiB is
the natural candidate inside that window, but these twenty-four objects cannot narrow it further. A sweep
that varies unrolling at fixed arithmetic -- which changes footprint without changing work -- would pin the
exact byte at which the rate steps, and would also test whether the step is the total kernel footprint or
only the loop body's. Until that runs, "about 16 KiB" is the honest statement and the mechanism, not the
constant, is what is established.

**Compiler consequence.** At low occupancy, a fused kernel's instruction footprint is a first-order cost,
independent of its arithmetic. Fusing two stages that together exceed the threshold can cost 2.3x per
instruction, which is how a fused chain comes to lose to a three-kernel split whose parts each fit -- the
effect section 140 recorded as "at 1 SG and 8 tiles the 3-kernel bridge wins" without identifying why.

## 157. X1 identified: the float-to-integer conversion saturates on this hardware and maps NaN to zero, deterministically -- and the ledger's own proposed fix, clamping before converting, is the one form that gets NaN wrong

Ledger row X1 asked what an int8 block containing NaN or Inf converts to. It was narrowed correctly as far as
the language goes -- LLVM makes `fptosi` of a NaN or out-of-range operand **poison**, so "unspecified bytes"
is the compiler's licence and not a hardware nondeterminism -- and then closed the question as "a different
and lower-value question". It is not lower-value. Poison says what the *optimiser* may assume; the emitted
instruction still does something definite, and a backend that emits it directly needs to know whether the
machine saturates (the clamp is redundant) or wraps (the clamp is mandatory).

**Measured, `x1probe.py`, four conversions on eight inputs, three dispatches each, 32 lanes.** Determinism was
checked before anything was concluded: every value is identical across all three runs and all 32 lanes, in all
four conversions. Results shown as the zero-extended byte the instruction produced.

| input | `fptosi` to i8 | `fptoui` to i8 | `fptosi` to i32 | clamp then `fptosi` to i8 |
| --- | --- | --- | --- | --- |
| quiet NaN | **0** | **0** | **0** | **129** (-127) |
| signalling NaN | **0** | **0** | **0** | **129** (-127) |
| +Inf | 127 | 255 | 2,147,483,647 | 127 |
| -Inf | 128 (-128) | 0 | -2,147,483,648 | **129** (-127) |
| +1e30 | 127 | 255 | 2,147,483,647 | 127 |
| -1e30 | 128 (-128) | 0 | -2,147,483,648 | **129** (-127) |
| +200 | 127 | 200 | 200 | 127 |
| -200 | 128 (-128) | 0 | -200 | **129** (-127) |

**The hardware saturates, at both widths, signed and unsigned.** Out-of-range finite values and infinities
clamp to the destination's limit -- `INT8_MAX`/`INT8_MIN`, `UINT8_MAX`/0, `INT32_MAX`/`INT32_MIN` -- rather
than wrapping. This is not IEEE-mandated and not implied by the IR; it is what this machine does.

**NaN converts to zero**, in every signed and unsigned width tested, from both quiet and signalling encodings.

**And the row's own recommendation is the one thing that breaks it.** X1's actionable rule was "clamp into
range before converting, or use a saturating conversion". The second half is now redundant, because the
conversion already saturates. The first half is **actively harmful**: `llvm.maxnum(NaN, -127)` returns the
non-NaN operand by definition, so a clamp maps NaN to the clamp's *lower bound*. A block containing NaN
converts to 0 without the clamp and to -127 with it -- a large negative weight where there was an absent one.
The fix proposed to make the hazard safe is the only tested form that turns a NaN into a plausible-looking
wrong number instead of a zero.

**What a compiler should do with this.** Emitting the conversion directly, as this campaign's hand-authored
AIR path does, the behaviour is defined, deterministic and benign, and no clamp is wanted. Going through an
optimiser that is entitled to exploit poison, the guarantee does not survive the IR even though the machine
would have honoured it, so the value must be made finite before the conversion -- and if it is, it must be
made finite by a **NaN-aware** select and not by `maxnum`/`minnum`, whose NaN rule is what causes the defect
above. The hazard in chapter 17 stands, but its remedy was wrong.

## 158. U3 characterized: the legacy bimodality is a two-state per-issue RATE with a 9/8 ratio, selected per dispatch rather than per machine or per kernel -- and the same-process tensor control excludes every machine-wide cause, DVFS included

Ledger row U3 recorded the solo legacy MMA rate falling into two clusters at 32 simdgroups per core and asked
"what selects the mode". Re-reading `mxdualf_u3.log` -- five repetitions of eight variants, no new dispatches
-- settles three of the four things one would want to know about it, and excludes the mechanism most likely to
be proposed.

**It is a rate, not an overhead.** Normalizing each legacy variant by its own issue count gives two clusters
that do not move with chain length:

| legacy issues per iteration | fast cluster | slow cluster | ratio |
| --- | --- | --- | --- |
| 4 | 0.081 | 0.089 | 1.109 |
| 8 | 0.079 | 0.089 | 1.122 |
| 16 | 0.078 | 0.088 | 1.123 |
| 64 | 0.077 | 0.087 | 1.129 |

A fixed startup cost would shrink as a fraction of a longer chain. Across a sixteen-fold range of chain
length the two levels and their separation are constant, so the difference is in the **per-issue rate**.
The mean ratio is **1.121**, against 9/8 = 1.125 and 8/7 = 1.143 -- close enough to a small integer quantum
to suggest a discrete issue interval of eight against nine units, and not close to any ratio of published
clock bins.

**It is selected per dispatch, not per machine and not per kernel.** Within repetition 2 the four-issue
variant is in the slow cluster while the eight-, sixteen- and sixty-four-issue variants are all in the fast
one, so it is not a property of the repetition. Each variant also appears in both clusters across the five
repetitions, so it is not a property of the kernel. Whatever selects the mode is re-decided for each
dispatch and then holds for that dispatch's duration.

**Every machine-wide cause is excluded by a control taken in the same five repetitions.** The tensor solo
kernels measured alongside move by **0.02 and 0.06 percent** (634.2 ns four times and 634.3 once; 1,266.6
four times and 1,267.3 once) while the legacy kernels move by 11 to 13 percent. A clock change, a thermal
or DVFS transition, a residency change or contention from another process must move both streams. It moves
only one. **This excludes DVFS as the mechanism** -- which matters because DVFS is the natural first guess,
was proposed independently by the prior-art lane, and cannot be excluded from published sources: the vendored
Turner material gives nominal peak clocks per chip and no bin ladder at all. The exclusion comes from this
record's own control, not from the literature.

**What remains, stated as a finite candidate set rather than as another narrowing.** The selector is
re-decided per dispatch, affects only the legacy pipe, and changes a per-issue rate by about one part in
eight. That leaves three candidates:

1. **Placement** -- which cores and which of the four accelerator slots per core the workgroups land on, with
   two placements differing by one issue slot's worth of arbitration.
2. **Operand-fetch conflict** -- a bank or port conflict in the legacy pipe's operand path that depends on the
   register assignment the driver chooses for that dispatch.
3. **A latched arbitration state** in the legacy issue path, set at dispatch and held.

The experiment that separates them needs no new kernel: dispatch one legacy variant many times **within a
single process**, recording the cluster each time, and again across processes. If the mode varies within a
process it is placement (1); if it is constant within a process but varies between them it is driver-chosen
allocation (2 or 3), and the driver's reported register assignment then separates those two. That is one
sweep of one existing kernel, and it is the whole remainder of U3.

## 159. The instruction-footprint cliff measured directly: the capacity is 16 KiB and it applies to the LOOP BODY, not the whole kernel -- section 156's bracket was right by accident and is corrected here

Section 156 identified U1 and U2 as an instruction-footprint cliff and bracketed it to (16,066, 17,158]
*whole-kernel* bytes from twenty-four fused kernels that were never built to answer this. That reading needed
testing rather than defending, for two reasons: the twenty-four kernels differ in instruction mix as well as
size, so size could have been a correlate; and `mxcal.py`'s own purpose-built `icache` sweep shows a
**smooth** rise at one simdgroup, not a step. The two were not in conflict only because that sweep has **no
sample at all between 9,606 and 17,844 bytes** -- it straddles the entire claimed threshold.

**`icfine.py`: seventeen points across that gap, varying only the fma chain length, so footprint changes and
the instruction mix does not.** The cliff is real, and it is sharper than the coarse sweep could show:

| total bytes | loop-body bytes | 1 SG ns/instr | 64 SG ns/instr |
| --- | --- | --- | --- |
| 15,376 | 14,013 | 0.878 | 0.152 |
| 16,064 | 14,701 | 0.879 | 0.153 |
| 16,368 | 15,005 | 0.882 | 0.153 |
| 17,030 | 15,667 | 0.889 | 0.153 |
| 17,334 | **15,971** | **0.893** | 0.154 |
| 17,958 | **16,595** | **1.859** | 0.162 |
| 18,610 | **17,247** | **2.454** | 0.165 |
| 20,522 | 19,159 | 2.478 | 0.170 |

**The quantity is the loop body.** Fitting total size against chain length over all seventeen points gives
`code_bytes = 1,363 + 63.90 x chain_length`, so 1,363 bytes are prologue and epilogue outside the loop.
Subtracting them: the last fully fast loop body is **15,971 bytes**, the first fully slow one **17,247**, and
**16,384 lies between them**. The capacity is **16 KiB**, and it holds the loop body rather than the kernel.
That is why section 156's whole-kernel bracket landed near the right number without being the right variable.

**There is a partial-residency point, which is what makes this a cache and not a switch.** At 16,595 loop
bytes -- 211 bytes over 16 KiB -- the rate is 1.859, almost exactly midway between the 0.893 plateau and the
2.47 floor. A hard mode change would step once; a cache that just fails to hold the body degrades over the
first few hundred bytes of overflow, which is what the single intermediate point shows.

**The occupancy control holds and is stronger here than in section 156.** Over the same seventeen kernels the
one-simdgroup rate rises **3.3x** while the sixty-four-simdgroup rate rises **1.15x**. Instruction fetch is
hidden almost completely when there are other warps to switch to, and not at all when there are none.

**Standing, and what changes.** The capacity is now measured rather than bracketed: **16 KiB, applied to the
loop body, on a kernel whose instruction mix is held constant**. Section 156's mechanism stands unchanged and
its bracket is superseded by this. The prior-art lane reports Turner measuring a **12 KB** instruction cache
on Apple 7 and 8 (M1, M2, A14, A15); 16 KiB on G17 is consistent with that as a generational increase, and
the 12 KB figure is independently refuted for this part by section 156's five fast cells, which exceed it and
are fast.

**Compiler consequence, sharpened.** The rule is not "keep the kernel small" but **"keep the innermost loop
body under 16 KiB"**. Prologue and epilogue are free of this cost; unrolling is what spends it. At one
simdgroup, crossing the boundary costs 2.8x per instruction regardless of what the instructions do, which is
a larger factor than most of the arithmetic choices this record measures.

## 160. U3's discriminator run: the legacy mode flips dispatch to dispatch inside one process, which excludes register assignment and process-scope latching and leaves only a dispatch-time decision

Section 158 reduced U3 to three candidates and named the experiment that separates them: dispatch one legacy
kernel repeatedly **within** a single process and again **across** processes. If the mode varies inside a
process it is re-decided per dispatch; if it is constant inside and varies between, it is chosen once by the
driver. `u3sweep.py` ran it, with the tensor kernel measured in the same dispatch as the control throughout.

**Three processes, six dispatches each:** legacy spread **12.26, 12.13 and 12.23 percent** within each
process, tensor spread **0.02, 0.04 and 0.03 percent** in the same dispatches. The mode varies inside a
single process, and the control does not move while it does.

**Ten dispatches in one process, in order:** 727.7, 649.2, 728.5, 649.1, 649.2, 649.5, 650.0, 722.2, 649.1,
728.2 ns. Four slow, six fast, no periodicity, and the two levels stay tight and separate -- the fast values
span 649.1 to 650.0 and the slow 722.2 to 728.5, with nothing between. The tensor kernel in those same ten
dispatches reads 1,266.4 to 1,266.8.

**What this excludes.** The **register assignment** candidate is dead: the binary is identical on every
dispatch, so an operand-fetch conflict fixed by the driver's register choice cannot flip between them. The
**process-scope latch** is dead for the same reason the first is -- the mode changes without the process
changing. And every machine-wide cause was already excluded in section 158 by this same control, which holds
again here at 0.03 percent while the legacy stream moves 12.

**What remains is one thing, decided at dispatch time**: which cores and which of the four accelerator slots
per core the workgroups are placed on, or an arbitration state latched when the command buffer is scheduled.
Those two are not separable by repetition, because both are re-decided per dispatch; they are separable by
**grid size**, since a placement effect should change its frequency as the number of threadgroups changes
while a latched state should not. That is the remaining experiment, and it is one sweep of the same kernel at
several grid widths.

U3 therefore goes from "what selects the mode" -- with DVFS the natural and unexcluded guess -- to a single
dispatch-time mechanism with two named variants and a one-sweep discriminator, and with the clock, thermal,
residency, contention and register-assignment explanations all measured out rather than argued out.

## 161. E6's third contending stream run: removing the accumulate from the ALU side changes nothing, which refutes the fp32-accumulate-path reading and leaves result-write bandwidth as the surviving mechanism

E6 asked what the legacy MMA and the ALU share, given that schedclass 43 (`op838`, f16 accumulate) overlaps
fp32 fma by 0.66 to 0.95 while the schedclass 118 forms (f32 accumulate) overlap by 0.37 or less. The
prior-art lane narrowed it by elimination -- not operand width, since `op2842` takes fp32 inputs and contends
exactly like the fp16 and bf16 forms; not multiply throughput, since the issue interval is 2.79 to 2.80 ns at
32 simdgroups per core in all four forms -- and proposed that the shared resource is the **fp32 accumulate
path**, with a prediction that separates it from generic ALU issue: an ALU multiply that does **not**
accumulate should not contend.

**Run, `e6mul.py`.** The existing `mxdualf` kernels, with the `llvm.fma.v8f32` call rewritten in the emitted
IR to a plain `fmul` on the same two operands -- same SSA names, same dependent chain, same loop, so the only
difference is the accumulate. The rewrite is asserted to have applied (8 calls in each of the two kernels)
before anything is timed. 32 simdgroups per core, median of 9 rounds:

| kernel | ns per iteration |
| --- | --- |
| legacy alone (L8) | 649.1 |
| fma alone (F8) | 494.1 |
| **fmul alone (F8)** | **357.6** |
| legacy + fma | 1,116.1 |
| **legacy + fmul** | **1,020.8** |

Overlap, as the fraction of the smaller stream that disappears when the two run together:

| ALU stream | overlap beside the legacy MMA |
| --- | --- |
| `ffma` | **0.055** |
| `fmul` | **-0.04** |

**Both are zero.** The `fmul` stream is 28 percent cheaper on its own and hides no better beside the legacy
MMA than `ffma` does; the two streams serialize in both cases. The prediction was that removing the
accumulate would remove the contention. It does not, so **the shared resource is not something only an
accumulating ALU operation needs**, and the fp32-accumulate-path reading is refuted in the form it was
stated.

**What survives, and why it is sharper than "generic ALU issue".** The variable that does move contention is
on the **MMA side**, not the ALU side: `op838` accumulates into one 32-bit register per lane and barely
contends, while the class-118 forms accumulate into two-register fp32 tuples and contend almost completely.
Since any fp32 vector ALU operation contends equally regardless of its own accumulate, the resource is
something **every fp32 vector result must pass through and that a wider result consumes more of** -- a
register-file write port or result bus, rather than an accumulate datapath.

That reading predicts something neither lane has tested: an ALU operation producing a **narrower** result
should contend less. An f16 or bf16 multiply against the same legacy stream separates "write bandwidth scales
with result width" from "any vector result costs the same", and it is one dispatch of the same shape as this
one. E6 is therefore not closed, but its candidate set is now one mechanism with a width prediction, and two
readings -- operand width, multiply throughput, and now the ALU-side accumulate -- have been measured out
rather than argued out.

## 162. E2's expectation is refuted by counting instructions instead of elements: the in-register cross-engine bridge is eight `simd.shuffle` instructions with no selects, and should be expected to win

E2 was narrowed to a design answer without a build: with E1 resolved, both layouts are known, and of the 64
legacy `(lane, half)` slots **only 2 hold an element sitting in the same lane of the tensor fragment**. The
row concluded that an in-register bridge "requires cross-lane movement for 62 of 64 elements -- a near-total
permutation, not a relabeling" and that building it "should be expected to lose" against section 153's
memory bridge, framed as "one fragment-order store and load against roughly sixty cross-lane moves".

**That conflates elements with instructions.** A cross-lane move is not an instruction: `simd.shuffle` is in
the contract as a general per-lane gather, so one instruction relocates one value in **every** lane at once,
each lane naming its own source. The right question is how many shuffles the permutation decomposes into.

**Computed from `pos()` and E1's closed form, for the 8x8 block inside a 16x16 fragment.** The control first:
this mapping reproduces the row's own figure exactly -- **2 of 64** elements are already in the correct lane
-- so it is the same permutation the row described, counted differently.

| destination slot | source half | elements | distinct destination lanes | instructions |
| --- | --- | --- | --- | --- |
| 0, 2, 4, 6 | 0 | 8 each | 8 each | 1 `simd.shuffle` each |
| 1, 3, 5, 7 | 1 | 8 each | 8 each | 1 `simd.shuffle` each |

**Eight `simd.shuffle` instructions for the whole 8x8 bridge, and no selects.** The structure is why: each
destination slot draws from exactly **one** source half, chosen by the slot's parity -- even slots take half
0, odd slots half 1. A permutation that needed both halves per destination slot would have cost a select as
well; this one does not.

**So the expectation inverts.** Eight register-to-register shuffles is not "roughly sixty cross-lane moves",
and the thing it is being compared against is not one store and one load: section 153's bridge is a
**host-mediated two-dispatch round trip**, and even an in-kernel version costs a store, a load, a barrier and
their latency. Eight ALU-class instructions per fragment should be expected to **win**, not lose, and
decisively at low occupancy, where section 159 has just shown that instruction footprint rather than
arithmetic is what costs -- and eight instructions is a very small footprint.

**Standing.** The kernel is still unbuilt, so this is a cost computed from two measured layouts rather than a
measurement. But the row's substantive claim -- that the in-register form is structurally disadvantaged --
is refuted, and what remains is no longer a design question with a discouraging answer. It is a build whose
instruction count is known exactly in advance: eight shuffles, no selects, source half given by destination
slot parity.

## 163. C5 given a falsifiable prediction: baseline native pressure steps between capacity 16 and 24, not between 24 and 32, so if pressure drives the reduction-direction flip then capacity 24 must already be ascending

Section 25.8 refuted C5's fold-away explanation and moved the residue to "a native scheduling/register-pressure
threshold", observing that read-all costs 8 instructions and no extra spills at per-lane capacity 16 but 201
instructions and 64 extra spills at capacity 32, while the direction boundary sits between the same two
capacities. That is a correlation across a single interval with nothing sampled inside it.

**Capacity 24 is inside it, and reachable with non-power-of-two shapes.** Built through the same path
(`gen_red_shape.py`), baselines only, nothing dispatched. The control first: the two retained shapes
reproduce section 25.8's recorded baseline instruction counts exactly, 266 and 338, so these are the same
objects.

| shape | per-lane capacity | instructions | registers | `op595` stack stores |
| --- | ---: | ---: | ---: | ---: |
| 16x32 | 16 | **266** | 56 | 20 |
| 16x48 | 24 | 329 | 72 | 28 |
| 24x32 | 24 | 328 | 80 | 24 |
| 32x32 | 32 | **338** | 80 | 24 |

**The baseline pressure step is between 16 and 24, not between 24 and 32.** Both capacity-24 shapes sit with
capacity 32 and not with capacity 16: 328 and 329 instructions against 338, and 80 registers matching 32x32
exactly, where capacity 16 is 266 instructions and 56 registers. Whatever changes in the native schedule
between the descending and ascending regimes has already changed by capacity 24.

**That makes C5 falsifiable rather than merely bounded.** If native pressure is what selects the reduction
direction, then **capacity 24 must already be ascending**, because its pressure profile is the capacity-32
one. If capacity 24 measures descending, pressure is excluded as the selector and the residue moves to
something that tracks capacity itself. Either outcome ends the row.

**The remaining work is one build, and it is not the one just done.** These are baseline kernels; the
discriminator needs the *read-all* variant at capacity 24, and `gen_red_variants.py` emits fixed shapes
only, so it needs a shape parameter before the read-all delta and the direction can be read at 16x48 or
24x32. Stated plainly so the next reader does not repeat the baseline build believing it answers the
question: it establishes the prediction, not the result.

## 164. U3 identified: the legacy time is quantized in steps of about 79.4 ns, and the "bimodality" is dispatches landing on adjacent rungs -- which also refutes section 158's constant 9/8 ratio as a sampling artifact

Sections 158 and 160 characterized U3 as a two-state per-issue **rate** with a constant ratio of about 1.121,
selected per dispatch. Both were read from measurements taken **only at 32 simdgroups per core**. Sweeping
occupancy shows that framing is wrong, and mine to correct.

**Occupancy sweep, `u3grid.py` and its onset variant.** Same kernel, same everything, only the threadgroup
count varies. The tensor kernel is the control in every dispatch.

| simdgroups per core | fast ns | slow ns | ratio | bimodal? |
| ---: | ---: | ---: | ---: | --- |
| 4, 8, 16 | 94.1 / 170.4 / 330.5 | 94.2 / 170.6 / 330.6 | 1.0011 / 1.0010 / 1.0002 | **no** |
| 18 | 410.1 | 410.3 | 1.0004 | no |
| 20 | 410.2 | 410.3 | 1.0004 | no |
| 24 | 568.7 | 569.1 | 1.0007 | no |
| 26 | 569.4 | 645.5 | 1.134 | **yes** |
| 28 | 569.5 | 649.7 | 1.141 | **yes** |
| 30 | 648.9 | 649.1 | 1.0003 | no |
| 32 | 649.1 | 728.0 | 1.122 | **yes** |

**The measurements take only four values.** Across seven occupancies from 18 to 32, every reading is one of
**410, 569, 649, 728 ns**. The gaps are **158.9, 79.8 and 79.0** -- one step of about **79.4 ns**, with the
first gap exactly two steps. The time is quantized.

**The slow level at one occupancy is the fast level at the next.** 28's slow reading is 649.7 and 30's fast
reading is 648.9; 26's slow is 645.5 against the same rung. So the two "modes" are not two states of a rate.
They are **the two rungs a dispatch can land on**, and a dispatch is bimodal exactly when the occupancy sits
near a rung boundary (26, 28, 32) and unimodal when it sits inside one (18, 20, 24, 30, and everything at or
below 16).

**This refutes the constant ratio.** Section 158 reported 1.121, close to 9/8, and drew a discrete issue
quantum from how clean that was. Measured across occupancies the ratios are **1.388, 1.141 and 1.122** -- not
constant, because they are just consecutive rungs of an additive ladder divided by each other. The apparent
9/8 was an artifact of sampling one occupancy, and the prior-art lane's warning applies squarely to my own
reasoning: a clean ratio constrains the candidate set, it is not evidence for a mechanism that would produce
it.

**What stands from 158 and 160.** The exclusions all hold and are what made this findable: it is not machine
state (the tensor control is 0.01 to 0.06 percent at every occupancy here, with one 2.9 percent outlier at 24
simdgroups where another process was active), not DVFS, not the register assignment, not process-scope
latching. What was wrong was calling the two levels a *rate* with a fixed ratio rather than two rungs of a
quantized ladder.

**What U3 now is.** The legacy MMA's time at a given occupancy is quantized in units of about 79.4 ns, and
which rung a dispatch lands on is decided at dispatch. With eight legacy issues per iteration that is about
9.9 ns per issue per step. The remaining question is narrow and no longer about "modes": what the quantum
counts -- most plausibly a wave or allocation granule of resident simdgroups -- and it is answerable by
sweeping occupancy finely enough to locate every rung boundary rather than the three found here.

### 163.1 The shape-parameterised read-all generator exists now, and its first result is a negative: per-lane capacity does not predict the native pressure jump

Section 163 said C5's discriminator needed a read-all variant at capacity 24 and that `gen_red_variants.py`
emits fixed shapes only. `gen_red_ra.py` removes that blocker: it derives both the baseline and the read-all
body from `gen_red_shape.py`'s template and takes `MxN`, so any shape can now be built both ways.

**Built at capacities 16, 24 and 32, nothing dispatched:**

| shape | capacity | baseline instr | read-all instr | delta | baseline spills | read-all spills | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16x32 | 16 | 266 | 308 | +42 | 20 | 20 | +0 |
| 16x48 | 24 | 329 | 391 | +62 | 28 | 28 | +0 |
| 24x32 | 24 | 328 | 471 | **+143** | 24 | 88 | **+64** |
| 32x32 | 32 | 338 | 415 | +77 | 24 | 24 | +0 |

**Two shapes at the same per-lane capacity behave completely differently.** 16x48 and 24x32 are both capacity
24: one adds 62 instructions and no spills, the other 143 instructions and 64 spills. And 32x32, at capacity
32, adds no spills at all. In this construction the pressure jump **does not track per-lane capacity** -- it
tracks the shape, and specifically the one with 24 rows.

**This construction is not section 25.8's, and the two must not be compared.** The baselines agree exactly
(266 and 338, as in section 163), but the read-all deltas do not: 25.8 records +8 instructions at 16x32 and
+201 with 64 extra spills at 32x32, against +42 and +77 with no extra spills here. The read-all loop written
here stores `cT` to `C[4096 + ix[0]*64 + ix[1]]`; whatever 25.8's variant did, it was not this. So the
prediction section 163 made -- that capacity 24 must be ascending if pressure selects the direction -- is
**not** tested by these numbers, and reading them as if it were would be the error this record keeps
catching.

**Where that leaves C5.** The tool it was blocked on exists and is shape-general. The next step is to
reconcile this read-all with 25.8's -- one of the two is reading `cT` in a way the other does not -- and then
read the reduction direction, which no harness here extracts yet and which is what the row actually turns
on. C5 is not closed, and the useful thing added is a generator plus one negative that constrains any
pressure-based explanation: capacity alone does not predict the jump.

## 165. E6 identified: the shared resource is result-write bandwidth, consumed in proportion to result width on either side, and the same two contention bands appear whether the wide result comes from the MMA or from the ALU

Section 161 refuted the fp32-accumulate-path reading by showing that `fmul` contends with the legacy MMA
exactly as `ffma` does, and left one surviving mechanism with a prediction: if the resource is a register-file
write port or result bus, then a **narrower** ALU result should contend less. That prediction is now run.

**`e6width.py`.** The whole F-stream accumulator is declared at the narrow type -- chain, phi and store all
`<8 x half>` -- rather than narrowing only the multiply, which would add `fptrunc`/`fpext` and confound the
instruction count. The legacy stream, the loop, the chain depth and the instruction count are identical
between the two arms. 32 simdgroups per core, median of 9 rounds:

| kernel | ns per iteration |
| --- | --- |
| legacy alone | 723.2 |
| fp32 ALU alone | 385.9 |
| **fp16 ALU alone** | **206.8** |
| legacy + fp32 ALU | 978.9 |
| **legacy + fp16 ALU** | **757.3** |

| ALU result width | overlap beside the legacy MMA |
| --- | --- |
| 32-bit | **0.337** |
| 16-bit | **0.835** |

**Halving the result width more than doubles the overlap.** A flat per-result cost would have left the two
equal. The resource is consumed in proportion to the **bits written**, which is what a write port or result
bus does and what an accumulate datapath does not.

**The two bands are the same on both sides, which is the part that identifies it.** Section 152 measured the
legacy forms against fp32 fma: `op838`, which accumulates into **one 32-bit register per lane**, overlaps
**0.66 to 0.95**; the class-118 forms, which accumulate into **two-register fp32 tuples**, overlap **0.37 or
less**. Measured here from the other direction, with the legacy MMA fixed and the ALU width varied: the
32-bit ALU result overlaps **0.337** and the 16-bit result **0.835**. Wide against wide is about 0.34 either
way; narrow against wide is 0.66 to 0.95 one way and 0.835 the other. **It does not matter which engine
produces the wide result.** That symmetry is why this is an identification and not another correlation: the
resource has no engine of its own, it is charged per result bit to whoever writes them.

**What the elimination sequence now reads as.** Operand width excluded (`op2842` takes fp32 inputs and
contends like the 16-bit-input forms). Multiply throughput excluded (the issue interval is 2.79 to 2.80 ns at
32 simdgroups per core in all four legacy forms). Per-MAC and per-instruction consumption excluded by
arithmetic. The ALU-side accumulate excluded by section 161. What is left, and now positively measured, is
**result-write bandwidth**.

**Compiler consequence.** Beside a saturated matrix stream, the free ALU budget is not a count of
instructions or of MACs but of **result bits**. Narrowing an ALU result buys contention back at roughly
two-for-one in the measured range, which is a scheduling lever the record did not previously have: a
half-precision epilogue beside a matrix stream costs far less than its instruction count suggests.

### 163.2 C2's residual is one loop nesting whose carried state is mode-independent, which is narrower than the row implies

C2's remaining clause reads as "transposed consumer readings inside the hidden-chunk loop", which invites the
reading that the transposed modes are untested across a quantize boundary. They are not. Two facts already in
the tree bound the gap much more tightly.

**The whole stage-1 to stage-2 chain is validated for all four readings.** `mxloop_modes_suite.log` runs A,
At, B and Bt with the stage-1 K loop as a **runtime** loop and checks **D1, the quantized bytes, the scale
codes and D2 bitwise** against the composed reference -- every section-140 configuration plus longer K at 8
and 32 blocks, all seven data variants, both fp8 formats: **416 of 416 exact**. Its repeat test runs three
timing iterations over two input sets holding different data (set 0, set 1, set 0) and requires the last to
equal set 0's reference, so a stale accumulator or register carried between iterations would show:
**112 of 112**. So the quantize boundary, the block scaling, the pack and the transposed relabelings are all
exercised for every reading, inside a runtime loop.

**The chunk loop's only carried state is the D2 accumulator.** In `mxpipe.py` the loop is
`K.loop(trip, [ZERO] * nD2, ['<8 x float>'] * nD2, chunk_body)` -- nothing crosses an iteration but the fp32
D2 tiles. And D2's layout is the D layout whatever the consumer reading is: the reading determines how D1 is
**read as a stage-2 operand**, not how D2 is **written**. So the outer loop's carried state cannot interact
with the mode.

**What that leaves.** Not "are the transposed readings exact across a quantize boundary" -- measured, 416 of
416 -- but "does adding one outer loop, carrying only a mode-independent accumulator, change anything". Mode
A says no at chunk scale (sections 141 to 144, 1,258 of 1,258). There is no identified mechanism by which At,
B or Bt would differ, because the thing that differs between modes is entirely inside the iteration and is
already covered.

**This is an argument, not a measurement, and the row should say so.** The build remains worth doing --
`mxpipe.py` implements only mode A for that boundary, so it is At/B/Bt in `chunk_back`'s consumer reading
with section 140's relabelings and then the same sweep. But the residual risk it would retire is now named
and small, rather than the open-ended one the row's phrasing suggests.

## 166. H4's occupancy sweep run: no register-pressure penalty appears from 20 to 116 registers per lane, which is inconsistent with a knee at 63 -- but the instrument co-varies parallelism with pressure, so it bounds the net effect rather than isolating occupancy

The prior-art lane fitted a two-regime model to H4, `occupancy = min(slot_cap, capacity / (regs * 32 * 4))`,
anchored on 640 simdgroups at 126 registers, predicting a 64-simdgroup slot cap and a **knee at 63 registers
per lane**: flat below, falling as `1/regs` above. They named the sweep that confirms or kills it and offered
it to this lane. Run here.

**Instrument.** `gen_icache`'s accumulator count sets register pressure -- each `<8 x float>` accumulator is
8 registers -- while the fma chain length sets the work. Chain fixed at 96, well under section 159's 16 KiB
footprint cliff (code size 6,380 to 7,056 bytes throughout, so the cliff cannot contaminate this). 1,280
threadgroups, far past saturation. Time normalized to the lowest-pressure point.

| accumulators | registers | spills | ns per fma per core | normalized |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 20 | 0 | 1.3009 | 1.000 |
| 4 | 36 | 0 | 1.3038 | 1.002 |
| 6 | 52 | 0 | 1.2887 | 0.991 |
| 8 | 68 | 0 | 1.2844 | 0.987 |
| 10 | 84 | 0 | 1.2565 | 0.966 |
| 12 | 100 | 0 | 1.2777 | 0.982 |
| 14 | 116 | 0 | 1.2833 | 0.986 |

**Flat, within 3.4 percent and with no trend, across a near-sixfold change in register pressure.** The model
predicts occupancy falling by 116/63 = 1.84 above the knee, so the 116-register point should cost about 1.84
times the 20-register one. It costs 0.986.

**The confound, which is mine and has to be stated.** Accumulator count sets register pressure *and*
instruction-level parallelism together: two accumulators give two independent fma chains, fourteen give
fourteen. A fall in occupancy could therefore be masked by a rise in ILP, and this sweep cannot separate
them. So the honest reading is not "occupancy is flat" but **"there is no net cost to using 116 registers per
lane in this shape"** -- which is a useful bound for a compiler and a weaker statement than the model's
refutation.

**What would isolate it.** Raise register pressure *without* adding independent chains: hold the accumulator
count fixed at two and add live values that are loaded before the loop and stored after it but never read
inside it. Registers then rise while ILP does not, and the model's rising branch above 63 either appears or
does not. That is one more build of the same generator, and it is the version of this experiment that should
have been run first.

**Standing.** The prediction is not observed, but it is not cleanly refuted either, and H4 stays open. What
this does establish, independent of the confound, is that **a kernel can run 116 registers per lane spill-free
at a saturating grid with no measurable penalty against 20** -- which is a fact a scheduler can use whatever
the residency mechanism turns out to be.

### 166.1 The confound-free version was attempted and the instrument was null: the compiler sinks values that are live but unread

Section 166 named the fix for its own confound -- hold the accumulator count at two and add values loaded
before the loop and stored after but never read inside it, so registers rise while parallelism does not. It
was built (`h4clean.py`, carrying the extra values through the loop's phi nodes) and it does not work.

| dead values added | 0 | 2 | 4 | 6 | 8 | 10 | 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| registers touched | 20 | 20 | 20 | 20 | 20 | 20 | 20 |
| code bytes | 6,380 | 6,380 | 6,380 | 6,380 | 6,380 | 6,380 | 6,380 |
| normalized time | 1.000 | 1.026 | 1.012 | 1.021 | 1.017 | 1.009 | 1.022 |

**Register pressure did not change at all, and neither did a single byte of code.** A value that is not read
inside the loop is not kept live across it: the backend sinks the load past the loop and the whole phi chain
disappears, whatever the IR says. The 1 to 3 percent time variation is the noise floor of this harness, not a
pressure effect, and reading it as one would have been a mistake.

This was caught only because the sweep prints `registers_touched` and code size beside the timing. Had it
printed timings alone it would have produced seven plausible numbers, a flat line, and a confident wrong
conclusion about register pressure -- from a kernel where the pressure never varied.

**So H4's clean experiment is still not run**, and the approach in section 166 will not produce it. Forcing
liveness requires the values to be *read* inside the loop, which is exactly what adds either parallelism or
work, so isolating pressure from both needs a different construction than either sweep here used -- reading
each extra value into the single existing chain would serialize it and change the chain length instead.
Section 166's bound stands, with its confound; this is a dead end recorded so the next attempt starts past it.

## 167. H4's two-regime model refuted cleanly: with parallelism and instruction count held fixed, register pressure from 20 to 116 per lane costs nothing, so there is no knee at 63

Section 166 measured no register-pressure penalty but co-varied parallelism with pressure, and section 166.1
found that the obvious fix -- values live but unread -- is a null instrument, because the backend sinks them.
The construction that works is to read each extra value **inside** the loop as the fma's *multiplicand*:
every value must then stay live, while the number of independent chains, the instruction count and the
arithmetic are all unchanged.

**`h4live.py`.** Two accumulators throughout, so two independent chains at every point; chain length 96, so
96 fmas at every point; zero spills at every point; code 6,380 to 7,224 bytes, far under section 159's
16 KiB cliff. 1,280 threadgroups, past saturation.

| live values added | 0 | 2 | 4 | 6 | 8 | 10 | 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **registers touched** | **20** | **36** | **52** | **68** | **84** | **100** | **116** |
| spills | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| normalized time | 1.000 | 0.993 | 0.991 | 0.990 | 0.979 | 1.010 | **1.021** |

**Register pressure now varies 5.8-fold and the time does not move.** The instrument is verified by the
register column itself, which is exactly what section 166.1's failure lacked: pressure rises 16 registers per
step, monotonically, as intended.

**The model predicts 1.84 and measures 1.021.** `occupancy = min(slot_cap, capacity / (regs * 32 * 4))` with
a knee at 63 registers per lane requires occupancy at 116 registers to be 63/116 of its capped value, so the
time should be about 1.84 times the low-pressure point. It is 1.021. There is **no knee at 63**, and no
register-pressure occupancy penalty anywhere in 20 to 116 registers per lane.

**What this settles and what it does not. CORRECTED: the original wording here overreached.** It settles
the **net effect**: a scheduler may use 116 registers per lane, spill-free, at a saturating grid, and pay
nothing against using 20. It does **not** settle that occupancy held, and this section first said it did
("occupancy in this range is slot-limited, not register-limited"), which the measurement cannot support.
The prior-art lane supplied the confound: Turner records ALU utilisation maxing out at **24 simdgroups per
core**, so a kernel above that floor shows flat time while occupancy falls from 64 to 24. Flat time is
consistent with occupancy holding *and* with it falling by nearly a factor of three. What the sweep rules
out is a *time* penalty, which is what a scheduler needs; what it cannot rule out is the occupancy drop,
which is what H4 asks. It does not settle the residency mechanism, and it sharpens the arithmetic that
made H4 interesting: the prior-art lane observed that 504 KiB of addressable live register bits per core
exceeds every physical register file in Turner's cross-vendor table, RDNA 3's 384 KB included. If occupancy
really is unchanged at 116 registers per lane, then 64 simdgroups per core at that pressure would need about
950 KB of backing, which no plausible physical file provides. So either the slot cap is well below 64
simdgroups per core at this shape, or the backing is not fully physical. This measurement cannot separate
those two, and that -- not the knee -- is what remains of H4.

## 168. C5's baseline direction censused across capacities 16, 24 and 32: it does not change, so the flip belongs to the read-all condition and not to capacity

Section 163 said C5 needed a direction extractor no harness provides, and the row's later note said the
census would need a dispatch budget. **Both were wrong.** `census_shape.py M N axis` reconstructs each
output's reduction tree empirically from hardware -- rebuilt with no assumed order, verified against every
observation with clade bitmasks, classified against the shape-general model -- and it runs in **4 to 13
seconds**, not the thousands of seconds a raw triple count suggests.

**Censused on the baseline `rs_MxN` kernels at the three capacities that bracket C5's boundary:**

| shape | capacity | axis | dispatches | outputs verified | distinct trees | (local, stage) |
| --- | ---: | --- | ---: | --- | ---: | --- |
| 16x32 | 16 | rows | 14,880 | 16/16 | 1 | `('desc','asc')` |
| 16x32 | 16 | cols | 1,680 | 32/32 | 1 | `('desc','asc')` or `('asc','asc')` |
| 16x48 | 24 | rows | 51,888 | 16/16 | 1 | `('desc','asc')` |
| 16x48 | 24 | cols | 1,680 | 48/48 | 1 | `('desc','asc')` or `('asc','asc')` |
| 24x32 | 24 | rows | 14,880 | 24/24 | 1 | `('desc','asc')` |
| 24x32 | 24 | cols | 6,072 | 32/32 | 1 | `('desc','asc')` |
| 32x32 | 32 | rows | 14,880 | 32/32 | 1 | `('desc','asc')` |
| 32x32 | 32 | cols | 14,880 | 32/32 | 1 | `('desc','asc')` |

**Every shape, both axes, one distinct tree, and the same classification: local descending, stage ascending.**
The two rows admitting a second variant are the ones the census cannot discriminate -- 1,680 dispatches, the
smallest leaf counts -- so they are ambiguous, not different. Nothing flips between capacity 16 and 32.

**What that establishes.** C5's boundary is **not a property of capacity in the baseline**. Section 25.8
observed the flip under **read-all**, and this census shows the same capacities produce one uniform direction
without it. So the read-all condition is doing the work, and any explanation that reaches for per-lane
capacity alone is excluded. That is a real constraint the row did not have, and it cost 30 seconds of GPU
once the right instrument was identified.

**What remains, and it is now small.** The census must be run on the **read-all** kernels, which means
`census_shape.py` taking a kernel prefix instead of hard-coding `rs_`. My `gen_red_ra.py` read-all does not
reproduce section 25.8's deltas (+42/+77 against +8/+201), so that reconciliation still has to happen first
or the census will be of the wrong kernel. Two small edits and two runs, not a campaign.

**And a correction to my own two prior claims.** Section 163 said no harness extracts the direction; one
does. The C5 row then said the census needed dispatch budget; it needs seconds. Both were stated without
checking the cost of the thing being declined, which is the same error as declining a measurement because it
looks expensive from its worst-case combinatorics.

## 169. C5 identified: the reduction-direction flip tracks native SPILLING, not per-lane capacity -- shown by two shapes at the same capacity that differ, and it reconciles the two read-all constructions

Section 25.8 bounded C5's flip to "a native scheduling/register-pressure threshold" from a correlation across
a single interval: read-all at capacity 16 cost 8 instructions and no spills and stayed descending, at
capacity 32 cost 201 instructions and 64 spills and flipped ascending. Nothing was sampled inside, and
capacity and pressure moved together, so neither was distinguished. Section 168 then censused the **baseline**
direction at capacities 16, 24 and 32 and found it uniform, which located the flip in the read-all condition
but not its cause.

**Censusing the read-all kernels separates them, because two shapes at the same capacity behave differently:**

| shape | capacity | read-all extra instructions | **read-all extra spills** | censused direction |
| --- | ---: | ---: | ---: | --- |
| 16x32 | 16 | +42 | **0** | `('desc','asc')` |
| 16x48 | **24** | +62 | **0** | `('desc','asc')` |
| 24x32 | **24** | +143 | **+64** | **`('asc','asc')`** |
| 32x32 | 32 | +77 | **0** | `('desc','asc')` |

Each direction is a single distinct tree reproducing all of its 14,880 or 51,888 observations.

**The one shape that spills is the one shape that flips.** 16x48 and 24x32 have identical per-lane capacity,
24, and opposite directions. 32x32 has the highest capacity and does not flip. Capacity predicts nothing
here; **the presence of spill stores predicts it exactly, four cases out of four.** The local stage goes
ascending precisely when the read-all pushes the native schedule into spilling, and stays descending whenever
it does not, whatever the capacity.

**This also reconciles the two read-all constructions, which looked like a contradiction.** Section 163.1
recorded that `gen_red_ra.py`'s deltas (+42 at 16x32, +77 at 32x32) do not match section 25.8's (+8 and +201
with 64 spills), and concluded the two are not the same kernel -- correctly. But they do not need to be. The
rule is not about the shape or the capacity, it is about whether **that particular read-all** spills: 25.8's
spills at 32x32 and flips there; mine spills at 24x32 and flips there. Two constructions, two different
shapes, one rule. The apparent disagreement was two instruments sampling the same mechanism at different
points, and the mechanism is what they agree on.

**What C5 was asking, answered.** The trigger for the reduction-direction flip is **native register spilling
in the object**, not the per-lane capacity `M*N/32` that the boundary appeared to track. The capacity
correlation in 25.8 was an artifact of one construction in which pressure happened to cross the spill
threshold between capacity 16 and 32. The compiler-facing form: a read-all that spills gets an ascending
local stage, and a reduction's summation order therefore depends on register pressure in the object -- which
is a stronger version of chapter 17's existing rule that the library reduction's order belongs to the
compiled object and must be probed per object rather than assumed.

### 163.3 C2's gap is a three-way corner: every pairwise combination of (fp8, transposed reading, outer loop) is already measured

C2 reads as though transposed consumer readings are untested in a loop, which is not what the evidence says.
Three factors are in play -- the fp8 operand class, a non-A consumer reading, and an outer runtime loop
carrying state across iterations -- and **all three pairwise combinations are measured**:

| | mode A | modes At, B, Bt |
| --- | --- | --- |
| **fp8, stage-1 K loop** | exact | **exact** -- 416 of 416, both fp8 formats, plus 112 of 112 on the repeat test (`mxloop_modes_suite.log`) |
| **fp8, outer chunk loop** | exact -- 1,258 of 1,258 (sections 141 to 144) | **the gap** |
| **non-fp8, outer unit loop** | exact | **exact** -- `chain_loop.py` at grid 2x2x2x2, where the unit loop's trip count is **2 for every mode**, so a multi-iteration runtime outer loop was exercised for all four |

The grid the row cites for its non-fp8 closure is not a single-iteration case: computing the trip count from
`chain_loop`'s own mode table at `mt1 = nt1 = 2` gives **units = 2** for A, At, B and Bt alike. So the outer
loop with transposed readings is measured; what is not measured is that combination **together with** fp8.

**And the third factor cannot interact with the second.** `mxpipe.py`'s chunk loop carries only the fp32 D2
accumulators, whose layout is the D layout whatever the consumer reading is -- the mode decides how D1 is read
as a stage-2 *operand*, not how D2 is *written*. So the outer loop's carried state is mode-independent by
construction, which is why no pairwise result is surprising and why the corner would be too.

**C2 stays open** -- a three-way corner is still unmeasured, and this record does not close corners by
composing pairs. But the residual is now stated as what it is: not "are the transposed readings safe in a
loop", which is measured twice over, but "does the one combination nobody has run behave like the three that
have". The build remains the At/B/Bt consumer reading in `mxpipe`'s `chunk_back`.

## 170. H4 closed by direct measurement: occupancy IS register-sensitive, falling 44.6 to 32.1 simdgroups per core, and 506 KiB per core of live register state is resident at once

H4 asked for physical register-storage capacity per core, residency allocation granularity, and whether
Dynamic Caching explains the absence of an occupancy cliff. Route 1 (the thread limit) closed negative in
section 25.46; route 2 closed in section 25.48, because `MTLDevice.counterSets` on this part exposes one set
holding one counter, `GPUTimestamp`. Route 3 -- a residency calibration through threadgroup memory --
**bounds** the row and does not close it, and says so in its own text (section 25.49): it resolves residency
in whole threadgroups, eight simdgroups against four, and reaches 94 registers.

**What closed H4 is none of the three.** It is a GPU-side high-water counter, and the instrument was already
in the tree: `rf_occ.py`, written 2026-09-18 for a different question, has every simdgroup's lane 0 count
arrival with an atomic add, record the running peak with an atomic max, and depart after a fixed spin.
Nobody had connected it to this row. Calling it "route 3" -- which this section first did -- collapses a
fourth method into the one it happens to follow; routes 1 and 2 closed negative, route 3 bounds, and a
counter nobody had listed is what produced the number.

**The instrument is verified before its data is used.** Launching grids below capacity, the peak reads
exactly what was launched: 20 simdgroups read 20, 40 read 40, 100 read 100, 400 read 400, with `active_end`
zero in every case so the adds and subs balance. That is the control that separates "occupancy is 32" from
"the max is stuck", and without it the saturating rows below would be uninterpretable.

**Occupancy falls with register pressure.** One simdgroup per threadgroup, 2,048 launched, peak over 20 cores:

| registers | peak resident | per core | implied live KiB/core | 1/regs from the first row |
| ---: | ---: | ---: | ---: | ---: |
| 22 | 893 | 44.6 | 123 | 44.6 |
| 30 | 889 | 44.5 | 167 | 32.7 |
| 46 | 793 | 39.6 | 228 | 21.4 |
| 62 | 711 | 35.5 | 276 | 15.8 |
| 82 | 646 | 32.3 | 331 | 12.0 |
| 98 | 641 | 32.0 | 393 | 10.0 |
| 110 | 645 | 32.2 | 443 | 8.9 |
| 126 | 643 | 32.1 | 506 | 7.8 |

**This refutes both standing readings, including mine.** Section 167 concluded from flat timing that
occupancy in this range is "slot-limited, not register-limited". It is not: occupancy falls by a factor of
**1.39** across the same sweep. The prior-art lane's correction was right and the mechanism is the one they
named -- ALU utilisation maxes out around 24 simdgroups per core, so a kernel above that floor shows flat
time while occupancy falls underneath it. Section 167's time result stands as a bound on *time*; its
occupancy claim is withdrawn, and this is the measurement that withdraws it.

**And it refutes the capacity model in the other direction.** `occupancy = capacity / (regs * 32 * 4)` would
put 126 registers at **7.8** simdgroups per core. Measured: **32.1**. The implied live bytes per core are not
constant either -- they rise monotonically from 123 KiB to 506 KiB -- so a fixed register file being divided
up is not what is happening. Occupancy falls from about 45 to a floor near 32 and then stops falling, which
is two constraints changing hands rather than one capacity being shared.

**The number H4 was actually after, and its two halves are not equally strong.** At 126 registers per lane
the machine holds **32.1 simdgroups per core resident simultaneously**, which is
`32.1 x 126 x 32 lanes x 4 bytes` = **506 KiB per core of live register state**, measured rather than
estimated and a lower bound rather than a fit.

*The half that stands on this data alone is the shape.* A fixed pool divided by per-simdgroup demand
predicts **7.8** simdgroups at 126 registers against **32.1** measured, and the implied live bytes per core
are not constant but rise monotonically from 123 to 506 KiB. Occupancy falls to a floor near 32 and then
stops falling. **That refutes the capacity model without reference to any other chip**: whatever the pool
is, it is not being divided by per-simdgroup register demand, because two constraints change hands rather
than one capacity being shared.

*The half that depends on an outside assumption is the comparison.* Turner's cross-vendor table has no
physical register file as large as 506 KiB -- RDNA 3's 384 KB is the largest entry, Apple 7/8 about 208 KB --
so either G17 carries a file half again larger than any published part, or the backing is not fully physical.
This is **corroboration, not proof**, and the evidence lane is right that it should be labelled as such: it
assumes Apple's file is no larger than AMD's, which is an inference from other silicon rather than a
measurement of this one. 506 KiB is large but not physically impossible, and if Apple simply built a bigger
file this argument evaporates while the shape argument above does not.

**The evidence lane's confound does not reach this instrument, and that was tested rather than assumed.**
Their threadgroup-memory handle infers residency from the size of a timing step, and they found the step's
*magnitude* tracks arithmetic intensity rather than residency -- it falls from 1.260 at nine registers to
1.047 at ninety-four, which reads as register-limited residency and is not, since a control issuing the same
twenty-four fmas per iteration at ten registers reproduces 1.041. Only the step's *presence* carries
residency information. That is a real trap for any method that reads residency out of a duration.

This probe reads a counter, not a duration, so the confound should not apply -- but "should not" is the kind
of claim this record keeps finding wrong, so it was measured. Holding registers fixed at 82 and varying only
the work per simdgroup from 4.1 ms to 21.3 ms, the peak reads **566, 628, 618, 618**: it rises once to a
plateau and then does not move, over a five-fold change in duration. The single low reading is the shortest
run, where the resident set has not reached its plateau, which is the condition `rf_occ.py`'s own comment
names. Every run in the sweep above used `L` from 600 to 7,500, all at or above that threshold, so no row in
that table is a short-run artifact.

So the two lanes' results are independent rather than redundant, and they agree where they overlap: the
threadgroup-memory handle bounds residency at 94 registers to at least eight simdgroups per core, and this
counter reads 32.0 at 98 registers. A lower bound of eight and a direct count of thirty-two are consistent,
and the counter is what supplies the number at 126 registers that the bound cannot reach.

**Spilling does not behave monotonically and is left as an observation.** The four spilled rows at 126
registers read 32.9, 39.5, 25.2 and 34.6 simdgroups per core at 64, 80, 144 and 257 stack stores. Spilling
frees registers and should raise occupancy; two rows do and two do not. Those kernels differ in more than
spill count and the sweep was not designed to separate them, so this is recorded as unexplained rather than
fitted.

### 169.1 Sections 168 and 169 re-censused per chain family: the C5 rule is untouched, and the reader has a resolution limit at eight short chains

Sections 168 and 169 reported "the direction of an object", which the evidence lane identified as an
abbreviation: an object carries a long chain family and, for some shapes, a short one, and the single-verdict
reader keeps only chains with four or more slot leaves, so it reports the long family and silently drops the
short. Their `families()` in `tools/g17reduxdir.py` splits the two. Re-censusing the four shapes those
sections rest on, baseline and read-all:

| object | M | long chains | short chains | long | short | spills |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `rs_16x32` | 16 | 2 | **8** | descending | *unresolved* | 20 |
| `rs_16x48` | 16 | 2 | **12** | descending | **ascending** | 28 |
| `rs_24x32` | 24 | 3 | 0 | descending | -- | 24 |
| `rs_32x32` | 32 | 12 | 0 | descending | -- | 24 |
| `ra_16x32` | 16 | 2 | **8** | descending | *unresolved* | 20 |
| `ra_16x48` | 16 | 2 | **12** | descending | **ascending** | 28 |
| `ra_24x32` | 24 | 3 | 0 | **ascending** | -- | **88** |
| `ra_32x32` | 32 | 12 | 0 | descending | -- | 24 |

**Their presence rule reproduces exactly.** Short chains appear precisely where `M` is 16 -- eight at
`16x32`, twelve at `16x48` -- and are absent at `M` of 24 and 32. Four of four, on objects built for a
different question than the 93 they measured it on.

**C5's rule is untouched, and the reason is structural rather than lucky.** The object that flips is
`ra_24x32`, and it carries **zero** short chains. So the flip is unambiguously in the long family, and the
discriminating pair that carries section 169 -- `16x48` not flipping against `24x32` flipping, at the same
per-lane capacity 24 -- cannot be an artifact of two families being pooled. The conflict the evidence lane
flagged at `24x16` is likewise between long chains, for the same reason.

**A limit in the reader, found by using it.** At `16x32` the short family is *present* (eight chains) and its
direction comes back **unresolved**; at `16x48` twelve chains resolve to ascending. So a null short direction
means "not readable at this chain count", not "no short family" -- and the two are worth distinguishing,
because a reader that returned null for both would let an unread family pass as an absent one. The
sections' own `('desc','asc')` pairings at `M` of 16 are now named: they are the long and short families,
not an ambiguity in a single verdict.

**What this changes in 168 and 169.** Nothing in the capacity conclusion, because `24x32` and `32x32` carry
no short family at all and they are the cells the argument turns on. What it changes is the scope of the
`M = 16` rows: their reported direction is the long family's, with a short family underneath that runs the
other way where it can be read at all.
