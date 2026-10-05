# AGXForge: a native compiler and runtime for the Apple M5 GPU

AGXForge compiles native code for Apple's M5 GPU (G17) and the tensor units inside each of its cores, and runs that code
two ways: through Metal, and below Metal with its own runtime. The compiler has its own instruction selection,
register allocation, encoder and executable image author. It rests on a model of the machine that was recovered and
then checked: Apple's compiler and decoder served as oracles, and GPU execution and independent numerical references
tested the recovered rules. All measurements are from **one M5 Pro (H17s, 48 GB)**. The GPU's tensor
units are distinct from the Apple Neural Engine.

![The two execution paths. AGXForge's compiler takes model kernels built in Python, or Metal source via Apple's front end
and AIR, and produces native G17 machine code. Through Metal, Metal creates the pipelines and command buffers; this path
carries every performance result. Below Metal, AGXForge's runtime writes the resources, launch state and Submit records
for MiniLM and Qwen with no Metal in the process. Both paths then go through Apple's IOGPU framework, kernel driver and
firmware to the GPU.](figures/showcase-paths.svg)

Three demonstrations, each readable on its own:

| | What it shows | Where to look first |
|---|---|---|
| [1. Complete models below Metal](#1-complete-models-below-metal) | Pretrained retrieval and generation through AGXForge's compiler and runtime, outputs checked independently | the recorded outputs |
| [2. A decode design that survives a compiler swap](#2-a-decode-design-that-survives-a-compiler-swap) | Faster than mlx-lm on one controlled workload, and Apple's compiler makes the same design faster still | the three-bar chart |
| [3. What the tensor unit actually does, measured](#3-what-the-tensor-unit-actually-does-measured) | Saturating int8 accumulation that AGXForge's compiler emits, a silent hazard in Apple's library chaining, and the register layout AGXForge uses instead | the chaining-hazard table |

## 1. Complete models below Metal

**Two pretrained applications run end to end with no Metal in the process.** MiniLM-L6 retrieves sentences;
Qwen2.5-0.5B-Instruct generates text. Asked to "Write a short greeting.", Qwen answers "Hello! How can I assist you
today?" ([receipt](../evidence/g17-native-qwen-guarded-generation.json)). For the query "A dog is running through the
grass.", MiniLM ranks "A puppy runs across a green field." (cosine 0.615) above "The spacecraft entered orbit around
Mars." (0.171) ([receipt](../evidence/g17-native-encoder-guarded-retrieval.json)).

**What AGXForge supplies and what Apple still does.** AGXForge writes the native code, the resources, the launch state and
the Submit records. Apple's IOGPU framework, kernel driver and firmware remain in the path, and the compiler's
validation uses Apple's decoder on macOS (no GPU execution calls it). The process creates no Metal device or command
buffer and checks that AGXMetal is absent. Activations stay on the GPU; the CPU tokenizes, picks the greedy token and
ranks by cosine.

**What establishes correctness.** The final vectors are compared with the original checkpoints evaluated
independently in FP64, against bounds fixed before the first dispatch: all 37 Qwen logit vectors are within
`0.05 + 0.003 * abs(reference)` and agree on the top token, with the native tokens teacher-forced; MiniLM's embedding error is about 1.29e-4, and 680/680 stage
checks pass (MM [25.210](g17-tensorops-machine-model.md#25210-pretrained-models-below-metal-minilm-l6-retrieval-and-qwen25-05b-generation-on-the-resident-agx-runner-every-application-vector-independently-checked-slower-than-matched-metal-and-mps-2026-09-30), [25.210.2](g17-tensorops-machine-model.md#252102-numerical-scope-and-the-controls-that-fail)).

**Scope.** Matrix operands pass through half precision with FP32 accumulation; an auxiliary Qwen half-precision
control fails, and that failure is part of the result. **Native execution is slower**: slower than matched Metal
blocks and about 7-15x slower than Transformers on MPS, so bypassing Metal shows control, not speed. The runtime
handles these two models' graph layouts, not arbitrary models, and has not been tested on another OS build.

**Next:** how submission works, [technical reference](g17-technical-reference.pdf) chapter 13 and section 17.2 ·
the checks and timings, MM [25.210.2](g17-tensorops-machine-model.md#252102-numerical-scope-and-the-controls-that-fail) and [25.210.5](g17-tensorops-machine-model.md#252105-performance-record) ·
reproduce it, MM [25.210.1](g17-tensorops-machine-model.md#252101-reproduce).

## 2. A decode design that survives a compiler swap

**The comparison holds the token history fixed and changes the implementation.** InternLM2.5-1.8B-chat with 4-bit
weights and 1,792 tokens of context. Three implementations, all through Metal, decode mlx-lm's own 128-token greedy
sequence: mlx-lm's `generate_step`; AGXForge's kernels compiled by AGXForge; and the same kernels compiled by Apple's compiler.

![Median decode throughput: mlx-lm 162.2 tokens per second; AGXForge's kernels with AGXForge's compiler 184.5 (1.14x);
the same kernels with Apple's compiler 213.4 (1.32x). Four repetitions each, tightly clustered.](figures/showcase-decode.svg)

**The design carries the advantage.** Parallel attention reduction, fused projections and residuals, seven dispatches
per layer against mlx-lm's sixteen, and a static graph with one command buffer per token: these survive the compiler
swap. Apple's compiler keeps the same threads, work partition and FP32 operation order, produces bit-identical
outputs, and raises throughput another 15-16 percent. So the lead over mlx-lm comes from the algorithms and structure,
not from instruction-level control; what Apple's compiler does better (instructions, registers or scheduling) is still
open. Someone building inference on this GPU can take the design without taking the compiler.

**Scope.** One decode graph at one context, four alternating repetitions. The shared history makes the arms
comparable; it does not claim free generation is identical to mlx-lm's (AGXForge's argmax differs at three of 128
positions, all ties or one-bf16-step ranks). At a 196-token context AGXForge is around parity with mlx-lm (0.91-1.02x, in the same study).

**Next:** the comparison and its controls, MM [25.211](g17-tensorops-machine-model.md#25211-the-matched-study-apple-compiled-twins-of-our-decode-kernels-give-10-15-percent-higher-median-throughput-than-ours-with-identical-tokens-so-the-lead-over-mlx-comes-from-the-algorithms-and-structure-not-instruction-control-2026-09-30) ·
the [recorded runs](../evidence/g17-matched-study-v1/decode-shared-history.json) · what limits each workload,
[technical reference](g17-technical-reference.pdf) section 12.10.

## 3. What the tensor unit actually does, measured

Three separate findings about the tensor unit, each established by execution against a reference that a wrong
reading fails. They are independent: each has its own evidence, and none explains another.

**Saturating int8 accumulation, emitted by AGXForge's compiler.** The int8 MMA has a saturating form: with the
accumulator present, each 16-product issue computes `clip(C + A·B, -2^31, 2^31 - 1)` instead of wrapping. Apple's
compiler emits it, but in our search only from a hand-written AIR call (`air.simdgroup_matrix_16x16x16_widening_multiply_accumulate_saturate`).
The public matrix interfaces declare no saturating operation. We searched the MPP headers in the macOS 27.0 SDK and
the `metal_simdgroup_matrix`, `metal_tensor` and `metal_cooperative_tensor` headers of the Metal 32023 toolchain;
`matmul2d`'s modes are `multiply` and `multiply_accumulate`. AGXForge's encoder emits the saturating form directly.
On hardware, AGXForge's program was run on inputs near both int32 rails, where four readings differ (wrap, clip after
every product, clip after every issue, clip once at the end). The output matched clip-after-every-issue, and the same
program checked against a wrap reference fails on 1,024 of 1,024 elements
(MM [25.89](g17-tensorops-machine-model.md#2589-compiler-emitted-tensor-programs-on-hardware-set-a-2026-09-23-what-runs-three-defects-the-runs-caught-and-the-checks-now-guarding-them)). Setting the same bit in Apple's own compiled `matmul2d` objects shows it also clamps with the left-transpose bit set
(the one transpose arm dispatched), and changes nothing when the sum stays in range (MM [25.42](g17-tensorops-machine-model.md#2542-x2-resolved-on-hardware-the-saturating-clamp-does-apply-under-transpose)).

**A silent hazard in Apple's cooperative-tensor chaining.** MPP lets one `matmul2d` feed its result to a second as an
input cooperative tensor, and offers `is_compatible_as_left_input` / `is_compatible_as_right_input` to check the pair.
Measured with one simdgroup, tile sizes 16, 32 and 64, and tagged inputs that show which producer element each consumer
operand actually holds:

| Chain (one simdgroup) | Configurations | Expected | Observed | Compatibility check |
|---|---|---|---|---|
| D1 as left input: GEMM1 M x N1 x N1, GEMM2 M x N2 x N1 | 27 (M, N1, N2 in 16, 32, 64) | GEMM2 reads all of D1 | When N2 >= N1, it does. When N2 < N1 (9 cases), only the first (M/16)(N2/16) tiles in row-major tile order are D1; the rest read zero or another D1 element | true in all 27 |
| D1 as right input: GEMM2 M2 x N1 x M1 | 27 (M1, N1, M2 in 16, 32, 64) | same | When M2 >= M1, exact. When M2 < M1 (9 cases), only the first (M2/16)(N1/16) tiles are valid; the rest read zero | true in all 27 |

One rule predicts all 54: the consumer sees the first min(producer tiles, consumer tiles) of the producer. Nothing
fails at compile or run time; an earlier reference check found 0 of 8 whole matrices correct in a narrower chain. The
fix is to pad the narrower consumer up to the producer's tile count, or pass the result through memory. The hazard is
in the library's cooperative tensor, not the hardware: hand-written AIR chains that feed the accumulator registers
straight into the next MMA are exact for narrower consumers too (MM [13](g17-tensorops-machine-model.md#13-composition-register-resident-chaining-and-the-memory-bridge); the measurements are recon sections 127 and 132). These runs predate the
2026-10-02 OS update and have not been repeated on macOS 27.0.1, and the scripts that produced them were not kept.

**Producer packing: one register, two meanings.** This concerns AGXForge's own register chains, not the library. A
tensor MMA leaves its 16 x 16 result spread over 32 lanes, eight registers ("slots") each. The next MMA reads those
registers as its B operand with a different coordinate map: in lane 0, slot 4 holds D element (1, 0) and is read as B
element (8, 0).

![Lane 0's eight slots. The hardware's canonical D coordinates are (0,0) to (0,3) then (1,0) to (1,3); the next MMA reads
the same registers as B elements (0,0) to (0,3) then (8,0) to (8,3). Canonical packing puts D row 1 in slot 4, so B
row 8 receives D row 1 and must be corrected. AGXForge's packing puts D row 8 there, so it is used
directly.](figures/showcase-packing.svg)

The two maps differ by a one-bit rotation of the row index, `rotl1(8) = 1`. Apple-compiled `simdgroup_matrix` code
packs D canonically, so a register-fed B arrives row-rotated and the other operand must be permuted. AGXForge packs the
producer so that application row 8 of D lands where the consumer reads B row 8, and the chained product needs no
shuffle. Measured in one 32 x 32 stage: AGXForge's B and transposed feeds read the logical operand bit for bit, while a
reference that applied the rotation fails (MM [25.103](g17-tensorops-machine-model.md#25103-register-feed-modes-read-the-logical-operand-in-this-compilers-kernels-and-section-132s-relabelings-belong-to-the-kernel-not-the-mma)). This is a layout result; it has no timing comparison.

**What this is not.** It is not a speed claim. AGXForge also has an exact int8 GEMM, but Apple's public `matmul2d` is
exact at the same shapes and its tuned medians are lower than AGXForge's. Those medians come from different sessions and
OS builds, not a paired study (MM [25.212](g17-tensorops-machine-model.md#25212-apples-public-int8-gemm-mpp-matmul2d-is-exact-at-25161s-four-shapes-and-its-tuned-medians-396-1556-us-are-below-this-projects-historical-int8-medians-not-a-paired-study-2026-10-04)).

**Next:** the tensor unit's rules, [technical reference](g17-technical-reference.pdf) chapter 10: packing in 10.5,
chaining and the library hazard in 10.9, int8 arithmetic in 10.11 · saturation, MM [25.89](g17-tensorops-machine-model.md#2589-compiler-emitted-tensor-programs-on-hardware-set-a-2026-09-23-what-runs-three-defects-the-runs-caught-and-the-checks-now-guarding-them) and
[25.42](g17-tensorops-machine-model.md#2542-x2-resolved-on-hardware-the-saturating-clamp-does-apply-under-transpose) · the int8 GEMM comparison, MM [25.212](g17-tensorops-machine-model.md#25212-apples-public-int8-gemm-mpp-matmul2d-is-exact-at-25161s-four-shapes-and-its-tuned-medians-396-1556-us-are-below-this-projects-historical-int8-medians-not-a-paired-study-2026-10-04) and the [archived receipt](../evidence/g17-mpp-int8-v1/mpp-int8.json).

## How the system was checked

- **The executed code is AGXForge's.** Ten kernels' GPU code matches AGXForge's `program.bin` byte for byte, and changing one
  constant after pipeline creation changes the output as predicted (MM [25.147](g17-tensorops-machine-model.md#25147-what-metal-still-does-for-our-programs-the-code-is-ours-verbatim-the-pipeline-config-is-metals-the-validator-checks-only-the-format-2026-09-29)).
- **The compiler is general beyond its own kernels.** Of 197 real Metal sources, 125 compiled and 101 ran and were
  checked, against their own references or Apple's build of the same source; none crashed the driver
  (MM [25.149.1](g17-tensorops-machine-model.md#251491-the-general-serializer-completed-101-of-125-run-and-checked-none-crash-2026-09-29)).
- **A CPU emulator replays the GPU.** It runs the compiler's bytes and reproduces full decode positions dispatch by
  dispatch (316/316 batched, 172/172 single-stream); it uses Apple's decoder and is an internal macOS tool
  (MM [25.197](g17-tensorops-machine-model.md#25197-a-shipped-model-graph-on-the-cpu-dispatch-by-dispatch-against-the-gpu-a-race-check-the-emulator-faster-2026-09-30)).
- **Failed controls are kept.** Checks state their precision domain, and one checked input set per kernel does not
  establish correctness for every input.

## Where to go next

| To | Read |
|---|---|
| Understand the machine, the compiler and both execution paths | [the technical reference](g17-technical-reference.pdf) (PDF, under review) |
| Check a number, or see every measurement and its history | [the machine model](g17-tensorops-machine-model.md), cited above as MM by section |
| Run the four example workflows | [examples/README.md](../examples/README.md) |
| See what this release contains and how it was chosen | [RELEASE.md](../RELEASE.md) |
| Build it | [the repository README](../README.md) |

The figures are drawn from the evidence files by `tools/g17showcasefigs.py`.
