# Example workflows

Four workflows, each built on the implementation and the retained evidence in this repository. The first two run
on any Mac with Xcode; the last two re-derive published numbers from receipts on any machine with Python, and
describe, but do not perform, the GPU reproduction.

| | Workflow | Reads evidence | Calculates on the CPU | Compiles | Dispatches GPU work |
|---|---|---|---|---|---|
| 1 | [Compile the reference's 17 x 19 x 16 tensor product](#1-compile-the-17-x-19-x-16-tensor-product) | | | **yes** | no |
| 2 | [A tensor rule against hardware evidence](#2-a-tensor-rule-against-hardware-evidence) | **yes** | **yes** (simulation) | | no |
| 3 | [The matched decode study](#3-the-matched-decode-study) | **yes** | | | no - reproduction does (below) |
| 4 | [MiniLM and Qwen below Metal](#4-minilm-and-qwen-below-metal) | **yes** | | | no - reproduction does (below) |

**Setup, once:** macOS with Xcode installed (`xcode-select -p` prints a path), then

```
python3 -m pip install -r requirements.txt
make native-tools
```

`make examples` runs all four. Nothing in `make examples` or `make test` dispatches GPU work.

---

## 1. Compile the 17 x 19 x 16 tensor product

**Purpose.** Follow the technical reference's running example (section 15.2) through the compiler: the IR, the
emitted G17 instructions, the executor contract (ABI) and the image the Metal path would load.

**Kind.** Compiles only. **Nothing is executed and the product is never computed.**

**Prerequisites.** macOS with Xcode. The compiler's release checks decode every emitted program with **Apple's G17
decoder** (`GPUCompiler.framework`, reached through `tools/agx3dis`, which `make native-tools` builds); without it
compilation refuses. No GPU is needed.

**Command.**

```
python3 examples/tensor_17x19x16.py /tmp/agxforge-tensor     # the directory must not exist
```

**Expected result.** The IR, the first instructions of the listing, and these facts, which the script compares with
the reference and exits non-zero on any difference:

```
"bytes": 1232, "instructions": 101,
"sha256": "8bb67a3c65919cb7fa9a6f2e70d152cb2d8086fba0cade7437f18d1d8114c667",
"abi_version": 5, "register_count": 76
matches the technical reference, section 15.2 (compiled, not executed)
```

The output directory holds `ir.txt`, `program.bin`, `instructions.txt` (Apple's decoder's framing, with each opcode's
name from `isa/g17-opcode-glossary.json`), `abi.json` (three bindings, A and B half and read-only, C float and
written; exact launch geometry required) and `image.object`, `image.library`, `image.archive`.

**Limitations.** The decoder dependency makes compilation macOS-only. The reference program's bytes did run below
Metal (reference chapter 15); this workflow does not run them.

---

## 2. A tensor rule against hardware evidence

**Purpose.** One coordinate rule of the tensor unit, a small calculation, and the rival reading that a retained GPU
run rejected. The rule: when an MMA's accumulator is fed straight into the next MMA as its B operand, lane l, slot j
is read as canonical B row `k = 4*(l>>4) + ((l>>1)&3) + 8*(j>>2)`, but that register holds canonical D row
`rotl1(k)`. Whether the consumer sees the logical operand depends on how the producer labelled its rows.

**Kind.** Part 1 is arithmetic. Part 2 is a **simulation**: the package's MMA arithmetic model, run on the CPU, against
an output the GPU recorded (MM 25.103). Nothing is compiled or dispatched.

**Prerequisites.** Python with `requirements.txt`. The retained run is included at
`results/g17-tensor-feedmodes-v1/` (extracted from the research checkout's evidence archive and listed in
`RELEASE-MANIFEST.json` with its hash).

**Command.**

```
python3 examples/tensor_feed_rule.py
```

**Expected result.**

```
slot 4: B row 8 reads a register that holds canonical D row rotl1(8) = 1; this compiler stored application row 8 there
  logical operand (this compiler's packing)        differs from the GPU in    0 of 1024 elements, max |error| 0
  row-rotated operand (section 132's relabelling)  differs from the GPU in 1024 of 1024 elements, max |error| 289.8
dispatch receipt: feed_B_half (rotated reference) FAILED; neg_B_identity (same program, logical reference) passed
verdict: the hardware output is the logical product bit for bit; the rotated reading is rejected
```

**Limitations.** One square 32 x 32 stage, one SIMD group, a half-narrowed feed; a layout result with no timing.
It is unrelated to the cooperative-tensor chaining hazard in Apple's library (a separate finding, MM 13), and to
saturation (MM 25.42, 25.89).

---

## 3. The matched decode study

**Purpose.** Inspect the study behind the decode comparison (MM 25.211): would AGXForge's decode design run as fast
if Apple's compiler built the kernels? Three implementations, **all through Metal**, decode mlx-lm's own 128-token
greedy sequence after a 1,792-token InternLM2.5-1.8B-chat prompt.

**Kind (the command below).** Reads evidence only: it re-derives the medians and per-repetition ratios from the
receipt's raw rows, and the per-kernel identity checks, and names every source. No GPU.

**Command.**

```
python3 examples/decode_study.py
```

**Expected result.**

```
mlx-lm generate_step                                    162.2              1.00-1.00
this project's kernels, this project's compiler         184.5              1.13-1.15
the same kernels, Apple's compiler                      213.4              1.31-1.33
...
all published numbers re-derived from the per-repetition rows
```

**What to inspect.** The graph configuration `tools/models/internlm2_q4_spec_code.json`; the kernel builders
`tools/g17qmv.py`, `g17decodeops.py`, `g17attn.py`, `g17gen.py` (assembled into bundles by `tools/g17deliver.py`);
graph assembly `tools/g17q4graph.py` and `tools/g17modelbuild.py`; the Metal twins `tools/twins/*.metal`; the
harness `tools/g17twin.py`, `tools/g17twinrun.m`, `tools/g17decodegen.m`; and the receipts in
`evidence/g17-matched-study-v1/`.

**Status of reproduction.** The recorded comparison can be inspected (above); **exact reproduction of its historical
inputs is incomplete.** The shared 128-token history is retained, but the 1,792-token prompt's ids are not, and the
checkpoint's identity is likely but not recorded
([inputs-recovery.json](../evidence/g17-matched-study-v1/inputs-recovery.json) lists what was searched and found). What
follows is a complete route for a **new** comparison of the same three arms, with every input identified by a file.
It dispatches GPU work and has not been run for this release.

**Prerequisites.** A G17 GPU; `requirements-reproduce.txt`; the InternLM2.5-1.8B-chat checkpoint at
`~/models/internlm2_5-1_8b-chat` and its `mlx_lm.convert -q` 4-bit copy at `~/models/internlm2_5-1_8b-chat-mlx-q4`
(the paths the tools read; the recovery file gives the hashes of the copies the study most likely used). Tokenize with
`tools/g17promptids.py`, not `transformers.AutoTokenizer`: under transformers 5.16 the latter splits this
checkpoint's words into single characters.

```
# 1. the prompt, as an explicit ids file and its provenance (IDS.provenance.json: text hash, template, tokenizer hashes) (CPU)
python3 tools/g17promptids.py TEXT.txt --instruction "Summarize the following section." --tokens 1792 --out IDS.json
# 2. a model config that carries that prompt: tools/models/internlm2_q4_spec_code.json with "prompt_ids" replaced by
#    IDS.json's prompt_ids and a new "name" (the graph builder uses a config's explicit prompt_ids)
python3 -c "import json; c=json.load(open('tools/models/internlm2_q4_spec_code.json')); c['name']='decode_1792'; \
  c['prompt_ids']=json.load(open('IDS.json')); json.dump(c, open('CONFIG.json','w'), indent=1)"
# 3. kernels and graph (GPU: each bundle is verified once on hardware; --check 0 skips the CPU simulator's long prefill)
python3 tools/g17deliver.py build --bits 4 --cap 2048 --out DELIVER
python3 tools/g17modelbuild.py --config CONFIG.json --deliver DELIVER --check 0
G=results/g17-model-internlm2/graph_q4_decode_1792                     # where modelbuild writes this graph
# 4. the Apple-compiled twins (xcrun metal), then a per-kernel bit comparison and timing (GPU)
python3 tools/g17twin.py graph --graph $G/graph.json --kinds all --tag all --out $G
python3 tools/g17twin.py kernels --graph $G/graph.json --out KERNELS
# 5. mlx-lm's own greedy 128 tokens on this prompt: one mlx arm, then take its out_ids (GPU, through MLX)
python3 tools/g17twin.py e2e --ctx 1792=$G=IDS.json --mlx-model ~/models/internlm2_5-1_8b-chat-mlx-q4 --arms mlx --reps 1 --out MLX1
python3 -c "import json; r=json.load(open('MLX1/e2e_rows.json'))[0]; json.dump(r['out_ids'], open('SHARED.json','w'))"
# 6. the shared-history graphs: those tokens written into the token log after the prompt (CPU)
python3 tools/g17forcehistory.py $G/graph.json SHARED.json --name forced
python3 tools/g17forcehistory.py $G/graph_twin_all.json SHARED.json --name twin_all_forced
# 7. the three arms, alternating, four repetitions (GPU)
python3 tools/g17twin.py e2e --ctx 1792=$G=IDS.json --mlx-model ~/models/internlm2_5-1_8b-chat-mlx-q4 \
        --arms graph_forced,graph_twin_all_forced,mlx --reps 4 --out E2E
```

`tools/g17twin.py e2e` passes the ids file to mlx-lm and the graph directory to `tools/g17decodegen`; the forced arms
are the graph files step 6 wrote. Compare `E2E/e2e_rows.json` with the recorded receipt's structure; the numbers will
differ from the recorded run's because the prompt, and possibly the checkpoint, differ.

**Limitations.** The recorded study is one model at one context, four repetitions at load 2.1-2.9. The route above has
not been run end to end: steps 1 and 6 are tested on the CPU, the rest are the study's own instruments. The result
shows that the implementation's advantage survives Apple compilation; it shows no benefit from native instruction
control.

---

## 4. MiniLM and Qwen below Metal

**Purpose.** Inspect bounded execution of two complete pretrained models with no Metal in the process (MM 25.210):
the pinned model inputs, the expected outputs, the numerical bounds and their independent checks, the controls that
fail, the measured platform, and the Apple components that remain.

**Kind (the command below).** Reads evidence only. No GPU.

**Command.**

```
python3 examples/native_inference.py
```

**Expected result.**

```
  'Write a short greeting.'        -> 'Hello! How can I assist you today?'
  independent check (no GPU): 37 of 37 logit vectors within 0.05 + 0.003*|reference| of the original checkpoint in FP64; ...
    0.6147  'A puppy runs across a green field.'
    0.1706  'The spacecraft entered orbit around Mars.'
MEASURED PLATFORM: macOS 26.6.2 (build 25G83), arm64; one M5 Pro (H17s)
the published outputs and checks are re-derived from the receipts
```

**Reproducing it (dispatches GPU work; not re-run for this release).** Needs a G17 GPU, `requirements-reproduce.txt`
and about 3 GB of disk; the checkpoints are fetched at the revisions and SHA-256s pinned in
`evidence/g17-inference-models-v1/`. **Only macOS 26.6.2 (25G83) has been measured**: the runtime writes launch
layouts recorded on that build, and running it on another OS build is untested. Every native run takes the
machine-wide GPU lock and refuses to proceed if the GPU's recovery or event counters change (MM 25.210.9); read
[docs/execution-validation.md](../docs/execution-validation.md) first. From MM 25.210.1:

```
mkdir -p results/g17-inference-demo
python3 tools/g17checkpoint.py evidence/g17-inference-models-v1/qwen results/g17-inference-demo/model.safetensors --fetch --receipt results/g17-inference-demo/checkpoint.json
python3 tools/g17tokenizerfetch.py evidence/g17-inference-models-v1/qwen results/g17-inference-demo/tokenizer
python3 tools/g17decoderlower.py evidence/g17-inference-models-v1/qwen results/g17-inference-demo/lowering --tokens 32
python3 tools/g17inferenceprepare.py evidence/g17-inference-models-v1/qwen results/g17-inference-demo/model.safetensors results/g17-inference-demo/prepared
python3 tools/g17inferencepreparedcheck.py evidence/g17-inference-models-v1/qwen results/g17-inference-demo/model.safetensors results/g17-inference-demo/prepared results/g17-inference-demo/lowering/lowering.json results/g17-inference-demo/preparation-check.json
python3 tools/g17inferencebundle.py results/g17-inference-demo/prepared results/g17-inference-demo/bundle --model qwen --optimization reuse_projection_packs_v1
# dispatches GPU work below Metal:
VECLIB_MAXIMUM_THREADS=1 python3 tools/g17nativegenerate.py results/g17-inference-demo/bundle results/g17-inference-demo/prepared results/g17-inference-demo/tokenizer --prompt 'Explain why the sky is blue.' --prompt 'Write a short greeting.' --prompt 'Explain why the sky is blue.' --max-new-tokens 12 --capture-logits results/g17-inference-demo/logits --receipt results/g17-inference-demo/generation.json
# offline, no GPU: every captured logit vector against the original checkpoint's eager FP64 model
VECLIB_MAXIMUM_THREADS=1 python3 tools/g17generationcheck.py results/g17-inference-demo/generation.json results/g17-inference-demo/model.safetensors results/g17-inference-demo/generation-independent.json
```

The MiniLM sequence is the same with `minilm`, `tools/g17inferencelower.py --policy half_transport_fp32_accumulate_v1`,
`tools/g17nativeretrieval.py` and `tools/g17retrievalcheck.py`; its full commands are in [MM 25.210.1](../docs/g17-tensorops-machine-model.md#252101-reproduce).

**Limitations.** Matrix operands travel in half precision with FP32 accumulation; an auxiliary half-policy control
and the original-FP32 block references fail and are kept as failures. Native execution is slower than matched Metal
and than Transformers on MPS. Two model graphs, not arbitrary models. Apple's IOGPU framework, kernel driver and
firmware remain in the path, and compilation uses Apple's decoder.
