# The chronicle: what measurement taught the compiler

What measurement established about this compiler and the machine, and what it retracted. The machine model
([g17-tensorops-machine-model.md](g17-tensorops-machine-model.md)) is the current account; this file keeps the
episodes that changed how the work is done. The co-execution chronicle that used to open this file (an abandoned
direction) is at the tag `archive/pre-coexec-removal-2026-09-29`.

## The accelerators are reachable, and the API is Metal 4's

`mpp::tensor_ops::matmul2d` (MetalPerformancePrimitives, the API MLX's NAX path
is built on) compiles at runtime on this machine without Xcode and runs a
20-line fp16 GEMM at 25 TFLOPS, exact - 3.4x the GPU's fp32 ALU ceiling and
86% of MLX's tuned kernel (results/accel-h17s.md). `simdgroup_matrix` is the
M1-M4 path and does not by itself prove accelerator use. MLX's convolutions do
not use the units, which is part of the MLX-vs-MPS conv gap and a place a
custom conv could reclaim it torch-free.

## The G17 backend, 2026-09-06: one bias at four levels, and a ban that caught nothing

An overnight session whose output was almost entirely corrections. Worth reading as a unit
because each defect was found only by fixing the one above it.

### The front of a sorted listing is not a sample

Six tools capped their work with `for n, line in enumerate(open(CORPUS)): if n >= lim: break`.
That is the first N programs of a **sorted** listing, and it silently decided which opcodes were
ever looked at. Measured both ways on the same maps, same cap, same tool:

| | instructions | byte-exact |
| --- | --- | --- |
| 400 programs, front of corpus | 9,507 | 96.0% |
| 400 programs, sampled across | 10,631 | 90.0% |
| all 6,594 programs | 184,349 | 89.5% |

Two things are in that table. The front 400 programs hold a thousand *fewer* instructions than
400 sampled ones, because the front of the corpus is small kernels - the same composition effect
that made the first held-out score read 26.4%. And the front sample did not predict the
population it was drawn from while the strided one does. Every capped figure this project had
quoted was measured on the easy end of its own corpus.

The same bias then turned up three levels further down, each exposed by the previous fix:

- **the form.** `form_bits` characterised a bit by flipping it in **one** instance, `raws[0]`.
  op586's b3.3 changes the opcode at 2 of 24 real instances and an operand's kind at 22; one
  base point saw nothing, and that bit was 221 of op586's 222 near misses. `g17diff` had spread
  twelve bases since it was written; the discipline had been dropped here.
- **the field.** The encode-direction gate certified each map on the first **12 instances** of
  its own form. op10372's operand 2 fails 113 of 120 sampled and 2 of the first 12 - a map that
  writes the wrong value 94% of the time, recorded as wrong twice.
- **the opcodes.** `--limit` selected the first N opcodes in four tools, including `g17ab`,
  which is the tool that *prices a change before it ships*.

**Rule.** When a truncation is found, do not fix it and move on. The same construct is almost
certainly present at the next level down: population, then form, then instance, then value.

### A ban that stopped nothing, shipped with a message saying it worked

`isa/g17-form-bits.json` was committed with 102 entries carrying `selects_opcode: []` - not one
bit in the file - under a commit message asserting the maps had been re-fitted under a measured
ban. The +9 instructions that commit reported came from the build cache growing by 2,633 kernels
underneath the fit while another session was compiling: a population change wearing a
mechanism's clothes.

The cause is neat. `form_bits()` fills that field only when handed a decoder, and the ordinary
fit called it **without** one, so the fitter read the ban at its start and destroyed it at its
end. The ban could only ever apply to the single fit following an explicit `--form-bits`, and to
no other, silently. Fixed, plus three search paths that ignored it: 90.1% -> 91.8%, op586's
blocked instructions 222 -> 40.

A regression case now asserts the file records selecting bits **at all** and finds ones the
classification does not already give - either alone is passable by an empty file. It failed on
its first run and located the erasure.

### An artefact that is not contradicted may simply not be there

The ISA peer found 12,386 mined rules on their side that change **0 of 473 builds** - validated
as *uncontradicted* by Apple's bytes while the builder ignored them entirely. Their sharpening
of the rule: state what an artefact DOES, not only that nothing disagrees with it.

Applied here by withdrawing the other two halves of the ban file and re-scoring: without
`unforced` and `constant`, 9,755 -> 9,662. 165 unforced bits and 4 constant corrections are
worth 93 instructions. Not decoration - and now a number rather than an assumption.

### Certified in both directions, and still wrong

Ten maps wrote back a different value than they read. They reproduce Apple's corpus perfectly
and would emit a wrong instruction the first time a compiler asked for a value Apple never wrote
in that slot. Demoting them cost **1,139** instructions on the whole corpus - not the 34 the
400-program sample showed, and both numbers are in the record because quoting only the cheap one
would have been the same error as everything above.

Two of the ten were not broken. A code table can spell one value several ways and the encoder
took the first match, which is sorted order: op10372's operand 2 has eight spellings of 12, six
read back correctly on all 49 instances, and the one it chose does so on 6. **Fix the encoder
first, then demote** - 8 maps instead of 10, and 8 more instructions.

### The endpoint's inherited-bit clause read 18.7% and is 0.6%

The census scored a bit Apple *never* varies identically to one that agrees three times in four.
The first is a constant of the form established by measurement - the same standing as a bit the
specification calls `forced`, and this project already prefers measurement to the specification
where they disagree.

| | |
| --- | --- |
| form constant | 64,291 (18.1%) - measured over its witnesses |
| INHERITED | 2,207 (0.6%) - the endpoint gap |

Of the remainder, **85.1% is invisible to the decoder** - only a dispatch can settle it. The
measurement saying, in its own terms, that the endpoint is blocked on execution.

### Five attributions, four wrong

Six execution results held out for hours (see `execution-validation.md`). The attributions:
the missing bank bit (op17757 already had it); the map's base (the certification gate failed on
the next run and Apple's decoder was unambiguous); "the harness authors from our maps"
(`g17as.field_encode` is called zero times during a dispatch); "the slot is missing its low bit"
(tested in memory, moved the register 290 -> 285, still wrong). The fifth held.

Every wrong one was caught by a **test** rather than by re-reading: the certification gate, an
in-memory trial, git refusing an empty commit for having no changes, and a control form failing
a check it should have passed. The one that held was fitted to 236 measurements rather than
reasoned from a plausible mechanism.

**Rule.** A mechanism that explains the observation is not evidence. A relation that predicts
hundreds of instances you did not use to build it is.
