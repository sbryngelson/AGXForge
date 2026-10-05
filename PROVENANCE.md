# Provenance of the released files

Every file in this release, by what it is and where it came from. Each file's category is also in RELEASE-MANIFEST.json (`provenance`).

This project's own code and writing are licensed under the MIT License ([LICENSE](LICENSE)). Third-party material retains its applicable terms ([THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)). The categories below record where each file came from; files produced by Apple's tools or derived from Apple's decoder metadata are listed individually.

| Category | Files | Bytes | What it is |
|---|---|---|---|
| apple-compiled program object | 65 | 190,904 | a program object produced by Apple's Metal compiler from a Metal source this project wrote (a witness the compiler's tensor lowering reads its templates from) |
| decoder listings of this project's programs | 65 | 1,681,513 | instruction listings Apple's G17 decoder produced from the witness objects above (programs compiled from this project's Metal sources); the listing's content is the decoded program |
| Apple decoder metadata dumps | 3 | 2,295,693 | dumps of Apple's own decoder metadata - the AGX3 instruction descriptors, register classes and register names inside GPUCompiler.framework's libLLVM - written by tools/agx3meta.c, which reads them from Apple's library at run time; the compiler's decoder interface reads them for operand counts and register names |
| derived from Apple's decoder metadata | 1 | 155,210 | a translation table this project generated (tools/g17renumber.py --write): it maps the opcode and register numbering of the macOS 27 (26A434) decoder back to the numbering of the original build, by matching register and register-class NAMES between the two builds' metadata dumps, aligning the two instruction-descriptor tables as monotone sequences anchored on 3,244 opcodes decoded under both builds, and fitting the immediate tags the new decoder adds; it contains id pairs and fitted constants, not Apple's tables |
| third-party model metadata | 7 | 45,551 | upstream configuration files and checkpoint headers copied or extracted from public Hugging Face repositories (Apache-2.0); see THIRD-PARTY-NOTICES.md; no weights |
| this project's compiled programs | 49 | 77,396 | programs and images this project's compiler and image author produced (in Apple's container formats) |
| recovered tables | 314 | 72,753,975 | this project's tables of the instruction set and machine, built by its own measurement and analysis (much of it from decoding programs with Apple's decoder and comparing with Apple's compiler) |
| measurement receipts | 186 | 9,371,788 | records of this project's measurements and checks; some carry model outputs, Apple tool version strings, driver log lines or decoder text as recorded |
| release-authored | 13 | 59,757 | written for this release |
| project source and writing | 704 | 16,124,645 | this project's code, tests, model configurations, Metal sources, documents and figures (the documents quote short passages of Apple SDK headers where a claim rests on them) |

## apple-compiled program object

- `results/g17-tensor-common-witness-v1/tensor-b-stride.o`
- `results/g17-tensor-common-witness-v1/tensor-common.o`
- `results/g17-tensor-common-witness-v1/tensor-renumbered.o`
- `results/g17-tensor-stride-prediction-v1/a-rows-10.o`
- `results/g17-tensor-stride-prediction-v1/a-rows-11.o`
- `results/g17-tensor-stride-prediction-v1/a-rows-12.o`
- `results/g17-tensor-stride-prediction-v1/a-rows-13.o`
- `results/g17-tensor-stride-prediction-v1/a-rows-14.o`
- `results/g17-tensor-stride-prediction-v1/a-rows-15.o`
- `results/g17-tensor-stride-prediction-v1/a-rows-17.o`
- `results/g17-tensor-stride-prediction-v1/a-rows-18.o`
- `results/g17-tensor-stride-prediction-v1/a-rows-20.o`
- `results/g17-tensor-stride-prediction-v1/a-rows-24.o`
- `results/g17-tensor-stride-prediction-v1/a-rows-31.o`
- `results/g17-tensor-stride-prediction-v1/a-rows-32.o`
- `results/g17-tensor-stride-prediction-v1/a-rows-33.o`
- `results/g17-tensor-stride-prediction-v1/a-rows-63.o`
- `results/g17-tensor-stride-prediction-v1/a-rows-64.o`
- `results/g17-tensor-stride-prediction-v1/a-rows-7.o`
- `results/g17-tensor-stride-prediction-v1/a-rows-9.o`
- `results/g17-tensor-stride-prediction-v1/a16-b9-c13.o`
- `results/g17-tensor-stride-prediction-v1/a5-c6.o`
- `results/g17-tensor-stride-prediction-v1/a8-b4.o`
- `results/g17-tensor-stride-prediction-v1/a9-c10.o`
- `results/g17-tensor-stride-prediction-v1/b-rows-10.o`
- `results/g17-tensor-stride-prediction-v1/b-rows-11.o`
- `results/g17-tensor-stride-prediction-v1/b-rows-12.o`
- `results/g17-tensor-stride-prediction-v1/b-rows-13.o`
- `results/g17-tensor-stride-prediction-v1/b-rows-14.o`
- `results/g17-tensor-stride-prediction-v1/b-rows-15.o`
- `results/g17-tensor-stride-prediction-v1/b-rows-16.o`
- `results/g17-tensor-stride-prediction-v1/b-rows-4.o`
- `results/g17-tensor-stride-prediction-v1/b-rows-5.o`
- `results/g17-tensor-stride-prediction-v1/b-rows-6.o`
- `results/g17-tensor-stride-prediction-v1/b-rows-7.o`
- `results/g17-tensor-stride-prediction-v1/b-rows-8.o`
- `results/g17-tensor-stride-prediction-v1/b-rows-9.o`
- `results/g17-tensor-stride-prediction-v1/b3-c7.o`
- `results/g17-tensor-stride-prediction-v1/b5-c11.o`
- `results/g17-tensor-stride-prediction-v1/c-rows-10.o`
- `results/g17-tensor-stride-prediction-v1/c-rows-11.o`
- `results/g17-tensor-stride-prediction-v1/c-rows-12.o`
- `results/g17-tensor-stride-prediction-v1/c-rows-13.o`
- `results/g17-tensor-stride-prediction-v1/c-rows-14.o`
- `results/g17-tensor-stride-prediction-v1/c-rows-15.o`
- `results/g17-tensor-stride-prediction-v1/c-rows-16.o`
- `results/g17-tensor-stride-prediction-v1/c-rows-4.o`
- `results/g17-tensor-stride-prediction-v1/c-rows-5.o`
- `results/g17-tensor-stride-prediction-v1/c-rows-6.o`
- `results/g17-tensor-stride-prediction-v1/c-rows-7.o`
- `results/g17-tensor-stride-prediction-v1/c-rows-8.o`
- `results/g17-tensor-stride-prediction-v1/c-rows-9.o`
- `results/g17-tensor-stride-prediction-v1/k128-a16.o`
- `results/g17-tensor-stride-prediction-v1/k128-b3.o`
- `results/g17-tensor-stride-prediction-v1/k160.o`
- `results/g17-tensor-stride-prediction-v1/k192-b4.o`
- `results/g17-tensor-stride-prediction-v1/k224.o`
- `results/g17-tensor-stride-prediction-v1/k96-c3.o`
- `results/g17-tensor-variant-witness-v1/tensor-a-stride-128.o`
- `results/g17-tensor-variant-witness-v1/tensor-a-stride-256.o`
- `results/g17-tensor-variant-witness-v1/tensor-a-stride-80.o`
- `results/g17-tensor-variant-witness-v1/tensor-a-stride-96.o`
- `results/g17-tensor-variant-witness-v1/tensor-c-stride-48.o`
- `results/g17-tensor-variant-witness-v1/tensor-leading-buf.o`
- `results/g17-tensor-variant-witness-v1/tensor-shape-16.o`

## decoder listings of this project's programs

- `results/g17-tensor-common-witness-v1/tensor-b-stride.instructions.json`
- `results/g17-tensor-common-witness-v1/tensor-common.instructions.json`
- `results/g17-tensor-common-witness-v1/tensor-renumbered.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a-rows-10.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a-rows-11.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a-rows-12.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a-rows-13.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a-rows-14.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a-rows-15.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a-rows-17.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a-rows-18.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a-rows-20.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a-rows-24.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a-rows-31.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a-rows-32.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a-rows-33.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a-rows-63.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a-rows-64.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a-rows-7.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a-rows-9.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a16-b9-c13.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a5-c6.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a8-b4.instructions.json`
- `results/g17-tensor-stride-prediction-v1/a9-c10.instructions.json`
- `results/g17-tensor-stride-prediction-v1/b-rows-10.instructions.json`
- `results/g17-tensor-stride-prediction-v1/b-rows-11.instructions.json`
- `results/g17-tensor-stride-prediction-v1/b-rows-12.instructions.json`
- `results/g17-tensor-stride-prediction-v1/b-rows-13.instructions.json`
- `results/g17-tensor-stride-prediction-v1/b-rows-14.instructions.json`
- `results/g17-tensor-stride-prediction-v1/b-rows-15.instructions.json`
- `results/g17-tensor-stride-prediction-v1/b-rows-16.instructions.json`
- `results/g17-tensor-stride-prediction-v1/b-rows-4.instructions.json`
- `results/g17-tensor-stride-prediction-v1/b-rows-5.instructions.json`
- `results/g17-tensor-stride-prediction-v1/b-rows-6.instructions.json`
- `results/g17-tensor-stride-prediction-v1/b-rows-7.instructions.json`
- `results/g17-tensor-stride-prediction-v1/b-rows-8.instructions.json`
- `results/g17-tensor-stride-prediction-v1/b-rows-9.instructions.json`
- `results/g17-tensor-stride-prediction-v1/b3-c7.instructions.json`
- `results/g17-tensor-stride-prediction-v1/b5-c11.instructions.json`
- `results/g17-tensor-stride-prediction-v1/c-rows-10.instructions.json`
- `results/g17-tensor-stride-prediction-v1/c-rows-11.instructions.json`
- `results/g17-tensor-stride-prediction-v1/c-rows-12.instructions.json`
- `results/g17-tensor-stride-prediction-v1/c-rows-13.instructions.json`
- `results/g17-tensor-stride-prediction-v1/c-rows-14.instructions.json`
- `results/g17-tensor-stride-prediction-v1/c-rows-15.instructions.json`
- `results/g17-tensor-stride-prediction-v1/c-rows-16.instructions.json`
- `results/g17-tensor-stride-prediction-v1/c-rows-4.instructions.json`
- `results/g17-tensor-stride-prediction-v1/c-rows-5.instructions.json`
- `results/g17-tensor-stride-prediction-v1/c-rows-6.instructions.json`
- `results/g17-tensor-stride-prediction-v1/c-rows-7.instructions.json`
- `results/g17-tensor-stride-prediction-v1/c-rows-8.instructions.json`
- `results/g17-tensor-stride-prediction-v1/c-rows-9.instructions.json`
- `results/g17-tensor-stride-prediction-v1/k128-a16.instructions.json`
- `results/g17-tensor-stride-prediction-v1/k128-b3.instructions.json`
- `results/g17-tensor-stride-prediction-v1/k160.instructions.json`
- `results/g17-tensor-stride-prediction-v1/k192-b4.instructions.json`
- `results/g17-tensor-stride-prediction-v1/k224.instructions.json`
- `results/g17-tensor-stride-prediction-v1/k96-c3.instructions.json`
- `results/g17-tensor-variant-witness-v1/tensor-a-stride-128.instructions.json`
- `results/g17-tensor-variant-witness-v1/tensor-a-stride-256.instructions.json`
- `results/g17-tensor-variant-witness-v1/tensor-a-stride-80.instructions.json`
- `results/g17-tensor-variant-witness-v1/tensor-a-stride-96.instructions.json`
- `results/g17-tensor-variant-witness-v1/tensor-c-stride-48.instructions.json`
- `results/g17-tensor-variant-witness-v1/tensor-leading-buf.instructions.json`
- `results/g17-tensor-variant-witness-v1/tensor-shape-16.instructions.json`

## Apple decoder metadata dumps

- `isa/g17-agx3meta-classes.txt`
- `isa/g17-agx3meta-instrs.txt`
- `isa/g17-agx3meta-regs.txt`

## derived from Apple's decoder metadata

- `tools/agx3renumber.h`

## third-party model metadata

- `evidence/g17-inference-models-v1/minilm/config.json`
- `evidence/g17-inference-models-v1/minilm/modules.json`
- `evidence/g17-inference-models-v1/minilm/pooling.json`
- `evidence/g17-inference-models-v1/minilm/safetensors-header.json`
- `evidence/g17-inference-models-v1/minilm/sentence-config.json`
- `evidence/g17-inference-models-v1/qwen/config.json`
- `evidence/g17-inference-models-v1/qwen/safetensors-header.json`

## this project's compiled programs

- `results/g17-tensor-feedmodes-v1/feed_B_half/program.bin`
- `results/g17-tensor-feedmodes-v1/feed_B_half/scan.arc.metallib`
- `results/g17-tensor-feedmodes-v1/feed_B_half/scan.lib.metallib`
- `results/g17-tensor-feedmodes-v1/feed_B_half/scan.o`
- `results/g17-tensor-feedmodes-v1/neg_B_identity/program.bin`
- `results/g17-tensor-feedmodes-v1/neg_B_identity/scan.arc.metallib`
- `results/g17-tensor-feedmodes-v1/neg_B_identity/scan.lib.metallib`
- `results/g17-tensor-feedmodes-v1/neg_B_identity/scan.o`
- `results/g17-tensor-stride-prediction-v1/a-rows-10.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/a-rows-11.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/a-rows-13.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/a-rows-14.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/a-rows-15.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/a-rows-17.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/a-rows-18.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/a-rows-20.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/a-rows-24.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/a-rows-31.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/a-rows-32.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/a-rows-33.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/a-rows-63.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/a-rows-9.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/a16-b9-c13.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/a8-b4.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/a9-c10.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/b-rows-11.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/b-rows-13.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/b-rows-14.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/b-rows-15.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/b-rows-16.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/b-rows-4.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/b-rows-5.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/b-rows-8.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/b-rows-9.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/b5-c11.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/c-rows-10.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/c-rows-11.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/c-rows-13.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/c-rows-14.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/c-rows-15.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/c-rows-16.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/c-rows-4.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/c-rows-5.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/c-rows-8.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/c-rows-9.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/k128-a16.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/k128-b3.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/k160.predicted.bin`
- `results/g17-tensor-stride-prediction-v1/k224.predicted.bin`

## Not included

No Apple framework, SDK file or binary is redistributed: the decoder is loaded from the reader's own macOS at run time, and the native tools are built from source on the reader's machine. No model weights or tokenizer files are included; the workflows fetch them at pinned revisions and hashes.
