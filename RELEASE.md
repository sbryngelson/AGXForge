# About this release

This repository is a **curated research release**: a selection of the project's research checkout, exported by a
script, not a separately maintained compiler. Implementation paths are the research checkout's own (`agxforge/`,
`tools/`, `isa/`, `docs/`), so a path cited in the technical reference or the evidence record means the same file here.

[RELEASE-MANIFEST.json](RELEASE-MANIFEST.json) records the source revision, whether the source working tree had
uncommitted changes, and for every file its SHA-256, size, origin (`committed` at the source revision, `modified` or
`untracked` in the working tree, `archive` extracted from an evidence archive, `overlay` written for the release, or
`transformed`: a research file with a few lines rewritten to point at release material, its source hash kept)
and the reason it is included (`seed`, `import`, `trace`, `reference`, `package`).

## What is included, and how it was chosen

- **Documents:** the technical reference (PDF and LaTeX source), the showcase and its figures, the full evidence
  record, the recon record it cites, the dispatch rules and the corrections chronicle.
- **Implementation:** the `agxforge` package (IR, compiler, encoders, image author, decoder interface, runtime
  contracts), the Metal executors and the native runtime sources, the model pipeline for MiniLM and Qwen, the matched
  study's builders, twins and harness, the CPU emulator, and the native sources `make native-tools` builds.
- **Dependencies, traced rather than listed by hand:** every included Python file's imports, followed to closure; the
  data files the four workflows and the retained tests actually opened when they ran in the research checkout; and
  small data files the code names by path. This is why `tools/` and `isa/` hold more than the workflows name directly.
- **Evidence:** the receipts behind every public claim, including the failures and controls kept as failures (the
  evidence record's section 25.210.10 lists them for the native path), the matched decode study, the Apple int8
  baseline and the historical int8 measurement it qualifies, and the retained B-feed runs workflow 2 reads.
- **Tests:** a REQUIRED set that substantiates the public claims and workflows (the tensor feed, saturation and
  chaining findings, the executor contract, the arithmetic model workflow 2 simulates with, the native pipeline's
  contracts, the GPU lock, the decode study's reproduction helpers and the platform record), which are never dropped
  for failing; and the other research-suite tests whose imports lie inside the release and that pass from a fresh git
  checkout of it with an empty home directory. None dispatches GPU work. `make test` runs exactly the selected set.
  The research tests left out are listed in the research checkout (`release/tests-not-kept.json`) with their actual
  errors and a category: an omitted historical fixture (a retained run the release does not carry), an omitted
  feature (research code or data outside the release), an environment dependency (a local cache), or the research
  ledger's stale review. Six of them were classified by reading their failures (an AI review, recorded as such).

## What is left out, and why

- **Campaign plans, handoffs and one-moment notes** (`docs/archive/`, `docs/specs/`, `plan/`), abandoned probes
  (`spike/` except the native runtime sources), and intermediate dumps that no public claim rests on.
- **The evidence archives** (`evidence/*.zip`, about 1.5 GB in the research checkout). Only the members a workflow
  needs are included, extracted and verified against the archive's recorded SHA-256. Some archives hold objects
  Apple's compiler produced.
- **Built binaries.** `make native-tools` builds them from the included sources.
- **The largest ISA artifacts** - the full recovered specification `isa/g17.yaml` (50 MB) and the bit specification
  `isa/g17-bit-spec.jsonl` (49 MB) among them - which no included workflow or test reads; the compiler reads the
  smaller tables that are here. `RELEASE-MANIFEST.json` lists every file a reference named but the size cap left out.
- **Tests that need the research checkout's full evidence store** or its git history; they are not part of this release.

The evidence record and the recon document are included as written, and they use the project's earlier name, triad,
in places. Where they link to a file the release does not carry, the export either brings the file in (a small code,
data or receipt file a claim points at) or rewrites the link to an entry of [docs/OMITTED.md](docs/OMITTED.md) that
names the target and says why it was left out; no link is left broken.

[PROVENANCE.md](PROVENANCE.md) classifies every file by origin: this project's source and writing, its recovered
tables, its measurement receipts, the programs its compiler produced, and, listed file by file, the objects Apple's
compiler produced, the text Apple's decoder produced, the one table derived from the decoder's numbering, and
third-party model metadata.

## Licensing

- **This project's own code and writing: MIT** ([LICENSE](LICENSE)).
- **Third-party model metadata: Apache-2.0**, under its upstream terms ([THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md),
  [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)).
- **Files produced by Apple's tools or derived from Apple's decoder metadata** are included and listed with their origin
  in [PROVENANCE.md](PROVENANCE.md): the 65 witness objects Apple's compiler built from this project's Metal sources
  (the compiler's tensor lowering reads them), their 65 listings by Apple's decoder, the three dumps of Apple's decoder
  metadata (`isa/g17-agx3meta-*.txt`, which the decoder interface reads), and the translation table
  `tools/agx3renumber.h`, which this project generated from that metadata (PROVENANCE.md says how).

## External dependencies on Apple

| Dependency | Needed for |
|---|---|
| Xcode command-line tools (`clang`, `xcrun metal`, `metallib`) | `make native-tools`; the Metal twins; Metal-source input |
| Apple's G17 decoder, loaded from `GPUCompiler.framework` at run time | **every compilation** (the compiler decodes what it emits), the CPU emulator's framing, the instruction listings |
| Retained Apple-compiled witness objects (included; listed in [PROVENANCE.md](PROVENANCE.md)) | **some tensor-lowering paths**: `agxforge/g17/tensorlower.py` reads them as encoding templates when it builds tensor instruction streams |
| Metal (`Metal.framework`) | the through-Metal path: every performance result |
| Apple's private IOGPU framework, the AGX kernel driver and the GPU firmware | the below-Metal path |
| Metal Performance Primitives (`matmul2d`) | the Apple int8 baseline (MM 25.212) |

## What has and has not been reproduced

The four workflows and the retained tests were run from a fresh copy of this release, on the CPU, with no private
cache. **No GPU work was run to prepare this release.** The measurements it reports come from one M5 Pro: the
through-Metal results on macOS 26.6.2 and Xcode 27.0, the native results on macOS 26.6.2 (build 25G83), and the Apple
int8 baseline on macOS 27.0.1. The native runtime's launch layouts were measured on build 25G83 only, and running it on
another OS build is untested; `python3 tools/g17platform.py --check` says how a machine differs from the measured one.

For the decode study, **the recorded comparison can be inspected, and exact reproduction of its historical inputs is
incomplete**: the 1,792-token prompt's ids were not retained, and the checkpoint's identity is likely but not recorded
([inputs-recovery.json](evidence/g17-matched-study-v1/inputs-recovery.json)). [examples/README.md](examples/README.md)
gives a complete route for a new comparison with explicitly identified inputs, and the native-inference reproduction
commands.
