# The AGX3 oracle: running Apple's own G17 decoder

Written 2026-09-04. Everything here is on H17s / `applegpu_g17s` (M5 Pro), macOS 26,
Metal toolchain 32023.

This project spent its G17 work measuring framing rules, because the premise recorded in
`tools/isadb.py` was that Apple's toolchain will not disassemble its own GPU. That premise is
half right, and the wrong half is the expensive one. The EMIT path is genuinely removed. The
DECODE path was never removed, ships in the OS, and can be called. This file is the record of
how it was reached, what it does and does not give, and which parts will break on an OS update.

## 1. What is actually stubbed

`applegpu-nt` is a symlink to `air-nt`, which loads `wrapper-nt.dylib`, which dispatches to a
per-generation plugin. For macOS 26 targets the plugin is `libapplegpu-nt.dylib`, covering
`applegpu_g13p` through `applegpu_g18p`.

Its 32 `AIRNT*` entry points were disassembled one by one. Exactly two are stubs:

    AIRNTEmitAssembly
    AIRNTEmitObject

Each is about twelve instructions that build the string `"[AGX] Plugin interface not
implemented: "`, append the function name, and return 0. Every other entry point, including
`AIRNTEmitExecutableImage` and `AIRNTEmitPipelineImages`, is real code. So `-S` cannot be
reached by any flag, environment variable or plugin routing, and this is true for every AGX
arch from `g13g` to `g18p`, not only G17.

One trap worth recording because it costs a day. Targeting `-platform_version macos 13.0` or
`15.0` routes to `libapplegpu23-nt.dylib` or `libapplegpu24-nt.dylib`, which reject a current
module with `incompatible module`. That looks like an AIR version problem and invites patching
`air.version` from 2.8 down to 2.5. It is not a version problem, it is plugin selection. With
`-platform_version macos 26.0 26.0` the 2.8 module is accepted unpatched, and the run then
fails on the stub above. Also, `-S` reads IR rather than an MTLB container, so the input has to
be textual or bitcode IR: `air-opt -S t.air -o t.ll` first.

## 2. The refusal that is not a refusal

`air-objdump --version` registers three targets: `agx1`, `agx2`, `agx3`. Pointing it at a
native object gives different errors per target:

    --triple=agx1   no disassembler for target agx1
    --triple=agx2   no instruction printer for target agx2
    --triple=agx3   no instruction printer for target agx3

llvm-objdump builds the MCDisassembler before it builds the MCInstPrinter. Reaching the printer
error therefore proves a decoder was constructed and is present. `agx1` is the control that
makes the two messages mean different things: it has no decoder at all, and says so.

This is the whole finding. Everything below is engineering.

## 3. Where the decoder lives

Not only in that binary. The shared-cache libLLVM under GPUCompiler.framework exports the AGX3
MC stack:

    /System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/libLLVM.dylib

    LLVMInitializeAGX3TargetInfo     LLVMInitializeAGX3Target
    LLVMInitializeAGX3TargetMC       LLVMInitializeAGX3AsmPrinter
    LLVMInitializeAGX3Disassembler   LLVMInitializeAGX3AsmParser
    LLVMInitializeAGX3TargetMCA      LLVMInitializeAGX3MCInstLifter
    LLVMInitializeAGX3APSTraceSystem

all resolvable with `dlsym`. Calling every one of them still leaves `LLVMCreateDisasm`
returning null, so the missing piece is narrower than a missing component.

`llvm::Target` for agx3 was dumped and each field pointer resolved against the shared cache
symbol table. The struct is a list of constructor function pointers, and exactly one relevant
slot is empty:

    field +0x80   MCDisassemblerCtorFn   written by LLVMInitializeAGX3Disassembler
    field +0x88   MCInstPrinterCtorFn    null

Apple withheld the instruction printer and nothing else.

## 4. Supplying the missing slot

Writing a printer factory into `+0x88` completes the object and `LLVMCreateDisasm` succeeds.
The returned printer needs no real behaviour, because `LLVMDisasmInstruction` calls `printInst`
only after the decode has already happened, and hands it the decoded `MCInst`. Reading opcode
and operands off that `MCInst` is the entire tool.

The vtable slot for `printInst` was found rather than assumed: point all 32 slots at recording
stubs, run one decode, and see which slot is called with an `MCInst` in argument 1. It is
slot 4.

`tools/agx3dis.c` is that tool, and it is 73 lines.

### The lifetime trap

The `MCInst` lives in `LLVMDisasmInstruction`'s frame, and `SmallVector<MCOperand, 8>` keeps
its first eight operands in inline storage inside that object. Capturing the `MCInst` pointer
in `printInst` and reading the operands after the call therefore WORKS for short instructions,
because the stack frame usually survives unchanged, and reads freed heap for anything longer.

Read that way, 69.13 percent of operands carried invalid `MCOperand` kinds, ten-operand
instructions were entirely garbage, and four-operand ones looked plausible except for a bad
last entry. That selective failure is what made it look like a struct layout problem. A wrong
offset would have failed uniformly.

Copying opcode, count and operands inside `printInst` instead: 290 objects, 89,645 operands,
0 invalid kinds.

## 5. The metadata behind the opcode ids

Once `llvm::Target` is in hand, its other constructor slots hand over the generated TableGen
tables. None of the struct layouts below were taken from public LLVM headers, because this is
Apple's fork. Each was recovered by self-validation, which is also the check that it has not
moved after an OS update:

    MCInstrDesc      stride 48   the first u16 must equal the array index for all 17,796
                                 entries. Only stride 48 satisfies that.
    MCOperandInfo    stride 6    validated against the decoder rather than by structure: over
                                 555,785 operand slots, RegClass >= 0 predicts a register
                                 operand and RegClass == -1 predicts an immediate, agreeing
                                 96.08 percent of the time. Every disagreement is a
                                 register-class slot that decoded as MCExpr instead of MCReg,
                                 which is what a relocation-bearing operand looks like, so the
                                 residue is a real distinction and not layout error.
    MCRegisterClass  stride 32   the u16 at +24 must equal the array index for all 500 entries.
    MCRegisterDesc   stride 24   smallest 4-aligned width whose Name index stays inside the
                                 string blob for all 3,567 registers.

`MCRegInfoCtorFn` takes a `const Triple &`, which has to be built. `llvm::Triple(const Twine&)`
is exported; a `Twine` is two 8-byte children followed by two 1-byte kind tags, with
CStringKind 3 and EmptyKind 1, so one can be constructed by hand in 5 lines.

`tools/agx3meta.c` does all of this and prints three tables.

## 6. What is available and what is not

Instruction NAMES do not exist anywhere. Apple built this LLVM with instruction names disabled,
so `MCInstrInfo`'s name table holds the strings "0", "1", "2" and so on, one per opcode. The
same blob is present in `libapplegpu-nt.dylib` and `air-opt`, equally stripped. There is no
`AsmStrs` mnemonic table to recover because the InstPrinter that would own it is not linked
into any shipping binary. Treat mnemonics as gone.

Register names and register-class names DID survive, because the switch that strips instruction
names does not touch them. This is why an operand can be typed while the instruction using it
stays a number:

    3,567 registers      R0, R14, CTLFLOWST, FLAGFALSE, ...
    500 register classes GPR16 (16 bits, 288 regs), GPR32 (32, 144), IRGPR32 (32, 160),
                         SIR32 (32, 68), FLAGR (16, 17), GPR32tup2 (64, 143),
                         GPR32tup4 (128, 141), GPR32tup8_alignedrc (256, 18), ...

Register operand values resolve. The decoder's register numbers are `MCRegister` ids in the
same namespace as `MCRegisterInfo`, and the GPR block starts at 105, so `R_n = 105 + n`.
Checked over 60 driver shaders: 300 distinct register numbers used, none out of range of the
3,567. Sub-registers are separate ids and appear in real code (429 is `R4L`), so the 16-bit and
32-bit views of the same file are distinguishable. The special block sits at the low ids:
`CTLFLOWST`, `FLAGFALSE`, `FLAGTRUE`, `LR`, `SMP_BATON`, `SP`, then the `SR_*` system registers.

Implicit operands are a dead end and are recorded as such. `ImplicitUses` at +24 and
`ImplicitDefs` at +32 are populated for 402 and 7 opcodes respectively, and for none of the 216
in the corpus. Exec-mask and predicate state is not hidden state in this ISA; it is explicit in
`FLAGR`-class operands.

IMMEDIATE values are still uninterpreted, and several are plainly encoded rather than literal.
Do not read them as numbers.

## 7. What the corpus looks like in these terms

`tools/g17opcodes.py` walks the cache and joins each opcode against the descriptor tables, into
`isa/g17-opcodes.toml`. Over 1,169 objects and 85,806 instructions:

    distinct opcodes            216 of 17,796
    distinct scheduling classes 44

    opcode  count   defs sched  operand signature
    554     12,488  1    23     IRGPR32 imm
    12674   8,107   1    286    GPR32tup2 imm imm GPR32tup2 imm GPR32 imm imm imm imm
    10282   6,470   1    5      IRGPR32 imm GPR32 imm GPR32 imm
    5106    6,073   1    172    GPR32tup8_alignedrc imm imm GPR32tup4_alignedrc imm imm
                                GPR32tup4_alignedrc imm imm GPR32tup8_alignedrc imm imm
    462     1,352   0    6      imm imm.t4

Two of those are worth reading closely. Opcode 5106 takes a 256-bit destination, two 128-bit
sources and a 256-bit accumulator, which is the shape of the Neural Accelerator MAC this
project derived independently. Opcode 462 is the branch, and its second operand carries
operand type 4 rather than 0, which is the PC-relative displacement.

**Length is not a function of opcode.** 64 of the 216 opcodes appear at more than one length:
opcode 12674 is 12 or 16 bytes, opcode 554 is 4 or 8, opcode 11842 is 2, 8 or 10. This is the
direct explanation for why length rules keyed on an instruction class kept failing and kept
being retracted. The width is an encoding property, not a class property, and the only thing
that reads it correctly is the decoder.

## 8. Reproducing

    clang -O2 -o tools/agx3dis  tools/agx3dis.c
    clang -O2 -o tools/agx3meta tools/agx3meta.c

    tools/agx3dis <applegpu-object> <offset> <length>   decode a byte range
    python3 tools/g17ref.py <object>                    walk _agc.main
    python3 tools/g17ref.py --check                     compare against g17dis over the cache
    tools/agx3meta instrs|classes|regs                  the descriptor tables
    python3 tools/g17opcodes.py                         rebuild isa/g17-opcodes.toml

Nothing needs sudo, entitlements, or a patched binary on disk.

## 9. Fragility

The route is version-pinned in three places, and all three are load-bearing:

1. The libLLVM path contains `Versions/32023`. A toolchain update changes it.
2. The `llvm::Target` field indices (8, 10, 16, 17) and every struct stride in section 5 are
   properties of Apple's build. `agx3meta` re-validates the strides on every run and refuses to
   print if one no longer holds, which is the intended early warning. `agx3dis` checks that
   `MCDisassemblerCtorFn` is non-null for the same reason.
3. The `printInst` vtable slot is 4 in this build and is not checked at runtime. If a future
   build reorders `MCInstPrinter`'s virtuals, `agx3dis` will silently capture nothing rather
   than fail, and the symptom will be opcode 0 on every line.

The corpus itself has a separate fragility already recorded in `tools/isadb.py`: the Metal
toolchain ships in a cryptex whose mount path carries an asset id, and that id changes on
update. It moved from `...Llj4yY` to `...M2ATNw` during this work.

## 10. What this changes

Framing stops being inferential. Before this, an instruction boundary was the output of a
measured rule, a tiling constraint solve, or a signature scan, and `tools/g17dis.py` was built
to fail loudly rather than guess because a desynchronised walk had already produced one false
architectural finding. Now a boundary is checkable per instruction against the decoder that
Apple's compiler shares with the encoder that produced the bytes.

The first check was not flattering and is recorded in
`ledger/g17-apple-decoder-ground-truth.toml`: g17dis walks 112,952 instructions where the
decoder finds 85,806, so roughly 27,000 reported instructions do not exist, and every coverage
and ENCODE percentage computed before this was scored against an inflated denominator.

## 11. The model layer

`tools/agx3dis.c` is deliberately the raw frontend: bytes to boundary, opcode and raw operands,
with no interpretation. `tools/g17model.py` is the layer above it, and it is where the tables
and the decode are joined:

    opcode id           stable, 17,796 of them, 216 seen in the corpus
    operand count       MCInstrDesc.NumOperands, matches the decoder exactly
    defs vs uses        MCInstrDesc.NumDefs - the first NumDefs operands are definitions
    register class      per operand, named
    scheduling class    44 in the corpus
    register names      MCRegister id to Apple's name

None of that is inference. Together it makes a shader readable:

    00000180  4 op14059  sched=463  R2 <- 1048576, SR_TP_IN_GRID_X, 0
    00000184  4 op14059  sched=463  R1 <- 2097152, SR_TP_IN_GRID_Y, 0
    00000188 10 op5077   sched=168  FLAG0 <- 0, 0, expr:0xb7500d020, 0, 0
    00000196 12 op10295  sched=5    R3H <- 16777248, expr:0xb7500d0b0, 0, R2L, 16
    000001b6  4 op582    sched=25   0, FLAG0, 1
    000001ba 10 op462    sched=6    0, 76

Scheduling class 463 is the grid-index read, and that is Apple's own register name rather than a
guess. The compare, flag and branch sequence is legible end to end. The immediates are raw and
several are packed field words, so they are printed but not interpreted.

`g17model.py` makes no semantic claim on purpose. A reading of the scheduling classes is
recorded in `ledger/g17-apple-opcode-metadata.toml` as inference from operand shape, and belongs
in `isa/g17-scalar-isa.toml` only once something causal backs it.

## 12. The limits of the oracle's authority

The decoder is ground truth for the corpus and is NOT ground truth for the ISA. Both directions
of that have been measured, and treating either as an error check will retire correct work.

**A rejection is not proof of error** - but the argument for that is structural, not
executional, and the executional one was withdrawn.

RETRACTED 2026-09-04, same day it was written. This section originally cited authored `sub` and
`mul` forms that the decoder rejects or renders at an unexpected width yet "execute correctly,
7 of 7 against preregistered values". That evidence was about a different program: the passing
test exercised a load and an add, not the mul or the sub, and the two had diverged silently
behind a shared name. Tested one operation per program, `sub` returns -10 where 7 was predicted.
It negates its source instead of subtracting. The decoder rejected exactly that encoding and was
right. See `ledger/g17-oracle-authority-boundary.toml` for the retraction in full.

What survives is the structural argument, which never depended on execution. Opcode 10701 is not
an anomaly, it is one cell of a systematic grid: opcodes 10696 through 10709 are the same
operation enumerated over operand shapes, `IRGPR32 imm imm {imm|GPR32|GPR16} imm
{imm|GPR32|GPR16} imm`, and scheduling class 312 holds 342 opcodes of which only 20 appear in
the corpus. So "absent from the 216-opcode census" means the compiler never chose that shape,
which is a statement about Apple's selection and not about the encoding's validity. The census
is a usage histogram, not an inventory.

The execution leg was later replaced by one that holds, from a different bit and re-run before
being believed. byte9[2] of the 14-byte load: Apple's decoder rejects the mutation, and byte9 is
0 in all 421 corpus instances of the compiler's load opcode. Dispatched one program per process
with both answers written down first, and with the control inside the pair:

    narrow = 0    predicted 0x12345678    got 0x12345678    decoder ACCEPTS
    narrow = 1    predicted 0x00000078    got 0x00000078    decoder REJECTS

The two programs differ in exactly that bit, and the full-word case is the control that would
look different if the narrow load did nothing. So the hardware implements a load width the
compiler does not select through that bit - it uses a different opcode for narrow loads - and the
tables have no reason to cover the combination.

The correct reading is therefore: a decoder rejection is a signal worth chasing and never a
verdict in either direction. Two rejections were chased in one day and went opposite ways. The
authored `sub` was rejected and was genuinely computing 0 - x. This load is rejected and computes
exactly what was predicted. What settles either case is a dispatch with the answer written down
first, plus a control that would look different if the feature did nothing.
See `ledger/g17-oracle-authority-boundary.toml`.

**An acceptance is not proof of correctness either, and this is the sharper limit.** The
decoder reports operands, and any byte that is not an operand field is invisible to it. Both
branch opcodes are 10 bytes with exactly 2 operands, and a per-byte variance census over every
corpus instance shows how little of the instruction that covers:

    op462  1352 instances   bytes 0-4 vary   bytes 5-9 constant  00 00 00 00 00
    op458   515 instances   bytes 0-4 vary   bytes 5-9 constant  00 ff c7 ff 7f

The two directions carry different constant tails, and an authored back edge with a zeroed tail
decodes cleanly as a well-formed op458 while being wrong in its whole second half. That defect
is undetectable by the decoder by construction, and it took execution to find.

`tools/g17variance.py` reports this per opcode. The extreme case is opcode 684, whose four bytes
are identical in all 1169 objects: zero varying bits, so the decoder can confirm nothing about
it at all. At the other end, opcode 12674 at 16 bytes has 38 varying bits out of 128.

One caveat on that census, of the same kind as the opcode one: a bit that is constant across the
corpus is constant because the compiler never varied it, which is not the same as the hardware
requiring it. Use it to find where authored bytes are unchecked, not as a claim about encoding.

## 13. What was built on it: dataflow

The oracle's purpose was never the disassembly. With framing settled and operands typed, the
work moved to connecting instructions rather than reading them, in three layers:

    tools/g17regs.py   register aliasing over LEAF registers - 410 of them, matching NumRegUnits
    tools/g17cfg.py    basic blocks, dominators and dominance frontiers
    tools/g17ssa.py    SSA, versioned per leaf so partial writes behave

The branch rule that makes the CFG possible is `target = offset + displacement`, which lands on
a real instruction boundary for 1867 of 1867 corpus branches against 12.9% for the
offset-plus-size alternative. Built for all 1180 objects with no failures: 85,879 instructions,
4,892 blocks, 129,591 phi nodes.

Every branch contributes both edges, because a G17 branch names no condition. That
over-approximates, which is the safe direction, and it is the direction the evidence demands:
op582 and op579 have identical descriptors and behave differently, so a CFG that inferred
conditionality from the preceding flag op would have encoded a known error. Details in
`ledger/g17-ssa-dataflow.toml`.

## 14. Three state domains in one graph

Register state was the first domain. Memory and exec followed, in the same SSA machinery, with
phi folding first so the graphs stayed readable (129,591 to 55,480 phi nodes, 57.2% removed,
zero orphaned uses).

    memory   one chain per RESOURCE, not one blob. Load versus store is MCInstrDesc.NumDefs;
             the resource is the relocation on the address operand, present on 18,673 of 19,996
             accesses. 96.3% of loads carry a known resource, and of the reads that a store in
             the same function reaches, 68.6% resolve to exactly one store.
    exec     scheduling class 25 consumes a FLAGR and defines no register, so it is modelled as
             transforming a synthetic EXEC location, with branches reading it. That is the whole
             claim - no push, no pop, no and-mask. The tables cannot say more, because op582 and
             op579 are identical in them and differ under execution.

`tools/g17slice.py --from SR_TP_IN_GRID_X` then walks the graph forward and prints
`sr_read -> alu -> alu -> store`, which is the chain end to end with no mnemonic anywhere in it.
Details in `ledger/g17-unified-state-ssa.toml`.

## 15. Semantic recovery by slices

The graph exists to be walked backward. `tools/g17slice.py --to store` takes an endpoint in a
real driver shader and prints the expression that computes it, together with the control
dependence that decides whether it runs, with every operation this project cannot name left as
an explicit hole:

    guarded by:
      t000000f4 = UNKNOWN_12715(..., reloc:0x...d140, M:0x...d140)
      t00000124 = UNKNOWN_12688(..., reloc:0x...d218, M:0x...d218)
      t0000068a = UNKNOWN_11310(0, 10, t000000f4, 32, 1, t00000680, 16, 0)
    computes:
      t00000766 = shr(32, 0, t000000fc, 16, t0000075c, 0, 32)
      t00000958 = sub(32, t00000944, 0, 1)

Control dependence, not just dominance, is what makes the guard line possible. A predicated GPU
guards work by mask, so an instruction can be conditional with no branch around it. Over 305
objects, 870 of 1310 blocks are control dependent and 739 of those (85%) are decided by a branch
whose guard defines a FLAG; 1048 stores are guarded and 346 are unconditional.

The output is a bound DAG rather than a nested expression, because real slices reuse
subexpressions heavily and inlining them produced thousands of unreadable characters per store.
Each contributing instruction gets one binding, which is also the form a differential probe
wants. Details in `ledger/g17-backward-slices-and-control-dependence.toml`.
