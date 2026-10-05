# The G17s ISA artifacts

Every file here is derived. `tools/g17export.py --check` re-derives the name-bearing ones from
`tools/g17slice.py` and exits non-zero on drift, so a stale artifact is a build failure rather
than a silent wrong answer.

## Start here

| file | what it is |
|---|---|
| `g17-contract.jsonl` | **The operational contract for all 6,718 admitted opcodes.** Certified encoding, the read set including implicit operands, the write set including whether it writes mov.a's file, the reordering and duplication properties, the evidence class, the candidate set. A backend can schedule, allocate for and encode an *unnamed* opcode from this alone. |
| `g17-authoring.jsonl` | Per-opcode field map: which bits carry which operand, the witness, dead bits, and `apple_witness` where a real instance exists. |
| `g17-certification.jsonl` | Evidence class per opcode, and the `structurally_ambiguous` flag. |

## What the evidence classes mean

    verified    77    a probe isolating this opcode ALONE returned the value its name predicts, on
                      hardware. Select from these without further checking.
    measured   442    a Metal construct emits it and the count scales exactly at 1x, 2x and 4x.
    corpus     124    it appears in a shipped object.
    table-only 2724   named from Apple's tables and structure. VERIFY BEFORE SELECTING - predicting
                      a name from a structural twin is right 0 times in 44 on the verified set.
    unnamed   3351

## runnable is not authorable

All 6,718 opcodes have a complete field map, so all of them can be *encoded*. Only **717** have an
encoding Apple actually wrote (`apple_witness`). The other 6,001 are known through a repair-walk
witness — an encoding built by mutating a neighbour until the decoder accepted it. Forty class-51
opcodes authored that way ignore both their inputs and return a constant of the encoding.

A walked witness *can* work — op1062 did — so treat `runnable: false` as a warning with a named
failure mode: **if the answer does not move when the inputs move, that is the witness rather than
the instruction.**

## For dispatching

| file | what it is |
|---|---|
| `g17-runnable-unnamed.txt` | The 45 unnamed opcodes with an Apple-written encoding. The dispatches worth running. |
| `g17-dispatch-candidates.txt` | Structural groups holding a hardware-verified member, with the neighbours a dispatch would separate. |
| `g17-separation-programs.txt` | Per group, the inputs on which every candidate returns a different value — and which members are runnable. |
| `g17-structurally-unsafe-groups.txt` | Groups where measurement proves structure cannot decide. Retracted names keep their candidate set here. |
| `g17-corpus-unnamed.txt` | The 12 opcodes Apple ships that this table cannot name. |

## Measured facts that are not names

| file | what it is |
|---|---|
| `g17-special-register-numbers.txt` | The hardware special-register numbering, which appears nowhere in Apple's tables. Executed and confirmed. |
| `g17-barrier-scopes.txt` | The barrier's memory scope is an immediate; the imageblock fence differs from the threadgroup one in one field. |
| `g17-imageblock-execution.txt` | The imageblock holds data, is indexed by thread position, and Apple emits **no** barrier for a same-threadgroup round trip. |
| `g17-imageblock-pair.txt` | A matched load/store of one slot from one kernel, so the operand roles are known by construction. |
| `g17-local-indexed-memory.txt` | The 132 memory opcodes that implicitly read SR_LOCAL_X and SR_LOCAL_Y. |
| `g17-class-evidence.txt` | Every scheduling class with unnamed members and all the evidence about it. |
| `g17-half-alu-predictions.txt` | Five predictions and their falsifiers, committed *before* the probes that tested them. All five held. |

## The bound, so nobody re-derives it

Apple's compiler emits **669 of 6,718** opcodes — across 1,795 shipped objects and 1,796 probes in
fifteen families. The reach stopped moving at 554 probe-reached three families ago. Roughly nine
tenths of this ISA has no oracle but execution, and that is why the artifacts above are addressed
to a dispatcher rather than to a reader.

## The assembler and what it can author

| file | what it is |
|---|---|
| `g17-exec-mask.txt` | The execution-mask family: one encoding word for `end`, `if`, `else`, `pop`, `while` and the branches. Count is a code table, the predicate register is an 8-entry lookup, `b2.1` inverts the predicate, branch target is `own offset + displacement`. 284 of 284 re-encoded byte-exact. |
| `g17-operand-maps.jsonl` | Every operand field tested against **every cached instance** of its opcode rather than the single witness it was fitted to. Verdicts `verified` / `conditional` / `degenerate` / `refuted`; only `verified` is authorable. Register fields advance one printed register per **two** encoded units, which is why maps fitted to one witness produced bytes Apple's own decoder rejected. |
| `g17-modal-bits.jsonl` | What Apple actually puts in each bit of each `(opcode, length)`, over 184,349 instructions. A bit no operand claims takes this value rather than zero, because an all-zero `op423` is an instruction Apple never emits. A default for staying on the manifold, not a semantic claim. |
| `g17-corpus-programs.jsonl` | 6,594 distinct programs with instruction boundaries (built by the whole-program agent), the basis for the authorability number below. |

`tools/g17as.py` assembles from those maps plus the specification's opcode, forced and mode bits.
No template, and the decoder is never consulted while encoding. **8.8%** of the corpus's 184,349
instructions currently re-assemble to Apple's exact bytes; five opcodes — `movimm` 554, `add` 10282,
`add` 10279, `load` 12682, `load` 12674 — block 46.6% of the rest. That percentage, weighted the
way real programs are weighted, is the honest measure of how much of this ISA can be authored
today, and it is the number to drive up.
