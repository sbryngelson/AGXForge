# Execution validation: dispatching authored G17 instructions

Everything else this backend knows traces to Apple's disassembler. That is evidence about
Apple's instruction *description*, not about silicon. Byte-exact reconstruction can be perfect
while the model is wrong in a way no decode-side instrument can see, because reading agrees
with itself: a map reads Apple's register number, writes bits that read back as the number it
was given, and never learns that the number designates a different physical register.

Only a dispatch settles that. This file is how one is done here, what the first one found, and
the rules that exist because a previous one rebooted the machine.

First dispatch: 2026-09-06, authorised by Spencer. Current agreement: **45 of 45 case results**
across nine already-witnessed arithmetic opcodes - **27 independent**: the batch is fifteen records
over nine opcodes, and the six extra records (`andn`, `mul`, `nand`, `xnor` under a second mnemonic,
`or` and `xor` at a second declared length) emit byte-identical programs to their twins
(`isa/g17-execution-control.json`, `decoded.encoded`), so 18 of the 45 are re-runs.

## The gate, and why it is a person

**Do not dispatch without Spencer's explicit OK.** On 2026-09-05 dispatching Apple's own
looping kernels with foreign buffer contents wedged the GPU and rebooted this machine; a
separate incident killed WindowServer. An automated prompt, a hook, a stop-check or an agent's
own judgement is not that authorisation, and this file is not a licence to skip asking.

## The five rules, and where each is enforced

| Rule | Enforced in |
| --- | --- |
| One dispatch per process | `spike/accel/re/oracle.py` runs every record in its own child |
| A fault ends the batch | the parent breaks on `status: fault`, a negative return code, or a new `gpuEvent-*.ips` |
| Refuse loops, op578, op579, op450 | `g17oracle.refusal()` plus `g17safe`, walking the WHOLE emitted program |
| Store nothing when the command buffer is non-zero | the child attaches `values` only after status and canary both pass |
| Several runs must agree | `RUNS = 3`, reduced by `oracle.agree()` |

`agree()` is a **pure function** so its refusals can be proved without dispatching anything.
`tools/g17regress.py` drives it directly and asserts that a disagreement, a failed command
buffer, a missing canary, a fault and an empty run list are each refused AND carry no values.
A guard that refuses nothing proves nothing.

### The canary

`g17oracle` writes `0x5A17C0DE` to slot 6 **after** every case store. The output buffer is
pre-filled with `0xDEADBEEF`, which is -6.26e18 as a float and "no candidate matches" to a
classifier, so a program that faults part-way leaves plausible-looking numbers in the case
slots. Absence of the canary means *did not finish* and is never a result. Verified in the
emitted program rather than assumed: a three-case record emits four stores.

## Running one

```
python3 tools/g17prove.py                  # the plan and what is refused; dispatches nothing
python3 tools/g17prove.py --write b.json   # the control batch, still dispatching nothing
python3 spike/accel/re/oracle.py b.json --out results.json
```

`g17prove.py` proves each guard refuses a program built to trip it before it will prepare
anything. The control batch is the fifteen already-witnessed arithmetic forms - add, sub, mul,
or, xor, and, nand, xnor, andn - because a batch in which *those* disagree is a batch whose
harness is broken, and nothing else in it can be believed.

## What the first dispatch found

### The forms are 16-bit, and the harness had been predicting 32

Against the harness's predictions the first batch agreed on **10 of 45**. Those predictions
were masked to `0xFFFFFFFF` because someone assumed it when the file was written and nothing
had ever run to contradict it. Masked to `0xFFFF`: **39 of 45**.

The evidence is one number rather than the score. `add` of `0x12345678` and `0x0F0F0F0F`
returned **25,991**, which is `0x5678 + 0x0F0F` exactly; `mul` returned `0x5678 * 0x0F0F`
truncated. A wrong mask cannot produce agreement on a value that carries the low half of *both*
inputs through a multiply. `tools/g17prove.py CONTROL` now masks to 16 bits, as measured.

### Two opcodes read the wrong register, and the decode side could not see it

`op437 and` and `op17757 xnor` read their second source from a register 144 away from where the
value was loaded. Both forms are `verified` on every corpus instance, pass certification in
both directions, and reproduce Apple's bytes. Nothing short of execution could have found it.

The answer needed one fact that had to be learned rather than guessed. The authoring table
reports a **slot**; the disassembler names a **register id**; the conversion is

```
register id = base + (slot >> 1) - 144 * (slot & 1)
```

Slot bit 0 is the **bank**, worth -144 in id space. A bit worth 144 cannot appear in a
positional number - the same fact the ledger records for `op10864`'s coefficient bit - which is
why every attempt to read the slot as a plain integer failed.

With the relation known, one uniform model (positions `b8.6, b8.7, b9.0..b9.6`, base 425)
predicts operand 4 exactly on **236 corpus instances across all six ten-byte forms**, the two
failing ones included. So the field is the same in all of them, and `isa/g17-authoring.jsonl`
was wrong in two places for those two forms:

- `4:slot` lacked `b8.6` and had every weight one position low
- `operand_slope` said `units: 1, verdict: "exact"` where all four working siblings say
  `units: 2, verdict: "interleaved"`

The second is what moved the hardware. A GPR16 slot interleaves two register ranges, so
register index 9 is slot **18**, not slot 9. Declared exact, the compiler asked for slot 9 and
got a register in the other bank.

```
and.and@437     [0, 0, 1544]       ->  [1, 1, 1544]
xnor.xnor       [65532, 0, 42632]  ->  [65529, 1, 42632]
                39 of 45           ->  45 of 45    (23 -> 27 of 27 independent)
```

The correction was confirmed **without dispatching** - the emitted instruction names `reg:434`
like its siblings - and only then re-dispatched.

## Two encoders, and which one a dispatch tests

This surprised the author of the first three attributions and is worth stating plainly:

| Path | Table | Measured by |
| --- | --- | --- |
| the assembler, `g17as` | `isa/g17-operand-maps.jsonl` | byte-exact reconstruction, both-direction certification, encoder independence |
| the compiler, `g17cc` | `isa/g17-authoring.jsonl` | **execution agreement** |

`g17as.field_encode` is called **zero times** during a dispatch. An execution result is not
evidence about the operand maps, and a byte-exact score is not evidence about what the compiler
emits. Instrument before attributing.

## Reading a result

Per record: `values` only when the run is trustworthy, `cb_status`, `finished` (the canary),
and `runs` (how many agreeing dispatches back it). A record without `values` carries a
`status` saying why - `refused`, `decode-mismatch`, `build-failed`, `cb-failed`,
`did-not-finish`, `disagreed`, `fault` - and a disagreement is a statement about the harness,
not data.

Results land in `isa/g17-execution-control.json`.

## What execution has NOT established

Nine opcodes at ten and twelve bytes, on integer and bitwise arithmetic, with two inputs each.
Nothing about loads, stores, control flow, tensor forms, float forms, or any opcode the control
does not contain. `execution agreement` in the metric table means exactly this batch and no
more, and it is quoted with the other four metrics because none of them means much alone.
