#!/usr/bin/env python3
"""Cartographer: the G17 evidence ledger as a machine-readable ISA specification.

This tool discovers nothing. It HARVESTS what the ledgers already establish into one record per
FORM, with an explicit evidence level on every axis, and reports three denominators separately
instead of one percentage. `--check` re-derives the artifacts and refuses if the committed copies
differ, so the specification cannot drift away from its evidence.

THE THREE DENOMINATORS, NEVER COMBINED

    D1 structural   forms this decoder can address at all.
    D2 encoded      forms whose encoding is established: an encoding APPLE ACTUALLY WROTE, no
                    unforced bits, and no invisible bit left undetermined.
    D3 semantics    forms whose semantics rest on hardware: evidence class `verified`, meaning a
                    probe isolating that opcode alone returned the value its name predicts.

D2 deliberately does NOT accept a complete field map. All 6,718 admitted opcodes have one, and
6,001 of them are known only through a repair-walk witness - an encoding built by mutating a
neighbour until the decoder accepted it. Forty class-51 opcodes authored that way ignore both
their inputs and return a constant of the encoding. A form that decodes is not a form that works.

D3 deliberately does NOT accept `measured`, which in this ledger means "a Metal construct emits it
and the count scales at 1x, 2x and 4x" - evidence about Apple's compiler, not about the hardware.
Nor does it accept a decoder name: predicting a name from a structural twin is right 0 times in 44
on the verified set. Naming is D1 evidence and nothing more.

Forms that were DISPATCHED but whose output was never compared with a prediction are counted and
reported apart, as `executed_unchecked`. Running is not checking.

TWO RULES THE HARVEST ENFORCES ON ITSELF

    An axis can never carry evidence above the level of the ledger that supplied it.
    An ABSENT ledger yields evidence 'absent', never a clean or empty answer. A fail-open default
    is how a coverage claim becomes fiction.

    python3 tools/g17isamap.py --write     regenerate isa/g17.yaml and the two side ledgers
    python3 tools/g17isamap.py --check     refuse if a committed artifact differs from a harvest
    python3 tools/g17isamap.py             print the denominators by family
"""
import argparse
import hashlib
import json
import collections
import functools
import os
import re
import sys

import g17auth  # the field-map API, for field_map_width_scope
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
# THE REPOSITORY ROOT MUST BE IMPORTABLE. `dispatched_byte_forms` decodes receipts through
# agxforge.g17.model, and this tool is run both as `python3 tools/g17isamap.py` (where nothing puts
# ROOT on the path) and as an import from a process whose CWD happens to be the checkout. Without
# this the decoder import failed in the first case and the reader returned an EMPTY population -
# silently, because the import sat inside a try/except - so the artifact reported 38 unique and 2
# ambiguous widths instead of 26 and 15, and `--check` PASSED, because it compares this tool
# against itself and both sides were equally blind.
sys.path.insert(0, str(ROOT))
from agxforge.g17.model import decode as decode_instruction        # noqa: E402  - loud on failure
ISA = ROOT/'isa'
SPEC = ISA/'g17.yaml'
UNIVERSE = ISA/'g17-universe.json'
COVERAGE = ISA/'g17-coverage.json'
LEDGER_INDEX = ISA/'g17-ledger-index.json'
CLAIMS = ISA/'g17-ledger-claims.json'
PEER_CLAIMS = ISA/'g17-peer-claims.json'
REFERENCE = ROOT/'docs'/'g17-isa-reference.md'
# Every file this tool writes. The isa/ file count excludes these, and the test reads this set
# rather than restating it: adding a sixth artifact must not be able to break that count again.
OUTPUTS = (SPEC, UNIVERSE, COVERAGE, LEDGER_INDEX, CLAIMS, PEER_CLAIMS)

# Weakest first. 'corpus' and 'oracle' are both VENDOR-BEHAVIOUR evidence and neither is hardware:
# corpus means Apple shipped it, oracle means Apple's compiler emits it and the count scales.
# 'executed' means a retained receipt shows it ran. 'checked' means an output was compared with an
# independent prediction. The gap between the last two is where a specification is won or lost.
LADDER = ('absent', 'table', 'decoder', 'corpus', 'oracle', 'executed', 'checked')
RANK = {name: i for i, name in enumerate(LADDER)}

# How this ledger's own evidence classes map onto the ladder, with why.
EVIDENCE_CLASS = {
    'verified': ('checked', 'a probe isolating this opcode alone returned the value its name '
                            'predicts, on hardware'),
    'measured': ('oracle', 'a Metal construct emits it and the count scales at 1x, 2x and 4x; '
                           'evidence about Apple\'s compiler, not about the hardware'),
    'corpus': ('corpus', 'it appears in a shipped object'),
    'table-only': ('table', 'named from Apple\'s tables and structure; name prediction from a '
                            'structural twin is right 0 times in 44 on the verified set'),
    'unnamed': ('table', 'structure only; Apple\'s tables carry no name'),
}

AXES = ('length', 'opcode_selection', 'operand_encoding', 'legal_forms', 'implicit_operands',
        'register_files', 'immediates', 'semantics', 'flags_exec', 'memory_effects',
        'latency_wait', 'lifetime', 'barrier_sync', 'resources', 'hardware_evidence',
        'unknown_bits', 'emitted_by', 'ledger_claims', 'peer_claims')

# Investigation order, deliberately not frequency order. 'unknown' is a COUNTED bucket: a form
# nobody has classified is a result, and filing it under a plausible family is how a map acquires
# imaginary coastline. SIMD, tensor, graphics and ray tracing have no mechanical discriminator in
# Apple's descriptor, so forms that would belong there sit in unknown until a probe separates them.
FAMILIES = ('scalar_alu', 'memory_addressing', 'simd_shuffle_reduce', 'control_predicate_exec',
            'atomic_sync', 'texture_image', 'tensor_accelerator', 'graphics', 'ray_tracing',
            'call_control', 'system_register', 'unknown')


# Facts harvested BY HAND from ledger/, each citing its file and that file's own status string.
# This is a curated subset, not a harvest: ledger/ holds 611 files and 247 of them name an opcode
# in prose, so reading them automatically is a project of its own. The subset is explicit so that
# what it omits is visible - see `unharvested_sources` in the coverage report.
# WHY A PLAN/RESULT PAIR IS NOT SELF-PROVING, quoted by every fact that rests on one. Named so
# the caveat cannot be dropped by paraphrase in one fact while the others keep it.
_SAME_COMMIT_CAVEAT = (
    "the plan file and the results file were added in the SAME COMMIT, so git history does not "
    "establish that the expectation predates the run, and `values == expect` alone would be a "
    "comparison with no way to fail. What makes this non-vacuous is that the expected values are "
    "RECOMPUTED here from an independent reference rather than read from the record")

LEDGER_FACTS = {
    447: [dict(axis='barrier_sync', evidence='corpus',
               source='ledger/g17-barrier-is-op447.toml + isa/g17-barrier-scopes.txt',
               state='the barrier; byte1 selects its scope and the memory scope is an immediate '
                     'operand, measured for five Metal scopes (mem_none 20, mem_device 154, '
                     'mem_threadgroup 276, mem_texture 1144, threadgroup_imageblock 532)',
               ledger_status='source-level correspondence over four kernels, with bar_both as the '
                             'discriminating case; not executed',
               note='surveying every opcode that defines nothing and takes only immediates returns '
                    'exactly one candidate across the corpus; simdgroup_barrier scopes were not '
                    'built, so four of the nine scope spellings have no encoding')],
    3978: [dict(axis='semantics', evidence='executed', widths=(10,),
                source='isa/g17-transcendental-unit.toml [code_2_is_not_sqrt]',
                state='RETURNS THE RECIPROCAL SQUARE ROOT, established on hardware: the encode '
                      'session executed it on three inputs and got 2.75 -> 0.60302269, 16 -> 0.25, '
                      '0.25 -> 2. This RETRACTED its previous name. It had been called `sqrt` '
                      'because that is the Metal construct Apple selects it for, and the '
                      'retraction carries a general rule the file states itself: isolation names '
                      'what Apple SELECTS for a source construct, and that is the instruction\'s '
                      'meaning only when the lowering is ONE INSTRUCTION LONG. Here it is not - '
                      'sqrt(x) is x * rsqrt(x), so the multiply that follows is the algorithm and '
                      'not a rounding step, and SQRT IS NOT AN INSTRUCTION on this machine',
                ledger_status='executed - three inputs on hardware, against a name the same file '
                              'withdraws',
                note='THE LIMIT IS IN THE SAME FILE AND IS NOT CLOSED: codes 2 and 3 - op3978 and '
                     'op3850 - BOTH return 1/sqrt on 2.75, and what separates them is not '
                     'established. So this fact names op3978 and does not distinguish it from its '
                     'twin. Third instance of this shape in the repository: op9986 was named clz '
                     'and computes msb, op10822 was named multiply, now this'),
           dict(axis='opcode_selection', evidence='oracle',
                source='isa/g17-transcendental-unit.toml [the_table]',
                state='the transcendental unit\'s operation is a THREE-BIT code at byte6[2:0], '
                      'swept through Apple\'s decoder from one encoding with everything else '
                      'held: 0 rint, 1 recip, 2 rsqrt, 3 rsqrt, 4 log2, 5 exp2, 6 trig, 7 does '
                      'not decode. The control that makes it a fact about the ISA rather than '
                      'about the decoder is Apple\'s own emission: exp2 carries byte6 = a5, log2 '
                      'a4, sqrt a2, rsqrt a3 and the sine primitive a6 - five of five',
                ledger_status='oracle - the decoder sweep cross-checked against what Apple\'s '
                              'compiler emits',
                note='code 6 is named `trig` and NOT `sin`: it is the one instruction common to '
                     'sin, cos, sinpi, cospi, fast::sin, fast::cos and precise::sin, the sin '
                     'lowering is twelve instructions, and what separates sine from cosine is '
                     'later - op9710 against op9711. What code 6 computes alone is not established')],
    5106: [dict(axis='operand_encoding', evidence='decoder',
                source='isa/tensor-isa.toml:31,59,108 vs linker/g17-tensorops-recon @ e9ec9f603b22',
                state='PARTLY CORROBORATED, PARTLY A SCOPE QUESTION - not a settled '
                      'contradiction. This repository\'s OWN bit ledger already records '
                      'byte8[1] as class=opcode with flips_to=5107 and byte6[0] as '
                      'class=opcode with flips_to=10384, so the peer lane\'s selector '
                      'claims for the no-accumulator and int8 forms are confirmed here '
                      'from independent data. byte8[2] and byte8[3] are class=operand at '
                      'op5106/len10, consistent with their transA/transB reading',
                ledger_status='isa/g17-bit-spec.jsonl and isa/g17-form-spec.jsonl, '
                              'independent of the peer branch',
                note='I first recorded isa/tensor-isa.toml:31 as contradicting this, and '
                     'that was over-stated. Its `inert = [byte8[0], byte8[2]]` sits in a '
                     'top-level [field_encoding] block whose own comment says "Both '
                     'families share ONE width field" - the two tensor.bound families at '
                     'unit_bytes 12 - while tensor.mac is unit_bytes 10. The same bit '
                     'position at two widths is exactly where this repository has been '
                     'burned before, so the inert claim is probably about the BOUND '
                     'form and neither contradicted nor confirmed by an op5106 check. '
                     'What IS about tensor.mac is the signature at :59 requiring byte8 '
                     '== 0x00, and a no-accumulator or transposed MMA breaks that - a '
                     'census of 112 that could not have refuted it, since that corpus '
                     'contains neither form'),
           dict(axis='legal_forms', evidence='table',
                source='isa/g17-contract.jsonl, checked against '
                       'linker/g17-tensorops-recon @ e9ec9f60',
                state='CORROBORATED INDEPENDENTLY. The peer lane reports the operand list '
                      'D:tup8, flags, imm, A:tup4, A-flags, A-type, B:tup4, B-flags, '
                      'B-type, C:tup8, imm, imm with D == C always. Apple\'s descriptor, '
                      'read here without reference to their work, gives nops=12, ndefs=1, '
                      'reads tup4/tup4/tup8 and writes tup8 - the same structure',
                ledger_status='two instruments agreeing: their field map and encoder, and '
                              'Apple\'s MCInstrDesc operand list',
                note='the sharper corroboration is op5107, their no-accumulator form. '
                     'Apple\'s descriptor gives it nops=9 with reads tup4/tup4 and NO '
                     'tup8 read at all, so the accumulator operand is absent from the '
                     'descriptor itself. Their "C ignored" claim did not need their '
                     'measurement to be believed - the table already says it, which is '
                     'what makes this agreement worth recording rather than copying'),
           dict(axis='latency_wait', evidence='decoder',
                source='ledger/g17-tensor-mac-wait-token.toml',
                state='the scattered tag is a WAIT TOKEN allocated in load order: a MAC takes the '
                      'next tag exactly when it consumes a load. An eight-state field whose bits '
                      'are scattered, not a one-hot in byte1',
                ledger_status='structural, exact on all ten shapes below the cap, and it fails on '
                              'all six at or above it',
                note='the failure above the cap is part of the claim, not an exception to it')],
    10094: [dict(axis='operand_encoding', evidence='decoder',
                 source='isa/g17-atomic-family.toml',
                 state='the atomic OPERATION is a four-bit SCATTERED field - b4[5], b5[3], b6[3], '
                       'b7[7] - not a contiguous opcode field. 208 opcodes share this opcode\'s '
                       '(schedclass, tsflags): schedclass 286 or 287, tsflags 0x402002400',
                 ledger_status='established by sweeping all sixteen combinations through Apple\'s '
                               'decoder, each bit flipped on a compiled op10094 and the resulting '
                               'opcode read back, then confirmed over all 64',
                 note='the companion NEGATIVE matters as much: 179 of the 208 declared opcodes in '
                      'this family were never produced by the decoder, so the declared family and '
                      'the reachable family are different populations'),
            dict(axis='legal_forms', evidence='decoder',
                 source='isa/g17-atomic-family.toml',
                 state='the family\'s structural axes were mapped by flipping each on a compiled '
                       'op10094; result-wanted and device-versus-threadgroup are separate axes',
                 ledger_status='confirmed by sweeping all 64 combinations',
                 note='179 of 208 declared members are not decoder-reachable'),
            dict(axis='semantics', evidence='executed', widths=(10,),
                 source='ledger/g17-the-atomic-works-and-the-observable-did-not.toml',
                 state='the UNIFORM-address device atomic executes: 32 lanes on one address '
                       'return exactly the set {0..31}, so each lane observed a distinct prior '
                       'value - this one does serialise, unlike the per-lane form at a uniform '
                       'address',
                 ledger_status='executed; the cited ledger is a RETRACTION that withdraws the '
                               'earlier inertness readings, which had judged buffer 0 - a buffer '
                               'the harness uploads and never copies back. Registered through '
                               'compiler.atomics.device, whose execution layer cites '
                               'compiler.work.e2e_atomicadd, e2e_waveagg and e2e_wavebcast',
                 note='the {0..31} result came through Apple\'s compiler on the same source; the '
                      'executed bytes of this compiler\'s own version were not retained, so the '
                      'form is attributed by a current compile and appears in '
                      'forms_without_isolated_record. The ORDER is a race and is compared as a '
                      'multiset, which is the contract the hardware owes')],
    # THE ATOMIC AXIS WAS ZERO BECAUSE THIS MAP NEVER READ THE EVIDENCE, NOT BECAUSE NOTHING
    # EXECUTED. Found 2026-09-17 while building a fresh atomic probe: the repository has carried
    # hardware-executed device and threadgroup atomics since 2026-09-08, registered in
    # results/g17-capability-ledger.json as `compiler.work.e2e_atomicadd` and
    # `compiler.work.e2e_tgatomic`, kind "workload GPU execution". Goal item 4 asked for a probe
    # to move this off zero; what moved it was reading what was already there. The probe's own
    # contribution is a NEGATIVE, recorded in
    # ledger/g17-the-atomic-probe-was-never-writing-memory.toml.
    #
    # THE LIMITATION IS THE CAPABILITY LEDGER'S OWN WORDS AND IT IS NOT DROPPED IN TRANSIT: "its
    # executed bytes were not retained, so the forms are those of the CURRENT compile", and both
    # capabilities list these forms under `forms_without_isolated_record`. So what executed is a
    # WHOLE PROGRAM agreeing with Apple's compilation and with the arithmetic - not this form in
    # isolation, and not with the executed bytes in hand. That is the `executed` rung, not
    # `checked`, and the attribution is by re-compilation rather than provenance.
    10090: [dict(axis='semantics', evidence='executed', widths=(10,),
                 source='ledger/g17-the-atomic-works-and-the-observable-did-not.toml',
                 state='a device atomic emitted by this compiler increments MEMORY: A[t] += 7 at '
                       'a per-lane address, loaded back in the same kernel and stored to C, '
                       'giving C = 7 in every one of 32 lanes; Apple\'s own compiler agrees on '
                       'the same source',
                 ledger_status='executed, and registered as compiler.work.e2e_atomicadd (kind '
                               '"workload GPU execution") with both /kernels/atomicadd/'
                               'agree_with_apple and agree_with_arithmetic true. The cited ledger '
                               'is itself a RETRACTION: it withdraws three earlier entries, '
                               'g17-the-atomic-is-inert.toml among them, whose "inert" reading '
                               'came from judging buffer 0 - which the harness uploads and never '
                               'copies back. The surviving half is the positive result relied on '
                               'here; the withdrawn half is the inertness',
                 note='the executed BYTES were not retained, so this form is attributed by a '
                      'current compile and the capability ledger lists it under '
                      'forms_without_isolated_record. A whole program agreeing with its reference '
                      'does not isolate one instruction, so this is execution of a program '
                      'CONTAINING the form, which is why it sits at `executed` and not `checked`. '
                      'It also does not prove serialisation: every lane touches its own slot, so '
                      'nothing contends'),
            dict(axis='resources', evidence='executed',
                 source='ledger/g17-the-atomic-works-and-the-observable-did-not.toml',
                 state='the buffer a device atomic targets must be BOUND in the image: an '
                       'unbound rank returns the fill value at status 0, indistinguishable from '
                       'a broken opcode, and the binding list has to come from the compiled '
                       'function rather than from a constant',
                 ledger_status='executed; the same ledger withdraws the readings that this '
                               'defect had manufactured',
                 note='this is the rule the fresh probe of 2026-09-17 violated by patching code '
                      'into a vendor host container, inheriting that host\'s declarations')],
    11765: [dict(axis='semantics', evidence='executed', widths=(12,),
                 source='ledger/g17-the-threadgroup-atomic-subtracts-and-needs-a-region.toml',
                 state='the threadgroup atomic executes: it SUBTRACTS, its operation is not a '
                       'writable field, and it needs the threadgroup region an ordinary load or '
                       'store allocates',
                 ledger_status='executing - the end-to-end kernel agreed with Apple and with the '
                               'arithmetic on the GPU; registered as compiler.work.e2e_tgatomic, '
                               'kind "workload GPU execution"',
                 note='executed bytes not retained, so the form is attributed by a current '
                      'compile and the capability ledger lists (11765, 12) under '
                      'forms_without_isolated_record; the region requirement is why a missing '
                      'threadgroup binding returns zeros that look like a broken opcode'),
            dict(axis='resources', evidence='executed',
                 source='ledger/g17-the-threadgroup-atomic-subtracts-and-needs-a-region.toml',
                 state='a threadgroup atomic needs BOTH a threadgroup binding in the image and a '
                       'threadgroup length at dispatch; missing either returns zeros',
                 ledger_status='executed end to end',
                 note='derived from the emitted opcodes rather than from a flag beside the '
                      'kernel, so a program that starts using threadgroup memory cannot forget '
                      'to ask for it')],
    10022: [dict(axis='operand_encoding', evidence='decoder',
                 source='isa/g17-atomic-family.toml',
                 state='a member of the atomic family whose OPERATION is the same four-bit '
                       'scattered field - b4[5], b5[3], b6[3], b7[7]',
                 ledger_status='established by sweeping all sixteen combinations through Apple '
                               'decoder and confirmed over all 64',
                 note='shares (schedclass, tsflags) with op10094; 179 of the 208 declared '
                      'members of that family were never produced by the decoder')],
    11462: [dict(axis='immediates', evidence='decoder',
                 source='isa/g17-condition-codes.toml',
                 state='the register-form integer compare; its condition code is an operand in a '
                       'namespace SHARED with select, not a property of the compare instruction '
                       '(cc 9 unsigned less-than, cc 10 unsigned greater-than, ...)',
                 ledger_status='one relational operator per kernel, twenty-four kernels, reading '
                               'the code operand back',
                 note='this opcode was once called `carry`, from a 64-bit add where it sits '
                      'between the words: right about the use, wrong about the instruction, since '
                      'a 32-bit carry-out is exactly an unsigned less-than. The contract ledger '
                      'already carries the corrected name `cmp`')],
    11452: [dict(axis='immediates', evidence='decoder',
                 source='isa/g17-condition-codes.toml',
                 state='the immediate-form integer compare, same shared condition-code namespace',
                 ledger_status='one relational operator per kernel, twenty-four kernels',
                 note='register and immediate forms are different opcodes, not one form with a '
                      'mode bit')],
    9787: [dict(axis='immediates', evidence='decoder',
                source='isa/g17-condition-codes.toml',
                state='the float compare, in a DIFFERENT condition-code namespace from the '
                      'integer one: cc 0 equal, cc 1 less-than, cc 2 greater-than',
                ledger_status='one relational operator per kernel, twenty-four kernels',
                note='the integer codes do not apply here; treating one table as universal is '
                     'the mistake this ledger exists to prevent')],
    11365: [dict(axis='immediates', evidence='decoder',
                 source='isa/g17-condition-codes.toml',
                 state='select carries a condition code from the SAME namespace as the compares, '
                       'so a composer can compute a code rather than look one up',
                 ledger_status='one relational operator per kernel, twenty-four kernels',
                 note='one namespace across the instruction set, not a property of the compare '
                      'instructions')],
    12674: [dict(axis='resources', evidence='decoder',
                 source='ledger/g17-tensor-operand-load-count-law.toml',
                 state='the tensor operand load count follows an exact linear law in the tile '
                       'shape, not the aspect ratio: count(op12674) - count(op12675) = 6n, which '
                       'depends on N alone, and op12674 grows by 7 per 16 of N and 4 per 16 of M',
                 ledger_status='structural, exact - 13 of 16 corrected shapes fit with zero '
                               'residual; THREE NAMED MISSES',
                 note='this ledger is itself a RETRACTION: it withdraws '
                      'ledger/g17-tensor-load-mix-tracks-aspect.toml, which claimed the mix '
                      'follows tile aspect. The three misses are named in the source and are '
                      'part of the claim, not exceptions to it')],
    12675: [dict(axis='resources', evidence='decoder',
                 source='ledger/g17-tensor-operand-load-count-law.toml',
                 state='the second operand load in the same count law; its difference from '
                       'op12674 is 6n and depends on N alone',
                 ledger_status='structural, exact on 13 of 16 shapes with three named misses',
                 note='replaces a withdrawn aspect-ratio reading')],
    12688: [dict(axis='resources', evidence='decoder',
                 source='ledger/g17-constant-address-load-is-in-the-prologue.toml',
                 state='a load from a CONSTANT address is resolved in the constant program and '
                       'not in main; this opcode carries the index at 128 per element',
                 ledger_status='structural, by compilation differential over twelve constant '
                               'indices, where main was identical across all twelve',
                 note='the authoring consequence is the point: a composer that sees u[7] and '
                      'looks for an addressing mode in the kernel body will not find one')],
    583: [dict(axis='flags_exec', evidence='decoder',
               source='ledger/g17-flag-selector-writer-and-reader-differ.toml',
               state='the flag READER carries a compact 3-bit flag index at byte1[4], byte1[5], '
                     'byte1[6] in its 4-byte form - the same position on op582, op579, op578 and '
                     'this opcode. The instruction that WRITES a flag carries the selector in '
                     'DIFFERENT bits, in a 6-byte form',
               ledger_status='structural, from the corpus; the 6-byte writer position has NOT '
                             'been executed by that side',
               note='it diagnoses a failed authoring attempt: authoring a non-zero flag did not '
                    'work because writer and reader positions were assumed to be the same. '
                    'Recorded with the writer half explicitly unexecuted')],
    17244: [dict(axis='semantics', evidence='executed', widths=(14,),
                 source='ledger/g17-a-texture-read-executes.toml',
                 state='a texture read executes from this compiler: the fetch writes and op17244 '
                       'reads the result; no constant program is required',
                 ledger_status='executed; the coordinate was fixed the same day',
                 note='this retires the earlier ledger claim that a texture read needs a constant '
                      'program, which had rested on one substitution. The coordinate delivery was '
                      'a separate defect, closed the same day. The companion ledger '
                      'g17-a-texture-kernel-needs-a-constant-program.toml is itself flagged as '
                      'carrying a retraction: it had concluded "the fault is in the image, not '
                      'the instructions" from a control described as Apple\'s exact bytes, and '
                      'that conclusion is withdrawn - Apple\'s whole 98-byte __text, constant '
                      'program and main together, reads the texture through this image builder')],
    458: [dict(axis='semantics', evidence='executed', widths=(10,),
               source='ledger/g17-the-back-edge-gates-on-op582.toml',
               state='the back edge: it repeats while lanes are active and gates on op582 alone, '
                     'not on the `while` exec Apple pairs it with',
               ledger_status='executed, 8 dispatches, every one bounded by construction',
               note='measured in a program with no reachable cycle, so the machine was never at '
                    'risk; it corrects an earlier negative that had tested op579 in front of a '
                    'FORWARD branch')],
    582: [dict(axis='semantics', evidence='executed', widths=(4,),
               source='ledger/g17-the-back-edge-gates-on-op582.toml',
               state='the conditional gate op458 depends on: taken when the predicate is true and '
                     'fell through when false, unlike op578/op579 which are taken either way',
               ledger_status='executed, 8 dispatches, every one bounded by construction',
               note='op582 is the gate; the `while` exec is not'),
          dict(axis='resources', evidence='executed',
               source='ledger/g17-the-mask-stack-is-twenty-deep.toml',
               state='op582 PUSHES an exec-mask level, and that stack is a true LIFO at least '
                     'twenty deep. A loop lowered as `body; cmp; op582; op458` therefore pushes '
                     'once per iteration',
               ledger_status='executed; every program straight-line except the back-edge probes, '
                             'which have no reachable cycle',
               note='the same ledger records that op582 does NOT gate the branch in the bounded '
                    'back-edge program, and says so as an awkward fact measured both ways rather '
                    'than reconciled by argument. That tension is carried here, not flattened')],
    577: [dict(axis='semantics', evidence='executed', widths=(4,),
               source='ledger/g17-a-counted-loop-executes.toml',
               state='the third member of the executed loop triple op582 -> op458 -> op577; a '
                     'counted loop returns the right value at 1, 3, 4, 5, 8 and 17 trips',
               ledger_status='executed; six trip counts, cb_status 0 throughout, correct value '
                             'in the requested slot',
               note='a cyclic program cannot be made safe by construction the way the back-edge '
                    'probe was, so this one ran behind a stated preflight. The first run '
                    'terminated cleanly and returned 0 instead of 1 - not the loop, the STORE - '
                    'which is why a clean termination is not by itself a correct result')],
    14061: [dict(axis='register_files', evidence='executed',
                 source='ledger/g17-the-destination-was-a-file-not-a-register.toml',
                 state='the destination is a FILE, not a register. Class 413 holds thirty-two '
                       'opcodes, sixteen named `flagsel`, and sixteen take the same operands',
                 ledger_status='executed; slot write and read-back, with the indexing left open',
                 note='HOW THE FILE IS INDEXED IS OPEN: writing slots 68 and 70 and reading them '
                      'back returned the second value and then nothing, which fits a latch or a '
                      'queue. Two instruments had to agree before this was a name and not a '
                      'guess, and neither was this project\'s own name table')],
    14120: [dict(axis='register_files', evidence='executed',
                 source='ledger/g17-the-destination-was-a-file-not-a-register.toml',
                 state='same class-413 file destination as op14061',
                 ledger_status='executed; slot write and read-back, with the indexing left open',
                 note='the indexing question is open for this opcode too')],
    10282: [dict(axis='operand_encoding', evidence='checked',
                 source='ledger/g17-alu-slot-model.toml',
                 state='an ALU operand is ONE SLOT, and `sub` is a slot ORDER - the compiler\'s '
                       'sub computed K-x because the order was wrong',
                 ledger_status='causal - four operations executed against preregistered values, '
                               'one dispatch per process',
                 note='the slot model says WHERE operands live and not what their MODIFIERS are: '
                      'each register operand is followed by a modifier this ledger does not '
                      'claim. Slot A had been recovered by correlation over Apple\'s corpus and '
                      'is executed now rather than only correlated')],
    9836: [dict(axis='semantics', evidence='executed', widths=(10,),
                source='ledger/ (extracted then read: single-construct attribution); width 10 '
                       'established by TWO independent instruments agreeing - the isolated '
                       'execution sweep\'s receipt id and Apple\'s witness byte count',
                state='computes `step`: reached by exactly one Metal construct and execution '
                      'confirmed it, which is a clean single-construct attribution',
                ledger_status='executed',
                note='the cleanest attribution shape in the corpus - one construct emits the '
                     'opcode and nothing else does, so the construct name and the semantics '
                     'cannot be confounded by a lowering that shares the opcode')],
    767: [dict(axis='legal_forms', evidence='executed',
               source='ledger/ (extracted then read: the op767 lesson)',
               state='the operand fields may be in a MODE where they hold constants rather than '
                     'registers, so the same field positions mean different things by mode',
               ledger_status='executed',
               note='this is why legal_forms and operand_encoding are separate axes here: a '
                    'field map that ignores mode describes one mode and claims all of them')],
    424: [dict(axis='operand_encoding', evidence='checked',
               source='ledger/g17-bitwise-reg-selector-was-inherited.toml',
               state='byte2 takes four values across the corpus for this opcode - 0x32, 0x12, '
                     '0x22, 0x2A - and its low three bits carry the operation selector',
               ledger_status='a preregistered FAILURE: `and` returned the xor answer, because '
                             'the operation selector was an inherited field',
               note='recorded as the negative it is. The selector was inherited from a template '
                    'rather than authored, so `and` computed xor - the same defect class as an '
                    'undeclared operand carrying a probe kernel\'s value')],
    578: [dict(axis='flags_exec', evidence='executed',
               source='ledger/g17-the-back-edge-gates-on-op582.toml',
               state='the `while` exec: taken whether the predicate is true or false, so it does '
                     'not gate on the compare',
               ledger_status='executed, 8 dispatches, every one bounded by construction',
               note='`while` sets the mask absolutely and does not nest')],
    579: [dict(axis='flags_exec', evidence='executed',
               source='ledger/g17-the-back-edge-gates-on-op582.toml',
               state='`while` with invert: taken whether the predicate is true or false, so it '
                     'does not gate on the compare either',
               ledger_status='executed, 8 dispatches, every one bounded by construction',
               note='the earlier refusal in g17cc rested on measuring this in front of a FORWARD '
                    'branch, which is why it read as not gating')],
    # THREE FORMS PROMOTED 2026-09-17 BY RECOMPUTING THE EXPECTATION, NOT BY TRUSTING IT.
    # isa/ holds 48 plan/result pairs of isolated dispatches. 17 forms have an `expect` list that
    # the observed values equal exactly, over 3 runs - but every plan was committed in the same
    # commit as its results, so nothing in the history says the expectation came first, and
    # "observed == expect" is otherwise a control that cannot fail. The three below are the ones
    # whose expectation is REPRODUCIBLE from a rule: I recomputed all four cases of each from an
    # independent reference and got the record's values. The other 13 are PRESERVATION checks
    # whose expected value is a prior observation ("the debt bits cleared, same result"), which no
    # rule regenerates; they stay in executed_unchecked and are counted in `limits`.
    # FOUR MORE, AND FINDING THEM CORRECTED MY OWN SPLIT. These forms were first classified as
    # "expectation is a prior observation" because the scan reached their deadbits3 arm - where
    # the expected value IS an earlier measurement - and I read one arm's note as the form's
    # status. A form can appear in several plan/result pairs, and isa/g17-execution-functions.json
    # holds a second arm for each of these whose expectation is a stated RULE. Recomputed here,
    # all four cases each, from an independent reference:
    #
    #     op998/12    float32 a + b        op2190/16   float32 a*b + c
    #     op3290/14   float32 a * b        op10826/14  uint32  a*b + c
    #
    # Each record's four answers are DISTINCT and none equals any of its operands, which is what
    # separates the reading from an absorbing or projecting function - the plan says so itself
    # ("three sources is where 'it returned something plausible' stops being an argument"). Each
    # encoded string holds only the target instruction, `from_template` is false, and the opcode
    # was found once per case, so "authored alone" is literal rather than a label.
    998: [dict(axis='semantics', evidence='checked', widths=(12,),
               source='isa/g17-execution-functions.json + '
                      'g17-execution-functions-results.json',
               state='op998 at 12 bytes is float32 ADD: all four cases bit-exact over 3 runs, the '
                     'instruction authored alone',
               ledger_status='executed and CHECKED - every case recomputed here as float32 '
                             'a + b and equal to the observed words',
               note='the plan notes that op998 was "delivery-proven" while its FUNCTION had never '
                    'been measured, and that those are different claims. Its OTHER arm, '
                    'deadbits3/op998.dead-bits, is a preservation check whose expected value is a '
                    'prior observation - a different and weaker claim about the same form. ' +
                    _SAME_COMMIT_CAVEAT)],
    3290: [dict(axis='semantics', evidence='checked', widths=(14,),
                source='isa/g17-execution-functions.json + '
                       'g17-execution-functions-results.json',
                state='op3290 at 14 bytes is float32 MULTIPLY: all four cases bit-exact over 3 '
                      'runs, the instruction authored alone',
                ledger_status='executed and CHECKED - every case recomputed here as float32 '
                              'a * b and equal to the observed words',
                note='case 2 carries a negative operand and case 3 a product smaller than either '
                     'factor, so neither a sign-dropping nor a max-like reading survives. ' +
                     _SAME_COMMIT_CAVEAT)],
    # SIX FORMS DETERMINED BY ELIMINATION, WITH NO STORED EXPECTATION TO COMPARE AGAINST - which
    # makes them the cleanest class here rather than the weakest. These records carry NO `expect`
    # field, so nothing could have been copied from the observation and the same-commit caveat
    # does not apply to them at all. Each has four input pairs, three runs, the instruction
    # authored alone. I computed ALL SIXTEEN two-input bitwise functions over the cases and
    # compared against the observed words: in each of the five below exactly one function fits
    # all four cases and fifteen are eliminated by the data.
    #
    # THE HYPOTHESIS CLASS IS THE LIMIT, and each fact says so: uniqueness holds among the sixteen
    # BITWISE functions of two 32-bit words. A function outside that class - one reading the
    # operands differently, or consulting a third - is not excluded by four points.
    13460: [dict(axis='semantics', evidence='checked', widths=(10,),
              source='isa/g17-execution-names.json + its -results.json',
              state='op13460 at 10 bytes computes nand = ~(a & b) over 32-bit words: four input pairs, '
                    'all bit-exact over 3 runs, the instruction authored alone',
              ledger_status='executed and CHECKED by elimination - of the sixteen two-input '
                            'bitwise functions nand is the only one reproducing all four observed '
                            'words, and the record carries NO expect field, so nothing was '
                            'compared against a stored expectation',
              note='the table does not name it, and the plan heading is DELIVERY PROOF IS NOT A NAME. The uniqueness is among the SIXTEEN BITWISE functions of two words and '
                   'nothing wider: four points do not exclude a function that reads the operands '
                   'differently or consults a third')],
    13521: [dict(axis='semantics', evidence='checked', widths=(10,),
              source='isa/g17-execution-names.json + its -results.json',
              state='op13521 at 10 bytes computes nor = ~(a | b) over 32-bit words: four input pairs, '
                    'all bit-exact over 3 runs, the instruction authored alone',
              ledger_status='executed and CHECKED by elimination - of the sixteen two-input '
                            'bitwise functions nor is the only one reproducing all four observed '
                            'words, and the record carries NO expect field, so nothing was '
                            'compared against a stored expectation',
              note='nor is symmetric in its operands, so these pairs do not order them. The uniqueness is among the SIXTEEN BITWISE functions of two words and '
                   'nothing wider: four points do not exclude a function that reads the operands '
                   'differently or consults a third')],
    13548: [dict(axis='semantics', evidence='checked', widths=(10,),
              source='isa/g17-execution-names.json + its -results.json',
              state='op13548 at 10 bytes computes orn = a | ~b over 32-bit words: four input pairs, '
                    'all bit-exact over 3 runs, the instruction authored alone',
              ledger_status='executed and CHECKED by elimination - of the sixteen two-input '
                            'bitwise functions orn is the only one reproducing all four observed '
                            'words, and the record carries NO expect field, so nothing was '
                            'compared against a stored expectation',
              note='orn is ASYMMETRIC, so the operand order is part of what these four pairs pin. The uniqueness is among the SIXTEEN BITWISE functions of two words and '
                   'nothing wider: four points do not exclude a function that reads the operands '
                   'differently or consults a third')],
    17744: [dict(axis='semantics', evidence='checked', widths=(10,),
              source='isa/g17-execution-names.json + its -results.json',
              state='op17744 at 10 bytes computes xnor = ~(a ^ b) over 32-bit words: four input pairs, '
                    'all bit-exact over 3 runs, the instruction authored alone',
              ledger_status='executed and CHECKED by elimination - of the sixteen two-input '
                            'bitwise functions xnor is the only one reproducing all four observed '
                            'words, and the record carries NO expect field, so nothing was '
                            'compared against a stored expectation',
              note='xnor is symmetric, so these cases do not order the operands. The uniqueness is among the SIXTEEN BITWISE functions of two words and '
                   'nothing wider: four points do not exclude a function that reads the operands '
                   'differently or consults a third')],
    13488: [dict(axis='semantics', evidence='checked', widths=(10,),
              source='isa/g17-execution-walked.json + its -results.json',
              state='op13488 at 10 bytes computes nandn = ~a & b over 32-bit words: four input pairs, '
                    'all bit-exact over 3 runs, the instruction authored alone',
              ledger_status='executed and CHECKED by elimination - of the sixteen two-input '
                            'bitwise functions nandn is the only one reproducing all four observed '
                            'words, and the record carries NO expect field, so nothing was '
                            'compared against a stored expectation',
              note='a two-source negated form Apple does not name in its table at all, so '
                   'this is a NAME rather than a confirmation. The uniqueness is among '
                   'the SIXTEEN BITWISE functions of two words and '
                   'nothing wider: four points do not exclude a function that reads the operands '
                   'differently or consults a third')],
    1062: [dict(axis='semantics', evidence='checked', widths=(10,),
                source='isa/g17-execution-walked.json + g17-execution-walked-results.json',
                state='op1062 at 10 bytes returns clamp(-x, 0, 1) at the walked witness source-'
                      'modifier setting: -0.25, -0.5, -0.75 and 2.0 give 0.25, 0.5, 0.75 and '
                      '0.0, bit-exact over 3 runs',
                ledger_status='executed and CHECKED - recomputed here as clamp(-x, 0, 1); of '
                              'clamp(-x,0,1), -clamp(x,0,1), clamp(abs x,0,1) and '
                              'clamp(-x,-1,1) only the first reproduces all four words, and the '
                              'record carries NO expect field',
                note='THE RECORD WAS BUILT TO DISCRIMINATE and its plan names the two readings: '
                     'with the source negate bit clear the function is clamp(-x, 0, 1), with it '
                     'set clamp(x, 0, 1) giving 0.0, 0.0, 0.0, 1.0 - "both are readable from '
                     'these four points and they are not confusable". The observed words select '
                     'the first, so this names the MODE the walked witness carries rather than '
                     'the opcode independently of its modifier, and the saturation bound is '
                     'pinned only from above, by the 2.0 case')],
    2190: [dict(axis='semantics', evidence='checked', widths=(16,),
                source='isa/g17-execution-functions.json + '
                       'g17-execution-functions-results.json',
                state='op2190 at 16 bytes is float32 FMA with three sources: a*b + c, all four '
                      'cases bit-exact over 3 runs, the instruction authored alone',
                ledger_status='executed and CHECKED - every case recomputed here as float32 '
                              'a * b + c and equal to the observed words',
                note='THREE sources is what makes this more than plausibility: a two-source '
                     'reading cannot produce these four answers, and none of them equals any '
                     'operand. Whether the product is rounded once or twice is NOT settled by '
                     'these inputs - no case was chosen to separate fused from unfused. ' +
                     _SAME_COMMIT_CAVEAT)],
    10826: [dict(axis='semantics', evidence='checked', widths=(14,),
                 source='isa/g17-execution-functions.json + '
                        'g17-execution-functions-results.json',
                 state='op10826 at 14 bytes is the INTEGER three-source multiply-add, a*b + c '
                       'modulo 2^32: all four cases exact over 3 runs, authored alone',
                 ledger_status='executed and CHECKED - every case recomputed here as '
                               '(a*b + c) & 0xFFFFFFFF and equal to the observed words',
                 note='the four answers 17, 65, 6, 101 are small and distinct, and none equals '
                      'any operand or any pairwise product, so no two-source reading fits. '
                      'Signedness is NOT settled: every case is small and positive, so a signed '
                      'multiply would give the same words. ' + _SAME_COMMIT_CAVEAT)],
    3802: [dict(axis='semantics', evidence='checked', widths=(10,),
                source='isa/g17-execution-rounding.json + g17-execution-rounding-results.json',
                state='op3802 at 10 bytes is float32 CEIL: 2.5 -> 3.0, -0.3 -> -0.0, 3.9 -> 4.0, '
                      '-1.5 -> -1.0, all four bit-exact over 3 runs, the instruction authored '
                      'alone',
                ledger_status='executed and CHECKED - every case recomputed here with numpy '
                              'float32 ceil and equal to the record\'s observed words',
                note='THE SIGNED ZERO IS THE CASE THAT CARRIES THE RESULT, and the plan says so '
                     'itself ("case 1 is the one to read rather than assert"). The hardware '
                     'returns 0x80000000 for ceil(-0.3): IEEE-754 roundToIntegral keeps the '
                     'operand\'s sign. My first reference disagreed there and was WRONG - '
                     'math.ceil returns an integer and drops the sign of zero - so the reference '
                     'has to be a float-preserving one. ' + _SAME_COMMIT_CAVEAT),
           dict(axis='flags_exec', evidence='checked',
                source='isa/g17-execution-rounding.json',
                state='no exceptional input was dispatched, so nothing here measures flag '
                      'behaviour: the four cases are ordinary finite floats',
                ledger_status='scope statement, not a measurement',
                note='recorded so the checked semantics above is not read as covering NaN, '
                     'infinity or subnormal inputs at this form')],
    3818: [dict(axis='semantics', evidence='checked', widths=(10,),
                source='isa/g17-execution-rounding.json + g17-execution-rounding-results.json',
                state='op3818 at 10 bytes is float32 TRUNC: 2.5 -> 2.0, -0.3 -> -0.0, 3.9 -> 3.0, '
                      '-1.5 -> -1.0, all four bit-exact over 3 runs, the instruction authored '
                      'alone',
                ledger_status='executed and CHECKED - every case recomputed here with numpy '
                              'float32 trunc and equal to the record\'s observed words',
                note='the same four inputs as op3802, which is what makes the PAIR informative: '
                     'ceil and trunc differ on 2.5 and 3.9 and agree on -0.3 and -1.5, so a '
                     'single absorbing function cannot produce both records. ' + _SAME_COMMIT_CAVEAT),
           dict(axis='flags_exec', evidence='checked',
                source='isa/g17-execution-rounding.json',
                state='no exceptional input was dispatched; the four cases are ordinary finite '
                      'floats',
                ledger_status='scope statement, not a measurement',
                note='so the checked semantics above says nothing about NaN, infinity or '
                     'subnormal inputs at this form')],
    10279: [dict(axis='semantics', evidence='checked', widths=(12,),
                 source='isa/g17-execution-compiler.json + '
                        'g17-execution-compiler-results.json',
                 state='op10279 at 12 bytes returns src + 4 on 4096, 4097, 4352 and 8192, '
                       'bit-exact over 3 runs - the positive control of the isolated-dispatch '
                       'instrument, authored alone',
                 ledger_status='executed and CHECKED - all four cases recomputed here as '
                               '(src + 4) & 0xFFFFFFFF and equal to the observed words',
                 note='this record is the instrument\'s own canary: the plan says "if this does '
                      'not replicate, the instrument is not running and every other result in '
                      'this file is void". The addend 4 is the authored operand, so the rule is '
                      'src + operand2 rather than a constant of the opcode. ' + _SAME_COMMIT_CAVEAT),
            dict(axis='operand_encoding', evidence='executed',
                 source='ledger/g17-two-more-fitted-constants-were-the-witness.toml',
                 state='operand 2 is the addend',
                 ledger_status='executed',
                 note='recorded by a ledger whose subject is that two fitted constants were '
                      'measuring their own witness; the operand role is the part that survived')],
    11375: [dict(axis='operand_encoding', evidence='executed',
                 source='ledger/g17-two-more-fitted-constants-were-the-witness.toml',
                 state='operand 2 is the condition',
                 ledger_status='executed',
                 note='same ledger; the select\'s fitted relation did not survive, this role did')],
}

# ISA-level facts that belong to a FIELD rather than to an opcode, so they are reported once
# instead of being attributed to forms the ledger does not name.
GLOBAL_FACTS = [
    dict(fact='the add family\'s remaining residue is causally inert: byte4[5] alone, and '
              'byte0[5] with byte6[5] written together because Apple never separates them',
         axis='unknown_bits', evidence='checked',
         source='ledger/g17-add-family-residue-is-metadata.toml',
         ledger_status='causal - both bits inverted through the canonical encoder, 15 '
                       'preregistered programs, all correct',
         why_unattributed='THIS DOES NOT RESOLVE op10279. Its remaining unforced bits at width 12 '
                          'are 4.6 and 11.0, and the bits shown inert here are byte4[5] and '
                          'byte0[5]/byte6[5] - different bits. The names are close enough that '
                          'reading this as a D2 promotion is the obvious mistake, so it is '
                          'recorded once, unattributed, and D2 is unchanged'),
    dict(fact='byte0[3] is the ALU load-use wait, and the peer lane has now placed it: it is the '
              'SLOT-7 BIT of an eight-bit wait mask at flags bits 24..31 - a load names its slot in byte4[3] + 2*byte4[4] + 4*byte6[4] and the decoder prints slot + 1 - the same mask whose '
              'slots 0..4 live in byte1[2..6]. A hardware census of op5106 (80 single flips, one '
              'process each) found clearing byte0[3] gives all zeros because the MMA reads its '
              'fragments before the loads land, which is the causal reading of the wait',
         axis='latency_wait', evidence='checked',
         source='ledger/g17-alu-load-use-wait.toml',
         ledger_status='causal - executed, positive control and a discriminating negative on the '
                       'same program',
         why_unattributed='the ledger establishes a BIT semantic over ALU instructions and names '
                          'no opcode set, so attributing it per form would invent a scope the '
                          'evidence does not have'),
]


# ------------------------------------------------------------------------------------------------
# THE CURATED isa/ LAYER, plus the three ledgers that speak to the axes LEDGER_FACTS left empty.
#
# Why this is a SEPARATE dict from LEDGER_FACTS rather than more entries in it: the two harvests
# have different provenance and should be countable apart, and appending to a 300-line literal is
# how a fact silently replaced an axis here once already.
#
# THE RULE THAT GENERATED THIS BLOCK. A fact is attributed to a form only where its source names
# an opcode SCOPE. The wait MASK over "every ALU and bitwise instruction" names no opcode set, so
# it stays in GLOBAL_FACTS unattributed; byte6[4:3] on op12674/op12675 names two, so it lands on
# those two. Attributing an unscoped fact per form would invent a scope the evidence lacks, and
# that is the difference between filling an axis and inflating it.
#
# Evidence is each source's OWN status, never upgraded. Two of these five are NEGATIVE results,
# recorded as content because a refused reading is a fact about the axis: an axis that says
# "this was tested against a discriminating control and refused" is strictly more informative
# than `absent`, and it stops the next session re-running the same test.
# op10019 `atomic.idx` IS THE COMPARE-EXCHANGE and this list did not contain it. Found by census,
# not by reading the table: 13 cache objects carry it and every one is a compare-exchange source
# (at-cmpx, atm-f-cmpxchg-dev-*, atm-i-cmpxchg-dev-*). It is op10018's neighbour, and op10018 is
# the member whose ten-byte operand map was measured here - so the family was walked straight past
# its compare-exchange. ledger/g17-the-device-atomic-is-declared-in-ld-md-byte40.toml
ATOMIC_OPS = (10018, 10019, 10022, 10023, 10090, 10091, 10094, 10095,
              11701, 11703, 11765, 11767, 11768, 11769)
TEXTURE_OPS = (10909, 14469, 14661, 14665, 14757, 15813, 15909)
TENSOR_MAC_OPS = (5106, 5107)
TENSOR_LOAD_OPS = (12674, 12675)

ISA_FACTS = {}

for _op in TENSOR_LOAD_OPS:
    ISA_FACTS.setdefault(_op, []).append(dict(
        axis='latency_wait', evidence='corpus',
        source='ledger/g17-load-counter-is-not-a-scoreboard-slot.toml',
        state='REFUSED, against a control that discriminates. byte6[4:3] is a mod-4 counter over '
              'loads in program order that advances in GROUPS rather than per instruction, and it '
              'is NOT a scoreboard slot. The defining property of a slot is reuse only after the '
              'previous occupant is consumed; over 12,840 sharing pairs byte6[4:3] scores 14.9%, '
              'position mod 4 scores 15.1% and random 0..3 scores 14.7% - indistinguishable from '
              'an arbitrary rotation. The degenerate controls show the test does discriminate: '
              'one slot for everything scores 4.4% and a different field b1[3:2] scores 5.3%. So '
              'the rotation carries the whole 15% and the field values add nothing. What survives '
              'is the description, not the reading',
        ledger_status='NEGATIVE, against a control that discriminates',
        note='recorded because the hypothesis was good and the data looked like it: without the '
             'position-mod-4 control the 14.9% would have been reported as "loads mostly respect '
             'slot discipline". A test named after a thing is not a test OF that thing'))
    ISA_FACTS.setdefault(_op, []).append(dict(
        axis='barrier_sync', evidence='corpus',
        source='ledger/g17-tensor-wait-token-is-a-rendezvous.toml',
        state='the LOAD publishes the consuming MAC\'s wait-token PARITY, inverted: '
              'b6[3] = 1 - (k & 1), holding on 382 of 384 load/MAC pairs across sixteen shapes. '
              'The controls are the point - a load\'s POSITION predicts the token at 50.3% and a '
              'shuffled control at 50.0% - so the agreement is not two counters advancing '
              'together; the load and the MAC write the same thing. ONLY ONE BIT IS SHARED: '
              'token bits 1 and 2 have no counterpart anywhere in the load encoding above 85%, '
              'searched over every bit',
        ledger_status='structural, against a shuffled control and a positional control that both '
                      'sit at chance',
        note='the polarity was nearly missed: the scoring function max(agree, 1 - agree) reports '
             '99.5% either way, and counted directly it is the COMPLEMENT. What the other two '
             'token bits select is open'))

for _op in TENSOR_MAC_OPS:
    ISA_FACTS.setdefault(_op, []).append(dict(
        axis='barrier_sync', evidence='corpus',
        source='ledger/g17-tensor-wait-token-is-a-rendezvous.toml',
        state='the MAC\'s wait token is a RENDEZVOUS with the load that feeds it, not an ordering '
              'counter: the load carries the token\'s low bit inverted (b6[3] = 1 - (k & 1)). The '
              'earlier allocation rule - a MAC takes the next token when it consumes a load newer '
              'than anything consumed before it in the repeat - is exact on all ten shapes with '
              'M*N <= 1536 and wrong on all six at or above 2048, and the boundary now has a '
              'cause: tokens are REUSED within a single repeat exactly there (32x64 reuses 3, '
              '48x48 reuses 10, 48x64 reuses 15) and zero times below it',
        ledger_status='structural, against a shuffled control and a positional control that both '
                      'sit at chance',
        note='what token bits 1 and 2 select is not established'))

for _op in ATOMIC_OPS:
    ISA_FACTS.setdefault(_op, []).append(dict(
        axis='barrier_sync', evidence='oracle',
        source='isa/g17-atomic-family.toml [selection_for_a_backend]',
        state='MEMORY ORDER IS NOT ENCODED, and this is an absence with a mechanism rather than a '
              'gap in the search: Metal REJECTS memory_order_seq_cst and memory_order_acq_rel on '
              'device atomics on this target, so relaxed is the only ordering the compiler can '
              'express and no field varies with it. Ordering therefore cannot be read out of, or '
              'written into, an atomic encoding by this project',
        ledger_status='oracle - what Apple\'s own compiler will and will not accept, not a '
                      'corpus census',
        note='so a backend authoring an atomic supplies no ordering field; it is the front end '
             'that must refuse the stronger orderings. atomic_load_explicit does not use this '
             'family at all (it lowers to an ordinary load, op12688) and atomic_store_explicit '
             'lowers to op10022 carrying the EXCHANGE code with the result discarded'))

# THREADGROUP-SCOPE atomics: b4[1] = 1 selects the threadgroup address form (op10094 <-> op11769),
# so these are the threadgroup members of the family the device opcodes above belong to.
#
# SIX MEMBERS, NOT THREE, and the three added were established by a census rather than by the
# encoding. Cross-tabulating __GPU_LD_MD byte 40 bit 2 - the device-atomic declaration - against
# what each object's text decodes to left 117 objects looking like "a device atomic with the
# declaration clear". They carry ONLY op11701 (8), op11703 (92) and op11765 (17), and every one of
# their names is *-tg-*. Reclassifying those three as threadgroup removes all 117 exceptions at
# once, which is the evidence; the names are only what made it worth checking.
# ledger/g17-the-device-atomic-is-declared-in-ld-md-byte40.toml
THREADGROUP_ATOMIC_OPS = (11701, 11703, 11765, 11767, 11768, 11769)
DEVICE_ATOMIC_OPS = tuple(op for op in ATOMIC_OPS if op not in THREADGROUP_ATOMIC_OPS)

# ------------------------------------------------------------------------------------------------
# ESTABLISHED HERE while integrating the tensor lowering, not read out of a ledger. Recorded with
# that provenance because the reader should know these came from an authoring path's refusals and
# a compile, not from someone else's census.
READ_SR_OPS = (14059, 14060)
INT8_MMA_OPS = (10384, 10385)

for _op in READ_SR_OPS:
    ISA_FACTS.setdefault(_op, []).append(dict(
        axis='resources', evidence='decoder',
        source='agxforge/g17/authorobj.py:_metadata, refusal text (established here 2026-09-17)',
        state='A PROGRAM READING A SYSTEM REGISTER OWES THE SECTION A SLOT-29 ENTRY, and the '
              'measured map has NINE members: 156, 157, 158, 160, 161, 162, 164, 165 and 166, '
              'each measured on objects reading that register alone. SR 61 - SR_TP_IN_GRID_X, '
              'thread_position_in_grid - is NOT among them. The authoring path refuses rather '
              'than approximating: "no measured slot-29 entry for system register 61 ... the '
              'general serializer emits no slot-29 vector, so authoring here would silently drop '
              'the declaration"',
        ledger_status='decoder - the authoring path\'s own refusal, naming its measured map',
        note='WHETHER 61 IS ABSENT BECAUSE NOTHING MEASURED IT OR BECAUSE THE CLASS CANNOT CARRY '
             'IT IS OPEN, and the refusal is careful not to decide. Nine measured and one named '
             'absent is the whole of what is known. The nine are contiguous except for 159 and '
             '163, which is not explained either. '
             'AND THE KEY ITSELF IS WRONG, measured 2026-09-17: the map is keyed on '
             '`decode_sr`\'s `sr`, which is byte1 alone, and byte1 does not identify the '
             'register. Over every cache object with exactly one read_sr and one slot-29 entry, '
             '(byte1, width, byte3) gives 35 keys with ZERO mapping to more than one entry, while '
             '(byte1, width) gives 20 keys of which FOUR are ambiguous - byte1 0x80, 0x81, 0x82 '
             'and 0x83 at width 8, e.g. 0x82 -> {52: 99, 50: 9, 10: 4}. The nine measurements are '
             'not thereby wrong, each having been taken on a population narrow enough to be '
             'unambiguous; the KEY is. Which bits of byte3 carry the register is NOT established - '
             '0x38, 0x20 and 0x40 differ in bits 3, 4 and 6 - so the measured rule is keyed on the '
             'whole byte and a subset would be a fit, not a measurement. '
             'ledger/g17-the-system-register-is-byte1-and-byte3.toml. '
             'AND THE SINGLETON REQUIREMENT IS OURS, NOT THE CLASS\'S: over the 89 cache objects '
             'whose text carries a tensor MAC, slot-29 vectors of ONE, TWO and THREE entries '
             'appear in 24, 17 and 48 of them, with extent exactly 4 + 4*count. Apple\'s tensor '
             'class carries multi-entry vectors routinely; tensormetadata.layout() refuses them. '
             'No pair can measure this - one more register read needs one more instruction, so '
             'grouping by identical instruction count and text length yields zero groups '
             'differing in entry count - which is why it is read from each object\'s structure '
             'instead. ledger/g17-the-tensor-class-already-carries-three-slot29-entries.toml'))

for _op in TENSOR_MAC_OPS:
    ISA_FACTS.setdefault(_op, []).append(dict(
        axis='legal_forms', evidence='decoder',
        source='agxforge/g17/abi.py ExecutionABI + scanlink.author_view (established here 2026-09-17)',
        state='ABI v5 IS THE TENSOR ABI, and the contract enforces that SYMMETRICALLY: a program '
              'executing tensor forms MUST carry the v5 execution block, and a program with no '
              'tensor forms must NOT ("an execution requirement on a program with no tensor '
              'forms: the field is stated only where the lane layout requires it"). scanlink '
              'refuses abi_version 5 without a captured tensor contract, so a buffer-only program '
              'declares v4 and omits the block entirely',
        ledger_status='decoder - the typed contract\'s own validation, exercised in both directions',
        note='the symmetry is the useful part: it means the execution block is not an optional '
             'annotation but a claim about the program, and asserting tensor=True of a program '
             'with no tensor forms is refused by name rather than serialised. Learned by trying '
             'exactly that'))

for _op in INT8_MMA_OPS:
    ISA_FACTS.setdefault(_op, []).append(dict(
        axis='opcode_selection', evidence='oracle',
        source='agxforge/g17/tensorgemm.py lowering, decoded emission (established here 2026-09-17)',
        state='THE INT8 MMA IS A DIFFERENT OPCODE, not op5106/op5107 with a type field: an int8 '
              'GEMM emits op10384 where the half GEMM emits op5106, twelve of them at 32x32x64 '
              'against the half form\'s twelve, with the same surrounding shape (op12656 loads '
              'where half uses op12674, op17257 stores, 16 MMA-family instructions either way)',
        ledger_status='oracle - the lowering\'s emission decoded at three shapes, not a census',
        note='recorded because it cost me a false alarm worth generalising: I filtered a decoded '
             'int8 program for op5106/op5107, found none, and briefly believed the int8 path '
             'emitted no multiply at all. The filter was wrong, not the code. An opcode-set filter '
             'over a family whose members differ BY TYPE will read as a missing capability'))

for _op in THREADGROUP_ATOMIC_OPS:
    ISA_FACTS.setdefault(_op, []).append(dict(
        axis='resources', evidence='corpus',
        source='isa/g17-resource-abi.toml [class.threadgroup] + [required.write_and_threadgroup_use]',
        state='a threadgroup atomic costs a threadgroup ALLOCATION that the instruction stream '
              'cannot decide, and the rule is a law in ONE DIRECTION ONLY. Slot 18\'s ABSENCE is '
              'exceptionless over both populations - 30,548 of 30,548 - so absence of the slot '
              'proves absence of threadgroup use. Its PRESENCE is a cell the two writers disagree '
              'on: an allocation implies it on all 9,556 Apple sections and has 84 corpus '
              'counterexamples, every one a tgatomic_load kernel that declares threadgroup memory '
              'and only LOADS from it. So presence needs `uses_threadgroup` from the front end and '
              'is REFUSED on this side for 851 sections',
        ledger_status='corpus - measured on 9,556 Apple and 20,992 corpus __compute sections, '
                      'with the refusal stated rather than defaulted',
        note='the asymmetry is the content: a declaration is not a use, and the counterexamples '
             'are all the same shape - declaring threadgroup memory and never writing it. A '
             'harness that bound zero threadgroup bytes would drop such a store silently'))

for _op in TENSOR_MAC_OPS:
    ISA_FACTS.setdefault(_op, []).append(dict(
        axis='resources', evidence='corpus',
        source='isa/g17-resource-abi.toml [class.tensor]',
        state='the per-kernel cost of a tensor MMA is a BOOLEAN and nothing more, and this is a '
              'measured negative rather than an unexplored axis: over 528 corpus tensor sections '
              'NO per-kernel field is a function of M, N or K. The only slots that are functions '
              'of them are 15, 16 and 44, all constant 1 - and a constant is a function of every '
              'input, which is why they do not count. Slot 44 is the marker, value 1 wherever '
              'present, and it must come from IR semantics: a TRANSPOSED matmul emits zero tensor '
              'units and still carries it, so it cannot be inferred from whether a tensor block '
              'was emitted',
        ledger_status='corpus - 528 tensor sections, with the three constant-1 slots named and '
                      'excluded for being constants',
        note='so a backend authoring an MMA owes the linker one boolean and no shape. The '
             'accompanying limit is that nothing in the vectors this builder composes records a '
             'tensor record at all'))

ISA_FACTS.setdefault(14120, []).append(dict(
    axis='resources', evidence='table',
    source='isa/g17-resource-abi.toml [class.acceleration_structure]',
    state='REFUSED, with the population named: 854 Apple sections carry an acceleration structure '
          'and they use a DIFFERENT internal binding ordering, detectable by indices in the '
          '0x83d0 namespace. Nothing here authors one',
    ledger_status='table - declared and counted, never built',
    note='recorded so the ray-tracing family reads as a refusal with a known shape rather than '
         'as an axis nobody looked at'))

# LATENCY_WAIT, FROM A FILE THIS TOOL HAD NEVER READ. `unharvested_sources` lists 21 isa/ files
# under `not_consulted` with the note that an axis reading `absent` may be answered there. Checked
# rather than assumed: for the SIMD reduction block it answers nothing, because every axis it
# speaks to is already populated at an equal or better evidence level - "not read" is not the same
# as "unexploited". isa/g17-scalar-isa.toml is different: it carries wait and scoreboard content
# for three forms whose latency_wait reads `absent` on every width, and the content is mostly
# NEGATIVE, which is the half this axis most needs.
#
# EACH FACT NAMES ONE WIDTH, because the source's own form names carry it -
# `load.through.pair.14`, `load.vec4.indexed.8`, `store.vec4.indexed.8`. A file that states the
# width is a file that cannot be spread across an opcode's other forms by accident, which is the
# defect three columns of this map have already had.
ISA_FACTS.setdefault(12688, []).append(dict(
    axis='latency_wait', evidence='decoder', widths=(14,),
    source='isa/g17-scalar-isa.toml [instruction load.through.pair.14]',
    state='the wait is LOCATED AND NOT RECOVERED: bytes 4..5 hold a composite the decoder prints '
          'inside one immediate, reading 0x01, 0x03 and 0x07 on the members\' LAST load - one bit '
          'per earlier load - and the rule behind it is not known, so this compiler inherits the '
          'value from the witness at that position',
    ledger_status='decoder sweep on the S1/S2/S3 witnesses '
                  '(results/g17-s23-whole-program-v1/sweeps/ld14_*.json); no hardware',
    note='the destination and pair fields ARE located by the same sweep, so this is a wait field '
         'whose carriers are known and whose semantics is not - the useful shape for this axis, '
         'because it says where to look rather than that nobody looked'))
ISA_FACTS.setdefault(12709, []).append(dict(
    axis='latency_wait', evidence='oracle', widths=(8,),
    source='isa/g17-scalar-isa.toml [instruction load.vec4.indexed.8]',
    state='LENGTH IS NOT AN OPERAND HERE, IT IS THE WAIT: Apple picks 8 bytes after an ALU and 14 '
          'after a special-register read or a store, through the same unrecovered composite in '
          'bytes 4..5 - so the choice of width encodes a dependency the rule for which is not '
          'recovered',
    ledger_status='oracle - Apple\'s own selection across the vector-memory witnesses '
                  '(results/g17-vector-memory-v1)',
    note='the consequence is a named refusal rather than a guess: this compiler emits ONE vector '
         'load per program and refuses a second by name, because it cannot compute the composite '
         'the second would need'))
ISA_FACTS.setdefault(17256, []).append(dict(
    axis='latency_wait', evidence='corpus', widths=(8,),
    source='isa/g17-scalar-isa.toml [instruction store.vec4.indexed.8]',
    state='this form has NO load-wait: every witness stores ALU results, and a value taken '
          'straight from a load or a texture fetch is refused rather than emitted',
    ledger_status='corpus - the witness population carries no such store',
    note='an absence established from the witnesses rather than from a measurement, which is why '
         'it is recorded at corpus and not higher: nothing here shows the hardware would fault, '
         'only that Apple never asks it to'))

# THE FLAG REGISTER IS AN ALLOCATABLE RESOURCE WITH A HARD ENCODING BOUND, from
# isa/g17-special-registers.toml - another file `not_consulted` listed, and this one does answer an
# axis that was absent for these opcodes.
#
# The bound is the interesting part and it is not a corpus observation: Apple's FLAGwritable class
# declares FIFTEEN registers, FLAG0 through FLAG14, and the ENCODING is a compact three-bit index,
# so only FLAG0..FLAG6 are reachable in these forms whatever the class declares. A compiler that
# trusted the class list would allocate registers it cannot name.
#
# THE FILE ALSO CORRECTS ITSELF IN EXACTLY THIS PROJECT'S RECURRING SHAPE, and the correction is
# quoted into the fact because it is the reason to trust the rest: it first said there are SEVEN
# flags, "which was read off the registers this corpus USES", and the class list is the authority -
# "seven was a property of the sample".
_FLAG_OPS = (575, 579, 582, 583, 10369, 10370, 10372, 10378, 10381)
for _op in _FLAG_OPS:
    ISA_FACTS.setdefault(_op, []).append(dict(
        axis='resources', evidence='corpus',
        source='isa/g17-special-registers.toml',
        state='this opcode reads an ALLOCATABLE FLAG REGISTER, not a single condition code, so '
              'flags carry register pressure. Apple\'s FLAGwritable class declares fifteen '
              '(FLAG0..FLAG14) and FLAGR adds FLAGTRUE and FLAGFALSE, but the encoding is a '
              'compact THREE-BIT index - FLAGn encodes as n and 6 encodes FLAGTRUE, which sits at '
              'class index 15 - so only FLAG0..FLAG6 are reachable in these forms whatever the '
              'class declares. FLAGTRUE and FLAGFALSE are constants rather than allocatable: they '
              'appear only as sources, and op575 reading FLAGTRUE is how an unconditional region '
              'is expressed',
        ledger_status='corpus, decode-side over 679 objects: 200 special registers defined, 26 '
                      'read, and only FLAG0..FLAG5 among them. The class membership is Apple\'s '
                      'own MCRegisterInfo (table), the three-bit bound is the encoding (decoder), '
                      'and the read counts are the corpus - the file says plainly that nothing in '
                      'it is executed',
        note='THE FILE CORRECTS ITSELF IN THIS PROJECT\'S RECURRING SHAPE and the correction is '
             'why the rest is trustworthy: it first recorded SEVEN flags, read off the registers '
             'the corpus USES, and notes that "seven was a property of the sample" while the '
             'class list is the authority. Two counts are left unreconciled rather than averaged: '
             'the prose says EIGHT opcodes carry the flag index, while the per-register read_via '
             'lists name NINE distinct opcodes, of which this is one. And the source names no '
             'WIDTHS, so this is attached opcode-wide: whether every form of this opcode carries '
             'a flag operand is inherited from the opcode here, not established per form'))

for _op in TEXTURE_OPS:
    ISA_FACTS.setdefault(_op, []).append(dict(
        axis='resources', evidence='executed',
        source='isa/g17-resource-abi.toml [class.texture] + [class.texture_buffer]',
        state='a texture costs INTERNAL BINDINGS, and the requirement is established on hardware: '
              'every section with texture state names at least one internal binding - 6,495 of '
              '6,495 Apple and 399 of 399 corpus, no exceptions - the ordinary set being [44, 48] '
              '([44, 45, 48] is commonest in Apple\'s own kernels at 4,456 sections and the extra '
              'members are NOT explained), with texture_buffer adding 46. An authored texture '
              'kernel whose section was built to this rule loads, creates a pipeline and runs at '
              'status 0; the earlier section built to "an ordinary texture emits no record at all" '
              'HUNG THE GPU. The consequence for the instruction stream is that the two internals '
              'take ranks 0 and 1, so ADDING A TEXTURE RENUMBERS EVERY USER BUFFER',
        ledger_status='executed - a clean dispatch at status 0, and a device hang from the '
                      'section that omitted the internals',
        note='the per-kernel texture-state SIZE (slot 38) stays REFUSED: its BASE is measured at '
             'one access (1d 4, 2d 8, 3d 16, cube 16, ms 16, over five probes with ty-2d matching '
             'tu-use1 as a control) but base x accesses fits some families and not others - cube '
             'gives 16, 32, 32 where the third would be 64. Its input is the number of texture '
             'INSTRUCTIONS, which exists only after instruction selection, so neither end of the '
             'pipeline can derive it'))


# ------------------------------------------------------------------------------------------------
# ITEM 3 CONTINUED: latency_wait, read for CONTENT out of four ledgers nothing had cited.
#
# The axis stood at 5 filled forms. These four sources all speak to it and none had been harvested:
# the store's own wait bit, the indexed store that has none, the five-bit wait MASK, and the
# negative that closes off latency information altogether.
#
# A LIMITATION THAT IS STRUCTURAL, NOT AN OVERSIGHT: both fact dicts are keyed by OPCODE, so a
# fact reaches every WIDTH of that opcode - op17235/len8 carries the store-wait fact although len8
# is precisely the width that cannot wait. The facts below therefore state their width scope in
# their own text, which is what a reader of any one record sees. Re-keying the harvest on
# (opcode, width) would be the real fix and is not done here.
#
# THE STORE'S OPCODE WAS NOT IN ITS LEDGER, so it was resolved with a second instrument rather than
# guessed. ledger/g17-store-load-dependency.toml names only the FORM ("store.14"), and the sub-form
# table in agxforge/g17/asm.py maps byte5[3:2] to four opcodes - which is not the same question.
# Decoding the assembler's own WORDSLOT_TEMPLATE_8/10/14 through agxforge.g17.model gives op17235 at
# all three widths, and HALFSLOT_TEMPLATE_14 gives op17199, so the word slot store is op17235 and
# the wait bit exists only where byte9 does. The templates' own byte5 is 0x04, i.e. sub-form 1,
# which the table also maps to 17235 - two readings agreeing. NOT ADJUDICATED HERE: the comment
# above that table says "11 is the store this compiler emits", and 11 is sub-form 3 (op17244);
# whichever is stale, this fact rests on the decode, which is stated so the next reader can check it.
LEDGER_FACTS.setdefault(17235, []).append(dict(
    axis='latency_wait', evidence='checked',
    source='ledger/g17-store-load-dependency.toml (opcode resolved by decoding '
           'g17asm.WORDSLOT_TEMPLATE_8/10/14 through agxforge.g17.model)',
    state='byte9[5] IS THE STORE\'S WAIT-FOR-LOAD, and it exists only at the widths that have a '
          'byte9 - so op17235/len10 and op17235/len14 can wait and op17235/len8 cannot. Clearing '
          'it on a store whose source is a pending load, three stores in one kernel, each '
          'affecting only its own output: 1000 -> 0, 1001 -> 0, 1002 -> 2. The values are the '
          'source register\'s contents BEFORE the load landed, which is why "reads zero" would '
          'have been the wrong summary - slot 120 read 2',
    ledger_status='causal, with the reverse control: SETTING byte9[5] on a store whose source is '
                  'an ALU result, where no load is pending, is completely inert (preregistered '
                  'out[100] = 11, measured 11)',
    note='the reverse arm is what makes this a WAIT bit rather than a data bit, and it is the arm '
         'a one-directional test would have skipped. The encode gate found it rather than a '
         'hypothesis: store.14 scored 8.3% byte-exact and its residue was ENTIRELY byte9, 47 of '
         '54 mismatches differing in that one byte and nothing else'))

LEDGER_FACTS.setdefault(17229, []).append(dict(
    axis='latency_wait', evidence='checked',
    source='ledger/g17-only-one-alu-family-waited-for-a-load.toml',
    state='THE INDEXED STORE CANNOT WAIT AT ALL at width 8: no byte9, so no wait bit, where the '
          'slot store answers the same hazard by widening. Measured as a pair of kernels '
          'differing in one `add 0` between the load and the store - 0xBB000000.. per thread '
          'with it and zero without, status 0 and no fault either way, so the store runs and '
          'stores nothing yet. The backend refuses that program and names the two repairs',
    ledger_status='executed - identical kernels with and without the intervening ALU op',
    note='AND THIS IS THE STANDING CAVEAT ON byte0[3]: op17229\'s Apple witnesses already CARRY '
         'byte0[3] and its store after a load still returned stale data. So a form carrying the '
         'bit is not evidence that the form waits, which is the premise the float extension rests '
         'on - see ledger/g17-byte0-bit3-corpus-does-not-support-the-float-extension.toml. A wide '
         'op17229 encoding exists in the authoring table and may carry a wait; untested, and the '
         'refusal does not assume it'))

# THE FLOAT EXTENSION'S BOUND, MEASURED ON BOTH AVAILABLE POPULATIONS AND NOT SUPPORTED BY EITHER.
# Attributed per opcode because the census NAMES these six; the reading it declines to support is
# a latency_wait reading, so that is the axis, and the evidence is `corpus` because the instrument
# is a decode of shipped libraries - no arm of this was executed.
BYTE0_BIT3_FLOAT_OPS = (998, 2190, 3290, 10826, 11372, 11375)
_B3_RATE = {998: ('fadd', '36.7%', '30.2% after a load against 33.4% elsewhere'),
            2190: ('ffma', '36.2%', '32.9% after a load against 34.0% elsewhere'),
            3290: ('fmul', '36.4%', '37.4% after a load against 33.1% elsewhere'),
            10826: ('madd', '32.4%', '43.6% after a load against 32.7% elsewhere'),
            11372: ('csel.reg', '5.6% of 18 patterns', 'population too small to condition on'),
            11375: ('select', '30.3%', '39.3% after a load against 25.7% elsewhere')}

for _op in BYTE0_BIT3_FLOAT_OPS:
    _n, _rate, _cond = _B3_RATE[_op]
    ISA_FACTS.setdefault(_op, []).append(dict(
        axis='latency_wait', evidence='corpus',
        source='ledger/g17-byte0-bit3-corpus-does-not-support-the-float-extension.toml',
        state='byte0[3] IS EXECUTED AS THE LOAD-USE WAIT ON op3290/len14 AND ON NOTHING ELSE. '
              'The compiler extends it to twelve more opcodes on a corpus bound, and the bound '
              'does not hold on either population available here. Over the shipped system '
              'libraries (9,556 programs, 4,213,552 instructions) Apple sets byte0[3] on %s of '
              'this opcode\'s distinct patterns, not "most", and the build cache agrees to within '
              'two points. Conditioned on the previous instruction being a load: %s - nothing '
              'resembling the 19.7%% against 0.0%% that NAMED the bit on the add family'
              % (_rate, _cond),
        ledger_status='NEGATIVE, decode-side, two populations plus a per-bit control; the one '
                      'executed form it does not touch is named in the ledger',
        note='THIS DOES NOT REFUTE THE WAIT READING - nothing here is executed, and a '
             'distribution that fails to support a reading is not a reading refuted. What falls '
             'is the bound. Two further specifics: "Apple never sets it" on op11372/op11375 is '
             'false on the shipped library (3,458 of 11,429 select patterns carry it), and the '
             '"setting it keeps the instruction decoding as itself" criterion cannot discriminate '
             'this bit at Apple\'s narrow float widths - at op3290/6 all 48 bit positions change '
             'the decoded opcode in 560 to 587 of 587 patterns, so the criterion is evaluable '
             'only on the flip-neutral encodings the compiler itself writes'))

GLOBAL_FACTS.append(dict(
    fact='the five-bit field at byte1[2..6] on every ALU and bitwise instruction is a MASK over '
         'five dependency slots, not an index into 32 of them: over 84,017 instances the '
         'popcount distribution is 68.1% zero and falls off monotonically, where an index would '
         'give the binomial 3.1 / 15.6 / 31.2 / 31.2 / 15.6 / 3.1',
    axis='latency_wait', evidence='corpus',
    source='ledger/g17-the-wait-tag-is-a-mask.toml',
    ledger_status='corpus-measured; NOT executed, and the ledger says what that costs: it does '
                  'not show what fills a slot, how long one stays occupied, or whether one can be '
                  'reused before the wait clears',
    why_unattributed='the source scopes it to "every ALU and bitwise instruction" and names no '
                     'opcode set, so attributing it per form would invent a scope the evidence '
                     'does not have. It cross-checks the load-use wait from the other side: '
                     'wait 0 / tag 0 is 49,098 instances, wait 1 / tag nonzero is 17,887'))

GLOBAL_FACTS.append(dict(
    fact='LATENCY IS NOT REACHABLE FROM THE SHIPPED LIBRARY, so no form on this map can carry an '
         'issue cost. Apple\'s AGX3 MCSchedModel sits behind MCSubtargetInfo, whose constructor '
         'resolves (LLVMInitializeAGX3TargetMC) but needs a walk through a Triple, two '
         'std::strings, three ArrayRefs and four table pointers with every offset guessed - and '
         'libLLVM exports no AGX3 symbol matching sched, model, itinerary, latency or proc, so '
         'there is nothing to anchor a guess against. The scheduling class stays an opaque '
         'GROUPING key, which is all this project has ever used it as',
    axis='latency_wait', evidence='table',
    source='ledger/g17-the-scheduling-model-is-not-reachable.toml',
    ledger_status='NEGATIVE, timeboxed, with the reason it was worth trying',
    why_unattributed='it is the reason an axis is empty rather than a fact about any form, and '
                     'it is recorded so the next session does not re-run the same walk. The '
                     'failure mode of guessing the offsets is not a wrong answer, it is a '
                     'plausible one'))



# ------------------------------------------------------------------------------------------------
# ITEM 3 CONTINUED: barrier_sync and resources, four more ledgers nothing had cited.

# A SECOND INSTRUMENT ON THE BARRIER, and the reason it is worth a separate entry rather than a
# footnote: the op447 fact already on this axis is a CORPUS reading - four of Apple's own kernels
# whose names state their scope. This one is an ORACLE reading - kernels this project wrote,
# compiled by Apple, one source property varied at a time. Same day, same byte1 values, arrived at
# from the other side. Two instruments agreeing is the only reason to believe either.
LEDGER_FACTS.setdefault(447, []).append(dict(
    axis='barrier_sync', evidence='oracle',
    source='ledger/g17-barrier-identified.toml',
    state='SCOPE SELECTS WHICH BARRIER IS EMITTED, NOT A FIELD INSIDE ONE, and the combined case '
          'is what proves it: mem_device|mem_threadgroup emits the device sequence 0f 69 00 '
          'followed by the threadgroup sequence 27 51 00 - two instructions, __text six bytes '
          'longer than either single-flag variant - where a flags field would have produced one '
          'sequence with a different value. Reading across the three singles, byte1 is 0x51 for '
          'both threadgroup-scope forms and 0x69 for device, while byte0 separates '
          'mem_threadgroup (0x27) from mem_none (0x07) and device (0x0f). The barrier is SIX '
          'bytes, which is the width this map carries for op447',
    ledger_status='live - single-variable differential against a barrier-free control, every '
                  'variant sharing the same 72-byte prefix and 42-byte suffix so the insertion '
                  'is isolated by construction; not executed',
    note='simdgroup_barrier produces NO size change at all - __text stays 170, identical to the '
         'barrier-free kernel - so it emits nothing at this dispatch. Recorded as an observation '
         'about a 32-thread single-threadgroup dispatch, where one SIMD group is already '
         'lock-step, and NOT as a fact about the instruction. That is also why no barrier here is '
         'authored: a barrier\'s effect is visible only when removing it causes a race, and this '
         'harness cannot produce one, so the missing piece is a wider dispatch rather than an '
         'encoding'))

# THE ATOMIC'S RESOURCE DECLARATION. This source is RETRACTED IN ITS TITLE - the index flags it,
# `declares_a_retraction: ["title"]` - and the retraction is specific: "the section diff is sound
# and every conclusion drawn from dispatching it is void", because every dispatch judged buffer 0,
# which the harness never copies back. So the SECTION COMPARISON is harvested and nothing that
# rests on a dispatch is, which is why the state below says where the field is and says nothing
# about what happens when it is set.
for _op in ATOMIC_OPS:
    ISA_FACTS.setdefault(_op, []).append(dict(
        axis='resources', evidence='corpus',
        source='ledger/g17-the-atomic-declaration-is-in-ld-md.toml [RETRACTED IN TITLE - the '
               'section diff is sound, every dispatch conclusion is void]',
        state='AN ATOMIC IS DECLARED IN __GPU_LD_MD, not in the instruction stream and not in the '
              'metadata slot set. Apple against Apple settles which section carries it - the same '
              'kernel with and without an atomic: __GPU_ARCH_LD_MD is byte-identical at 40 (a '
              'constant, not a declaration), while __GPU_LD_MD goes 216 -> 224, eight bytes '
              'LONGER, and the two flatbuffers diverge first at byte 40 where the atomic\'s '
              'vtable carries a field the store leaves zero. That entry and the eight bytes it '
              'points at are the declaration. CORRECTED 2026-09-17: this read "this backend does not '
              'build the section at all", true when the cited ledger was written and false '
              'now - agxforge/g17/ldmd.py GENERATES it from the program\'s own layout. It '
              'emits 216 bytes differing from Apple\'s non-atomic version in 25 of 216 '
              'bytes and lacks the trailing "compute" string both Apple kernels carry',
        ledger_status='corpus - a section-by-section diff of Apple objects against each other and '
                      'against a generated image; the file\'s DISPATCH conclusions are retracted '
                      'and none of them are used here',
        note='seventeen kernels run on that section, so most of what __GPU_LD_MD declares is not '
             'load-bearing for loads, stores, branches, threadgroup memory or loops - and it IS '
             'load-bearing for an atomic. This is a RESOURCES fact and deliberately not a '
             'semantics one: it says what an atomic owes the linker metadata, never that any '
             'atomic operation has been observed to do anything. '
             'AND THE BLOCKER IS SMALLER THAN THAT FACT IMPLIES, measured 2026-09-17: '
             'mdgen.describe round-trips __GPU_LD_MD byte-for-byte, so the section is not '
             'opaque to us, and the declaration is SLOT 24 of its table 136 - present in '
             '511 of 511 objects whose text carries a device atomic and 0 of the 21,463 '
             'others, which is the same population byte 40 bit 2 selects. Slot 1, which '
             'also appears in the donor and looked equally likely, is NOT the '
             'discriminator: 20,493 non-atomic objects carry it. So the work is adding two '
             'slots to a describable table rather than generating a section from nothing. '
             'Still a RESOURCES fact, so atomic_sync stays at D3 = 0. '
             'AND THE GAP IS THREE SLOTS, not a section: ldmd.build\'s existing `restore` mode already '
             'writes eight of Apple\'s ten t136 slots, leaving slot 24 (the declaration, a '
             'CONSTANT one-byte 1 across all 461 objects of that class), slot 1 (a program '
             'fact, 218 values, present in 20,493 non-atomic objects too) and t44 slot 2. '
             'REVISED AGAIN after building it: the atomic is a DIFFERENT BODY LAYOUT, not '
             'ours plus a flag - six offsets move +4, slots 1 and 24 appear, tlen 40 -> 48. '
             'Our restore output is byte-layout identical to Apple\'s NON-atomic t136, and '
             'writing slot 24 at Apple\'s offset 22 into OUR layout lands inside slot 6, the '
             'entry PC, which occupies bytes 20..23 here and 24..27 there. The candidate '
             'round-tripped byte-identically and kept every structural property; what '
             'caught it was describe reporting width 2 where Apple has width 1. '
             'ledger/g17-the-atomic-declaration-is-slot-24-of-table-136.toml'))

GLOBAL_FACTS.append(dict(
    fact='METADATA SLOT 0 IS THE REGISTER-COUNT DECLARATION and the law is exact: the highest '
         '32-bit register index named in _agc.main, plus one, in 2,022 of 2,022 objects. The '
         'constant program\'s registers are NOT counted - taking the max over both spans gives '
         '1,942 of 2,022, and those 80 misses were the instrument',
    axis='resources', evidence='corpus',
    source='ledger/g17-slot-zero-is-the-main-register-count.toml',
    ledger_status='law exact and one-sided over 2,022 objects; the authoring path still emits a '
                  'per-witness constant, deferred behind an in-flight dispatch',
    why_unattributed='it is a property of a PROGRAM\'s metadata, not of any instruction form, so '
                     'there is no opcode to attribute it to. Kept because the 80 exceptions are '
                     'the lesson: all 80 were UNDER-declared and none over-declared, and a '
                     'one-sided error is an instrument artifact rather than a second rule - a '
                     'declaration that were merely unreliable would miss both ways. The 96% that '
                     'the wider walk produced never described the objects'))

GLOBAL_FACTS.append(dict(
    fact='SLOT 17 IMPLIES SLOT 38, and it is unbreakable by construction rather than merely '
         'unfalsified: six probes each varying one thing, and the sharpest - a write-access '
         'texture with no sampler and nothing sampled - carries slot 38 anyway. A texture write '
         'needs a texture binding and slot 38 tracks texture presence, so the implication is '
         'structural. Both no-texture arms (one written buffer, four written buffers) carry '
         'neither slot, so the probe set can distinguish',
    axis='resources', evidence='oracle',
    source='ledger/g17-slot-17-is-a-texture-write.toml',
    ledger_status='compiled, not dispatched: six probes built to break an implication a census of '
                  '9,547 Apple sections held without exception over 6,016 premises',
    why_unattributed='a slot implication is a fact about the metadata format, not about an '
                     'opcode. AND THE SAME FILE CARRIES A RETRACTION that must travel with it: '
                     'it first claimed slots 16 and 17 never co-occur - 0 of 18,356 cached '
                     'kernels - and read the pair as a two-valued indicator of whether a texture '
                     'is written. The real co-occurrence count is 186. The surviving claim is '
                     'only the implication; the indicator reading is dead. It is the difference '
                     'between what a constructed population can say and what a census can: not '
                     '"unbroken so far" but "here is the construction that would have broken it"'))



# ------------------------------------------------------------------------------------------------
# THE BARRIER'S ORDERING FUNCTION, EXECUTED - and the two accesses it orders, which is what puts
# new opcodes on barrier_sync rather than a third fact on op447 alone. Both facts already on that
# axis for op447 are non-hardware ("not executed", "not dispatched"); this one ran.
LEDGER_FACTS.setdefault(447, []).append(dict(
    axis='barrier_sync', evidence='executed',
    source='ledger/g17-threadgroup-exchange-at-32-lanes.toml',
    state='THE BARRIER ORDERS TWO THREADGROUP ACCESSES ACROSS 32 LANES, dispatched: read_sr ->  '
          'add 0x40 -> op13288 store.tg tgm[tid] -> op447 -> movimm 31 -> op12364 load.tg '
          'tgm[31] -> op17235 store, and u[0] came back 0x5f = 0x40 + 31. One word settles four '
          'things at once - the special register is per-lane, the threadgroup store\'s index '
          'operand indexes, the barrier orders the two accesses, and threadgroup memory carries '
          'data BETWEEN lanes rather than being per-thread scratch',
    ledger_status='causal and unambiguous BY CONSTRUCTION: every lane reads one fixed slot that '
                  'only the last lane writes, so the answer cannot be a lane winning a race',
    note='the observable had to be redesigned rather than repeated, and that is the transferable '
         'part. Two earlier versions could not tell "lane 0 won the race" from "every lane is '
         'lane 0" - storing the thread index gave 0 and storing index+0x40 gave 0x40, which is '
         'lane 0 either way. The device store on this host carries no index, so all 32 lanes race '
         'for u[0]. Reading a FIXED slot only the last lane writes makes the two answers '
         'different rather than making the input different'))

for _op, _role in ((13288, 'store'), (12364, 'load')):
    LEDGER_FACTS.setdefault(_op, []).append(dict(
        axis='barrier_sync', evidence='executed',
        source='ledger/g17-threadgroup-exchange-at-32-lanes.toml',
        state='THIS IS THE THREADGROUP %s AN op447 BARRIER ORDERS. In the 32-lane exchange the '
              'op13288 store and the op12364 load sit either side of one barrier and every lane '
              'read the value the LAST lane wrote (0x5f = 0x40 + 31), which is only possible if '
              'the barrier completes the store before the load is allowed to observe the slot. '
              'Threadgroup memory therefore carries data between lanes; it is not per-thread '
              'scratch' % _role.upper(),
        ledger_status='executed - dispatched at 32 threads in one threadgroup, with the race '
                      'removed by construction rather than by repetition',
        note=('AND A SEPARATE EXACT RULE FROM THE SAME PROBE, mechanism NOT established: a '
              'threadgroup store whose VALUE operand and INDEX operand are the same register does '
              'not index - every lane writes the base. tgm[tid] = tid + 0x40 from two registers '
              'reads back 0x5f; tgm[tid] = tid from one register reads back 0, slot 31 never '
              'written. Four candidate mechanisms were each worth one dispatch and each failed: '
              'not a missing emission pair, not a scheduling gap, not the xor immediate (at lane '
              '0 it stores 1 where a move would store 0), not the value lifetime. The repair is '
              'the one this backend already has for a related case - copy the value before the '
              'instruction that would consume it twice'
              if _op == 13288 else
              'the peer counted 866 threadgroup accesses in the corpus - 355 stores and 511 '
              'loads - and ZERO put one register in both the value and the index operand. '
              'Apple\'s own allocator never does it, which is what makes the same-register '
              'constraint recorded on op13288 the machine\'s rule rather than an artefact of how '
              'this backend authors')))

for _op in READ_SR_OPS:
    LEDGER_FACTS.setdefault(_op, []).append(dict(
        axis='resources', evidence='executed',
        source='ledger/g17-threadgroup-exchange-at-32-lanes.toml',
        state='A KERNEL GETS ONLY THE SPECIAL REGISTERS ITS OWN METADATA DECLARES, and reading '
              'one it does not declare FAILS SILENTLY rather than refusing. a6-tgidx declares '
              'thread_position_in_threadgroup and threadgroup_position_in_grid; a program patched '
              'into it reading thread_position_in_GRID - which this backend had used everywhere, '
              'since tb-host3wide declares it - returned ZERO in every lane, so a 32-lane program '
              'silently became 32 copies of lane 0. 0xa4 is '
              'thread_position_in_threadgroup, read off that host\'s own read_sr',
        ledger_status='executed - the zero was measured, and the correct register in the same '
                      'host returned a per-lane value',
        note='the constant matters less than the failure mode: patching a program into a host '
             'INHERITS that host\'s declared inputs, and an undeclared read is indistinguishable '
             'from a register that happens to hold zero. This is the same hazard the slot-29 '
             'refusal answers from the authoring side - a program reading a system register owes '
             'the section an entry - approached from the host-inheritance side instead'))



# ------------------------------------------------------------------------------------------------
# WHAT A DEVICE ATOMIC OWES THE LINKER, and the first MEASURED candidate for why this project's
# authored atomic does nothing. Attributed to the DEVICE members only: the census is what
# established the device/threadgroup split, so spraying it across the threadgroup twins would be
# asserting the opposite of what it measured.
for _op in DEVICE_ATOMIC_OPS:
    ISA_FACTS.setdefault(_op, []).append(dict(
        axis='resources', evidence='corpus',
        source='ledger/g17-the-device-atomic-is-declared-in-ld-md-byte40.toml',
        state='A DEVICE ATOMIC IS DECLARED BY __GPU_LD_MD BYTE 40 BIT 2, and read as a BIT rather '
              'than a byte value the rule is exceptionless over 21,981 cache objects: set in 511 '
              'of 511 whose text decodes to a device atomic, clear in 279 of 279 whose only '
              'atomic is threadgroup-scope. It follows the SOURCE and not the text - 49 objects '
              'carry no atomic opcode at all and have it set, which is atomic_load_explicit '
              'lowering to an ordinary load (op12688 in 6, op12682 in 37) - so the instruction '
              'stream alone cannot predict the declaration a program owes',
        ledger_status='corpus/oracle over the build cache, decode-side, exceptionless after two '
                      'classification corrections the census itself forced; NOT executed',
        note='THE REASON THIS WAS WORTH THE CENSUS. All three hosts spike/accel/re/atom12.py '
             'patches an authored atomic into - tb-host3wide, tb-host3buf, tb-host3hi - have the '
             'bit CLEAR, and every Apple device-atomic kernel has it set. So that probe emits a '
             'program whose code contains a device atomic and whose linker metadata declares '
             'none, which predicts exactly the symptom nine ruled-out mechanisms could not '
             'explain: every arm returning the seed, including the code-0 control. It is a '
             'CANDIDATE and not the cause - nothing here dispatched, and the section ledger it '
             'builds on is RETRACTED in its title for its dispatch conclusions, so its section '
             'diff is usable and its execution evidence is not. The discriminating run is named '
             'in the ledger: the same authored atomic in a host whose bit is SET, with the '
             'observable proved first by writing a known constant'))



# ------------------------------------------------------------------------------------------------
# ITEM 4, THE TEXTURE HALF - and it was reached by READING, not by dispatching. texture_image stood
# at D3 = 0 on all four instruments while two committed ledgers recorded a texture fetch running
# from this compiler and returning the right texel at four coordinates. Neither was cited.
#
# THE FORMS ARE DECODED, NOT INFERRED. The ledger's own listing of Apple's per-lane kernel gives
# `005c l8 op15813 texture.read`, which is Apple's program and not ours - so the attribution here
# comes from compiling the very IR the first dispatch used, g17texrun.executable_ir(5, 1), and
# decoding it: op592/len4 and op15813/len8 (with op423/len10, op10279/len12, op11842/len8,
# op14059/len4, op17235/len14 alongside). The coordinate (5, 1) in that call is the ledger's first
# arm, so the program decoded here IS the program that ran.
#
# EVIDENCE IS 'executed', NOT 'checked', AND THE CHOICE IS THE POINT. The outputs WERE compared
# against independent predictions - the texture is defined texel(x, y) = 1000*y + x + 7 and the
# runs returned 1012, 3009 and 3014 - which is 'checked' by this ladder's own words. But
# `d3_semantics_checked` is reserved for the contract's `verified` class, an ISOLATED one-opcode
# probe, and this is a whole kernel. So it lands in `d3_ledger_executed`, the column that means
# "a ledger records this running on hardware and it is not the isolated verified class", and the
# strength of the comparison is stated in the fact instead of being smuggled into a column.
TEXTURE_EXECUTED_FORMS = ((15813, 8, 'the texture read itself'),
                          (592, 4, 'the coordinate publish that makes it per-lane'))

for _op, _len, _role in TEXTURE_EXECUTED_FORMS:
    LEDGER_FACTS.setdefault(_op, []).append(dict(
        axis='semantics', evidence='executed', widths=(_len,),
        source='ledger/g17-the-texture-coordinate-is-a-register-published-into-op4.toml '
               '+ ledger/g17-a-texture-read-executes.toml (forms decoded from '
               'g17cc.compile_function(g17texrun.executable_ir(5, 1)), not taken from the '
               'ledger\'s listing of APPLE\'s kernel)',
        state='A TEXTURE READ EXECUTES FROM THIS COMPILER AND RETURNS THE RIGHT TEXEL, and the '
              'answer TRACKS the coordinate rather than sitting in a slot. Four dispatches over '
              'one r32Uint 2D texture with texel(x, y) = 1000*y + x + 7: coordinate (5, 1) gave '
              'A[400] = 1012, coordinate (2, 3) gave 3009, and the per-lane case with only lane '
              '31 storing gave 3014 = texel(7, 3) - lane 31\'s own x. This form is %s, at '
              'width %d' % (_role, _len),
        ledger_status='executed and discriminated; a texture kernel is this compiler\'s end to '
                      'end. NOT the contract\'s isolated `verified` class, which is why this '
                      'counts in d3_ledger_executed and not in d3_semantics_checked',
        note='THE DEGENERATE HYPOTHESIS WAS NAMED AND RULED OUT, which is what makes the per-lane '
             'arm worth anything. With all 32 lanes storing to slot 400 the result was texel(0, '
             '3) - exactly what a publish that BROADCAST lane 0\'s register would give - so that '
             'run decides nothing and the ledger records it as ambiguous rather than as support. '
             'Guarding the store with t > 30 leaves one writer whose own x is 7, and it returned '
             'texel(7, 3). Two further limits travel with this: the kernel runs on APPLE\'s '
             'ty-2d metadata section, so nothing here says this compiler can generate the texture '
             'resource metadata, and op592\'s register form is CONSTRUCTED - all 104 corpus '
             'witnesses of that operand write op0 destinations, because Apple\'s own per-lane '
             'texture read does not publish at all. What licenses it is the dispatch, not the map'))



# A SUSPECTED CONTRADICTION, CHECKED AND ABSENT - recorded because the check is the expensive part
# and the next session should not repeat it.
GLOBAL_FACTS.append(dict(
    fact='THE NINE OPCODES A LEDGER SAYS "STAY UNNAMED" DO NOT CONTRADICT THE CONTRACT\'S '
         '`verified` CLASS, and the resolution is the ledger\'s own sentence. '
         'ledger/g17-the-dark-classes-were-arithmetic-all-along.toml refuted naming op9704, '
         'op9795, op9805, op9835, op9796, op9806, op9836, op9832 and op9881 as fdim / step / min '
         '/ max / clamp - "a clean single-construct attribution is not sufficient; the class has '
         'to agree" - and concluded they are "the SELECT the lowering ends with, not the '
         'function". The contract names five of them fcsel.imm.f32.f32.to.f16, '
         'fcsel.imm.f32.f16.to.f16, fcsel.imm.f16, fcsel.f16 and fcsel.a, which IS the select '
         'reading, and leaves the other four at `measured` (op9704, op9795, op9835) or `unnamed` '
         '(op9805). So the five in D3 are verified AS SELECTS and the refuted names are the '
         'function names, which nothing here claims',
    axis='semantics', evidence='checked',
    source='ledger/g17-the-dark-classes-were-arithmetic-all-along.toml cross-read against '
           'isa/g17-contract.jsonl (checked here 2026-09-17)',
    ledger_status='executed; five predictions committed before the probes that tested them',
    why_unattributed='it is a fact about the SCOPE of five names, and attributing it per form '
                     'would move a denominator in whichever direction it was written: on the four '
                     'unverified opcodes a `semantics` fact at the ledger\'s own `executed` '
                     'status would raise D3 for names the ledger REFUSED, and on the five '
                     'verified ones it would overwrite a `checked` axis with a weaker one while '
                     'the D3 column - computed from the contract class, not from the axis - stayed '
                     'put, leaving the record and the count disagreeing. Recorded once, '
                     'unattributed, and every denominator is unchanged'))



@functools.lru_cache(maxsize=1)
def decoder_build_identity():
    """Bytes naming the DECODER BUILD these caches' answers came from: the native decoder's sources,
    the renumbering table it translates through (tools/g17renumber.py), and which numbering Apple's
    library answers in. macOS 27 renumbered every opcode; keyed on the Python sources alone, the
    repair-width cache kept the widths decoded in the new numbering (D1 read 1,923 for 7,549)."""
    h = hashlib.sha256()
    for name in ("agx3dis.c", "agx3dislib.c", "agx3remap.h", "agx3renumber.h"):
        f = ROOT/'tools'/name
        h.update(name.encode() + (f.read_bytes() if f.exists() else b"absent"))
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin") as t:
        t.write(bytes.fromhex("0e000000"))      # op684, `end`, in the original numbering
        t.flush()
        r = subprocess.run([str(ROOT/'tools'/'agx3dis'), t.name, "0", "4"], capture_output=True,
                           text=True, env=dict(os.environ, AGX3_RAW="1"))
    h.update(("raw:" + r.stdout.strip()).encode())
    return h.hexdigest().encode()


def repair_cache_key(contract, apple, decoder=None):
    """The repair-width cache key: the witnesses AND the decoder that reads them.

    A cache keyed on its input alone cannot tell that the code changed, so a parser fix keeps
    serving the broken answers and a parser REGRESSION keeps serving the good ones. That is not
    hypothetical here - the hex-token defect cost 72 opcodes their width and its repair touches
    only `agxforge/g17/model.py`, changing no witness at all. The previous version carried a
    hand-typed `v2` marker, which is the same idea maintained by memory.

    `decoder` is injectable so a guard can show the key MOVES when the decoder does; passing
    nothing reads the live file.
    """
    if decoder is None:
        decoder = (ROOT/'agxforge'/'g17'/'model.py').read_bytes()
    witnesses = json.dumps(
        {str(o): (r.get('encoding') or {}).get('witness') for o, r in sorted(contract.items())
         if o not in apple}, sort_keys=True).encode()
    return hashlib.sha256(witnesses + hashlib.sha256(decoder).hexdigest().encode()
                          + decoder_build_identity()).hexdigest()


def repair_walk_widths(contract, apple):
    """Width for the 5,991 opcodes with no Apple witness, from DECODING their repair-walk one.

    Every admitted opcode has a retained witness; 5,991 of them are repair-walk encodings, built
    by mutating a neighbour until the decoder accepted the bytes. Those witnesses are padded to
    sixteen, so the width is the DECODED length, which is weaker evidence than an Apple byte
    count in a specific way: it says the decoder accepts a form of that width, not that anything
    ever emitted one. Forty class-51 opcodes authored this way return a constant of the encoding.

    So the width is recorded and LABELLED, and D2 still requires an Apple witness - which is why
    admitting these raises coverage without moving the encoding denominator at all. Getting that
    wrong once took D1 from 970 forms to 7,013.
    """
    # 5,991 decodes, ~25s, and it runs on every rebuild - which took the test suite from 40s to
    # 108s. Cached on a digest of the witnesses it reads, so the cache cannot outlive its input.
    # THE KEY COVERS THE DECODER'S CODE, NOT ONLY ITS INPUT. These widths are produced by
    # `agxforge.g17.model.decode`, and a cache keyed on the witnesses alone cannot tell that the
    # decoder changed - so a parser fix keeps serving the broken widths and a parser REGRESSION
    # keeps serving the good ones. That is not hypothetical here: the hex-token defect cost 72
    # opcodes their width, and the repair for it changes exactly this file and nothing in the
    # witnesses. The hand-maintained `v2` marker below was the same idea done by memory, which
    # only works while someone remembers.
    key = repair_cache_key(contract, apple)
    cached = CACHE/('repairwidth-v3-%s.json' % key[:16])
    if cached.exists():
        stored = json.loads(cached.read_text())
        return {int(k): v for k, v in stored['widths'].items()}, stored['failed']
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from agxforge.g17.model import decode
    widths, failed = {}, []
    for opcode, row in contract.items():
        if opcode in apple:
            continue
        witness = (row.get('encoding') or {}).get('witness')
        if not witness:
            failed.append(dict(opcode=opcode, reason='no witness'))
            continue
        try:
            instructions = list(decode(bytes.fromhex(witness), 0))
        except Exception as exc:
            # THE MESSAGE, NOT JUST THE CLASS. This recorded `type(exc).__name__`, so twenty
            # opcodes were published with the reason "ValueError" and nothing else - a failure
            # class is not a failure. The text is what says whether the witness is malformed, the
            # length is wrong, or the decoder refuses the form.
            failed.append(dict(opcode=opcode, reason='%s: %s' % (type(exc).__name__, exc)))
            continue
        if not instructions or instructions[0].opcode is None:
            failed.append(dict(opcode=opcode, reason='decodes to no opcode'))
        elif instructions[0].opcode.id != opcode:
            failed.append(dict(opcode=opcode, reason='decodes to op%d' % instructions[0].opcode.id))
        else:
            widths[opcode] = len(instructions[0].raw)
    CACHE.mkdir(parents=True, exist_ok=True)
    _cache_write(cached, json.dumps(dict(widths={str(k): v for k, v in widths.items()},
                                      failed=failed), sort_keys=True))
    return widths, failed


def apple_widths(contract):
    """Width from the byte count of the encoding APPLE ACTUALLY WROTE. No decoding involved.

    `apple_witness` holds Apple's exact emitted bytes; `witness` is the same padded to sixteen.
    An earlier version of this function decoded the PADDED buffer and reported 120 opcodes whose
    width form-bits did not list - it was measuring the decoder's reading of padding, not a
    width. Apple's own byte count needs no interpretation, and on that basis the picture is:
    502 agree with form-bits, THREE contradict it, and 212 opcodes have an Apple witness with no
    form-level entry at all. Those 212 are forms Apple emitted that the form ledger does not
    cover, and they belong in D1.

    Only the 717 Apple-written witnesses are used. The other 6,001 are repair-walk encodings -
    built by mutating a neighbour until the decoder accepted them - and admitting a width on that
    basis would contradict the reasoning D2 rests on. An earlier pass did exactly that and took
    D1 from 970 forms to 7,013.
    """
    widths = {}
    for opcode, row in contract.items():
        apple = (row.get('encoding') or {}).get('apple_witness')
        if isinstance(apple, str) and apple and len(apple) % 2 == 0:
            widths[opcode] = len(apple)//2
    return widths


CACHE = ROOT/'results'/'g17-isamap-cache'


def _cache_write(path, text):
    """Write a cache file ATOMICALLY: a temporary beside it, then rename over it.

    `make check-ledgers` runs its checks concurrently now, and three of them (isamap --check,
    fitfromexecution --check, widthaudit --check) read and fill this cache. A plain write_text
    truncates first, so a concurrent reader of a cold key could load half a file and die on
    JSONDecodeError. Writers produce identical content, so last-rename-wins is correct."""
    tmp = path.with_name('%s.%d.tmp' % (path.name, os.getpid()))
    tmp.write_text(text)
    os.replace(tmp, path)


def decoder_identity_check(contract):
    """Ask Apple's decoder to decode each retained witness and confirm it names that opcode.

    An independent instrument, used for IDENTITY only - width comes from the Apple byte count
    above. Zero witnesses decode to a different opcode, and the count of witnesses the decoder
    cannot use is reported rather than skipped: the instrument has its own limits.
    """
    # Decoding 6,718 witnesses costs ~13s, which made a fast tool slow and a 23-case suite
    # 56s. Cached under an ignored directory and keyed on a digest of the witnesses.
    #
    # THE KEY COVERED THE INPUT AND NOT THE INSTRUMENT, AND THE ARTIFACT PUBLISHED THE STALE
    # ANSWER FOR TWO DAYS. The witnesses had not changed, so the key had not moved, so a cache
    # entry written before the DECODER learned to read 75 of them kept being returned. The
    # committed isa/g17-coverage.json therefore said `identity_agrees` 6,643 with 75 witnesses
    # the decoder "could not use" and twenty ValueErrors; a checkout with no cache computes
    # 6,718 and zero. This docstring said THREE. Three numbers for one measurement, and only
    # the pristine one was true - found by running the suite in a worktree pinned at the commit,
    # where `--check` had been green throughout because it read the same cache.
    #
    # The decoder is an input to this result as much as the witnesses are, so its source is in
    # the key: every module of the package the instrument names, not just model.py, because a
    # decode table lives beside it and changing one would otherwise go unnoticed exactly as this
    # did.
    decoder_src = sorted((ROOT/'agxforge'/'g17').glob('*.py'))
    key = hashlib.sha256(json.dumps(
        {str(o): (r.get('encoding') or {}).get('witness') for o, r in sorted(contract.items())},
        sort_keys=True).encode()
        + b''.join(hashlib.sha256(f.read_bytes()).digest() for f in decoder_src)
        + decoder_build_identity()).hexdigest()
    # v2: the result gained `apple_bytes_agree`, and a v1 entry lacks it - the instrument's
    # OUTPUT shape is part of the key for the same reason its source is.
    cached = CACHE/('identity-v2-%s.json' % key[:16])
    if cached.exists():
        return json.loads(cached.read_text())
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from agxforge.g17.model import decode
    agree, apple_agree, mismatched, errors = 0, 0, [], []
    for opcode, row in contract.items():
        witness = (row.get('encoding') or {}).get('witness')
        if not witness:
            errors.append(dict(opcode=opcode, reason='no witness'))
            continue
        try:
            instructions = list(decode(bytes.fromhex(witness), 0))
        except Exception as exc:
            errors.append(dict(opcode=opcode, reason=type(exc).__name__))
            continue
        if not instructions or instructions[0].opcode is None:
            errors.append(dict(opcode=opcode, reason='decodes to no opcode'))
        elif instructions[0].opcode.id != opcode:
            mismatched.append(dict(opcode=opcode, decoded=instructions[0].opcode.id))
        else:
            agree += 1
            # THE DISCRIMINATING PART OF `agree`. A walked witness was admitted because this same
            # decoder named it this opcode, so its agreement is the admission test re-run. Only a
            # witness that IS Apple's emitted bytes was not chosen by the decoder.
            apple = ((row.get('encoding') or {}).get('apple_witness') or '').lower()
            if apple and witness.lower().startswith(apple):
                apple_agree += 1
    result = dict(agree=agree, apple_bytes_agree=apple_agree, mismatched=mismatched,
                  errors=errors)
    CACHE.mkdir(parents=True, exist_ok=True)
    _cache_write(cached, json.dumps(result, sort_keys=True))
    return result


# Facts reported by the linker lane's accelerator reconnaissance, branch
# linker/g17-tensorops-recon @ 3a97205e. NOT MERGED, and NOT verified here. They are recorded
# because they are the only evidence this map has for the accelerator family, and they are kept
# in a SEPARATE column (d3_peer_reported) so that another lane's measurement can never quietly
# raise this artifact's own hardware denominator. Their strength is their own to defend.
PEER = 'linker/g17-tensorops-recon @ 3a97205e (unmerged, unverified here)'
PEER_FACTS = {
    5106: [dict(axis='semantics',
                state='the f16 16x16x16 multiply-accumulate intrinsic, sched 172, 10 bytes; '
                      'products exact with both dtype bits clear. Executes on a DIFFERENT '
                      'datapath from op2862, not one unit with two modes'),
           dict(axis='resources',
                state='an EIGHT-SLOT DEPENDENCY SCOREBOARD, not a result ring. A load names the '
                      'slot it fills in byte4[3] + 2*byte4[4] + 4*byte6[4], where the value the '
                      'decoder prints is slot + 1, and the consumer '
                      'MMA carries an eight-bit wait mask at flags bits 24..31: byte1[2..6] are '
                      'slots 0..4, byte7[7] slot 5, byte2[6] slot 6, byte0[3] slot 7. 110 '
                      'dispatches, 20 preregistered; exactly one mask bit set is identical iff '
                      'the bit is 24+s and zeros otherwise, with no exceptions in 63 runs. '
                      'SUPERSEDES the peer lane\'s own batch-11 reading of a round-robin result '
                      'ring, which they corrected: each MMA was waiting on its own load group, '
                      'which is why the field looked one-hot per MMA'),
           dict(axis='length',
                state='the 14-byte load is selected by bit 37 OR by a displacement above 255, '
                      'since the displacement field is eight bits. LENGTH IS NOT CHOSEN BY WHAT '
                      'IS IN FLIGHT, and a second vector load is an authoring rule rather than a '
                      'width question',
                note='CORRECTED by the peer lane, and the correction is the same shape as the '
                      'byte8 == 0x00 census: their first reading was "14 bytes iff bit 37" on '
                      '52/52 and 319/319, which held because every load in that sample had a '
                      'small displacement. The compiler\'s own packed-GEMM templates carry '
                      '14-byte loads at displacements 512..3584 with no bit 37 set, so the '
                      'sample could not have refuted the stronger rule. Recorded here because I '
                      'had already folded the too-strong version'),
           dict(axis='latency_wait',
                state='the MMA\'s byte0[3] is this repository\'s load-use wait bit - the peer '
                      'lane reaching the same field independently, and now placed as slot 7 of '
                      'an eight-bit wait mask'),
           dict(axis='unknown_bits',
                state='A HARDWARE CENSUS CALIBRATES THIS AXIS IN BOTH DIRECTIONS. 80 single-bit '
                      'flips of op5106, one process each, random tile with C != 0: all 16 bits '
                      'the DECODER calls inert are also hardware-inert, which is evidence '
                      'AGAINST this artifact\'s blanket treatment of decoder-invisible bits as '
                      'unknown, at least for this opcode. But of the bits the decoder REFUSES, '
                      'eight change the result - byte0[4], byte1[1], byte2[1], byte3[3], '
                      'byte4[7], byte7[3], byte7[4], byte8[7] - so the hardware reads fields '
                      'there that the decoder does not name, and six others execute as D == C. '
                      'byte5[7] and byte8[4], which this ledger calls forced, change the result, '
                      'as the register-class story predicts')],
    5107: [dict(axis='semantics',
                state='tensor.mac.init, sched 173; the initialising form of the same intrinsic')],
    5100: [dict(axis='semantics',
               state='DECODES BUT THE DRIVER NEVER EMITS IT, and it executes. Reached at one '
                     'flip from op5098 by clearing byte5[7] (A tup8, B tup4). The peer lane '
                     'authored and ran it: fp32 A truncated to 10 significand bits times half B, '
                     'and times the same bits as bfloat, 4/4 tiles each bit-exact against their '
                     'model. Control: the same bytes with byte0[1] flipped, which the decoder '
                     'refuses, dispatch without fault and leave D == C',
               note='the most interesting row in their sweep and the one I asked for - a form '
                    'the decoder admits, the hardware computes, and Apple\'s compiler never '
                    'emits. Peer-reported and unverified here'),
            dict(axis='length', state='10 bytes, decoded alone at its own length')],
    5101: [dict(axis='semantics',
               state='decodes at two flips from op5098 (byte5[7] with byte8[1]); the '
                     'no-accumulator sibling of op5100. Not executed',
               note='peer-reported, unverified here')],
    10384: [dict(axis='semantics',
                 state='the i8.i8 accelerator MMA, sched 170, 10 bytes: exact 16-term sums, and '
                       'C + 2^31 wraps to -2^31 with NO saturation')],
    2862: [dict(axis='semantics',
                state='the legacy fp32 MMA, sched 118, 14 bytes, reached from MSL '
                      'simdgroup_multiply_accumulate: a full-precision fused sequential chain')],
    11456: [dict(axis='semantics',
                 state='the mask generator behind masked tensor stores, for partial tiles and '
                       'per-SIMD-group masking. Apple\'s table names this opcode `csel`')],
}

# Opcodes the peer lane reports the driver compiling that this decoder does NOT admit. They are
# outside every denominator here, which is the sharpest statement available about D1's closure.
# VERIFIED HERE against Apple's own MCInstrDesc table, not taken on report: all five are rows in
# the 17,796-row dump, with the structure the peer lane described - A's register class widening
# from 41 (tup2, int8) through 159 (tup4) to 335 (tup8), and every no-accumulator form at nops=9
# against its accumulating sibling's 12.
#
# THIS CORRECTS THIS ARTIFACT'S EARLIER HEADLINE. I had recorded these as opcodes the driver
# compiles and "this decoder does not admit", and read that as the decoder-addressable surface
# being narrower than the toolchain's. The peer lane supplied the mechanism and it is a limit of
# the ENUMERATION, not of the decoder: byte5[7] and byte8[4] change an operand's register CLASS,
# so a single-bit flip on the op5106 witness leaves the old B register field in place, which
# under the wider class names a register that is not correctly aligned, and Apple's decoder
# rejects the whole encoding. The admission walk records "forced" and moves on. Reaching op5104
# needs the selector bit AND the re-encoded register field together - two or more flips.
#
# So the honest claim is: the admitted count of 6,718 UNDERCOUNTS the decoder-addressable surface
# wherever a selector bit also changes an operand's register class, and a single-flip walk cannot
# see those forms by construction. The decoder admits them; my instrument could not reach them.
WALK_UNDERCOUNT = {
    5098: 'sched 172, A widened to regclass 335 (tup8): both operands fp32',
    5099: 'sched 173: the no-accumulator form of op5098',
    5104: 'sched 172, A regclass 159 (tup4), B widened: A 16-bit, B fp32',
    5105: 'sched 173: the no-accumulator form of op5104',
    10385: 'sched 171, A regclass 41 (tup2): the no-accumulator int8 form',
}

# MEASURED, not asserted. Two independent sweeps now bound the undercount.
#
# Mine (tools/g17isareach.py, results/g17-isa-reach-v1): one bit of each of the 717 Apple-written
# witnesses, inside the instruction's real length. NINE opcodes come back that Apple's table
# declares and the admitted ledger lacks - op475, op502, op529 from their neighbours at flip
# 13.5, op13790 at 3.4, op14470/14662/14666/14758 at 15.7 in 22-byte forms, op17010 at 2.5 - all
# from seeds that are themselves admitted. So the admission run had these seeds and did not
# reach them; its cache is dated eleven days before the contract ledger, and this artifact does
# not claim to know whether that is staleness or a miss.
#
# THE LENGTH CONTROL IS WHAT MAKES THAT NUMBER MEAN ANYTHING. 446 further opcodes appear if a
# mutation is allowed to re-length the instruction, and those are the decoder reading a
# different instruction rather than a neighbour. This repository has paid for that mistake
# before, in a differential fit that enrolled a length bit and scored the mismatch as a
# coefficient. Without the control the sweep reported op5104 (14 -> 10) and op5100 (16 -> 10)
# as distance-1 neighbours, and they are not.
#
# The peer lane's (linker/g17-tensorops-recon @ 5cdd1b14): the accelerator class exhaustively.
# Apple declares 132 opcodes in sched 170-175; the decoder admits exactly TEN, and that holds
# under every CPU name this libLLVM knows - g17s, g17g and g18 admit the same ten, while g17,
# g16, g16s, g16g, g15 and g15s decode the identical bytes as op3773/op3789 and admit no
# accelerator opcode at all. So the 122 unadmitted forms are behind no CPU here, which is the
# same "declared for a subtarget this decoder is not" pattern the atomic family showed.
REACH_MEASUREMENT = dict(
    own_sweep=dict(
        instrument='tools/g17isareach.py, one bit of each Apple witness inside its real length',
        seeds=717, opcodes_reached=2435,
        outside_the_admitted_set=[475, 502, 529, 13790, 14470, 14662, 14666, 14758, 17010],
        rejected_by_the_length_control=446,
        note='all nine are Apple table rows reached from admitted seeds at the SAME length',
        what_the_nine_are=(
            'the peer lane reproduced all nine at distance 1 under the same own-length control '
            'and read their operand lists (linker/g17-tensorops-recon, peer_nine.json). They are '
            'WIDTH AND ADDRESSING-MODE STEPS, the next-wider or next-mode member of each family. '
            'byte15[7] in the 22-byte group widens the DESTINATION class by one step and changes '
            'nothing else: op14469->op14470 dest R0L 16-bit to R0 32-bit, op14661->op14662 and '
            'op14665->op14666 R0 to the pair R0_R1, op14757->op14758 tup4 to FIVE registers, a '
            'width Apple\'s decoder names and nothing here had. byte13[5] in the 14-byte trio '
            'replaces one address-expression operand with an immediate 16, eight operands to '
            'seven. op13808->op13790 at byte3[4] turns the first operand from an address '
            'expression into register R2, seven operands to six. op17013->op17010 at byte2[5] '
            'drops a register/immediate pair, seven to six. So the admission walk missed the '
            'WIDER and the differently-addressed members of families whose narrower members it '
            'already had')),
    peer_sweep=dict(
        source='linker/g17-tensorops-recon @ 5cdd1b14 (unmerged, unverified here)',
        scope='sched 170-175, the accelerator class',
        declared_by_apple=132, admitted_by_the_decoder=10,
        method='every 1/2/3-bit flip of each seed at 10 bytes and zero-extended to 12, about '
               '1.1M candidates, a bounded 5-deep BFS, and 1-2 flips of 14/16-byte extensions',
        claim_is='reachable within three flips, NOT "not admitted"',
        cpu_axis='g17s, g17g, g18 admit the same ten; g17, g16, g16s, g16g, g15, g15s admit no '
                 'accelerator opcode and decode the same bytes as op3773/op3789; the 122 others '
                 'are behind no CPU this libLLVM knows',
        decode_but_never_emitted=[5100, 5101]),
    what_neither_sweep_bounds=('how many forms sit behind a selector whose every single-bit '
                               'neighbour the decoder refuses. An accept-only search cannot '
                               'cross a rejected intermediate at ANY round count, which is why '
                               'the original walk\'s eight rounds do not help - and both sweeps '
                               'here are accept-only too'))


# Semantics established BY THIS SESSION on hardware: an isolated probe returned the value a
# prediction committed in advance said it would. That is the same standard as the ledger's
# `verified` class, but it is a different instrument and gets its own column - d3_probed_here -
# because merging my dispatches into a class I did not produce would misattribute both.
#
# Carrier: results/g17-tensorops-recon-v1/carrier.py from linker/g17-tensorops-recon, which loads
# R8..R11 from buffer A at lane*16 in scoreboard slot 0, executes the authored instruction
# verbatim, and stores R8..R11 to buffer C. Stimulus word 0 = 3*lane + 1 over 32 lanes in one
# threadgroup; words 1..3 are a pass-through control and were intact in every run.
#
# The wait mattered and the control caught it: without flags bit 24 (byte1[2], slot 0) every word
# came back zero, because the store read registers the load had not yet filled. Words 1..3 being
# zero is what said "the load did not land" rather than "the instruction returns zero".
PROBED_HERE = {
    # TEN FORMS DETERMINED BY MY OWN PROBES, 2026-09-17, both batches preregistered in their own
    # commits before any dispatch - the plans are 23260f38 and 1b93f7e2, the results 838232e0 and
    # b2109617, so the history dates every prediction before its run. Eleven and eight records
    # executed, CONTROL.op10279 reproducing [4100, 4101, 4356, 8196] in both, zero gpu events.
    #
    # WHAT MADE THEM DETERMINABLE WAS THE INPUTS, NOT THE LIBRARY. Every earlier record for these
    # opcodes used small positive words, on which identity, abs_s32, fabs, fsat and ffract all
    # return the operand unchanged - so the elimination census reported them as five-way ambiguous
    # and the map read `absent`. Batch 1 used -1.0f, 1.5f, -3.0f and 5; batch 2 used 0x0ABCDEF1,
    # -1.0f, 0xFF and 0x5A5A to separate what batch 1 left as a family.
    586: dict(name='move.32', width=8, predicted='0xBF800000, 0x3FC00000, 0xC0400000, 5', observed='0xBF800000, 0x3FC00000, 0xC0400000, 5',
              claim='a 32-bit move: the source returned unchanged. The five-way table was committed at 23260f38 before the dispatch and identity is the row that fits; abs_s32, fabs, fsat and ffract are each excluded by at least one case'),
    590: dict(name='zext16', width=8, predicted='0xDEF1, 0, 0xFF, 0x5A5A', observed='0xDEF1, 0, 0xFF, 0x5A5A',
              claim='zero-extension of the low sixteen bits, a & 0xFFFF - trunc16 in the table committed at 1b93f7e2, and the only one of eight offered candidates that fits all four cases'),
    10284: dict(name='sext16', width=12, prediction_held=False,
                predicted='0xDEF1, 0, 0xFF, 0x5A5A - trunc16, since sign extension was NOT in the '
                          'candidate set I committed',
                observed='0xFFFFDEF1, 0, 0xFF, 0x5A5A',
                claim='SIGN-extension of the low sixteen bits. MY PREREGISTERED SET DID NOT '
                      'CONTAIN IT: batch 2 offered six truncation widths and identity, all '
                      'zero-extending, so the closest row it could have matched was trunc16 and '
                      'the hardware returned the sign-extended value instead. Batch 1 could not '
                      'have caught the omission either - every low half in its inputs had bit 15 '
                      'clear, making sext16 and zext16 identical there. The corrected reading is '
                      'sign extension, and it is corrected by measurement rather than by my table'),
    408: dict(name='mask.ffef', width=10, prediction_held=False,
              predicted='0xDEF1, 0, 0xFF, 0x5A5A - trunc16, the nearest row of the set I committed',
              observed='0xDEE1, 0, 0xEF, 0x5A4A',
              claim='a & 0xFFEF, the low sixteen bits with BIT 4 CLEARED. MY PREREGISTERED SET DID '
                    'NOT CONTAIN IT: every candidate I offered was a contiguous mask or identity, '
                    'and this is neither. The cleared bit is measured and NOT explained - a single '
                    'hole in a mask is equally consistent with a source bit consumed by a field '
                    'this probe does not vary, and inherited operand bits have misled this project '
                    'four times. CONFIRMED A MASK BY BATCH 3 (plan 95f373e9, results '
                    'efce2b47): four inputs, two with bit 4 SET and two with it clear, and '
                    'the hole FOLLOWS THE INPUT - 0x0ABCDEF1 gives 0xDEE1 and 0x000000FF '
                    'gives 0xEF, while 0x8000ABCD whose bit 4 is already clear comes back '
                    'unchanged as 0xABCD. A source bit consumed by a field this probe does '
                    'not vary would have produced a FIXED value regardless of the input, so '
                    'the artefact reading is excluded by construction. a & 0xFFEF is the '
                    'only candidate of nineteen that fits, and its function still fell '
                    'outside the preregistered set, which that plan said in advance it '
                    'would'),
    # ELEVENTH FORM, 2026-09-17. Plans cc02af27 and 29b6d977, results b905d731 and 3d2c27c1, each
    # plan in its own commit before its dispatch. This one is here because its FIRST batch
    # (efce2b47) recorded it as an unreachable and was wrong about that: op612 returned 0 for all
    # four inputs, my candidate library held no function constant at 0, and elimination reported
    # nothing left. It is a range predicate and every input I had chosen was outside its window,
    # so zero was the correct answer four times. What was missing was inputs, not an instrument
    # and not a candidate.
    612: dict(name='range4', width=12, prediction_held=False,
              predicted='lo=0 w=16 src 0,13,14,15,16: 15, 7, 3, 1, 0 | lo=0 w=8 src 0,5,6,7,8: '
                        '15, 7, 3, 1, 0 | lo=8 hi=32 src 6,8,5,30,32: 12, 15, 8, 3, 0 | '
                        '[after the correction] lo=8 w=32 src 37,40,36: 7, 0, 15 | lo=16 w=8 src '
                        '21,24,16,13: 7, 0, 15, 8 | lo=0 w=0 src 0,5: 0, 0',
              observed='lo=0 w=16: 15, 7, 3, 1, 0 | lo=0 w=8: 15, 7, 3, 1, 0 | lo=8 w=32 src '
                       '6,8,5,30,32: 12, 15, 8, 15, 15 - TWO PLACES THE PREDICTION MISSED, '
                       'against 3 and 0 | lo=8 w=32 src 37,40,36: 7, 0, 15 | lo=16 w=8: 7, 0, '
                       '15, 8 | lo=0 w=0: 0, 0',
              claim='bit j of the destination, j in 0..3, is set iff lo <= src + j < lo + w, '
                    'unsigned, where lo is operand 4 and w is operand 5: a four-bit range '
                    'predicate over four consecutive element coordinates with the window given as '
                    'a base and a COUNT. 24 of 24 values across five immediate settings, no '
                    'exceptions. Three rival readings each died on a case chosen for it before '
                    'the run: a lower bound with no upper bound (needed 15,15,15 at lo=8 w=32 src '
                    '37,40,36), an absolute upper bound [lo, w) (needed zeros throughout lo=16 '
                    'w=8, where lo > w empties the window), and every shift- or mask-shaped rule '
                    '(dead on a LONE HIGH BIT, lo=8 w=32 src=5 -> 0b1000, reproduced at lo=16 w=8 '
                    'src=13 -> 8). MY FIRST PREDICTION OF THE SECOND IMMEDIATE WAS WRONG and is '
                    'kept above: I read the pair as (lo, hi) with an absolute bound, which is the '
                    'peer lane\'s published sentence at PEER_PIN, and it fits thirteen of the '
                    'fifteen values that batch measured. It is a width. Their examples cannot see '
                    'the difference - all but one sit at lo=0 where the readings coincide, and '
                    'their single lo=8 case (src 6 -> 0b1100) agrees under both - and that one '
                    'number IS reproduced here on this repository\'s instrument. Not asked: the '
                    'destination is taken as four bits because lo=0 w=16 at src=0 returned 15 and '
                    'not 0xFFFF, and no record varies the destination register class. CONFIRMED '
                    'ON ITS STRUCTURAL TWIN: op621 obeys the identical relation (plan 409c5d31) '
                    'and its own earlier ledger reading was this rule at lo=0'),
    # TWELFTH FORM, the same operation on its structural twin. Plan 409c5d31, results 3a-dated
    # below; op621 was ALREADY determined in a ledger of 2026-09-07 as a mask of
    # min(max(W - src, 0), 4) ones, which is this relation's lo=0 face, and every one of that
    # campaign's 27 sources ran with operand 4 at the witness's zero. Applying op612's reading to
    # it took one dispatch. The ledger is corrected at each point the old relation is stated, not
    # only at its end, and it now carries the answer to the question it left open.
    621: dict(name='range4', width=12,
              predicted='lo=0 w=32 src 29,30,31,32: 7, 3, 1, 0 | lo=16 w=8 src 21,24,16,13: '
                        '7, 0, 15, 8 | lo=8 w=32 src 5,6: 8, 12',
              observed='lo=0 w=32 src 29,30,31,32: 7, 3, 1, 0 | lo=16 w=8 src 21,24,16,13: '
                       '7, 0, 15, 8 | lo=8 w=32 src 5,6: 8, 12',
              claim='the same range predicate as op612: bit j set iff lo <= src + j < lo + w, '
                    'operand 4 the base and operand 5 the count. 10 of 10 values, every one '
                    'predicted. The first record reproduces the four points '
                    'ledger/g17-addr16-turns-a-count-into-a-mask.toml published for this opcode '
                    'ten days earlier, so the other two are readings rather than an instrument '
                    'that moved; the second parts the readings, since the ledger\'s '
                    'count-to-mask relation predicts 0,0,0,0 where a lower bound of 16 gives '
                    '7,0,15,8; and the third returns 0b1000, a lone high bit, which no mask of N '
                    'low ones can produce. op612 ran at the same settings in the same batch and '
                    'returned the same four values, so the two opcodes are ONE OPERATION and '
                    'nothing in either batch separates them - their corpus counts differ by two '
                    'orders of magnitude (891 against 7) and that is all that does. This also '
                    'answers what the ledger could not name: the FOUR is structural to the '
                    'operation, four consecutive element coordinates, which is why 97 per cent '
                    'of op612\'s results are consumed by loads and stores - it produces their '
                    'element mask, not their address, so the `addr16` name and the '
                    'memory_addressing family describe position rather than function',
              corrects='ledger/g17-addr16-turns-a-count-into-a-mask.toml, whose relation holds '
                       'only at operand 4 = 0 - the second time this opcode\'s fitted constant '
                       'was a value sitting in its witness, after the saturation point that was '
                       'really operand 5'),
    # FIVE FORMS FROM THE ARITY-1 SPLITTING CAMPAIGN, 2026-09-17. Plans 49ade47b and 1c774101,
    # results 6cb21223 and 61d82ef9. The first batch fitted them on four inputs; the second
    # re-asked at SIXTEEN that separate every one of the library's fifty-three candidates, twelve
    # of them values the fit had never seen, with all sixteen expected values committed before the
    # dispatch. Both controls were in each batch at the same inputs as the records.
    #
    # WHAT MADE THEM DETERMINABLE WAS THE LIBRARY, NOT THE INPUTS ALONE. The census offered only
    # integer and single-precision readings, so an fp16 opcode or a masked one could only ever
    # come back "no candidate fits" - 32 of the first batch's 58 records did. Adding half
    # precision and a mask family is what let these resolve, and adding the masks immediately
    # killed a fit that had looked unique (op3775's htrunc, which `a & 0xFFFC` matches on all four
    # of the old inputs).
    16805: dict(name='sar1', width=12,
                predicted='0, 0, 1, 2, 7, 8, 127, 16383, 16384, 32768, 1073741823, 3221225472, '
                          '3221225472, 90075000, 757935405, 3755999232',
                observed='0, 0, 1, 2, 7, 8, 127, 16383, 16384, 32768, 1073741823, 3221225472, '
                         '3221225472, 90075000, 757935405, 3755999232',
                claim='an arithmetic right shift by one: the sign bit is replicated, confirmed at '
                      '0x80000000 -> 0xC0000000 and 0x7FFFFFFF -> 0x3FFFFFFF. 16 of 16, twelve of '
                      'them inputs the fit never saw, against 26 competing integer candidates'),
    3807: dict(name='ceil.f16', width=10,
               predicted='hceil at all 16 inputs - 0, 1, 2, 4, 16, 1, 1, 2, -1, -0, 3, 1 in half, '
                         'then four high-bit words, the last of which is a NaN and returns the '
                         'canonical quiet 0x7E00',
               observed='hceil at all 16 inputs - 0, 1, 2, 4, 16, 1, 1, 2, -1, -0, 3, 1 in half, '
                        'then four high-bit words, the last of which is a NaN and returns the '
                        'canonical quiet 0x7E00',
               claim='the ceiling of a half: the unique survivor of 57 candidates over three '
                     'interpretations. THE NAME HOLDS. The only case my model missed was a NaN '
                     'input, where I predicted 0 because `_integral` did not propagate NaN - a '
                     'bug IEEE settles a priori, fixed as a bug rather than fitted'),
    3791: dict(name='floor.f16', width=10,
               predicted='hfloor at all 16 inputs', observed='hfloor at all 16 inputs',
               claim='the floor of a half, unique survivor of 57 candidates. THE NAME HOLDS'),
    3823: dict(name='trunc.f16', width=10,
               predicted='htrunc at all 16 inputs', observed='htrunc at all 16 inputs',
               claim='truncation of a half toward zero, unique survivor of 57. THE NAME HOLDS'),
    3855: dict(name='rsqrt.f16', width=10,
               predicted='hrsqrt at all 16 inputs', observed='hrsqrt at all 16 inputs',
               claim='the reciprocal square root of a half, unique survivor of 57 candidates. '
                     'THE NAME HOLDS, and it is IEEE about the sign of zero: rsqrt(-0.0) returns '
                     '-inf, preserving the sign through the reciprocal, where my model said +inf. '
                     'That case is why this form also CORRECTS an earlier claim of mine - the '
                     'half path does not canonicalise negative zero, individual OPERATIONS do'),
    3307: dict(name='fmul.f32.f16.to.f16', width=4,
               # THE GUARD CAUGHT THIS ONE. `predicted` first read "no expectation was
               # committed", which is true of the numeric vector and made the record a held
               # prediction whose predicted and observed text differed - so it counted as
               # confirmed while saying it had predicted nothing. The plan DID name a reading:
               # "The name predicts the first", f32_x_f16_to_f16 of three offered. That is the
               # prediction, and it is what the hardware picked.
               predicted='f32_x_f16_to_f16 - named in the plan as the reading the table\'s name '
                         'predicts, offered against hmul and hmul_f32_operands, with no numeric '
                         'expect vector because the record was to choose among the three',
               observed='f32_x_f16_to_f16 - named in the plan as the reading the table\'s name '
                        'predicts, offered against hmul and hmul_f32_operands, with no numeric '
                        'expect vector because the record was to choose among the three',
               claim='an f32 source times an f16 source, narrowed to f16 - exactly what the name '
                     'says, and a mixed-precision form this library could not express until this '
                     'batch. THE NAME HOLDS. I preregistered the reason it might be unreadable, '
                     'that the f32 operand could be a register PAIR being fed half its input, and '
                     'that worry was wrong: the mostly-zero output IS the correct answer, because '
                     'a half bit pattern read as f32 is a denormal near 1e-41 and underflows when '
                     'multiplied'),
    1277: dict(name='exp2.f16', width=10, prediction_held=False,
               predicted='hexp2, correctly rounded, at 16 inputs - including 0x7C00 (infinity) '
                         'for exp2(16), which overflows half',
               observed='13 of 16 exact. Every EXACT case right - 2^0, 2^1, 2^2, 2^4. One 1-ulp '
                        'miss at 2^0.25 (0x3CC1 against 0x3CC2). And exp2(16) returned 0x7BFF, '
                        'half\'s largest finite value 65504, NOT infinity',
               claim='base-2 exponential of a half. THE NAME HOLDS AND MY PREDICTION OF THE '
                     'UNIT DID NOT, which the scoring policy committed before the run named as '
                     'the expected outcome for a transcendental: the exact cases all match, so '
                     'the name is confirmed without relying on the unit\'s accuracy, and the '
                     'rounding differs by 1 ulp. THE SEPARATE FINDING IS THE SATURATION: this '
                     'unit clamps to the largest finite half instead of overflowing to infinity, '
                     'and that is kept as its own candidate (hexp2_sat) rather than folded into '
                     'hexp2, because editing the model would let a fit swallow the finding'),
    2575: dict(name='log2.f16', width=10, prediction_held=False,
               predicted='hlog2, correctly rounded, at 16 inputs',
               observed='15 of 16 exact, the miss 1 ulp (0x47AA against 0x47AB). log2(0) returned '
                        '-inf and log2 of a negative returned the canonical quiet NaN, both as '
                        'predicted. Most of the 16 are exact under any rounding: over every '
                        'retained record the census has 2575/log2.f16 at 22 of 25 interior cases '
                        'exact and 2 of the 5 that can tell rounding apart',
               claim='base-2 logarithm of a half. THE NAME HOLDS; the unit is approximate by 1 '
                     'ulp, so the bit-exact prediction is refuted and the name is not'),
    766: dict(name='fadd.sat.f16', width=10, prediction_held=False,
              predicted='hadd_sat - a half add clamped to [0, 1] - at 10 input pairs, where the '
                        'clamp RANGE was the guess and not the add',
              observed='9 of 10 exact. The clamp is confirmed as [0, 1]: (1.0, 2.0) gives 1.0 and '
                       '(-1.5, 1.5) gives 0. The miss is (-0.0) + (-0.0), which returned +0 where '
                       'IEEE and my model say -0',
              claim='a saturating half add, and `.sat` is a clamp on the RESULT to [0, 1] rather '
                    'than a modifier this record inherits - which was the alternative the plan '
                    'named. THE NAME HOLDS. The single miss is this form canonicalising negative '
                    'zero, where op3855 in the same batch preserves it, which is what establishes '
                    'that the behaviour is per-operation'),
    # RETRACTED 2026-09-18, SIX MORE AND FOR THE SAME REASON, found by making the test precise.
    # op13853 quad.sum.f16 and op13855 quad.product.f16 were both promoted with the observation
    # `0xDEF1, 0, 0xFF, 0x5A5A` - the input. So were op13875 quad.fmax.f16 and op13883
    # quad.fmin.f16, and op13873 quad.fmax.f32 and op13881 quad.fmin.f32 with their own input.
    # A sum and a product, and a maximum and a minimum, cannot both be the identity.
    #
    # The precise test is NOT "this opcode appears in a degenerate pair in some batch" - that
    # over-reports, and it would have condemned op16890 simd.smax and op16898 simd.smin, whose
    # promoted observations are 94 and 1 and therefore came from the distinct-lane batch. The test
    # is whether a form's OWN promoted observation equals its complement's. Eleven opcodes are
    # touched by a degenerate probe somewhere; six of them are actually degenerate here.
    #
    # RETRACTED 2026-09-18: op14040, op14301, op16860 and op16868 were promoted here as
    # `trunc16`, confirmed at sixteen inputs, and the fit is a PROBE ARTEFACT. Their table names
    # are quad.shuffle_up1, simd.shuffle_up1, simd.fmax.f16 and simd.fmin.f16 - every one a
    # CROSS-LANE operation - and this harness gives every lane the same value, so a reduction or a
    # shuffle returns that value and my library reads it as the low sixteen bits.
    #
    # The proof is in the retained results and needed no new dispatch: simd.fmax.f16 and
    # simd.fmin.f16 returned BYTE-IDENTICAL vectors in both batches, as did simd.sum.f16 against
    # simd.product.f16. A maximum and a minimum cannot agree unless the lanes are all equal.
    #
    # What caught it was root's rule of 2026-09-18 - score Apple's NAME as a separate evidence
    # dimension rather than as a footnote. Fifteen forms came back with the fitted function
    # disagreeing with the table name, and twelve of those fifteen are cross-lane names fitted as
    # lane-local functions. The hardware agreed with my prediction sixteen times out of sixteen
    # and the prediction was of the wrong function; only the name disagreed, and only because it
    # was being scored.
    #
    # The cross-lane forms that ARE determined here (op16874 simd.sum = 1520, op16890 simd.smax,
    # op16898 simd.smin and the rest) came from a batch built for the purpose, with DISTINCT
    # per-lane values - docs/archive/g17-isa-crosslane-batch2-predictions.md - which is why their sums are
    # not their inputs. That is the arm this batch lacked.
    16898: dict(name='simd.smin', predicted=1, observed=1,
                claim='a signed minimum reduction across the simdgroup'),
    16906: dict(name='simd.umax', predicted=94, observed=94,
                claim='an unsigned maximum reduction across the simdgroup'),
    16914: dict(name='simd.umin', predicted=1, observed=1,
                claim='an unsigned minimum reduction across the simdgroup'),
    16922: dict(name='simd.xor', predicted=64, observed=64,
                claim='an xor reduction across the simdgroup'),
    # second batch, predictions committed in docs/archive/g17-isa-crosslane-batch2-predictions.md
    16874: dict(name='simd.sum', predicted=1520, observed=1520,
                claim='a sum reduction across the simdgroup'),
    16890: dict(name='simd.smax', predicted=94, observed=94,
                claim='a signed maximum reduction across the simdgroup'),
    16830: dict(name='simd.and', predicted=0, observed=0,
                claim='a bitwise AND reduction across the simdgroup'),
    16882: dict(name='simd.or', predicted=127, observed=127,
                claim='a bitwise OR reduction across the simdgroup'),
    13889: dict(name='quad.sum', predicted='22,22,22,22,70,70,70,70 per quad',
                observed='22,22,22,22,70,70,70,70 per quad',
                claim='a sum reduction scoped to each QUAD of four lanes - constant within a '
                      'quad and different between quads, which neither a simdgroup-wide '
                      'reduction nor a scalar operation can produce'),
    13913: dict(name='quad.smin', predicted='1,1,1,1,13,13,13,13 per quad',
                observed='1,1,1,1,13,13,13,13 per quad',
                claim='a signed minimum reduction scoped to each quad of four lanes'),
    14289: dict(name='simd.shuffle_down1',
                predicted='neighbour value 3l+4 for lanes 0..30, lane 31 not predicted',
                observed='neighbour value 3l+4 for lanes 0..30, lane 31 not predicted',
                claim='a shuffle down by one lane. Lane 31 has no neighbour above it and CLAMPS '
                      'to its own value rather than wrapping to lane 0 - recorded, not '
                      'predicted, because inventing a prediction for an undefined case is how '
                      'an exception becomes a rule'),
    # THE PREDICTION THAT FAILED, and it is the most informative result in either batch.
    16873: dict(name='simd.prefix_sum', predicted='inclusive: 1,5,12,22,35,51,...',
                observed='0,1,5,12,22,35,51,70 - EXCLUSIVE',
                prediction_held=False,
                claim='an EXCLUSIVE prefix sum: lane l receives the sum of lanes strictly BELOW '
                      'it, so lane 0 returns 0. The name `simd.prefix_sum` does not say which, '
                      'and Metal spells both - simd_prefix_exclusive_sum and '
                      'simd_prefix_inclusive_sum appear separately in the construct vocabulary. '
                      'My prediction was the inclusive scan and the hardware returned the '
                      'exclusive one, shifted by exactly one lane'),
}
PROBED_NOTE = (
    'every lane returned the same value, and for op16906 and op16922 that value is one no single '
    'lane held: 94 is the maximum of 3*lane+1 which only lane 31 holds, and 64 is the xor of all '
    'thirty-two inputs which no lane holds at all. A scalar operation returns the lane\'s own '
    'value and a shuffle returns a neighbour\'s, so neither can produce these. The names came '
    'from the isolation sweep that once called msb "clz", so the prediction was written as a '
    'five-way table before dispatch rather than as "the name is right"')


def probed_axis(opcode, name, length=None):
    """The semantics axis for an opcode I probed on hardware, at the width I probed it at.

    A ROW MAY NAME ITS WIDTH AND MUST BE HONOURED WHEN IT DOES. This table is keyed on the opcode
    and originally returned for every width of it, which was harmless only because every entry in
    it happened to be single-width. op586 and op590 are not: both carry widths 4 and 8, and the
    batch that determined them dispatched the EIGHT-byte form. Returning the row for len4 would
    assert a determination at a width no probe ever ran - the opcode-for-form spreading that has
    inflated three columns of this map already. A row without `width` keeps the old behaviour,
    which is correct for the single-width entries that predate this.
    """
    row = PROBED_HERE.get(opcode)
    if row is None or name != 'semantics':
        return None
    if row.get('width') is not None and length is not None and int(row['width']) != int(length):
        return None
    held = row.get('prediction_held', True)
    return dict(state=('%s - CONFIRMED ON HARDWARE, predicted %s before dispatch and observed %s'
                       if held else
                       '%s - MY PREDICTION WAS REFUTED. I predicted %s and the hardware returned '
                       '%s; the claim above is the corrected reading')
                      % (row['claim'], row['predicted'], row['observed']),
                evidence='checked', source='results/g17-isa-crosslane-v1 + '
                                          'docs/archive/g17-isa-crosslane-restated-predictions.md',
                probed_here=True, predicted=row['predicted'], observed=row['observed'],
                prediction_held=held, note=PROBED_NOTE)


def peer_axis(opcode, name):
    for fact in PEER_FACTS.get(opcode, ()):
        if fact['axis'] == name:
            return dict(state=fact['state'], evidence='corpus', source=PEER,
                        peer_reported=True, verified_here=False,
                        means=('reported by another lane on an unmerged branch. Recorded because '
                               'it is the only evidence here for this family; counted in its own '
                               'column, never in this map\'s hardware denominator'))
    return None


def ledger_axis(opcode, name, length=None):
    """The curated ledger fact for this opcode and axis, as an axis dict, or None.

    A fact may carry `widths`, and then it reaches ONLY those widths. Both fact dicts are keyed by
    OPCODE, so without this a fact established at one width lands on every width of that opcode.
    For an axis that is only informative that is a stated limitation; for `semantics` it INFLATES
    A DENOMINATOR, because D3 reads that axis. The texture read is the case that forced it: the
    fetch executed at op15813/len8 and the opcode also has a len14 form nothing has run, and
    without `widths` texture_image came off zero at TWO forms on evidence for one.
    """
    for fact in tuple(LEDGER_FACTS.get(opcode, ())) + tuple(ISA_FACTS.get(opcode, ())):
        if fact['axis'] != name:
            continue
        widths = fact.get('widths')
        if widths and length is not None and length not in widths:
            continue
        return axis(fact['state'], fact['evidence'], fact['source'],
                    ledger_status=fact['ledger_status'], note=fact['note'],
                    **({'established_at_widths': list(widths)} if widths else {}))
    return None


# How a ledger's OWN status string places it on the ladder. Derived from the whole vocabulary,
# not guessed. Two classes are excluded from evidence entirely rather than downgraded, because a
# withdrawn claim cited as support is worse than no citation: `retracted` and `superseded`.
# `measured` is deliberately NOT placed - in this corpus it covers both decode-side and on-hardware
# measurement, and an axis may not carry evidence its source does not establish.
STATUS_CLASS = {
    'causal': 'checked', 'executed': 'executed', 'structural': 'decoder',
    'mechanical': 'table', 'compiled': 'oracle', 'measured': 'ambiguous',
    'live': 'lifecycle', 'negative': 'refutation', 'negative result': 'refutation',
    'open': 'open', 'partial': 'open', 'method': 'process', 'correction': 'process',
    'harness defect': 'instrument defect', 'instrument defect': 'instrument defect',
    'measurement defect': 'instrument defect', 'measurement': 'ambiguous',
}
EXCLUDED_STATUS = ('retracted', 'superseded')
# ledger/STATE.md is RENDERED from the entries (tools/ledger.py state): indexing it counted every
# opcode an entry names a second time, under a file that is not a ledger, and its "retracted"
# heading listed it among the ledgers declaring a retraction.
LEDGER_VIEWS = frozenset({'STATE.md'})


def status_head(value):
    return str(value).split(':')[0].split(' -')[0].split(',')[0].split(';')[0].strip().lower()


def not_citable(doc):
    """Why a parsed ledger may not be cited as support, or None: 'status' when its status head is
    retracted or superseded, 'truth' when its truth header is (tools/ledger.py). A milestone or note
    that a later entry overturned keeps the status it was written with - 'measured', 'causal' - so
    the status alone indexed every one of them as citable; the truth header is what says so."""
    status = doc.get('status') if isinstance(doc.get('status'), str) else None
    head = status_head(status) if status else ''
    if head and any(head.startswith(x) for x in EXCLUDED_STATUS):
        return 'status'
    if doc.get('truth') in EXCLUDED_STATUS:
        return 'truth'
    return None


def ledger_entries():
    """(name, text, doc) for every ledger entry. The entries live as tables in six area files since
    2026-09-26 (tools/ledger.py); `name` is '<id>.toml', the name every citation of an entry uses, and
    `text` is the entry's own text as its old file read, so an id in a header is not read as prose."""
    sys.path.insert(0, str(ROOT/'tools'))
    import ledgerfiles
    for stem, e in sorted(ledgerfiles.entries(str(ROOT), with_text=True).items()):
        yield stem + '.toml', e.get('_text', ''), {k: v for k, v in e.items() if not k.startswith('_')}


def ledger_index():
    """Index ledger/ by opcode: a CITATION index, not a semantic harvest.

    For each ledger it records the file, its own title or claim, its own status string and the
    ladder class that status implies. For each opcode it records which ledgers mention it, split
    into two tiers: `about` when the opcode appears in the ledger's title or claim, and
    `mentions` when it appears only in the body. Neither tier asserts what the ledger established
    about that opcode - only that it is cited there, which is what a reader needs in order to go
    and read it.

    Retracted and superseded ledgers are EXCLUDED and counted separately, by status or by truth
    header (not_citable). A withdrawn claim cited as support would be worse than no citation at all.
    """
    import tomllib
    root = ROOT/'ledger'
    if not root.exists():
        return dict(present=False)
    per_opcode, ledgers = defaultdict(lambda: dict(about=[], mentions=[])), {}
    excluded, unparsed, unmapped = [], [], Counter()
    for name, text, doc in ledger_entries():
        status = doc.get('status') if isinstance(doc.get('status'), str) else None
        head = status_head(status) if status else None
        # STARTSWITH, not equality. Three ledgers read "retracted against Apple's decoder",
        # "retracted and reverted" and "retracted and replaced by measurement the same day" -
        # the last of which says its own claim was false - and an exact match indexed all three
        # as citable evidence.
        why = not_citable(doc)
        if why:
            excluded.append(dict(file=name, status=status, excluded_by=why,
                                 superseded_by=doc.get('superseded_by') if why == 'truth' else None,
                                 headline=str(doc.get('title') or doc.get('claim') or '')[:160]))
            continue
        klass = STATUS_CLASS.get(head) if head else None
        if head and klass is None:
            unmapped[head] += 1
        headline = str(doc.get('title') or doc.get('claim') or '').strip()
        # A ledger can be partly withdrawn while its status still reads as evidence: 33 of them
        # are, and g17-tensor-execution-harness.toml is the clearest - status "causal - executed"
        # with a title beginning RETRACTED IN PART and a [retraction] section explaining that the
        # finding "IS NOT HARDWARE BEHAVIOUR". A status filter cannot see any of that, so it is
        # FLAGGED rather than excluded: such a ledger usually still holds sound parts, and this
        # one says which are unaffected.
        # TWO DEFECTS FIXED HERE 2026-09-17, both found by asking why a ledger whose status
        # literally begins "RETRACTION" was not flagged.
        #
        # THE STATUS WAS NEVER READ. Seven ledgers declare a retraction in `status` and none was
        # flagged - including g17-the-atomic-works-and-the-observable-did-not.toml, whose status
        # is "RETRACTION and result: the capability works; three entries above are wrong", which
        # is exactly the kind a citation must carry. The guard that makes a fact say so reads this
        # field, so an unflagged retraction is a citation the guard cannot check. No fact cited
        # any of the seven when this was fixed, so widening the detection changed no verdict -
        # which is the point of fixing it BEFORE adding a fact that would have walked through it.
        #
        # AND A PRECEDENCE BUG THAT MADE THE REASONS WRONG WHENEVER THE TITLE MATCHED. The
        # conditional expression binds LOOSER than `|`, so `{'title'} if t else set() | a | b`
        # parses as `{'title'} if t else (set() | a | b)` - a title match discarded the section
        # and body reasons rather than joining them. Membership was unaffected, because any
        # non-empty list flags, but every reported reason for a title-matching ledger was a
        # single 'title' that may have hidden a [retraction] SECTION.
        reasons = set()
        if re.search(r'retract', headline, re.I):
            reasons.add('title')
        if re.search(r'retract|supersed|withdraw', str(status or ''), re.I):
            reasons.add('status')
        if any(re.search(r'retract|supersed|withdraw', k, re.I) for k in doc):
            reasons.add('section')
        if re.search(r'RETRACTED IN PART|IS NOT HARDWARE BEHAVIOUR', text):
            reasons.add('body')
        withdrawal = sorted(reasons)
        ledgers[name] = dict(status=status, status_class=klass, date=str(doc.get('date') or ''),
                               milestone=str(doc.get('milestone') or ''), headline=headline[:200],
                               declares_a_retraction=withdrawal or None)
        in_headline = set(re.findall(r'op(\d{3,5})', headline))
        for op in set(re.findall(r'op(\d{3,5})', text)):
            tier = 'about' if op in in_headline else 'mentions'
            per_opcode[int(op)][tier].append(name)
    return dict(present=True, files_indexed=len(ledgers), unparsed=unparsed,
                excluded_retracted_or_superseded=excluded,
                # a ledger with no status recorded is its own category, named rather than null
                status_classes=dict(Counter(v['status_class'] or '(no status recorded)'
                                            for v in ledgers.values())),
                status_heads_unmapped=dict(unmapped),
                declaring_a_retraction=sorted(
                    n for n, v in ledgers.items() if v['declares_a_retraction']),
                declaring_a_retraction_count=sum(
                    1 for v in ledgers.values() if v['declares_a_retraction']),
                opcodes_with_a_citation=len(per_opcode),
                opcodes_a_ledger_is_about=sum(1 for v in per_opcode.values() if v['about']),
                by_opcode={str(k): v for k, v in sorted(per_opcode.items())},
                ledgers=ledgers,
                note=('a citation index: it says where an opcode is discussed and at what status, '
                      'never what was established about it. Tier `about` means the opcode appears '
                      'in the ledger headline; `mentions` means only in the body.'))


# Axis keywords for MECHANICAL claim extraction. Each maps a word that appears in a ledger
# sentence to the axis that sentence most likely concerns. This is a triage aid, not a reading:
# the extracted sentence is published VERBATIM with its ledger and status so a consumer can go
# read the source, and it never feeds a denominator.
CLAIM_KEYWORDS = (
    (('wait', 'latency', 'cycles', 'in-flight', 'load-use'), 'latency_wait'),
    (('barrier', 'synchroni', 'fence'), 'barrier_sync'),
    (('exec mask', 'exec ', 'predicate', 'mask stack', 'lanes active'), 'flags_exec'),
    (('slot', 'operand', 'modifier', 'field is', 'immediate'), 'operand_encoding'),
    (('register file', 'file, not a register', 'tuple', 'GPR', 'address file'), 'register_files'),
    (('bytes', 'byte-exact', 'length', 'width'), 'length'),
    (('atomic', 'threadgroup', 'device memory'), 'memory_effects'),
    (('texture', 'sampler', 'texel'), 'memory_effects'),
    (('opcode selection', 'selects the opcode', 'reselect', 'flips to'), 'opcode_selection'),
    (('stack', 'depth', 'ring', 'resource'), 'resources'),
    (('returns', 'computes', 'executed', 'correct value', 'arithmetic'), 'semantics'),
)


def extract_claims(root=None):
    """Pull the verbatim sentence around every opcode mention in every indexed ledger.

    The hand-curated LEDGER_FACTS above took one ledger at a time and there are 598 of them, so
    this does the mechanical part: for each ledger and each opcode it names, the sentence that
    names it, the ledger's own status, and a keyword guess at which axis the sentence concerns.

    It is deliberately NOT a reading. Nothing here is promoted to an axis value or counted in a
    denominator; the sentence is published verbatim so that a consumer can go to the source, and
    every row is marked extracted and unreviewed. A keyword guess at an axis is worth exactly
    what a keyword guess is worth, and saying so is what keeps it useful.
    """
    import tomllib
    root = Path(root or ROOT/'ledger')
    if not root.exists():
        return dict(present=False)
    per_opcode, skipped = defaultdict(list), Counter()
    for name, text, doc in ledger_entries():
        status = doc.get('status') if isinstance(doc.get('status'), str) else None
        head = status_head(status) if status else ''
        if not_citable(doc):
            skipped['excluded_retracted_or_superseded'] += 1
            continue
        klass = STATUS_CLASS.get(head)
        withdrawn = bool(re.search(r'retract', str(doc.get('title') or ''), re.I)
                         or any(re.search(r'retract|supersed|withdraw', k, re.I) for k in doc))
        # sentences, flattened out of the prose sections
        blobs = [str(doc.get('title') or '')] + [
            v.get('text') if isinstance(v, dict) else v
            for v in doc.values() if isinstance(v, (str, dict))]
        for blob in blobs:
            if not isinstance(blob, str):
                continue
            for sentence in re.split(r'(?<=[.;])\s+|\n\n', ' '.join(blob.split())):
                mentioned = set(re.findall(r'op(\d{3,5})', sentence))
                if not mentioned or len(sentence) < 24:
                    continue
                lowered = sentence.lower()
                axes = sorted({axis for words, axis in CLAIM_KEYWORDS
                               if any(w in lowered for w in words)})
                for opcode in mentioned:
                    per_opcode[int(opcode)].append(dict(
                        ledger=name, status=status, status_class=klass,
                        ledger_declares_a_retraction=withdrawn,
                        axis_guess=axes or ['unclassified'], sentence=sentence[:400]))
    # deterministic, and capped so one chatty ledger cannot dominate an opcode
    trimmed = {}
    for opcode, rows in per_opcode.items():
        # the title is read once as itself and once as a doc value; dedupe on (ledger, sentence)
        seen, unique = set(), []
        for row in rows:
            key = (row['ledger'], row['sentence'])
            if key not in seen:
                seen.add(key)
                unique.append(row)
        rows = unique
        rows.sort(key=lambda r: (RANK.get(r['status_class'] or 'absent', 0) * -1,
                                 r['ledger'], r['sentence']))
        trimmed[str(opcode)] = rows[:12]
    return dict(present=True, opcodes=len(trimmed),
                claims=sum(len(v) for v in trimmed.values()),
                skipped=dict(skipped), by_opcode=trimmed,
                means=('EXTRACTED AND UNREVIEWED. The sentence is verbatim from the named ledger '
                       'and the axis is a keyword guess. Nothing here is an axis value and '
                       'nothing here is counted: it exists so that 598 ledgers are searchable '
                       'per opcode without anyone claiming to have read them.'))


# The peer lane's reconnaissance, PINNED to an immutable commit. Pinning is not fussiness: the
# branch is unmerged and moves several times an hour, and an artifact harvested from a moving ref
# cannot be regenerable. If the commit is not fetched the extraction reports itself ABSENT rather
# than returning nothing, because an empty harvest and an unavailable source must not look alike.
PEER_PIN = '447ddc2d6bb198be1b05f9cd67438b47d91f0255'
PEER_DOC = 'docs/g17-tensorops-accelerator-recon.md'

# A correction written BELOW a claim leaves the claim standing, and a verbatim line harvest
# republishes both with nothing to say which one survived. The peer's recon does exactly this:
# section 10 still reads "op11456 is the mask generator" and section 32 reads "the mask generator
# is op612, not op11456 (correction to section 10)". Quoting is still the whole method - this
# detects supersession using the document's OWN correction language and never by reviewing the
# claim, which would not be mine to do.
CORRECTION_SECTION = re.compile(r'correction to section (\d+)', re.I)
CORRECTION_TARGET = re.compile(r'\bnot op(\d{3,5})\b', re.I)
DOC_SECTION = re.compile(r'^#+\s*(\d+)\.')
# The role the correction takes away, read out of its own sentence: "The mask generator is op612,
# not op11456" yields `mask generator`, and the row that claims that role for op11456 is the one
# withdrawn. Matching on the section number the correction CITES does not work - this document's
# cross-references run one behind its own headings, so "correction to section 10" withdraws a
# claim printed under "## 11. Batch 8", and the number matched an unrelated barrier sentence
# instead. The opcode and the role phrase are both stated outright; the cited number is not
# trusted, only reported.
CORRECTION_ROLE = re.compile(
    r'^\**\s*(?:the\s+)?([a-z][a-z0-9 /_-]{2,40}?)\s+is\s+op\d{3,5},\s*not\s+op(\d{3,5})', re.I)


def asserts_role(line, opcode, role):
    """Does this line CLAIM the role for this opcode, rather than mention or deny it?

    A substring test on the role phrase is not this question, and it marked two lines of the
    peer's own correction as withdrawn claims: "op11456 is not a mask generator (op612 already is
    one)" and a line quoting "op11456 is not the mask generator" both contain the phrase and both
    say the opposite of the claim being retired. Marking a correction as the thing it corrects is
    the same defect one level down from the one this whole pass exists to fix.

    So the line must positively assert it: `op<n> is [the|a] <role>`, where an intervening "not"
    breaks the match because neither "the" nor "a" nor the role itself follows the verb.
    """
    return re.search(r'op%d\s+(?:is|=)\s+(?:the\s+|a\s+)?%s' % (opcode, re.escape(role)),
                     line, re.I) is not None


def classify_peer_lines(text):
    """Opcode-bearing lines with their section, and which of them the document itself withdraws.

    Pure in its input so the guard can be fired on a document that states a claim and then
    corrects it, rather than on whatever the live pin happens to contain: a supersession detector
    tested only against a document that HAS a correction cannot fail for the right reason.

    Two strengths, kept apart because they are not equally certain. `superseded_by` needs the
    correction to name the opcode it withdraws ("not op11456") AND to name the role it takes away
    ("the mask generator is op612, not op11456"), so the withdrawn line can be identified by that
    role rather than guessed. `contested_by` is the weaker reading applied to the opcode's other
    lines: the document disputes this opcode somewhere, and which line died is not determinable
    from the correction's words. Neither is a verdict on their work; both are the document
    disagreeing with itself in its own language. Where the correction cites a section number, it
    is recorded and checked, never used to choose the target - see `cited_section_mismatch`.
    """
    rows, corrections, section = defaultdict(list), [], None
    for raw in text.splitlines():
        head = DOC_SECTION.match(raw.strip())
        if head:
            section = int(head.group(1))
        line = ' '.join(raw.strip().lstrip('|-* ').split())
        mentioned = set(re.findall(r'op(\d{3,5})', line))
        if not mentioned or len(line) < 20:
            continue
        lowered = line.lower()
        axes = sorted({axis for words, axis in CLAIM_KEYWORDS if any(w in lowered for w in words)})
        withdraws = {int(o) for o in CORRECTION_TARGET.findall(line)}
        target_section = CORRECTION_SECTION.search(line)
        role = CORRECTION_ROLE.match(line)
        if withdraws:
            corrections.append(dict(line=line[:400], withdraws=sorted(withdraws),
                                    role_withdrawn=role.group(1).strip().lower()
                                    if role else None,
                                    cited_section=int(target_section.group(1))
                                    if target_section else None))
        for opcode in mentioned:
            rows[int(opcode)].append(dict(axis_guess=axes or ['unclassified'], line=line[:400],
                                          section=section,
                                          corrects=sorted(withdraws) or None))
    for correction in corrections:
        role = correction['role_withdrawn']
        found_in = set()
        for opcode in correction['withdraws']:
            for row in rows.get(opcode, []):
                if row['line'] == correction['line']:
                    continue
                if role and asserts_role(row['line'], opcode, role):
                    row['superseded_by'] = correction['line']
                    found_in.add(row['section'])
                else:
                    row.setdefault('contested_by', correction['line'])
        # The cited number is checked against where the withdrawn claim actually sits, and a
        # disagreement is published. Silently trusting it is what marked the wrong line.
        correction['withdrawn_claim_sections'] = sorted(s for s in found_in if s is not None)
        correction['cited_section_mismatch'] = bool(
            correction['cited_section'] is not None and found_in
            and correction['cited_section'] not in found_in)
    return rows, corrections


def extract_peer_claims():
    """Verbatim opcode-bearing lines from the peer lane's recon, at a pinned commit.

    Same mechanical treatment as this repository's own ledgers, and the same refusal to call it a
    reading: the line is quoted, the axis is a keyword guess, and every row says the work is
    another lane's, unmerged, and unverified here. It is included because it is the only evidence
    this map has for the accelerator family, and excluded from every denominator for the same
    reason - it is not this artifact's measurement to count.

    The pin is the peer's PUSHED tip. Their live reading runs ahead of it - on 2026-09-17 it did
    so twice, by five unpushed commits and then again - and that is deliberately not harvested: a
    commit that can still be amended or rebased is not a pin.

    NOTHING HERE MAY CONSULT A REF THAT CAN MOVE, and the first version of this did. It published
    a `pin_lag` measured against `origin/linker/...` and the peer's local branch, so the artifact
    changed the moment they committed - with nothing changing in this repository - and the
    regenerability test went red on a field that had no business being pinned. A pinned extraction
    must be a pure function of (PEER_PIN, PEER_DOC); the lag is a LIVENESS question and is printed
    by the report instead, where being out of date is the point. `peer_pin_lag` is still here and
    the test asserts this function does not call it.
    """
    import subprocess
    try:
        text = subprocess.run(['git', '-C', str(ROOT), 'show', '%s:%s' % (PEER_PIN, PEER_DOC)],
                              capture_output=True, text=True, check=True).stdout
    except Exception as exc:
        return dict(present=False, pinned_commit=PEER_PIN, document=PEER_DOC,
                    reason='%s - fetch the branch to regenerate this artifact'
                           % type(exc).__name__)
    per_opcode, corrections = classify_peer_lines(text)
    trimmed, superseded = {}, 0
    for opcode, rows in per_opcode.items():
        seen, unique = set(), []
        for row in rows:
            if row['line'] not in seen:
                seen.add(row['line'])
                unique.append(row)
        # Corrections first, then WITHDRAWN rows, then the rest - because the ten-row cap must
        # never drop either half of a disagreement. Sorting withdrawn rows last (which is what
        # this did) silently deleted all three of op11456's from the artifact and left
        # superseded_claims reading 0, so the published record showed no withdrawal at all: worse
        # than publishing the claim unmarked, since the reader cannot even see it was there.
        unique.sort(key=lambda r: (r.get('corrects') is None, 'superseded_by' not in r))
        superseded += sum(1 for r in unique[:10] if 'superseded_by' in r)
        trimmed[str(opcode)] = unique[:10]
    return dict(present=True, pinned_commit=PEER_PIN, document=PEER_DOC,
                branch='linker/g17-tensorops-recon (unmerged)',
                opcodes=len(trimmed), claims=sum(len(v) for v in trimmed.values()),
                corrections=corrections, superseded_claims=superseded,
                by_opcode=trimmed,
                means=('another lane\'s reconnaissance, quoted verbatim at a pinned commit. '
                       'EXTRACTED, UNREVIEWED, and NOT VERIFIED HERE. Counted in no denominator: '
                       'their measurement is theirs to defend, and folding it into this map\'s '
                       'numbers would be claiming their work as this artifact\'s evidence. Rows '
                       'carrying `superseded_by` are ones the document itself withdraws further '
                       'down - detected from its own correction language, not reviewed here.'))


def peer_pin_lag():
    """How far the pin sits behind the peer's pushed tip, and behind their local work.

    An extraction can be regenerable and stale at the same time, and only one of those is visible
    from inside it. Reported so the staleness is a published number rather than something a reader
    discovers by asking the peer.
    """
    import subprocess

    def count(rng):
        try:
            out = subprocess.run(['git', '-C', str(ROOT), 'rev-list', '--count', rng],
                                 capture_output=True, text=True, check=True).stdout
            return int(out.strip())
        except Exception:
            return None
    def exists(rev):
        return subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--verify', '--quiet', rev],
                              capture_output=True).returncode == 0

    # THE BRANCH IS REAPED WHEN ITS PR MERGES, and this measured the lag against the branch REF.
    # Deleting `linker/g17-tensorops-recon` after #52 merged made both numbers None and failed the
    # test that requires them to be measurable - a check pinned to a path rather than to content,
    # which is this repository's own lesson. Once the work is on main the lag against main is the
    # meaningful number, and it is 0 by construction; the reaped state is reported rather than
    # silently substituted.
    ref = 'origin/linker/g17-tensorops-recon'
    local = 'linker/g17-tensorops-recon'
    reaped = not exists(ref)
    if reaped and count('%s..origin/main' % PEER_PIN) is not None:
        ref, local = 'origin/main', 'origin/main'
    return dict(behind_pushed_tip=count('%s..%s' % (PEER_PIN, ref)),
                pushed_tip_behind_their_local=(0 if reaped else
                                               count('%s..%s' % (ref, local))),
                branch_reaped=reaped,
                measured_against=ref,
                means=('behind_pushed_tip 0 means the pin IS their published tip. The second '
                       'number is their unpushed local work, which this extraction refuses to '
                       'quote because an unpushed commit is not immutable. When the branch has '
                       'been reaped after merging, the lag is measured against origin/main and '
                       '`branch_reaped` says so - a deleted ref is not an unmeasurable lag.'))


def field_map_width_scope(src):
    """Opcodes witnessed at several widths for which the field-map API returns ONE length.

    A field map is keyed to a width, and `g17auth.length(op)` answers per OPCODE. For an opcode
    witnessed at one width those are the same question; for 163 of them they are not, and a
    consumer that asks `length(op)` and then reads bits at another of that opcode's widths is
    reading positions keyed to a different form. Nothing raises.

    THIS IS A MEASURED INSTANCE, NOT A WORRY. The peer lane closed op11456 (widths 8, 10 and 14)
    and reported: "needed my own bit-flip census to find imm_e/imm_f's physical bits on the
    10-byte form, since your contract's field map is keyed to the unrelated 14-byte encoding and
    didn't apply byte-for-byte". Checked here rather than taken on report - length(11456) is 14 -
    and then censused, because one instance of a scope error is an anecdote and the population is
    the fact. It is the same defect this artifact corrected in its own ledger today, where
    "op612 writes somewhere a GPR store cannot read" was true of the instruction that batch
    emitted and false of the opcode.
    """
    # WIDTHS COME FROM `formbits`, WHICH IS KEYED PER FORM. The first version of this read
    # `src['lengths'].get(opcode)`, which is keyed by STRING opcode and holds ONE length each, so
    # an int lookup returned None for every row and the census reported 0 opcodes - a zero that
    # measured the reader, which is why the count is asserted in the test rather than trusted.
    per_opcode = defaultdict(set)
    for key in src['formbits']:
        op_s, len_s = key.split(',')
        per_opcode[int(op_s)].add(int(len_s))
    rows = {}
    for opcode, rec in sorted(src['contract'].items()):
        widths = sorted(per_opcode.get(opcode) or ())
        if len(widths) < 2:
            continue
        try:
            keyed = g17auth.length(opcode)
        except Exception:
            keyed = None
        rows['op%d' % opcode] = dict(
            witnessed_widths=widths, field_map_keyed_to=keyed,
            corpus_instances=(rec.get('meaning') or {}).get('corpus_instances'),
            other_widths=[w for w in widths if w != keyed])
    return dict(
        opcodes=len(rows), by_opcode=rows,
        means=('for each of these the field-map API answers with ONE width while the decoder '
               'admits several, so a consumer reading bits at any other width of the same opcode '
               'is using positions keyed to a different form and gets no error. Confirmed on '
               'op11456 by another lane, which had to run its own bit-flip census on the 10-byte '
               'form because this repository\'s map is keyed to 14. Not a claim that any '
               'particular field is wrong - a claim that the question the API answers is narrower '
               'than the question its callers ask. AND THE REASON IT IS SILENT, from that lane: '
               'the decoded operand INDICES line up across the widths while the physical bit '
               'positions do not, so a consumer checks that operand 4 exists on both forms, finds '
               'it does, and reads the wrong bits. Their conclusion after hitting it three times '
               'in one day: never assume a decoded bit-N lives at the positionally-matching '
               'byte and bit'))


def named_unknowns(src, spec, universe, totals, d2b, unsep, fmws):
    """The unknown side of this map, sorted by whether it can be NAMED - not just counted.

    "6,833 forms have no semantics" is a true sentence that tells a reader nothing about what to
    do. What is worth publishing is the NAMEABILITY of each gap, because that is what says whether
    a gap is a work item, a research programme, or a limit of the method:

      tier 1  enumerable - the members are listed here, individually, with why each is stuck
      tier 2  characterised - too many to list usefully, but the MECHANISM is measured, so the
              class is named even though its members are not
      tier 3  unbounded - not nameable from inside this artifact at all, and saying so is the
              only honest treatment

    This section feeds NO denominator and is asserted to feed none. Filling a coverage axis with
    a statement of ignorance is how a count rises without anything being learned, and this
    repository has a guard against exactly that (an-absent-axis-is-not-a-gap-to-fill). What is
    counted here is what is NOT known; it is a liability register, not an asset.
    """
    contract = src['contract']
    # FROM `universe`, NOT `spec`. The family is an OPCODE-level field and `spec` is keyed by
    # form ('op612' -> {'len12': ...}), so `.get('family')` on its values returns None for every
    # entry and the count came back 6,718 - the whole universe, reported as having no family. A
    # wrong structure gives a confident wrong number that reads like a finding, and the only
    # reason this one was caught is that it equalled the denominator exactly.
    fams = Counter((r.get('family') or 'unknown') for r in (universe or {}).values())
    undetermined = ((d2b.get('undetermined_in_apples_whole_corpus') or {}).get('counts') or {})
    singletons = sorted(k for k, v in undetermined.items() if v <= 1)

    tier1 = dict(
        opcodes_with_no_semantics_from_any_instrument=dict(
            opcodes=unsep.get('opcodes_with_no_other_semantics') or [],
            why=('every other opcode in the unseparated records already has semantics from '
                 'another instrument, so probing it is a cross-check. These four do not - they '
                 'are the only places in the ambiguous pool where a determination would be new '
                 'coverage rather than corroboration'),
            what_would_close_it=('op1028 HAS now been dispatched and returned 0x0000, 0x7C00, '
                                 '0x0000, 0xFC00 - half +inf and -inf bit patterns - which no '
                                 'candidate in the library reproduces, so it has data and no '
                                 'function. op9788 needs a three-source record design this '
                                 'harness cannot author yet. op10267 has ZERO defs in the whole '
                                 'corpus and is unreachable by any probe that needs a witness. '
                                 'op9837 is an fp16 candidate and the class now exists')),
        forms_whose_undetermined_bits_no_corpus_can_settle=dict(
            forms=singletons,
            why=('a bit verdict needs two instances of the FORM, and across both committed '
                 'Apple corpora, each distinct text once, each of these appears once or not at '
                 'all. Measured, not assumed: these are the forms whose count in '
                 'undetermined_in_apples_whole_corpus is below two; the other %d forms there '
                 'are reachable by reading'
                 % ((d2b.get('undetermined_in_apples_whole_corpus') or {}).get(
                     'reachable_by_reading_more_corpus', 0))),
            what_would_close_it=('not corpus. Either a dispatch that varies the bit and reads '
                                 'the result, or Apple\'s compiler emitting the form at a '
                                 'parameter it has not been asked for')),
        forms_absent_from_apples_corpus_entirely=dict(
            forms=(d2b.get('undetermined_in_apples_whole_corpus') or {}).get(
                'absent_from_apples_corpus') or [],
            why=('the free-bits instance came from a probe or a repair-walk witness, not from '
                 'Apple. This form is in D1 because the decoder addresses it and in no Apple '
                 'denominator because Apple never wrote it'),
            what_would_close_it='nothing in the corpus; only execution'),
        opcodes_apples_table_declares_that_the_walk_cannot_reach=dict(
            opcodes=sorted(WALK_UNDERCOUNT),
            why=('real rows in the MCInstrDesc dump that this artifact\'s admission walk does '
                 'not admit - a known undercount with a named mechanism, not a decoder limit'),
            what_would_close_it='the reachability sweep in `peer_reported.reach_measurement`'),
        opcodes_whose_field_map_answers_for_one_width_only=dict(
            count=fmws.get('opcodes'),
            sample=sorted(fmws.get('by_opcode') or {})[:12],
            why=('witnessed at several widths while the field-map API returns ONE length, so '
                 'bits read at any other width of the same opcode are keyed to a different form '
                 'and nothing raises. Reported by another lane for op11456, verified, censused'),
            what_would_close_it=('a per-form field map, or a bit-flip census on the specific '
                                 'width - which is what that lane had to do')))
    # AN EMPTY UNKNOWN IS NOT AN UNKNOWN. The three forms this listed (11460/8, 9328/10, 9704/8)
    # were "absent" only from the `spans` index of one corpus; both corpora, decoded, hold 16,
    # 1,919 and 543 of them. With nothing to list the entry is dropped rather than published as
    # a tier-1 row that enumerates nothing - it comes back by itself if a form is ever absent.
    if not tier1['forms_absent_from_apples_corpus_entirely']['forms']:
        del tier1['forms_absent_from_apples_corpus_entirely']
    # AND THE SINGLETONS: 16778/12 and 17257/12 were the last two, and execution settled them
    # (tools/g17freebits.py execution_verdicts). An empty list is dropped, not published.
    if not tier1['forms_whose_undetermined_bits_no_corpus_can_settle']['forms']:
        del tier1['forms_whose_undetermined_bits_no_corpus_can_settle']

    tier2 = dict(
        forms_apple_never_wrote_at_that_width=dict(
            count=totals['d1_structural'] - (d2b.get('admissible') or 0),
            why=('in D1 because the decoder addresses the form; permanently outside D2, whose '
                 'definition is that Apple wrote THIS width. Not a gap in the measurement - a '
                 'statement about what Apple\'s shipped code contains'),
            what_would_close_it='nothing. D2 is not supposed to move here'),
        forms_excluded_from_d2_by_unforced_bits=dict(
            count=(d2b.get('by_bit_kind') or {}).get('unforced bits'),
            why=('Apple wrote the width, but bits in the encoding are not forced by anything '
                 'this artifact can see'),
            what_would_close_it=('for 62 of the 89 exclusions the honest answer is nothing - '
                                 'see d2_exclusion_basis.what_would_close_each')),
        opcodes_with_no_family=dict(
            count=fams.get('unknown'),
            of=sum(fams.values()),
            why='no family hypothesis reaches them; the assignment is calibrated, not guessed',
            what_would_close_it='a family rule with a measured hit rate, or execution'),
        opcodes_the_table_names_nothing_for=dict(
            count=Counter((r.get('meaning') or {}).get('evidence')
                          for r in contract.values()).get('unnamed'),
            why=('admitted by the decoder with no name and no recovered meaning - the largest '
                 'single class in the universe and the least characterised'),
            what_would_close_it='the elimination census, one dispatch at a time'),
        forms_dispatched_whose_values_nobody_checked=dict(
            count=totals['executed_unchecked'],
            why=('these RAN. A program that ran is not an instruction whose output was compared '
                 'against an expectation, and they are kept out of D3 deliberately'),
            what_would_close_it=('reading each receipt for a checked VALUE - not counting its '
                                 'presence in a passing program')))

    tier3 = dict(
        instructions_not_in_apples_table=dict(
            count=None,
            why=('D1 is the DISCOVERED, decoder-addressable surface. An instruction the hardware '
                 'implements that Apple\'s table does not declare is invisible to every '
                 'instrument here, and no count in this artifact bounds it'),
            what_would_close_it=('nothing this artifact can do. Saying the number is unbounded '
                                 'is the finding')),
        bits_apples_decoder_cannot_see=dict(
            count=None,
            why=('a bit invisible to Apple\'s decoder is not a bit the hardware ignores. These '
                 'are counted as unknown wherever they appear, and the population of them is '
                 'not knowable from a decoder'),
            what_would_close_it='execution that varies the bit, per form'))

    return dict(
        tier1_enumerable=tier1, tier2_characterised=tier2, tier3_unbounded=tier3,
        opcodes_apples_table_declares_and_the_decoder_does_not_admit=dict(
            count=17779 - len(contract),
            why=('rows in Apple\'s MCInstrDesc dump this decoder does not admit. Enumerable in '
                 'principle - they are table rows - and they are the reason D1 is 7,000 forms '
                 'over 6,718 opcodes rather than 17,779'),
            what_would_close_it=('admission requires a decodable witness; five of them are '
                                 'named individually in tier 1')),
        means=('the unknown side, sorted by NAMEABILITY. Tier 1 members are listed; tier 2 '
               'classes are measured but their members are not worth listing; tier 3 is not '
               'nameable from inside this artifact and has no count, which is itself the result. '
               'Nothing here feeds a denominator - a statement of ignorance must never raise a '
               'coverage number, and a test asserts it does not'))


def cited_isa_files():
    """Which isa/*.toml files the curated facts actually cite, read out of their own `source`.

    Derived rather than listed: a hand-kept list of harvested files is the same defect as the
    hardcoded `harvested=7` it replaces - it can agree with itself while the harvest moves.
    """
    names = {f.name for f in ISA.glob('*.toml') if f.is_file()}
    cited = set()
    facts = [f for v in LEDGER_FACTS.values() for f in v]
    facts += [f for v in ISA_FACTS.values() for f in v]
    facts += list(GLOBAL_FACTS)
    for fact in facts:
        src = fact.get('source', '')
        for name in names:
            if name in src:
                cited.add(name)
    return cited


def unharvested_sources():
    """Evidence that exists and this harvest does NOT read. Reported, not omitted.

    The first version of this tool printed "LEDGERS ABSENT: none", which was true of the seven
    files it listed and false as a statement about coverage. A source list is a filter, and a
    filter that reports only its own contents cannot report what it misses.
    """
    entries = list(ledger_entries())
    naming = [n for n, text, _ in entries if re.search(r'op\d{3,5}', text)]
    return dict(
        ledger_directory=dict(
            files=len(entries),
            files_naming_an_opcode=len(naming),
            harvested_by_hand=sorted(LEDGER_FACTS) + ['(global) ' + f['fact'] for f in GLOBAL_FACTS],
            note=('per-opcode evidence in prose TOML; automatic harvesting is a separate project. '
                  'Every axis reading `absent` may have evidence here that this tool has not read, '
                  'and latency_wait and barrier_sync demonstrably do.')),
        isa_directory=dict(
            # EXCLUDING this tool's own four outputs. Counting them made the artifact depend on
            # its own existence, so --check failed the moment the index was first written: a
            # regenerability check cannot pass if the content is a function of the last write.
            files=len([f for f in ISA.glob('*') if f.is_file()
                       and f.name not in {o.name for o in OUTPUTS}]),
            # DERIVED, not asserted. This read `harvested=7` as a literal, which is a number
            # the work cannot move: harvesting more files left it at seven. It is now computed
            # from the `source` strings the curated facts actually cite, so the count rises when
            # the harvest does and `not_consulted` names what is still unread.
            harvested=len(cited_isa_files()),
            harvested_files=sorted(cited_isa_files()),
            not_consulted=sorted(
                f.name for f in ISA.glob('*.toml')
                if f.is_file() and f.name not in cited_isa_files()
                and f.name not in {o.name for o in OUTPUTS}),
            note=('`harvested_files` is derived from the sources the curated facts cite; every '
                  'name under `not_consulted` is evidence this tool has not read. '
                  'THIS NOTE USED TO END "and an axis reading `absent` may be answered there", '
                  'WHICH OVERSTATES IT - measured on 2026-09-17 by reading four of these files '
                  'for content. Two paid: g17-scalar-isa.toml filled latency_wait on three forms '
                  '(a located-but-unrecovered wait composite, a width that IS the wait, and a '
                  'form with none) and g17-special-registers.toml filled resources on nine (the '
                  'flag register is allocatable, and a three-bit index bounds it to FLAG0..FLAG6 '
                  'below the fifteen Apple declares). Two did not: g17-simd-reduction-block.toml '
                  'answered NONE of the 38 cells it speaks to, because all of them were already '
                  'populated at an equal or better evidence level, and g17-typed-holes.toml '
                  'answered 2 of 149 for the same reason. So an unread file is a statement about '
                  'this tool rather than about unexploited evidence, and the way to tell which '
                  'is to check the axes it would touch BEFORE reading it')))


def load(path, kind='json'):
    p = Path(path)
    if not p.exists():
        return None, False
    if kind == 'jsonl':
        with p.open() as handle:
            return [json.loads(line) for line in handle if line.strip()], True
    return json.loads(p.read_text()), True


def sources():
    contract, ok_contract = load(ISA/'g17-contract.jsonl', 'jsonl')
    formbits, ok_formbits = load(ISA/'g17-form-bits.json')
    freebits, ok_freebits = load(ISA/'g17-free-bits.jsonl', 'jsonl')
    bitspec, ok_bitspec = load(ISA/'g17-bit-spec.jsonl', 'jsonl')
    lifetimes, ok_life = load(ISA/'g17-auth-lifetimes.json')
    lengths, ok_len = load(ISA/'g17-auth-lengths.json')
    sweep, ok_sweep = load(ISA/'g17-execution-sweep-results.json')
    found = dict(contract=ok_contract, form_bits=ok_formbits, free_bits=ok_freebits,
                 bit_spec=ok_bitspec, auth_lifetimes=ok_life, auth_lengths=ok_len,
                 execution_sweep=ok_sweep)
    if not ok_contract:
        raise SystemExit('isa/g17-contract.jsonl is required and absent')
    life = (lifetimes or {}).get('lifetimes') or {}
    by_opcode = {r['opcode']: r for r in contract}
    widths = apple_widths(by_opcode)
    repair, repair_failed = repair_walk_widths(by_opcode, widths)
    identity = decoder_identity_check(by_opcode)
    return dict(
        contract={r['opcode']: r for r in contract},
        formbits=formbits or {},
        # Keyed by FORM. The same bit position means different things at different widths, and a
        # review found 338 conflicting classifications across 112 forms from pooling lengths.
        freebits={(r['opcode'], r['length']): (r.get('bits') or {}) for r in (freebits or [])},
        bitspec={r['opcode']: (r.get('bits') or {}) for r in (bitspec or [])},
        lifetimes=life,
        certified_lifetimes={op for op, v in life.items() if isinstance(v, dict) and v.get('ok')},
        lengths=(lengths or {}).get('lengths') or {},
        **_execution_instruments(sweep or []),
        apple_width=widths, repair_width=repair, repair_width_failed=repair_failed,
        decoder_identity=identity,
        found=found)


_READER_CACHE = {}


def _memoized(function):
    """Both receipt readers are argument-free and glob 58 files; `sources()` is called by many
    tests. Memoizing here rather than at the call sites keeps the cost off every caller."""
    def wrapper():
        if function.__name__ not in _READER_CACHE:
            _READER_CACHE[function.__name__] = function()
        return _READER_CACHE[function.__name__]
    wrapper.__name__ = function.__name__
    wrapper.__doc__ = function.__doc__
    return wrapper


@_memoized
def dispatched_byte_forms():
    """(opcode, length) -> receipt, by DECODING the bytes a receipt retains in `decoded.encoded`.

    The strongest placement available: not a field claiming an opcode and not an id claiming a
    width, but the instruction that was dispatched, read back through Apple's decoder. 248 rows
    across the targeted files carry it and every hex string decodes.

    It settled op11375, the worst of the inferred widths - four admitted forms, and the two files
    that name the opcode carry `length: null` - by decoding its dispatched bytes to width 14 where
    the Apple-witness fallback had chosen 6.
    """
    out = {}
    for path in sorted(ISA.glob('g17-execution-*-results.json')):
        try:
            doc = json.loads(path.read_text())
        except Exception:
            continue
        rows = doc if isinstance(doc, list) else (doc.get('rows') or doc.get('results') or [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get('cb_status') != 0 or not row.get('finished') or row.get('status') != 'ok':
                continue
            decoded = row.get('decoded')
            encoded = decoded.get('encoded') if isinstance(decoded, dict) else None
            if not isinstance(encoded, list):
                continue
            for text in encoded:
                try:
                    raw = bytes.fromhex(text)
                except Exception:
                    continue
                try:
                    instructions = list(decode_instruction(raw))
                except Exception:
                    continue          # this row's bytes are not decodable; the reader is not
                for inst in instructions:
                    _keep_better_receipt(out, (inst.opcode.id, len(inst.raw)), dict(
                        values=row.get('values'), runs=row.get('runs'), decoded_ok=True,
                        source=path.name, receipt_id=row.get('id'),
                        placed_by='decoding the dispatched bytes'))
    return out


def _keep_better_receipt(out, key, candidate):
    """Keep the receipt with the stronger evidence, not the one whose filename sorts first.

    These readers used `setdefault`, so the FIRST file to mention a form won and every later one
    was discarded unread. That is filename ordering deciding what the map cites, and it picked
    the weaker record the moment a form was re-dispatched: `g17-execution-constantsweep-results`
    sorts before `...constantsweep2-results`, so the spec cited a batch whose eight cases were
    four distinct inputs repeated over the corrected batch that superseded it.

    Ranked on what can be read from the row itself - how many DISTINCT values it returned, then
    how many runs agreed. A receipt whose outputs vary is better evidence than one that is
    constant, which is the same judgement the census makes about degenerate records.

    HOW THIS RANKING MISLEADS, because it is a proxy and not the property. Distinct values is not
    discriminating power: a record that merely RETURNS ITS OPERAND over eighteen varied inputs
    scores eighteen and beats a careful eight-case probe built to separate candidates, even
    though the census would call the first degenerate and read nothing from it. The map does not
    compute degeneracy - that is the census's job - so this ranks on the strongest signal
    available at this layer. It is strictly better than the filename ordering it replaces, and it
    is not the right answer; the right answer would consult
    `isa/g17-execution-fits.json`'s per-row degeneracy flags and prefer a non-degenerate receipt
    first. Verified when it was introduced: changing the ranking moved no denominator - D1 7001,
    D2 628, D3 89/23/27, dispatched-unchecked 483 all unchanged - so it alters which receipt is
    CITED and never what the map claims.
    """
    def strength(rec):
        values = rec.get('values') or []
        try:
            distinct = len({int(v) for v in values})
        except Exception:
            distinct = 0
        return (distinct, int(rec.get('runs') or 0))
    if key not in out or strength(candidate) > strength(out[key]):
        out[key] = candidate


@_memoized
def targeted_execution_forms():
    """(opcode, length) -> receipt, from the FIFTY-SEVEN results files the sweep is not.

    isa/ holds 58 `g17-execution-*-results.json` files and the map read exactly one of them. The
    other 57 are targeted probes and they carry the same fact in a DIFFERENT SCHEMA: the sweep puts
    the width in the row id ("D10279.l12") and these put it in explicit `op` and `length` fields,
    which is why a reader written for one is blind to the other. Fourteen admitted (opcode, width)
    pairs over 13 opcodes live only here.

    It is not a large population and it is not why it matters. Two of those pairs REFUTE a width
    this map had inferred: op9832 was dispatched at 8 and 16 while the Apple-witness fallback had
    chosen 6, and op11507 was dispatched at 8 while the fallback had chosen 10. An inference about
    which width a probe used is exactly the thing a receipt naming the width settles.
    """
    out = {}
    for path in sorted(ISA.glob('g17-execution-*-results.json')):
        if path.name == 'g17-execution-sweep-results.json':
            continue
        try:
            doc = json.loads(path.read_text())
        except Exception:
            continue
        rows = doc if isinstance(doc, list) else (doc.get('rows') or doc.get('results') or [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get('cb_status') != 0 or not row.get('finished') or row.get('status') != 'ok':
                continue
            opcode, length = row.get('op'), row.get('length')
            if opcode is None or length is None:
                continue
            _keep_better_receipt(out, (int(opcode), int(length)), dict(
                values=row.get('values'), runs=row.get('runs'), decoded_ok=None,
                source=path.name, receipt_id=row.get('id')))
    return out


def _execution_instruments(sweep):
    """The three receipt readers, each called ONCE.

    Written first as five calls inside one dict literal - the byte reader twice, the targeted
    reader twice, the sweep reader three times - which took the suite from 14s to 99s. That is the
    same defect as the eight identical 34MB YAML parses that once took it from 47s to 335s, and it
    is the same fix: compute each population once and hand the parts out.
    """
    by_bytes = dispatched_byte_forms()
    targeted = targeted_execution_forms()
    swept = executed_forms(sweep)
    return dict(
        executed={**by_bytes, **targeted, **swept},
        executed_sweep_only=swept,
        executed_targeted_only={k: v for k, v in targeted.items() if k not in swept},
        executed_bytes_only={k: v for k, v in by_bytes.items() if k not in swept})


_SWEEP_ATTRIBUTION = collections.Counter()


def executed_forms(sweep):
    """(opcode, length) pairs with a retained receipt that finished. Dispatch, not correctness.

    The id carries the width ("D10279.l12"), so attribution is per form rather than per opcode.
    A receipt is admitted only when the command buffer completed and the run finished; the file
    retains observed values and NO prediction, which is why this feeds `executed`, never `checked`.
    """
    out = {}
    for row in sweep:
        if not isinstance(row, dict):
            continue
        if row.get('cb_status') != 0 or not row.get('finished') or row.get('status') != 'ok':
            continue
        match = re.match(r'^D(\d+)\.l(\d+)$', str(row.get('id') or ''))
        if not match:
            continue
        opcode, intended = int(match.group(1)), int(match.group(2))
        if row.get('op') is not None and int(row['op']) != opcode:
            continue
        # THE ID NAMES AN INTENTION; THE BYTES NAME WHAT RAN. This read the width out of the id,
        # which was already an improvement over per-opcode attribution - but it credits the form
        # the plan ASKED for rather than the one the encoder emitted, and those disagree for 89 of
        # these records: D1004.l12 emitted six bytes, D590.l4 emitted eight. The census derives
        # its width from the emitted bytes, so the two instruments were attributing the same
        # measurement to different forms, and the intended form was being credited with a receipt
        # it never earned while the dispatched form went without.
        #
        # `decoded.encoded` holds one entry per case, each the bytes of the instruction under
        # test. Every case must agree on the length or the record is not describing one form.
        # VERIFIED BY DECODING, not assumed from the hex length. Each entry holds the bytes of
        # the instruction under test - one instruction - and `decoded.opcodes` lists the whole
        # PROGRAM's opcodes, which is easy to confuse for the same thing. So the bytes are decoded
        # and the instruction's own opcode must match the record's before its length is trusted.
        encoded = (row.get('decoded') or {}).get('encoded') or []
        sizes = set()
        for text in encoded:
            if not isinstance(text, str) or not text:
                continue
            try:
                instructions = list(decode_instruction(bytes.fromhex(text)))
            except Exception:
                sizes = None
                break
            hit = [i for i in instructions
                   if getattr(getattr(i, 'opcode', None), 'id', None) == opcode]
            if len(instructions) != 1 or not hit:
                sizes = None
                break
            sizes.add(len(getattr(hit[0], 'raw', b'') or b'') or len(text) // 2)
        if sizes and len(sizes) == 1:
            length = sizes.pop()
            _SWEEP_ATTRIBUTION['length taken from the decoded emitted instruction'] += 1
            if length != intended:
                _SWEEP_ATTRIBUTION['moved off the length its id claimed'] += 1
        else:
            # NAMED, NOT SILENT: a fallback that fills unknowns is fine and an uncounted one grows
            # without anyone noticing.
            length = intended
            _SWEEP_ATTRIBUTION['fell back to the id: bytes absent, undecodable, or not one '
                               'instruction of this opcode'] += 1
        out[(opcode, length)] = dict(values=row.get('values'), runs=row.get('runs'),
                                     decoded_ok=bool((row.get('decoded') or {}).get('ok')),
                                     length_from=('the decoded emitted instruction'
                                                  if length != intended or sizes else 'the id'),
                                     intended_length=intended)
    return out


def construct_groups(record):
    """Which Metal construct groups cause this opcode to be emitted, and whether that is pure.

    THIS IS NOT A FAMILY SIGNAL AND MUST NOT BE USED AS ONE. The relation is many-to-many and
    records the COMPILER'S EXPANSION, not the instruction's function. Measured counterexamples:
    op458 `branch` is emitted by idiv/imod, because integer division lowers to a branch loop;
    op12715 `load` is emitted only by rt:rt.intersect; all 65 opcodes emitted only by simd_*/
    quad_* constructs include `msb`, `csel` and `sub`, which are the scalar steps a lane
    reduction lowers to; and the rt-pure set contains a hardware-verified `and.a`.

    It is kept because it answers a real question - which opcodes does a compiler need in order
    to support feature X - and that question is not "what is this instruction".
    """
    groups = set()
    for c in ((record.get('meaning') or {}).get('constructs') or []):
        head = str(c).split('.')[0]
        if ':' in head:
            groups.add(head.split(':')[0])
        elif head.startswith(('simd_', 'quad_')):
            groups.add('lane')
        else:
            groups.add('plain')
    return sorted(groups)


# Family by the FIRST SEGMENT of a recorded name, matched exactly rather than by substring.
# Derived from the complete vocabulary - all 65 first segments over the 637 opcodes whose name
# Apple's table records at a class better than table-only - not from imagination. Three separate
# misses taught that lesson: patterns hand-guessed as `simdgroup.` missed `tensor.mac`, then
# missed `simd.smin` and `quad.sum.f32`, each time filing named instructions as unknown. Exact
# segment matching also removes the simdgroup/simd prefix collision by construction.
SEGMENT_FAMILY = {
    # scalar ALU, conversions, selects, bitwise
    'add': 'scalar_alu', 'addsat': 'scalar_alu', 'and': 'scalar_alu', 'andn': 'scalar_alu',
    'asr': 'scalar_alu', 'ceil': 'scalar_alu', 'clamp': 'scalar_alu', 'cmp': 'scalar_alu',
    'csel': 'scalar_alu', 'cvt': 'scalar_alu', 'exp2': 'scalar_alu', 'fadd': 'scalar_alu',
    'fcmp': 'scalar_alu', 'fcsel': 'scalar_alu', 'ffma': 'scalar_alu', 'floor': 'scalar_alu',
    'fmul': 'scalar_alu', 'fselect': 'scalar_alu', 'funnel': 'scalar_alu', 'log2': 'scalar_alu',
    'madd': 'scalar_alu', 'mov': 'scalar_alu', 'movimm': 'scalar_alu', 'msb': 'scalar_alu',
    'mul': 'scalar_alu', 'mulhi': 'scalar_alu', 'nand': 'scalar_alu', 'nor': 'scalar_alu',
    'not': 'scalar_alu', 'or': 'scalar_alu', 'orn': 'scalar_alu', 'pack': 'scalar_alu',
    'popcount': 'scalar_alu', 'recip': 'scalar_alu', 'reverse': 'scalar_alu',
    'rint': 'scalar_alu', 'rsqrt': 'scalar_alu', 'select': 'scalar_alu', 'shl': 'scalar_alu',
    'shr': 'scalar_alu', 'sub': 'scalar_alu', 'subsat': 'scalar_alu', 'trig': 'scalar_alu',
    'trunc': 'scalar_alu', 'unpack': 'scalar_alu', 'xnor': 'scalar_alu', 'xor': 'scalar_alu',
    # cross-lane. `simd`/`quad` are reductions and shuffles; `simdgroup` is the matrix unit and
    # is a DIFFERENT family, which is why these are exact segments and not prefixes.
    'simd': 'simd_shuffle_reduce', 'quad': 'simd_shuffle_reduce',
    'simdgroup': 'tensor_accelerator', 'tensor': 'tensor_accelerator',
    'texture': 'texture_image',
    'load': 'memory_addressing', 'store': 'memory_addressing', 'addr16': 'memory_addressing',
    'base': 'memory_addressing',
    'atomic': 'atomic_sync', 'barrier': 'atomic_sync',
    'branch': 'control_predicate_exec', 'exec': 'control_predicate_exec',
    'end': 'control_predicate_exec', 'flag': 'control_predicate_exec',
    'read_sr': 'system_register',
}

# Segments deliberately left out, with why. Listed rather than dropped so that the completeness
# check below distinguishes "considered and not decidable" from "never looked at".
UNCLASSIFIED_SEGMENTS = {
    'publish': 'unclear whether this is a memory-visibility operation or a queue/export step; '
               'no declared effect settles it',
    'pad': 'a single opcode whose name suggests encoding padding rather than an operation',
}

TRUSTED_NAME_CLASSES = ('verified', 'measured', 'corpus')


def name_segment(record):
    """The first dot-segment of a recorded name, only at a trusted class. Else None."""
    name = record.get('name')
    cls = (record.get('meaning') or {}).get('evidence')
    if not name or cls not in TRUSTED_NAME_CLASSES:
        return None
    return str(name).split('.')[0]


def family_from_name(record):
    """(family, evidence) from the recorded name's first segment, or (None, None)."""
    segment = name_segment(record)
    if segment is None:
        return None, None
    family = SEGMENT_FAMILY.get(segment)
    if not family:
        return None, None
    cls = (record.get('meaning') or {}).get('evidence')
    return family, 'checked' if cls == 'verified' else 'oracle'


def effect_family(record):
    """The family the DECLARED EFFECTS alone would give, or None. Used to measure disagreement."""
    effects = record.get('effects') or {}
    reads, writes = record.get('reads') or {}, record.get('writes') or {}
    classes = [c for c in (reads.get('operands') or []) + (writes.get('operands') or []) if c]
    if effects.get('atomic'):
        return 'atomic_sync'
    if effects.get('texture'):
        return 'texture_image'
    if effects.get('control'):
        return 'control_predicate_exec'
    if 'SIR32' in classes or 'IR32' in classes:
        return 'system_register'
    if (effects.get('memory') or reads.get('may_load') or writes.get('may_store')
            or writes.get('address_file')):
        return 'memory_addressing'
    return None


def family_disagreements(contract):
    """Opcodes where the recorded name and the declared effects imply different families.

    Reported rather than silently resolved. The name leads, because a family describes the
    OPERATION and an effect flag describes a property of it - but the reader should see the
    count. Both groups here support that order: 86 are `*.a` forms such as `and.a`, ALU
    operations whose destination is the address file, and that destination is not lost because
    the register_files axis carries writes_address_file; the remaining one is `barrier`, which
    is synchronisation and not addressing.
    """
    out = Counter()
    for r in contract.values():
        named, _ = family_from_name(r)
        declared = effect_family(r)
        if named and declared and named != declared:
            out['declared %s, named %s' % (declared, named)] += 1
    return dict(out)


def unmapped_name_segments(contract):
    """Trusted name segments this map neither classifies nor explicitly sets aside.

    Reported so a vocabulary the harvest has not considered is visible as a number rather than
    dissolving into the unknown bucket, which is where the last three misses hid.
    """
    seen = Counter()
    for r in contract.values():
        segment = name_segment(r)
        if segment and segment not in SEGMENT_FAMILY and segment not in UNCLASSIFIED_SEGMENTS:
            seen[segment] += 1
    return dict(seen)


def family_of(record):
    """Classify from established structure only; return (family, evidence).

    Every rule cites something mechanical: an effect flag Apple's descriptor declares, an MCID
    flag, or a register class. Where nothing mechanical separates the candidates the answer is
    'unknown' at evidence 'absent'.
    """
    effects = record.get('effects') or {}
    reads, writes = record.get('reads') or {}, record.get('writes') or {}
    classes = [c for c in (reads.get('operands') or []) + (writes.get('operands') or []) if c]
    # A name Apple's table records at a trusted class describes the INSTRUCTION; an effect flag
    # describes only a property of it. So the name leads where one exists, and the descriptor is
    # the authority for the 6,081 opcodes that have no trusted name.
    named, named_evidence = family_from_name(record)
    if named:
        return named, named_evidence
    flags = set()
    for key in ('not_duplicable', 'rematerializable', 'commutable'):
        if effects.get(key):
            flags.add(key)
    if effects.get('atomic'):
        return 'atomic_sync', 'table'
    if effects.get('texture'):
        return 'texture_image', 'table'
    if effects.get('control'):
        return 'control_predicate_exec', 'table'
    if 'SIR32' in classes or 'IR32' in classes:
        return 'system_register', 'table'
    # Before the generic memory rule: three families Apple's descriptor cannot express at all,
    # each recognised by a name that could not plausibly mean anything else. An imageblock load
    # does touch memory, but "imageblock" is the more specific fact and 'memory_addressing'
    # would bury every graphics form in the largest family.
    named, named_evidence = family_from_name(record)
    if named in ('graphics', 'tensor_accelerator', 'ray_tracing'):
        return named, named_evidence
    if effects.get('memory') or reads.get('may_load') or writes.get('may_store'):
        return 'memory_addressing', 'table'
    if writes.get('address_file'):
        return 'memory_addressing', 'table'
    if 'FLAGR' in classes:
        return 'control_predicate_exec', 'table'
    if any('tup' in c or c.startswith('GPR16') for c in classes):
        # A tuple operand is a wide register group. Whether that is SIMD, a tensor stage or wide
        # scalar arithmetic is behavioural, and this descriptor cannot say - so consult the
        # recorded name before giving up, since this branch is exactly where the simdgroup
        # matrix opcodes live and returning early here hid every one of them.
        named, named_evidence = family_from_name(record)
        return (named, named_evidence) if named else ('unknown', 'absent')
    if classes and all(c in ('GPR32', 'IRGPR32') for c in classes):
        return 'scalar_alu', 'table'
    # Only now consult the recorded name. The descriptor describes the instruction; the name is
    # weaker evidence and must not override it, so it fills unknowns and nothing else.
    named, named_evidence = family_from_name(record)
    if named:
        return named, named_evidence
    return 'unknown', 'absent'


def unknown_bits(opcode, length, src):
    """Bits whose role is not established for THIS form, and the evidence that settled the rest.

    A bit Apple's decoder ignores is 'invisible', which means invisible TO THE DECODER: a bound on
    what decoding can observe, not a demonstration that the hardware ignores it. Such a bit stays
    UNKNOWN until independent corpus instances settle it as a constant or as genuinely varying.
    A verdict of 'undetermined' - fewer than two instances - is not a settlement.
    """
    key = '%d,%d' % (opcode, length)
    form = src['formbits'].get(key)
    verdicts = src['freebits'].get((opcode, length))
    if form is None and verdicts is None:
        return dict(state='no form-level bit evidence at this width', evidence='absent',
                    source='isa/g17-form-bits.json + isa/g17-free-bits.jsonl')
    unforced = ['%d.%d' % (b[0], b[1]) for b in ((form or {}).get('unforced') or [])]
    # `absent at this length`: an opcode-level invisible bit that lies beyond THIS form's bytes,
    # recorded per form by tools/g17formspecgen.py. It does not exist in the form, so it is settled
    # for it. Accepted only where a per-form entry SAYS so - no bit is filtered here by length, so
    # a form whose evidence predates the check is judged exactly as before.
    settled = {b for b, v in (verdicts or {}).items()
               if isinstance(v, dict)
               # `free`: settled by EXECUTION where one Apple instance cannot settle it
               # (tools/g17freebits.py execution_verdicts)
               and v.get('verdict') in ('constant', 'varies', 'free', 'absent at this length')}
    undetermined = sorted(b for b, v in (verdicts or {}).items()
                          if isinstance(v, dict) and v.get('verdict') == 'undetermined')
    invisible = [b for b, v in (src['bitspec'].get(opcode) or {}).items()
                 if isinstance(v, dict) and v.get('class') == 'invisible']
    remaining = sorted(set(unforced) | set(undetermined) | (set(invisible) - settled - set(undetermined)))
    return dict(state='established' if not remaining else 'incomplete',
                count=len(remaining), bits=remaining[:48],
                unforced=len(unforced), undetermined=len(undetermined),
                invisible_to_decoder=len(invisible), settled_by_corpus=len(settled),
                evidence='corpus' if verdicts else ('decoder' if form else 'absent'),
                source='isa/g17-form-bits.json + isa/g17-free-bits.jsonl + isa/g17-bit-spec.jsonl')


def axis(state, evidence, source, **extra):
    if evidence not in RANK:
        raise ValueError('bad evidence level: ' + repr(evidence))
    return dict(state=state, evidence=evidence, source=source, **extra)


def checked_width(opcode, src):
    """(width, how it was established) for the ONE form a `verified` class supports, or (None, why).

    D3's largest column read `cls == 'verified'` - "a probe isolating this opcode alone returned
    the value its name predicts, on hardware" - and applied it to EVERY admitted width of that
    opcode. A probe dispatches one encoding, so it supports one FORM. Uncorrected that made D3v
    140 forms over 91 opcodes: 49 of them width-spray.

    This is the same defect, in the same shape, that put 62 forms in D2 at widths Apple never
    wrote - D2 was gated on an OPCODE-level `apple_witness` while a form is (opcode, width). That
    one was found and fixed; this column was never checked for it.

    The width is chosen by the strongest available instrument and LABELLED, never guessed silently:
    the isolated execution sweep names it per form ("D10279.l12") for 38 of the 91; Apple's own
    witness byte count is the next best, being the encoding a probe would most likely have been
    built from; and where neither exists the smallest admitted width is used and said to be
    unestablished, so a reader can discount it.
    """
    # EVERY width this opcode was dispatched at, across all three receipt instruments. They do not
    # conflict - 12 opcodes have a sweep width and a byte-decoded width with no overlap, and both
    # are true, because they are different dispatches of different forms. What no instrument
    # records is which of them established the NAME, so where several ran the identity of the
    # single form D3 counts is AMBIGUOUS and is labelled that way rather than smoothed over.
    ran = sorted({ln for (op, ln) in src['executed'] if op == opcode})
    sweep_ran = sorted({ln for (op, ln) in src.get('executed_sweep_only') or {} if op == opcode})
    if len(ran) == 1:
        return ran[0], 'receipt, UNIQUE: the only width this opcode was ever dispatched at'
    if ran:
        pick = sweep_ran[0] if sweep_ran else ran[0]
        return pick, ('receipt, AMBIGUOUS: dispatched at %s and no instrument records which '
                      'width established the name; the %s is taken'
                      % (', '.join(str(x) for x in ran),
                         'sweep width' if sweep_ran else 'smallest'))
    aw = src['apple_width'].get(opcode)
    if aw:
        return aw, 'INFERRED: apple_witness byte count, no receipt names a width for this opcode'
    widths = sorted({w for o, w in
                     ((int(k.split(',')[0]), int(k.split(',')[1])) for k in src['formbits'])
                     if o == opcode} | ({src['repair_width'][opcode]}
                                        if src['repair_width'].get(opcode) else set()))
    if widths:
        return widths[0], 'UNESTABLISHED: smallest admitted width, no receipt and no Apple witness'
    return None, 'UNESTABLISHED: no width could be chosen'


def width_provenance(opcode, length, src, form):
    """(evidence, source) for a form's WIDTH. Three provenances, never merged into one.

    The width is the one axis every admitted opcode can have, so it is also the axis where a
    denominator is easiest to inflate: a repair-walk decode and an Apple byte count are both
    "the decoder accepts this length" and only one of them is an encoding anybody emitted.
    """
    if form:
        return 'decoder', 'isa/g17-form-bits.json'
    if src['apple_width'].get(opcode) == length:
        return 'decoder', 'isa/g17-contract.jsonl (apple_witness byte count)'
    if src['repair_width'].get(opcode) == length:
        return 'decoder', 'isa/g17-contract.jsonl (repair-walk witness, decoded length)'
    return 'table', 'isa/g17-auth-lengths.json'


def width_label(opcode, length, src, form):
    """The labelling item 2 asks for, as fields rather than prose."""
    # ADMISSIBLE MEANS ONE THING: Apple's own witness for this opcode is THIS MANY BYTES.
    # form-bits evidences a width from the DECODER - it is how this project learned the form
    # exists, not evidence anybody emitted it - so it is not admissible on its own. Marking it
    # admissible let 62 forms into D2 at widths Apple never wrote; see the D2 gate below.
    if src['apple_width'].get(opcode) == length:
        return dict(width_origin=('apple-witness byte count, corroborated by form-bits' if form
                                  else 'apple-witness byte count'),
                    admissible_for_d2=True)
    # THE SECOND WAY A FORM CAN BE APPLE-WITNESSED, and it is not the form-bits mistake above.
    # form-bits says THE DECODER ACCEPTS a form of this width, which is how this project learned
    # the form exists - admitting it once let 62 forms into D2 at widths Apple never wrote. A
    # CORPUS INSTANCE is the opposite kind of evidence: Apple's own shipped program contains this
    # exact (opcode, width), framed by Apple's own decoder. 12674/16 occurs 6,387 times and was
    # inadmissible only because the single exemplar retained for op12674 is a different length.
    #
    # The exemplar rule stays, because 170 opcodes have a retained witness the corpus scan never
    # sees at any width - this is a union, not a replacement.
    instances = corpus_form_instances().get((opcode, length), 0)
    if instances:
        return dict(width_origin='apple corpus instances at this exact form',
                    corpus_instances=instances, admissible_for_d2=True)
    if form:
        return dict(
            width_origin='form-bits', admissible_for_d2=False,
            means='the decoder reads a form of this width and this project mapped its bits, but '
                  "Apple's witness for this opcode is %s bytes, so no encoding of THIS width is "
                  'known to have been written' % (src['apple_width'].get(opcode),))
    if src['repair_width'].get(opcode) == length:
        return dict(
            width_origin='repair-walk decode', admissible_for_d2=False,
            means='the decoder accepts a form of this width, reached by mutating a neighbour '
                  'until the bytes were accepted. Nothing is known to have EMITTED one, and '
                  'forty class-51 opcodes authored this way return a constant of the encoding. '
                  'It evidences the width and nothing else')
    return dict(width_origin='declared length only', admissible_for_d2=False)


# THE ELIMINATION CENSUS, harvested from the artifact tools/g17fitfromexecution.py writes. A FIFTH
# D3 INSTRUMENT, kept in its own column because it is a different kind of evidence from the other
# four: not the contract's verified class, not a ledger's claim, not a probe of mine, not a peer's
# report, but what survives when every candidate function is computed against the words an
# isolated dispatch returned. The bar lives in that tool - four cases, three runs, at least eight
# competing candidates eliminated, outputs not equal to any input column, and every record of a
# form agreeing - so this reader only consumes `forms_promotable` and never relaxes it.
#
# WHY IT IS NOT FOLDED INTO d3_semantics_checked: the claim is weaker in a specific, nameable way.
# Uniqueness holds within a published candidate library and at the operand and modifier
# configuration the record was dispatched with. A function of another arity, one reading the
# operands differently, or one depending on a modifier is not excluded by four points. Merging the
# columns would make that caveat invisible, which is the failure this whole map exists to avoid.
# A CANDIDATE SIXTH D3 CONTRIBUTION, COMPUTED AND DELIBERATELY NOT COUNTED.
#
# The census determined three conditional-select forms by eliminating a library of 26 PREDICATES
# against a boolean output - op11372 as `a == b` returning integer 1, op11381 as the same
# predicate returning f32 1.0, and op11492 as `(a & 0xFFFF) <u b` once its declared 16-bit first
# source is applied. That is elimination of the same kind as the function census, with margins and
# competitor counts, and by the logic of the fifth column it belongs in a D3 column.
#
# IT IS NOT ADDED TO ONE, for two reasons that both point the same way. Merging it into
# `d3_fitted_by_elimination` would make that column mean two instruments with different failure
# modes - the whole reason the five columns are exclusive - and creating a sixth column changes
# THE REPOSITORY'S HEADLINE NUMBER, which is root's call and not a thing to do unilaterally at the
# end of a long session. An absent axis is not a gap to fill.
#
# So the count is computed, published beside the denominators, and marked as not counted. If it
# should be a column, promoting it is one edit on a number that is already visible.
def _predicate_forms():
    """(forms, ambiguous) from the PREDICATE census - forms keyed 'opcode/width'."""
    path = ISA / 'g17-execution-fits.json'
    if not path.exists():
        return {}, {}
    try:
        doc = json.loads(path.read_text())
    except Exception:
        return {}, {}
    out, ambiguous = {}, {}
    for key, entry in ((doc.get('csel_boolean_comparisons') or {}).get('forms')
                       or {}).items():
        per = entry.get('per_form') or {}
        # AN EXACT STRING MATCH EXCLUDED THREE DETERMINATIONS. The census reports two determined
        # states - `determined` and `determined at an unrecognised true-value` - and the second is
        # a determination of the PREDICATE whose only difference is that the word meaning "true"
        # is not one the recogniser lists (0x7 and 0x30 appear). This filter compared for equality
        # and silently dropped op11412, op11421 and op11472, so the candidate count read 11 where
        # it should read 14.
        #
        # I found it by writing 12 into a commit message from expectation and checking: the guess
        # was wrong in both directions, and being wrong is what made me look at the filter.
        if not str(per.get('state') or '').startswith('determined'):
            continue
        try:
            opcode = int(key.split('/')[0])
        except Exception:
            continue
        # KEYED BY FORM NOW, WHICH IT COULD HAVE BEEN ALL ALONG. This read "keyed by opcode,
        # not by form ... the predicate field does not carry the instruction LENGTH, so this
        # cannot say which form of a multi-width opcode was determined", and that was true of the
        # AGGREGATE while every row underneath it carried `widths`. The census now lifts the
        # width onto the form entry, so the fact belongs to a form - the opcode-level-fact
        # -applied-per-form error this map has caught three times no longer applies here.
        #
        # Four of the fourteen opcodes have two lengths in the map (op11372, op11412, op11462,
        # op11492); each was dispatched at exactly one of them, unanimously over dozens of
        # records and several batches, measured twice and in agreement - the census's `widths`
        # field and an independent decode of the retained `decoded.encoded` bytes.
        #
        # A form entry that saw TWO widths would be exactly the error above, so it is skipped
        # and named rather than attributed to a guessed width.
        widths = entry.get('emitted_widths') or []
        if len(widths) != 1:
            ambiguous['%d/%s' % (opcode, key.split('/', 1)[-1])] = dict(
                emitted_widths=widths,
                means=('this form was dispatched at %d distinct widths, so an opcode-level '
                       'determination cannot be attributed to one of them' % len(widths)))
            continue
        out['%d/%d' % (opcode, widths[0])] = dict(
                           opcode=opcode,
                           width=widths[0],
                           predicate=per['predicates'][0],
                           records=per.get('records'),
                           true_at_fewest_cases=per.get('true_at_fewest_cases'),
                           true_at_most_cases=per.get('true_at_most_cases'),
                           competitors=entry.get('competitors'),
                           counted_in_d3=False,
                           why_not_counted=(
                               'elimination over a PREDICATE library against a boolean output is a '
                               'different instrument from elimination over a function library '
                               'against values, with different failure modes. Folding it into '
                               'd3_fitted_by_elimination would make that column mean both; giving '
                               'it a sixth column changes this repository\'s headline D3 and is '
                               'root\'s call. Computed and published so promoting it is one edit '
                               'on a number already visible. THE WIDTH IS NO LONGER A REASON: '
                               'this names a form, not an opcode'))
    # THE AMBIGUOUS ONES ARE RETURNED SEPARATELY, NOT MIXED INTO THE FORMS. Putting them in
    # `out` under a reserved key made the diagnostic a member of the population every caller
    # iterates - one entry that is not a form, in a dict whose every value is supposed to be one.
    return out, ambiguous


def predicate_determined_forms():
    """{'opcode/width': reading} for forms the PREDICATE census determines, or {}."""
    return _predicate_forms()[0]


def predicate_forms_whose_width_is_ambiguous():
    """Forms the predicate census determines that were dispatched at more than one width.

    Published beside the candidate column rather than dropped. An empty dict here is a
    measurement - every determined form named exactly one width - and not the absence of a
    check; the guard asserts the two populations partition the census's determined entries.
    """
    return _predicate_forms()[1]


def fitted_semantics():
    """{(opcode, width): fit} for forms the elimination census names, or {} if it has not run."""
    path = ISA / 'g17-execution-fits.json'
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text())
    except Exception:
        return {}
    out = {}
    for key, entry in (doc.get('forms_promotable') or {}).items():
        try:
            opcode, width = (int(part) for part in key.split('/'))
        except Exception:
            continue
        out[(opcode, width)] = dict(entry, bar=doc.get('bar'), scope=doc.get('scope'))
    return out


def form_record(opcode, length, src):
    r = src['contract'][opcode]
    enc = r.get('encoding') or {}
    meaning = r.get('meaning') or {}
    effects = r.get('effects') or {}
    reads, writes = r.get('reads') or {}, r.get('writes') or {}
    form = src['formbits'].get('%d,%d' % (opcode, length)) or {}
    family, family_evidence = family_of(r)
    unknown = unknown_bits(opcode, length, src)
    receipt = src['executed'].get((opcode, length))
    apple = bool(enc.get('apple_witness'))
    cls = meaning.get('evidence')
    level, why = EVIDENCE_CLASS.get(cls, ('table', 'unrecognised evidence class'))

    # D2. An Apple-written encoding OF THIS FORM, no unforced bits, nothing undetermined. A
    # repair-walk witness proves the decoder accepts the bytes, which is not the same claim.
    #
    # `apple` is an OPCODE-level fact and a form is (opcode, width). An opcode has exactly one
    # Apple-written witness, so a form at any other width has no Apple-written encoding at all -
    # and gating D2 on `apple` counted 62 such forms, among them op998/len4 and op998/len8 where
    # Apple wrote six bytes, and op12679/len14 where Apple wrote eight. This is the same defect
    # as the audit that asked about another length: a property proven of an encoding does not
    # belong to the opcode.
    # APPLE-WITNESSED AT THIS EXACT FORM, by either of two populations of Apple's own code.
    # The paragraph above is right that an opcode-level fact must not be applied to every width -
    # that defect admitted 62 forms - but it drew the wrong conclusion from it: that an opcode
    # has exactly ONE Apple-written witness. It has one RETAINED exemplar. Apple's shipped corpus
    # contains 784 distinct forms, and 240 of them were inadmissible only because the exemplar
    # kept for their opcode happens to be a different length. 12674/16 occurs 6,387 times.
    #
    # Both tests below are per FORM, so the original defect stays fixed. What changes is that a
    # second population of Apple-written encodings is now read at the same granularity.
    #
    # form-bits is still NOT admitted: it says the decoder ACCEPTS a width, which is how this
    # project learned the form exists, not evidence anybody emitted one.
    corpus_here = corpus_form_instances().get((opcode, length), 0)
    apple_here = src['apple_width'].get(opcode) == length or bool(corpus_here)
    encoded = apple_here and unknown['state'] == 'established'
    # D3. Only the ledger's own `verified` class: isolated on hardware against its predicted value.
    # ONE FORM PER VERIFIED OPCODE, at a width chosen by the strongest instrument and labelled.
    # See `checked_width`: applying an opcode-level class to every width inflated this column by
    # 49 of 140 forms, which is the D2 defect of 62 forms repeated on D3.
    _cw, _cw_how = checked_width(opcode, src) if cls == 'verified' else (None, None)
    semantics_checked = cls == 'verified' and length == _cw

    if semantics_checked:
        semantics = axis('name %r confirmed: %s' % (r.get('name'), why), 'checked',
                         'isa/g17-contract.jsonl', name=r.get('name'),
                         checked_width=_cw, checked_width_origin=_cw_how)
    elif cls == 'verified':
        # the opcode IS verified and this is not the width its probe supports. Informative, and
        # deliberately NOT in D3: the class is per-opcode and a probe dispatched one encoding.
        semantics = axis('name %r confirmed at width %s, NOT at this width: %s'
                         % (r.get('name'), _cw, why), 'corpus',
                         'isa/g17-contract.jsonl', name=r.get('name'),
                         checked_width=_cw, checked_width_origin=_cw_how,
                         checked_at_this_width=False)
    elif meaning.get('constructs'):
        semantics = axis('authorable as %s; %s' % (', '.join(meaning['constructs'][:6]), why),
                         level, 'isa/g17-contract.jsonl', constructs=list(meaning['constructs']))
    elif r.get('name'):
        semantics = axis('carries the name %r but %s' % (r.get('name'), why), level,
                         'isa/g17-contract.jsonl')
    else:
        semantics = axis('unknown', 'absent', None)

    record = {
        'family': dict(state=family, evidence=family_evidence, source='isa/g17-contract.jsonl'),
        'length': axis(length, *width_provenance(opcode, length, src, form),
                       length_bits=enc.get('length_bits') or [],
                       **width_label(opcode, length, src, form)),
        'opcode_selection': axis(
            'bits that reselect the opcode are recorded' if enc.get('opcode_bits')
            else 'not established',
            'decoder' if enc.get('opcode_bits') else 'absent', 'isa/g17-contract.jsonl',
            count=len(enc.get('opcode_bits') or []),
            reselecting_bits_in_this_form=['%d.%d' % (b[0], b[1])
                                           for b in (form.get('selects_opcode') or [])]),
        'operand_encoding': axis(
            'per-operand bit placement is mapped' if enc.get('fields') else 'unmapped',
            'corpus' if apple else 'decoder', 'isa/g17-contract.jsonl',
            operands=enc.get('nops'), defs=enc.get('ndefs'),
            fields=sorted(enc.get('fields') or {}),
            certified=enc.get('certified'),
            witness_origin=('apple' if apple_here else
                            'apple-at-another-width' if apple else 'repair-walk'),
            unmapped_operands=enc.get('unmapped_operands') or []),
        'legal_forms': (lambda derived, witnessed: axis(
            sorted(derived | ({witnessed} if witnessed else set())) or 'no width is evidenced',
            'corpus' if witnessed else ('decoder' if derived else 'table'),
            'isa/g17-form-bits.json + the decoded Apple witness',
            from_form_bits=sorted(derived),
            from_apple_witness=witnessed,
            apple_width_not_in_form_bits=bool(witnessed and derived and witnessed not in derived),
            no_form_bits_entry=not derived))(
                {int(k.split(',')[1]) for k in src['formbits']
                 if k.split(',')[0] == str(opcode)}, src['apple_width'].get(opcode)),
        'implicit_operands': axis(
            'declared' if (reads.get('implicit') or writes.get('implicit')) else 'none declared',
            'table', 'isa/g17-contract.jsonl',
            uses=reads.get('implicit') or [], defs=writes.get('implicit') or []),
        'register_files': axis(
            sorted({c for c in (reads.get('operands') or []) + (writes.get('operands') or []) if c})
            or 'none declared', 'table', 'isa/g17-contract.jsonl',
            writes_address_file=bool(writes.get('address_file'))),
        'immediates': axis(enc.get('certified') or 'unknown',
                           'corpus' if apple else 'decoder', 'isa/g17-contract.jsonl'),
        'semantics': semantics,
        'flags_exec': axis(
            'reads or writes FLAGR' if 'FLAGR' in (reads.get('operands') or []) +
            (writes.get('operands') or []) else 'no flag operand declared',
            'table', 'isa/g17-contract.jsonl',
            control=effects.get('control') or [], may_raise_fp=bool(effects.get('may_raise_fp'))),
        'memory_effects': axis(
            sorted(k for k in ('memory', 'atomic', 'texture', 'side_effects') if effects.get(k))
            or 'none declared', 'table', 'isa/g17-contract.jsonl',
            may_load=bool(reads.get('may_load')), may_store=bool(writes.get('may_store')),
            address_space='unknown'),
        'latency_wait': axis('unknown', 'absent', None),
        'lifetime': axis((src['lifetimes'].get(str(opcode)) or {}).get('choice', 'unknown'),
                         'decoder' if str(opcode) in src['certified_lifetimes'] else 'absent',
                         'isa/g17-auth-lifetimes.json'),
        'barrier_sync': axis('unknown', 'absent', None),
        'emitted_by': axis(
            construct_groups(r) or 'no construct recorded', 'oracle',
            'isa/g17-contract.jsonl',
            means=('construct groups whose compilation emits this opcode: a COMPILER DEPENDENCY, '
                   'not a statement about what the instruction does'),
            pure=len(construct_groups(r)) == 1,
            constructs=len((meaning.get('constructs') or []))),
        'resources': axis('unknown', 'absent', None),
        'hardware_evidence': axis(
            cls, level, 'isa/g17-contract.jsonl', meaning_of_class=why,
            corpus_instances_for_the_opcode=meaning.get('corpus_instances') or 0,
            corpus_instances_for_this_form=corpus_form_instances().get((opcode, length), 0),
            instance_count_scope=(
                'TWO COUNTS FROM TWO POPULATIONS, and they are not a total and a part. '
                "`corpus_instances_for_this_form` is this width alone in the DECODED union of "
                'every committed Apple corpus: 5,010,028 instructions over 1,893 forms, being '
                'isa/g17-corpus-programs.jsonl decoded (365,590) plus the vendor shader corpus '
                '(4,644,438). It formerly read the stored `spans` index of the first file alone, '
                '184,349 instructions, and T1 replaced that - so a reader comparing this against '
                'any older figure is comparing two different populations, not two measurements. '
                "`corpus_instances_for_the_opcode` is the contract's per-opcode figure, "
                'replicated onto every width. It is now BOTH committed corpora with each distinct '
                'program text counted once (tools/g17contract.py corpus_instances, from the '
                '`distinct_texts` block of isa/g17-vendor-corpus-forms.json); it was the local '
                'build cache, which gave op13754 0 and left 1,812 of 1,893 witnessed forms with a '
                "per-form count above their own opcode's figure. Against the per-FORM count, "
                'which is by RECORD, 373 forms still exceed it, and that is the deduplication '
                '(13483/2: 312,123 records, 279,336 in distinct texts), not a second population: '
                'against the distinct-text per-form counts no form exceeds its opcode. So do not '
                'subtract them.'),
            dispatched=bool(receipt), executed_unchecked=bool(receipt) and not semantics_checked,
            receipt=receipt or None),
        'unknown_bits': unknown,
        # A line the peer's own document withdraws further down is not evidence, and the flag
        # reads STANDING rows only. It used to read `bool(rows)`: op11456's withdrawn "is the
        # mask generator" line would have carried the axis by itself, which is the extraction
        # republishing a retracted claim as this map's only reason to count something.
        'peer_claims': (lambda rows, standing: axis(
            ('%d standing line(s) from the peer lane name this opcode%s' % (
                len(standing),
                ', and %d the document itself withdraws' % (len(rows) - len(standing))
                if len(standing) != len(rows) else '')) if rows
            else 'no peer line names this opcode',
            'corpus' if standing else 'absent', 'isa/g17-peer-claims.json',
            extracted_unreviewed=True, peer_reported=bool(standing), verified_here=False,
            pinned_commit=PEER_PIN if rows else None,
            withdrawn_by_the_source=len(rows) - len(standing),
            axis_guesses=sorted({a for r in standing for a in r['axis_guess']})
            if standing else [],
            lines=rows[:6]))(
                *(lambda rows: (rows, [r for r in rows if 'superseded_by' not in r]))(
                    ((src['peer_claims'].get('by_opcode') or {}).get(str(opcode)) or [])
                    if src['peer_claims'].get('present') else [])),
        'ledger_claims': (lambda rows: axis(
            '%d extracted sentence(s) naming this opcode' % len(rows) if rows
            else 'no ledger sentence names this opcode',
            'table' if rows else 'absent', 'isa/g17-ledger-claims.json',
            extracted_unreviewed=True,
            axis_guesses=sorted({a for r in rows for a in r['axis_guess']}) if rows else [],
            strongest_status=(rows[0]['status_class'] if rows else None),
            claims=rows[:6]))((src['claims'].get('by_opcode') or {}).get(str(opcode)) or []),
        'ledger_references': (
            lambda cites: axis(
                'cited by %d ledger(s)' % (len(cites['about']) + len(cites['mentions']))
                if cites else 'no ledger cites this opcode',
                'corpus' if cites and cites['about'] else ('table' if cites else 'absent'),
                'isa/g17-ledger-index.json',
                about=cites['about'] if cites else [], mentions=cites['mentions'] if cites else [],
                cited_ledgers_declaring_a_retraction=sorted(
                    n for n in ((cites['about'] + cites['mentions']) if cites else [])
                    if (src['ledger'].get('ledgers') or {}).get(n, {}).get('declares_a_retraction')),
                means=('where this opcode is discussed and at what status, per opcode not per '
                       'width; it does not assert what those ledgers established. A cited ledger '
                       'that declares a retraction somewhere is named separately: read it before '
                       'relying on it'))
        )((src['ledger'].get('by_opcode') or {}).get(str(opcode))),
    }
    # A curated ledger fact replaces the default for its axis. Applied last so that the axis
    # values above are the fallback and the strongest available evidence wins, never the reverse.
    for name in AXES:
        supplied = ledger_axis(opcode, name, length)
        if supplied is not None:
            # MERGE, never replace. Replacing dropped operand_encoding's witness_origin and broke
            # the D2 definition test, because a curated fact overwrote fields it never carried.
            merged = dict(record.get(name) or {})
            merged.update(supplied)
            record[name] = merged
    # my own hardware probes come first: they are the strongest evidence here and must not be
    # displaced by a peer report or a ledger fact for the same axis
    for name in AXES:
        supplied = probed_axis(opcode, name, length)
        if supplied is not None:
            merged = dict(record.get(name) or {})
            merged.update(supplied)
            record[name] = merged
    peer_axes = []
    for name in AXES:
        supplied = peer_axis(opcode, name)
        if supplied is not None and not record.get(name, {}).get('ledger_status'):
            merged = dict(record.get(name) or {})
            # A DENOMINATOR-BEARING FIELD IS NOT OVERWRITABLE BY A NARRATIVE. unknown_bits.state
            # is what D2 reads, and a peer note landing on it silently decoupled the axis from
            # the flag it feeds - the suite caught that immediately. Their text goes beside the
            # measurement, never on top of it.
            if name == 'unknown_bits':
                supplied = dict(supplied)
                merged['peer_hardware_census'] = supplied.pop('state', None)
                supplied.pop('evidence', None)
            merged.update(supplied)
            record[name] = merged
            peer_axes.append(name)
    # The four D3 columns must be mutually exclusive or the total is nonsense. A probe of mine
    # sets evidence='checked', which is exactly what a ledger fact sets, so without this it is
    # counted twice - the same leak that peer reports had.
    ledger_semantics = (record['semantics']['evidence'] in ('executed', 'checked')
                        and not record['semantics'].get('peer_reported')
                        and not record['semantics'].get('probed_here'))
    fit = (src.get('fits') or {}).get((opcode, length))
    checked = bool(semantics_checked and not record['semantics'].get('probed_here'))
    led = bool(ledger_semantics and not semantics_checked)
    # ONE FORM PER PEER-REPORTED OPCODE, the same narrowing D3's verified class already has.
    # The peer's lines name an OPCODE - op11456's says "the mask generator behind masked tensor
    # stores" with no width - and the axis set peer_reported from those rows, so every width of
    # the opcode carried it. op11456 has three widths and contributed THREE forms for one
    # opcode-level claim: d3_peer_reported read 7 forms over 5 opcodes. This is the third column
    # to need the same fix, after D2 (62 forms) and D3's verified class (49).
    #
    # The width comes from `checked_width`, so the choice is the one the rest of the map already
    # makes and it arrives LABELLED - for op11456 that is 10, with the provenance "receipt,
    # AMBIGUOUS: dispatched at 10, 14 and no instrument records which". A claim whose opcode has
    # no width at all is counted in `peer_claims_with_no_placeable_width` rather than dropped.
    peer_rows = bool(record['semantics'].get('peer_reported'))
    # `_cw` above is computed ONLY for the contract's verified class, and none of the peer-reported
    # opcodes is in it - gating on `_cw` therefore took this column to ZERO on the first attempt,
    # which is the shape of a metric that cannot be moved rather than a narrowing. The width is
    # computed here for the peer opcodes specifically, by the same function and with the same
    # label, and only when there is a claim to place.
    _pw, _pw_how = checked_width(opcode, src) if peer_rows else (None, None)
    peer = peer_rows and _pw is not None and length == _pw
    probed = bool(record['semantics'].get('probed_here'))
    # THE FIFTH COLUMN IS EXCLUSIVE OF THE OTHER FOUR, like they are of each other. Twelve of the
    # census's forms are ones I had already written facts for by hand, and counting them in both
    # places would inflate D3 by exactly the work that validated the tool.
    fitted = bool(fit) and not (checked or led or peer or probed)
    # THE SIXTH COLUMN, EXCLUSIVE OF THE OTHER FIVE. A family law is a third attribution
    # instrument and it is placed LAST in the precedence deliberately: where a form already has
    # its own isolated probe, its own ledger fact, or a library fit against its own cases, that
    # evidence is about THIS form, while a family law is a rule stated over the family. When both
    # exist the narrower instrument should own the claim, and the wider one is a cross-check.
    # THE SEVENTH COLUMN, between the library fit and the family law in precedence: it is a claim
    # about THIS form from its own records, so it outranks a family-level rule, and it is a weaker
    # observable than a 32-bit output so it yields to a fit that survived one.
    _csel = csel_predicate_semantics().get((opcode, length))
    csel_predicate = bool(_csel) and not (checked or led or peer or probed or fitted)
    if _csel:
        record['csel_predicate'] = dict(
            evidence='checked', predicate=_csel['predicate'],
            competitors=_csel['competitors'], records=_csel['records'],
            true_at_fewest_cases=_csel['true_at_fewest_cases'],
            false_at_fewest_cases=_csel['false_at_fewest_cases'],
            source='isa/g17-execution-fits.json csel_boolean_comparisons',
            counted_at_this_width=csel_predicate,
            state=('which comparison this opcode encodes, by eliminating 33 predicates against a '
                   'boolean output. Counted only because the comparison was exercised BOTH ways '
                   'at or above the census\'s own min_cases - a predicate true at every probed '
                   'case cannot be told from the constant true, which is the characteristic '
                   'failure of a boolean observable'))
    # THE NINTH COLUMN. It sits after the exact-match instruments and before the family law: an
    # agreement to one ULP is weaker than an exact match on a form's own cases, and stronger than
    # a rule stated over a family.
    _equiv = equivalence_class_semantics().get((opcode, length))
    equivalence = bool(_equiv) and not (checked or led or peer or probed or fitted
                                        or csel_predicate)
    if _equiv:
        record['equivalence_class'] = dict(
            evidence='checked', candidates=_equiv['equivalence_class'],
            state=_equiv['state'], pairs=_equiv['pairs'],
            source='isa/g17-execution-fits.json the_input_that_would_split_it',
            counted_at_this_width=equivalence,
            means=('the form\'s observable behaviour at its DECLARED WIDTH is determined; the '
                   'residue is which of two library names to write, not what the instruction '
                   'does. No dispatch can separate them here and none should be designed to'))
    _ulp = transcendental_semantics().get((opcode, length))
    transcendental = bool(_ulp) and not (checked or led or peer or probed or fitted
                                         or csel_predicate or equivalence)
    if _ulp:
        record['transcendental'] = dict(
            evidence='checked', name=_ulp['name'], interior_cases=_ulp['interior_cases'],
            interior_exact=_ulp['interior_exact'],
            interior_nontrivial_cases=_ulp.get('interior_nontrivial_cases'),
            interior_nontrivial_exact=_ulp.get('interior_nontrivial_exact'),
            interior_max_ulp=_ulp['interior_max_ulp'],
            overflow_cases=_ulp['overflow_cases'],
            rivals_within_one_ulp=_ulp['rivals_within_one_ulp'],
            source='isa/g17-execution-fits.json transcendental_accuracy',
            counted_at_this_width=transcendental, verdict=_ulp['verdict'],
            state=('the hardware reproduces the correctly-rounded value of the operation its own '
                   'Apple name states, to at most one ULP over the interior cases, with overflow '
                   'counted apart - and no OTHER named operation in the table comes within a ULP '
                   'on the same records, which is the elimination an accuracy measurement does '
                   'not otherwise perform'))
    _lane = lane_probe_semantics().get((opcode, length))
    _mem = memory_probe_semantics().get('%d/%d' % (opcode, length))
    _ctl = control_probe_semantics().get('%d/%d' % (opcode, length))
    lane_probe = bool(_lane) and not (checked or led or peer or probed or fitted
                                      or csel_predicate or transcendental or equivalence)
    memory_probe = bool(_mem) and not (checked or led or peer or probed or fitted
                                       or csel_predicate or transcendental or equivalence
                                       or lane_probe)
    control_probe = bool(_ctl) and not (checked or led or peer or probed or fitted
                                        or csel_predicate or transcendental or equivalence
                                        or lane_probe or memory_probe)
    if _ctl:
        record['control_probe'] = dict(
            evidence='executed', reading=_ctl['reading'], probes=_ctl['probes'],
            relations=_ctl['relations'], apple_name=_ctl['apple_name'],
            distinct_encodings=_ctl['distinct_encodings'],
            source='isa/g17-comparison-semantics.json',
            means=('a vendor-compiled kernel containing this form was dispatched once per '
                   'relation, with operands chosen so that no rival relation and no rival '
                   'signedness agrees on all sixteen cases. The relation field itself is not '
                   'decoded, so the measured values are listed rather than summarised'))
    if _mem:
        record['memory_probe'] = dict(
            evidence='executed', reading=_mem['reading'], probes=_mem['probes'],
            apple_name=_mem['apple_name'], source='isa/g17-memory-semantics.json',
            means=('a vendor-compiled kernel containing this form was dispatched and the memory '
                   'read back, with a sentinel-filled output and the untouched slots checked. '
                   'The authored oracle cannot reach these opcodes at all - the safety gate '
                   'refuses them'))
    if _lane:
        record['lane_probe'] = dict(
            evidence='checked', reading=_lane['reading'], probes=_lane['probes'],
            lanes=_lane['lanes'], refuted=_lane['refuted'],
            source='isa/g17-lane-semantics.json',
            counted_at_this_width=lane_probe,
            state=('every lane read from a compiled Metal kernel with a thread-indexed '
                   'destination, the form named by decoding the emitted stream, and the '
                   'candidates required to be pairwise separated by the inputs before the '
                   'dispatch. The observable the isolated-dispatch oracle cannot provide'))
    _law = family_law_semantics().get((opcode, length))
    family = bool(_law) and _law['tested'] and not (
        checked or led or peer or probed or fitted or csel_predicate or lane_probe
        or transcendental or equivalence)
    if _law:
        record['family_law'] = dict(
            evidence='checked' if _law['tested'] else 'absent',
            laws=_law['laws'],
            laws_whose_records_do_not_test_them=_law['untested'],
            records=_law['records'],
            source='isa/g17-execution-fits.json family_laws.laws[*].explains',
            counted_at_this_width=family,
            also_counted_elsewhere=bool(_law['tested']) and not family,
            state=('a rule stated over a FAMILY reproduces this form\'s retained cases. The '
                   'values are hardware observations; what is family-level is the ATTRIBUTION, '
                   'so this is a different instrument from a probe that confirms a name and from '
                   'a library fit against this form\'s own cases, and it is counted apart from '
                   'both. A family law can be right about the family and wrong about a member '
                   'whose encoding differs in something the law does not read'))
    if fit:
        record['fitted_by_elimination'] = dict(
            evidence='checked', function=fit['function'],
            source='isa/g17-execution-fits.json',
            state='every candidate of the published library computed against the words an '
                  'isolated dispatch returned; exactly one survives every case',
            records=[dict(stem=r['stem'], id=r['id'], cases=r['cases'], runs=r['runs'],
                          competitors=r['competitors'], had_expect=r['had_expect'],
                          outputs_constant_across_cases=r.get(
                              'outputs_constant_across_cases'),
                          outputs_equal_an_input_column=r.get(
                              'outputs_equal_an_input_column'))
                     for r in fit['records']],
            also_counted_elsewhere=not fitted,
            scope=fit.get('scope'))
    if peer_rows:
        record['semantics'] = dict(record['semantics'], peer_width=_pw,
                                   peer_width_origin=_pw_how,
                                   peer_claim_counted_at_this_width=peer,
                                   peer_claim_placeable=_pw is not None)
    record['denominators'] = dict(
        d1_structural=True, d2_encoded=bool(encoded),
        d3_semantics_checked=checked, d3_ledger_executed=led,
        d3_peer_reported=peer, d3_probed_here=probed,
        d3_fitted_by_elimination=fitted, d3_csel_predicate=csel_predicate,
        d3_lane_probe=lane_probe, d3_transcendental=transcendental,
        d3_equivalence_class=equivalence, d3_family_law=family,
        d3_memory_probe=memory_probe, d3_control_probe=control_probe)
    return record


def build():
    src = sources()
    src['ledger'] = ledger_index()
    src['claims'] = extract_claims()
    src['peer_claims'] = extract_peer_claims()
    src['fits'] = fitted_semantics()
    spec, universe = {}, {}
    forms = {(int(k.split(',')[0]), int(k.split(',')[1])) for k in src['formbits']}
    forms |= set(src['freebits'])
    forms |= set(src['executed'])
    # An Apple-WRITTEN witness at a width evidences that form even when form-bits omits it.
    forms |= {(op, w) for op, w in src['apple_width'].items() if op in src['contract']}
    # Forms Apple's shipped corpus contains, which the exemplar-keyed set above cannot reach:
    # an opcode appears there at every width Apple emitted it at, not only the retained one.
    forms |= {(op, w) for (op, w) in corpus_form_instances() if op in src['contract']}
    # ITEM 2. Every admitted opcode gets a form record at the width its repair-walk witness
    # decodes to, so D1 counts the surface this decoder can address rather than the subset that
    # also has form-bits. The width is LABELLED on the length axis as a repair-walk decode and
    # carries admissible_for_d2=False, so D2 cannot move: form_record gates D2 on an Apple
    # witness independently of how the form entered this set.
    #
    # This is the same widening that once took D1 from 970 to 7,013 and was reverted, and the
    # reason it is right now and was wrong then is provenance: unlabelled, the jump reads as
    # progress on encodings Apple never wrote. D1 is therefore reported SPLIT by witness origin
    # (see `d1_apple_witnessed` / `d1_repair_walk_only`), and a test asserts the parts sum.
    forms |= {(op, w) for op, w in src['repair_width'].items()
              if op in src['contract'] and w}
    for opcode, length in sorted(forms):
        if opcode in src['contract']:
            spec.setdefault('op%d' % opcode, {})['len%d' % length] = form_record(opcode, length, src)
    for opcode, r in sorted(src['contract'].items()):
        family, family_evidence = family_of(r)
        enc = r.get('encoding') or {}
        meaning = r.get('meaning') or {}
        universe['op%d' % opcode] = dict(
            family=family, family_evidence=family_evidence, name=r.get('name'),
            schedclass=r.get('schedclass'), evidence_class=meaning.get('evidence'),
            evidence_level=EVIDENCE_CLASS.get(meaning.get('evidence'), ('table',))[0],
            corpus_instances=meaning.get('corpus_instances') or 0,
            witness_origin='apple' if enc.get('apple_witness') else 'repair-walk',
            certified=enc.get('certified'), nops=enc.get('nops'), ndefs=enc.get('ndefs'),
            widths=sorted({int(k.split(',')[1]) for k in src['formbits']
                           if k.split(',')[0] == str(opcode)}),
            dispatched_widths=sorted(l for (o, l) in src['executed'] if o == opcode),
            # a width for EVERY admitted opcode, each labelled with how it is known. The
            # apple-byte-count widths are the only ones D2 will ever accept.
            width_record=(
                dict(width=src['apple_width'][opcode], evidence='apple-witness byte count',
                     admissible_for_d2=True)
                if opcode in src['apple_width'] else
                dict(width=src['repair_width'][opcode],
                     evidence='decoded length of a repair-walk witness',
                     admissible_for_d2=False,
                     means='the decoder accepts a form of this width; nothing is known to have '
                           'emitted one, and forty class-51 opcodes authored this way return a '
                           'constant of the encoding')
                if opcode in src['repair_width'] else
                dict(width=None, evidence='absent',
                     means='no witness of this opcode decodes to itself')),
            full_axis_record='op%d' % opcode in spec)
    hypothesis = schedclass_hypothesis(universe)
    for name, row in (hypothesis['by_opcode'] or {}).items():
        universe[name]['family_hypothesis'] = dict(
            row, evidence='inference', means=('a scheduling-class co-membership hypothesis, '
                                              'never a classification; family stays unknown'))
    src['family_hypothesis'] = {k: v for k, v in hypothesis.items() if k != 'by_opcode'}
    return src, spec, universe


# THE PREREGISTERED-EXPECT CENSUS. isa/ holds plan/result pairs of isolated dispatches; a plan
# record may carry an `expect` list, and the result carries the words the hardware returned. Where
# they are equal over several runs that is the strongest per-form evidence in this repository -
# EXCEPT that every plan was committed in the same commit as its results, so nothing in the
# history says the expectation came first. The split below is therefore between expectations a
# RULE regenerates (promotable, and three are) and expectations that are a PRIOR OBSERVATION -
# "the debt bits cleared, same result" - which no rule reproduces and which stay unchecked.
# Counted rather than asserted, because `limits` quotes these numbers.
_RULE_REPRODUCED_FORMS = ((998, 12), (2190, 16), (3290, 14), (3802, 10),
                          (3818, 10), (10279, 12), (10826, 14))
# Determined with NO stored expectation, by eliminating every competing candidate against the
# observed words. Counted apart because the same-commit caveat above does not apply to these.
_FUNCTION_FITTED_FORMS = ((1062, 10), (13460, 10), (13488, 10), (13521, 10),
                          (13548, 10), (17744, 10))


def preregistered_expect_basis():
    """Forms whose isolated result equals a plan's `expect`, split by whether a rule regenerates
    it. Reads the files rather than a remembered count, so a new pair moves the number."""
    plans, results = {}, {}
    for path in sorted(ISA.glob('g17-execution-*.json')):
        name = path.name
        try:
            recs = json.loads(path.read_text())
        except Exception:
            continue
        recs = recs if isinstance(recs, list) else recs.get('results', list(recs.values()))
        is_result = name.endswith('-results.json')
        stem = name[:-len('-results.json')] if is_result else name[:-len('.json')]
        for record in recs:
            if isinstance(record, dict) and record.get('id'):
                (results if is_result else plans).setdefault(stem, {})[record['id']] = record
    matched, pairs = set(), 0
    for stem in sorted(set(plans) & set(results)):
        pairs += 1
        for rid, res in results[stem].items():
            plan = plans[stem].get(rid) or {}
            expect = plan.get('expect')
            match = res.get('match')
            if not expect or res.get('status') != 'ok' or not match or not all(match):
                continue
            if list(res.get('values') or []) != list(expect):
                continue
            op = res.get('op')
            for hexed in (res.get('decoded') or {}).get('encoded') or []:
                try:
                    raw = bytes.fromhex(hexed)
                except Exception:
                    continue
                for ins in decode_instruction(raw, 0):
                    if ins.opcode is not None and ins.opcode.id == op:
                        matched.add((op, len(ins.raw)))
    promoted = sorted(set(_RULE_REPRODUCED_FORMS) & matched)
    return dict(plan_result_pairs=pairs, forms_matching_a_planned_expect=len(matched),
                function_fitted_with_no_expect=len(_FUNCTION_FITTED_FORMS),
                function_fitted=[list(f) for f in sorted(_FUNCTION_FITTED_FORMS)],
                rule_reproduced_and_promoted=len(promoted),
                expectation_is_a_prior_observation=len(matched) - len(promoted),
                promoted=[list(f) for f in promoted],
                unreproduced=[list(f) for f in sorted(matched - set(promoted))],
                how=('equal over every case AND every run, the instruction authored alone; '
                     'promotion required recomputing the expectation from an independent '
                     'reference, because plan and results share a commit and "observed == '
                     'expect" cannot otherwise fail. A FORM CAN HAVE SEVERAL ARMS in different '
                     'plan/result pairs, and they can differ in strength: four of these were '
                     'first called unreproducible because the scan reached their preservation '
                     'arm, where the expected value is a prior observation, while a second arm '
                     'in g17-execution-functions.json states an arithmetic RULE. Check every '
                     'arm of a form before calling its expectation unreproducible.'))


# HOW OFTEN A FORM APPEARS IN APPLE'S WHOLE SHIPPED SET, not in the build cache. The free-bits
# verdicts were fitted on a cache scan; isa/g17-corpus-programs.jsonl is 184,349 instructions and
# carries (offset, length, opcode) spans, so the count is per FORM rather than per opcode - which
# matters, because `corpus_instances` in the contract is an opcode fact and an opcode with two
# widths can be two singletons rather than a pair.
# PER-FORM CORPUS INSTANCES, because the contract's count is per OPCODE and this map publishes it
# on FORM records. op17220 carries corpus_instances 2 at both len8 and len10: the opcode appears
# twice, and a reader of either form record would reasonably take it as two instances of THAT form.
# They are two singletons. That is the same opcode-for-form conflation that inflated D2 by 62 forms
# and D3's verified class by 49 - the third place it has turned up, and the first in a field nobody
# was gating on.
#
_VENDOR_INDEX = 'g17-vendor-corpus-forms.json'
# Bumped when the CODE below changes WHAT it counts, not only when its inputs change. The old cache
# key was a digest of one input file, so replacing `spans` with a decode and adding a second corpus
# would have been served the previous answer off disk - a key over the input but not over the code.
_CORPUS_SCAN_VERSION = 2


def _decode_forms(path, text_key='text'):
    """{(opcode, width): instances} by DECODING each program, not by reading a stored index."""
    count = Counter()
    if not path.exists():
        return count
    import agxforge.g17.model as _model
    with path.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except Exception:
                continue
            hexbytes = record.get(text_key)
            if not hexbytes:
                continue
            try:
                decoded = _model.decode(bytes.fromhex(hexbytes))
            except Exception:
                continue
            for inst in decoded:
                count[(inst.opcode.id, len(inst.raw))] += 1
    return count


def corpus_sources():
    """Every committed Apple corpus admission counts, kept SEPARATE so provenance survives.

    Two corrections live here, and both change published counts, so they are recorded rather than
    applied quietly (section 25.78).

    **It read a stored index, not the corpus.** `g17-corpus-programs.jsonl` carries a precomputed
    `spans` field and this counted that: 184,349 instructions where decoding the same committed file
    gives 365,590, a **49.6% undercount**, and 784 contract forms where decoding gives 984. Two
    hundred forms were already in a committed file and invisible to admission.

    **It ignored a second corpus entirely.** `results/g17-vendor-shader-corpus-v1` holds 9,556
    Apple-compiled functions and 4,644,438 instructions, and none of it was admitted.

    The decode itself lives in `tools/g17vendorforms.py`, which writes a committed index - reading
    it costs 0.001s against 104 seconds to decode, and this runs inside the test suite. That is the
    same shape as the `spans` field being repaired, with the one difference that made `spans` wrong:
    `g17vendorforms.py --check` re-decodes both corpora and compares. An index whose freshness is
    verifiable is a cache; one whose freshness is assumed is a second source of truth.
    """
    index = ISA/_VENDOR_INDEX
    if index.exists():
        sources = (json.loads(index.read_text()).get('sources') or {})
        if sources:
            out = {}
            for name, block in sources.items():
                counts = Counter()
                for key, n in (block.get('forms') or {}).items():
                    opcode, width = (int(x) for x in key.split('/'))
                    counts[(opcode, width)] = n
                out[name] = counts
            return out
    # NO INDEX: decode both corpora directly. Two orders of magnitude slower and never silently
    # wrong, and the only path a fresh clone without the extracted corpus can take.
    out = {'corpus_programs': _decode_forms(ISA/'g17-corpus-programs.jsonl')}
    vendor = Counter()
    for path in sorted((ROOT/'results'/'g17-vendor-shader-corpus-v1'/'sweeps').glob('*.jsonl')):
        vendor.update(_decode_forms(path))
    out['vendor_shader_corpus'] = vendor
    return out


# WAS: "carries (offset, LENGTH, opcode) spans over Apple's whole shipped set, so the count is
# available at the right unit for the price of one pass". It was not Apple's whole shipped set and
# the price was wrong in the other direction: `spans` reports 184,349 instructions where decoding
# the same file gives 365,590, and a second committed corpus of 4,644,438 was not read at all.
# Cached on the inputs AND on _CORPUS_SCAN_VERSION, because a key over the inputs alone would serve
# the pre-rewrite answer from disk after the code changed what it counts.
@_memoized
def corpus_form_instances():
    """{(opcode, width): instances} across every committed Apple corpus. Union of corpus_sources."""
    inputs = [ISA/'g17-corpus-programs.jsonl', ISA/_VENDOR_INDEX]
    key = hashlib.sha256(
        ('|'.join('%s:%s' % (p.name, p.stat().st_mtime_ns if p.exists() else 0) for p in inputs)
         + '|v%d' % _CORPUS_SCAN_VERSION).encode() + decoder_build_identity()).hexdigest()[:16]
    cached = CACHE/('corpusforms-%s.json' % key)
    if cached.exists():
        return {tuple(int(x) for x in k.split(',')): v
                for k, v in json.loads(cached.read_text()).items()}
    total = Counter()
    for counts in corpus_sources().values():
        total.update(counts)
    CACHE.mkdir(parents=True, exist_ok=True)
    _cache_write(cached, json.dumps({'%d,%d' % k: v for k, v in sorted(total.items())}))
    return dict(total)


def corpus_distinct_text_forms():
    """{(opcode, width): (instances, texts)} with each distinct program text counted ONCE.

    `corpus_form_instances` counts RECORDS, and records repeat: the vendor corpus is 9,556
    functions over 7,468 distinct texts and the two corpora share 2. Where a question needs two
    WITNESSES of a form, two copies of one function are one, so it reads this instead.
    """
    index = ISA/_VENDOR_INDEX
    if index.exists():
        block = (json.loads(index.read_text()).get('distinct_texts') or {}).get('forms') or {}
        if block:
            return {tuple(int(x) for x in key.split('/')): (n, texts)
                    for key, (n, texts) in block.items()}
    # NO INDEX: decode the distinct texts of both corpora directly.
    import agxforge.g17.model as _model
    seen = set()
    count, texts = Counter(), Counter()
    paths = [ISA/'g17-corpus-programs.jsonl'] + sorted(
        (ROOT/'results'/'g17-vendor-shader-corpus-v1'/'sweeps').glob('*.jsonl'))
    for path in paths:
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                try:
                    hexbytes = json.loads(line).get('text')
                except Exception:
                    continue
                if not hexbytes or hexbytes in seen:
                    continue
                seen.add(hexbytes)
                try:
                    got = Counter((i.opcode.id, len(i.raw))
                                  for i in _model.decode(bytes.fromhex(hexbytes)))
                except Exception:
                    continue
                count.update(got)
                for key in got:
                    texts[key] += 1
    return {k: (count[k], texts[k]) for k in count}


# WAS: read `spans` out of isa/g17-corpus-programs.jsonl alone, and published the result as
# "Apple's whole shipped set". It was the T1 defect (section 25.58) surviving in a second reader
# after section 25.78 repaired corpus_form_instances: 34 of 34 undetermined forms "singleton or
# absent", so "NOT more corpus ... only execution settles them". Over both corpora, one witness
# per distinct text, 32 of the 34 have two or more instances - the remedy the text ruled out.
def corpus_singletons(forms):
    """For each (opcode, width), how many instances every committed Apple corpus holds.

    `counts` is instances over DISTINCT program texts across both corpora - a function shipped in
    two libraries is one witness, not two. `per_source` keeps the record counts of each corpus,
    so a form's evidence stays attributable to the artifact carrying it.
    """
    if not forms:
        return dict(forms_checked=0, singleton_or_absent=0, counts={}, note='no forms')
    distinct = corpus_distinct_text_forms()
    if not distinct:
        return dict(forms_checked=0, singleton_or_absent=0, counts={}, note='corpus absent')
    sources = corpus_sources()
    keys = sorted(set(map(tuple, forms)))
    counts = {'%d/%d' % f: distinct.get(f, (0, 0))[0] for f in keys}
    absent = sorted(k for k, v in counts.items() if v == 0)
    return dict(forms_checked=len(counts),
                singleton_or_absent=sum(1 for v in counts.values() if v < 2),
                reachable_by_reading_more_corpus=sum(1 for v in counts.values() if v >= 2),
                counts=counts, absent_from_apples_corpus=absent,
                distinct_texts_carrying={'%d/%d' % f: distinct.get(f, (0, 0))[1] for f in keys},
                per_source={name: {'%d/%d' % f: src.get(f, 0) for f in keys}
                            for name, src in sorted(sources.items())},
                note=('a verdict needs two instances of the FORM; `counts` is instances over '
                      'the distinct program texts of BOTH committed Apple corpora '
                      '(isa/g17-corpus-programs.jsonl decoded, and the vendor shader corpus), '
                      'each text once - not the build cache the verdicts were fitted on, and not '
                      'the `spans` index of the first corpus this used to read. `per_source` '
                      'is record counts per corpus. A form no corpus contains is listed in '
                      '`absent_from_apples_corpus` - its free-bits instance came from somewhere '
                      'else, a probe or a repair-walk witness, and is not Apple evidence'))


def undetermined_free_bits_pass(forms, freebits):
    """What the free-bits rule gives for each form's UNDETERMINED bits over both corpora.

    Read from `isa/g17-free-bits-known-drift.json`, which `tools/g17freebits.py --pin` writes by
    regenerating every verdict from both committed corpora and which `--check-pinned` re-derives in
    `make check-ledgers`.

    A verdict still in the drift is NOT adopted into `isa/g17-free-bits.jsonl`. Section 25.88 made
    adopting a byte-changing verdict a GPU-checked decision; that check ran
    (isa/g17-adoption-check-results.json), the 13 forms it cleared were adopted and LEFT this
    population, and each form still here carries the reason it was not (`adoption`), from
    tools/g17adoptcheck.py's own record.

    Measured 2026-09-24, and not assumed: re-deriving these verdicts with each distinct program
    text counted once gives the same verdict and value on all 470 drift bits of these forms, so
    record-level duplication does not manufacture any of these constants.
    """
    path = ISA/'g17-free-bits-known-drift.json'
    if not path.exists() or not forms:
        return dict(forms_checked=0, note='no pinned drift, or no forms')
    drift = {}
    for op, ln, bit, _before, after in json.loads(path.read_text()).get('drift') or []:
        drift[(op, ln, bit)] = after
    per, bits = {}, Counter()
    for f in sorted(set(map(tuple, forms))):
        und = sorted(b for b, v in (freebits.get(f) or {}).items()
                     if isinstance(v, dict) and v.get('verdict') == 'undetermined')
        got = Counter()
        for b in und:
            after = drift.get((f[0], f[1], b))
            got[after[0] if after and after[0] in ('constant', 'varies') else 'undetermined'] += 1
        bits.update(got)
        per['%d/%d' % f] = dict(undetermined_bits=len(und),
                                adoption=(_singleton_execution(f)
                                          if und and got.get('undetermined') == len(und)
                                          else _adoption(f)),
                                **{k: got.get(k, 0) for k in ('constant', 'varies',
                                                              'undetermined')})
    settled = sorted(k for k, v in per.items() if v['undetermined_bits'] and not v['undetermined'])
    return dict(forms_checked=len(per), forms_every_undetermined_bit_settled=len(settled),
                forms_still_undetermined=sorted(set(per) - set(settled)),
                bits=dict(sorted(bits.items())), per_form=per, adopted=False,
                by_adoption=dict(sorted(Counter(v['adoption'].split(':')[0]
                                                for v in per.values()).items())),
                source='isa/g17-free-bits-known-drift.json (tools/g17freebits.py --pin, both corpora)',
                note=('verdicts the corpus gives for forms still outside D2 on undetermined bits; '
                      'none of these is adopted, and `adoption` says why per form'))


def _singleton_execution(form):
    """What execution says about a single-instance form's undetermined bits, from the retained runs."""
    base = 'singleton: the corpus cannot settle it'
    path = ISA/'g17-singleton-bits-results-v2.json'
    if path.exists():
        doc = json.loads(path.read_text())
        rows = [r for r in doc.get('rows') or [] if not r.get('control')
                and (r.get('op'), r.get('length')) == tuple(form)]
        if rows and doc.get('controls_ok'):
            free = sum(1 for r in rows if r.get('verdict') == 'as-predicted')
            live = sum(1 for r in rows if r.get('verdict') == 'BIT-IS-LIVE')
            return ('%s; EXECUTED (isa/g17-singleton-bits-results-v2.json, controls ok): %d of %d '
                    'undetermined bits FREE on hardware, %d live, the rest uninformative; not '
                    'written into isa/g17-free-bits.jsonl, whose verdicts are corpus-derived'
                    % (base, free, len(rows), live))
    tile = ISA/'g17-tile-store-probe-results-v2.json'
    if tuple(form) == (17257, 12):
        if tile.exists():
            return base + '; executed by tools/g17tilestoreprobe.py, see its results'
        return (base + '; tools/g17tilestoreprobe.py v1 was VOID (sign-symmetric input control), '
                'v2 awaits a dispatch')
    return base + '; execution probes in tools/g17singletonbits.py'


def _adoption(form):
    """Why a drifting form's verdicts are not adopted, from tools/g17adoptcheck.py's record."""
    import g17adoptcheck as _ac
    if tuple(form) in _ac.NOT_ADOPTED:
        return _ac.NOT_ADOPTED[tuple(form)]
    planned = {(f['op'], f['length']) for f in _ac.load_plan()['forms']}
    if tuple(form) not in planned:
        return _ac.NO_BYTE_DECISION
    return 'in the adoption plan with no recorded outcome'


# WHAT THE ELIMINATION CENSUS COULD NOT SEPARATE, DECOMPOSED BY WHETHER IT MATTERS. The census
# reports ambiguous, interpretation-ambiguous and constant-output records, and the raw totals read
# like a large pool of unknown instructions. They are not. Almost every one of those records belongs
# to an opcode whose semantics this map already carries from another instrument - so the ambiguity
# is a property of the RECORD's chosen inputs, not a gap in the specification.
#
# The two pairs the census names most often are the clearest case: `bitwise_A` against umin/smin
# over 19 records, where the second operand is always the smaller so "return b" and "return the
# minimum" agree; and `bitwise_C` against rotl/sar/shl/shr over 12, where the shift amount is zero
# so "return a" and "shift by nothing" agree. Twenty-two opcodes are involved and EVERY one of them
# already has semantics populated, so probing them would produce cross-checks rather than coverage.
# That is worth stating because the inseparable-pairs report reads like a work queue and is not one.
def unseparated_basis(spec):
    """The census's unseparated records, split by whether the opcode's semantics is already known."""
    path = ISA/'g17-execution-fits.json'
    if not path.exists():
        return dict(records=0, note='the elimination census has not run')
    try:
        doc = json.loads(path.read_text())
    except Exception:
        return dict(records=0, note='the elimination census is unreadable')
    known = {}
    for name, forms in spec.items():
        evidence = {(record.get('semantics') or {}).get('evidence', 'absent')
                    for record in forms.values()}
        known[int(str(name)[2:])] = evidence != {'absent'}
    kinds, unknown = Counter(), {}
    for row in doc.get('rows') or []:
        constant = bool(row.get('outputs_constant_across_cases'))
        verdict = row.get('verdict')
        if not constant and verdict not in ('ambiguous', 'interpretation-ambiguous'):
            continue
        # CONSTANT FIRST, because the populations OVERLAP: a record can be both ambiguous and
        # constant-output, and counting it in both would double it. Constant is the stronger
        # statement - an absorbing function fits it whatever the candidate list says.
        kind = 'constant output' if constant else verdict
        already = known.get(row.get('op'), False)
        kinds[(kind, 'opcode semantics already known' if already else
               'opcode semantics ABSENT')] += 1
        if not already:
            unknown.setdefault(row.get('op'), []).append(dict(kind=kind, id=row.get('id')))
    return dict(records=sum(kinds.values()),
                by_kind_and_novelty={'%s / %s' % k: v for k, v in sorted(kinds.items())},
                opcodes_with_no_other_semantics=sorted(unknown),
                note=('the raw totals read like a pool of unknown instructions and are not: the '
                      'ambiguity belongs to the records\' chosen INPUTS, and almost every opcode '
                      'in them already has semantics from another instrument. Constant-output is '
                      'counted first because the populations overlap and it is the stronger '
                      'statement. The named pairs - bitwise_A against umin over 19 records where '
                      'the second operand is always smaller, bitwise_C against rotl over 12 where '
                      'the shift is zero - involve 22 opcodes and ALL of them are already known, '
                      'so probing them yields cross-checks rather than coverage'))


_FAMILY_LAW_CACHE = {}


def family_law_semantics():
    """{(opcode, length): {laws, records, tested}} - the forms a FAMILY LAW determines.

    THE SIXTH D3 COLUMN'S SOURCE. A family law is a third attribution instrument, beside the
    candidate library and the name-confirming isolated probe: it predicts a form's output from the
    table name, the declared precision, an immediate decoded out of the record's own bytes, or the
    lane set the record declares, and then requires the retained observation to match.

    WHAT IT ASSERTS, PRECISELY, because the column's name has to be honest. The values ARE hardware
    observations - every record it reads is a retained dispatch result, so this is not inference
    from structure alone. What differs from `d3_semantics_checked` is the ATTRIBUTION: the verified
    class asserts that an isolated probe returned the value a NAME predicts, the elimination census
    asserts that exactly one candidate of a published library survives a form's own cases, and a
    family law asserts that a rule stated over a FAMILY reproduces this form's cases. The failure
    modes differ, which is the whole reason these columns are exclusive rather than summed: a
    family law can be right about the family and wrong about a member whose encoding differs in
    something the law does not read.

    `tested` comes from the census, which owns `_law_is_tested` - a law whose own title says its
    records do not test it (`DEGENERATE EVIDENCE`, `NOT DETERMINED`) determines nothing, and
    re-deriving that predicate here would be a control sharing its predicate with what it controls.

    The width is decoded from each explaining record's emitted bytes, with the instruction's own
    opcode required to match before the length is trusted - the discipline that moved 89 sweep
    receipts off the width their ids claimed.
    """
    if _FAMILY_LAW_CACHE:
        return _FAMILY_LAW_CACHE['forms']
    path = ISA / 'g17-execution-fits.json'
    forms = {}
    if not path.exists():
        _FAMILY_LAW_CACHE['forms'] = forms
        return forms
    try:
        doc = json.loads(path.read_text())
    except Exception:
        _FAMILY_LAW_CACHE['forms'] = forms
        return forms
    laws = ((doc.get('family_laws') or {}).get('laws') or {})
    rows = {(r.get('stem'), r.get('id')): r for r in (doc.get('rows') or [])}
    emitted = {}
    for results in sorted(ISA.glob('g17-execution-*-results.json')):
        stem = results.name[:-len('-results.json')]
        try:
            data = json.loads(results.read_text())
        except Exception:
            continue
        for row in (data if isinstance(data, list) else data.get('results') or []):
            if not (isinstance(row, dict) and row.get('id')):
                continue
            hexes = (row.get('decoded') or {}).get('encoded') or []
            if hexes and isinstance(hexes[0], str) and hexes[0]:
                emitted[(stem, row['id'])] = hexes[0]
    unknown = collections.Counter()
    for name, law in laws.items():
        title = name.split(':', 1)[0]
        for pair in (law.get('explains') or []):
            row = rows.get(tuple(pair))
            if not row or row.get('op') is None:
                continue
            opcode = int(row['op'])
            text = emitted.get(tuple(pair))
            length = None
            if text:
                try:
                    instructions = list(decode_instruction(bytes.fromhex(text)))
                    hit = [i for i in instructions
                           if getattr(getattr(i, 'opcode', None), 'id', None) == opcode]
                    if len(instructions) == 1 and hit:
                        length = len(getattr(hit[0], 'raw', b'') or b'') or len(text) // 2
                except Exception:
                    length = None
            if length is None:
                unknown['op%d' % opcode] += 1
                continue
            entry = forms.setdefault((opcode, length),
                                     dict(laws=set(), untested=set(), records=0))
            entry['laws'].add(title)
            if not law.get('tested', True):
                entry['untested'].add(title)
            entry['records'] += 1
    for entry in forms.values():
        entry['laws'] = sorted(entry['laws'])
        entry['untested'] = sorted(entry['untested'])
        # A form is determined when at least one law its records TEST places it. A form placed only
        # by laws whose titles disclaim their own evidence is not determined by any of them.
        entry['tested'] = bool(set(entry['laws']) - set(entry['untested']))
    _FAMILY_LAW_CACHE['forms'] = forms
    _FAMILY_LAW_CACHE['width_unknown'] = dict(unknown)
    return forms


_AUDIT_PLANS = {}


def _audit_plans():
    """{(stem, id): plan} from the execution plan files - the cases each record actually asked."""
    if _AUDIT_PLANS:
        return _AUDIT_PLANS
    for path in sorted(ISA.glob('g17-execution-*.json')):
        if path.name.endswith('-results.json'):
            continue
        try:
            rows = json.loads(path.read_text())
        except Exception:
            continue
        for row in (rows if isinstance(rows, list) else []):
            if isinstance(row, dict) and row.get('id') and row.get('cases'):
                _AUDIT_PLANS[(path.name[:-len('.json')], row['id'])] = row
    return _AUDIT_PLANS


_EQUIV_CACHE = {}


def equivalence_class_semantics():
    """{(opcode, length): class} for forms whose surviving candidates are ONE function at this width.

    THE TENTH D3 COLUMN, and the narrowest claim any of them makes. The census's split analysis
    reports, per ambiguous record, whether a concrete input separates each pair of survivors. Its
    strongest verdict is not "our inputs failed" but "SAME FUNCTION at this declared width - not an
    ISA ambiguity but two names in this library for one function once the operands are masked, so
    no dispatch can split them and none should be designed to". Where every surviving pair of a
    form carries that verdict, the form's observable behaviour AT ITS DECLARED WIDTH is determined;
    what remains ambiguous is which of two library NAMES to write down, which is a fact about the
    library.

    `16819/12` is the clearest case: it is Apple's `asr` and `17045/14` is Apple's `shr`, both
    declare 16-bit sources, and a 16-bit source zero-extends so the sign bit is always clear - which
    makes an arithmetic and a logical right shift the same function there. Recording that as an
    equivalence is the honest reading; recording it as "unresolved" invites an experiment that
    should not be designed, and this lane routed exactly such a request to another lane before
    withdrawing it.

    Counted only for the SAME-FUNCTION verdict. The weaker siblings - "different functions this
    destination cannot show apart" and "indistinguishable over the declared domain" - are SAMPLED
    rather than argued from the operand class, and the census says so in its own note, so they stay
    out.
    """
    if _EQUIV_CACHE:
        return _EQUIV_CACHE['forms']
    out = {}
    try:
        doc = json.loads((ISA / 'g17-execution-fits.json').read_text())
    except Exception:
        _EQUIV_CACHE['forms'] = out
        return out
    records = ((doc.get('the_input_that_would_split_it') or {}).get('records') or {})
    per_opcode = collections.defaultdict(list)
    for key, entry in records.items():
        per_opcode[key.split('/')[0]].append(entry)
    rows = collections.defaultdict(list)
    for row in (doc.get('rows') or []):
        if row.get('op') is not None:
            rows[int(row['op'])].append(row)
    for opcode_text, entries in per_opcode.items():
        pairs = [pair for entry in entries for pair in (entry.get('pairs') or [])]
        if not pairs or not all('SAME FUNCTION' in (p.get('state') or '') for p in pairs):
            continue
        opcode = int(opcode_text)
        widths = set()
        for row in rows.get(opcode, []):
            text = _emitted_bytes().get((row.get('stem'), row.get('id')))
            if not text:
                continue
            try:
                instructions = list(decode_instruction(bytes.fromhex(text)))
            except Exception:
                continue
            hit = [i for i in instructions
                   if getattr(getattr(i, 'opcode', None), 'id', None) == opcode]
            if len(instructions) == 1 and hit:
                widths.add(len(getattr(hit[0], 'raw', b'') or b'') or len(text) // 2)
        if len(widths) != 1:
            continue
        names = sorted({tuple(sorted(p.get('candidates') or [])) for p in pairs})
        out[(opcode, widths.pop())] = dict(
            equivalence_class=[list(n) for n in names],
            state=(pairs[0].get('state') or ''),
            pairs=len(pairs))
    _EQUIV_CACHE['forms'] = out
    return out


_EMITTED_CACHE = {}


def _emitted_bytes():
    """{(stem, id): hex} the bytes each retained record emitted, from the results files."""
    if _EMITTED_CACHE:
        return _EMITTED_CACHE
    for results in sorted(ISA.glob('g17-execution-*-results.json')):
        stem = results.name[:-len('-results.json')]
        try:
            data = json.loads(results.read_text())
        except Exception:
            continue
        for row in (data if isinstance(data, list) else data.get('results') or []):
            if not (isinstance(row, dict) and row.get('id')):
                continue
            hexes = (row.get('decoded') or {}).get('encoded') or []
            if hexes and isinstance(hexes[0], str) and hexes[0]:
                _EMITTED_CACHE[(stem, row['id'])] = hexes[0]
    return _EMITTED_CACHE


_ULP_CACHE = {}


def transcendental_semantics():
    """{(opcode, length): reading} for forms the ULP instrument determines AND eliminates.

    THE NINTH D3 COLUMN'S SOURCE. The instrument measures how far the hardware is from the
    correctly-rounded value of the operation its own Apple name states, with overflow cases counted
    apart - at the boundary the largest finite value and infinity are ADJACENT encodings, so a
    saturating opcode would otherwise score one ULP against an IEEE reference and read as a
    rounding difference.

    ACCURACY IS NOT ELIMINATION, and that gap is why this instrument fed no denominator for so
    long. A distance to ONE reference says the hardware is close to that function; it says nothing
    about whether another named operation is equally close. The census now measures every rival in
    its own transcendental table against the same records and marks a form promotable only when
    the named operation is the SOLE reference within a ULP everywhere. Seven forms qualified when
    this was written; 22 do on 2026-09-23, and no rival comes within a ULP of any of them. Their
    "N of M exact" is NOT the rounding evidence: most interior cases are exact under any rounding,
    so each form carries `interior_nontrivial_exact` of `interior_nontrivial_cases` beside it -
    2573/log2 is 15 of 16 raw and 1 of 2 there, 3661/recip the same, 3658/recip 16 of 16 and 0 of 0.

    What it can still get wrong, which is why it is its own column rather than folded into the
    exact-match census: agreement to one ULP is not agreement, and a form that computes a slightly
    different approximation of the same function would pass. The exact-match columns cannot make
    that mistake and this one can.
    """
    if _ULP_CACHE:
        return _ULP_CACHE['forms']
    out = {}
    try:
        doc = json.loads((ISA / 'g17-execution-fits.json').read_text())
    except Exception:
        _ULP_CACHE['forms'] = out
        return out
    for key, entry in (doc.get('transcendental_accuracy') or {}).items():
        if not entry.get('promotable'):
            continue
        try:
            opcode = int(str(key).split('/')[0])
        except Exception:
            continue
        # THE INSTRUMENT KEYS BY OPCODE AND NAME, NOT BY FORM, so the width is decoded out of the
        # records' own emitted bytes with the instruction's opcode required to match - the same
        # discipline that moved 89 sweep receipts off the width their ids claimed. A form whose
        # records disagree on the length is SKIPPED, because attributing an opcode-level fact to a
        # guessed width is the error this map has caught four times.
        widths = set()
        for pair in (entry.get('records') or []):
            text = _emitted_bytes().get(tuple(pair))
            if not text:
                continue
            try:
                instructions = list(decode_instruction(bytes.fromhex(text)))
            except Exception:
                continue
            hit = [i for i in instructions
                   if getattr(getattr(i, 'opcode', None), 'id', None) == opcode]
            if len(instructions) == 1 and hit:
                widths.add(len(getattr(hit[0], 'raw', b'') or b'') or len(text) // 2)
        if len(widths) != 1:
            continue
        width = widths.pop()
        out[(opcode, int(width))] = dict(
            name=entry.get('name'), interior_cases=entry.get('interior_cases'),
            interior_exact=entry.get('interior_exact'),
            # THE DISCRIMINATING DENOMINATOR, carried beside the raw one. A case within 1/64 ulp
            # of a representable value is exact under ANY rounding, so "15 of 16 exact" for
            # 2573/log2 is 1 of 2 on the cases that can tell rounding apart, and 3658/recip's
            # 16 of 16 is 0 of 0. Dropping these here published the raw figure alone.
            interior_nontrivial_cases=entry.get('interior_nontrivial_cases'),
            interior_nontrivial_exact=entry.get('interior_nontrivial_exact'),
            interior_max_ulp=entry.get('interior_max_ulp'),
            overflow_cases=entry.get('overflow_cases'),
            rivals_within_one_ulp=entry.get('rivals_within_one_ulp'),
            verdict=entry.get('verdict'))
    _ULP_CACHE['forms'] = out
    return out


_LANE_CACHE = {}


_MEMORY_CACHE = {}


def memory_probe_semantics():
    """{form: reading} from the EXECUTED memory probe, or {}.

    AN ELEVENTH COLUMN, AND ITS OWN INSTRUMENT. `g17safe` refuses every load, store and atomic
    opcode, so the authored oracle cannot reach these forms at all - but that gate is about bytes
    this project wrote pointed at an address nobody checked, not about whether memory instructions
    may run. `tools/g17memprobe.py` compiles ordinary Metal, lets Apple choose the address, runs
    it, and reads the buffer back, naming the form by decoding the program that executed.

    What it can get WRONG, which is why it is not folded into the lane-probe column: its
    observable is MEMORY rather than a register, so it needs a sentinel-filled output and an
    independent check that the slots outside the expected range did not move - a store that also
    wrote elsewhere is invisible to a probe that only reads where it expects a value. And the
    compiler chooses the address, so no displacement Apple would not emit is reachable.

    A form is counted only when a probe of it left exactly one surviving addressing law, the
    sentinel guard held, and the positive control was exact.
    """
    if _MEMORY_CACHE:
        return _MEMORY_CACHE['forms']
    path = ISA / 'g17-memory-semantics.json'
    forms = {}
    if not path.exists():
        _MEMORY_CACHE['forms'] = forms
        return forms
    try:
        doc = json.loads(path.read_text())
    except ValueError:
        _MEMORY_CACHE['forms'] = forms
        return forms
    if not (doc.get('control') or {}).get('exact'):
        _MEMORY_CACHE['forms'] = forms
        return forms
    readings = collections.defaultdict(list)
    for record in (doc.get('records') or {}).values():
        if record.get('status') != 'ok' or not record.get('untouched_hold_the_sentinel'):
            continue
        survivors = record.get('survivors')
        if not survivors or len(survivors) != 1:
            continue
        # ATTRIBUTE THE LAW TO THE FORM WHOSE ROLE IT DESCRIBES. Every one of these programs
        # contains BOTH a load and a store - `C[tid+4] = A[tid]` needs both - so crediting a
        # probe's single survivor to every memory form in the program gives the load a law about
        # the store. It did: `12682/14` collected one reading as a load and another as a store,
        # the two disagreed by construction, and the form was dropped for inconsistency it never
        # had. A measured law belongs to the instruction it was measured on.
        kind = record.get('kind')
        for shape in record.get('forms') or []:
            if kind and (shape['apple_name'] or '').split('.')[0] != kind:
                continue
            # THE READING IS CANONICALISED TO THE LAW IT NAMES, not to the sentence the probe
            # happened to write. `17238/10` was determined by a scalar store probe saying
            # "element displacement: writes elements 4..4+N" and by a vector one saying "element
            # displacement, live bytes only: vectors 4..4+N" - the same law, two parameterisations
            # - and the consistency check dropped the form for a disagreement that was entirely
            # in the prose. The same defect cost three cross-lane forms earlier in this work.
            law = survivors[0].split(':')[0].split(',')[0].strip()
            readings[shape['form']].append(
                dict(reading=law, detail=survivors[0], probe=record['id'],
                     apple_name=shape['apple_name']))
    for form, entries in readings.items():
        distinct = {e['reading'] for e in entries}
        if len(distinct) == 1:
            forms[form] = dict(reading=entries[0]['reading'],
                               apple_name=entries[0]['apple_name'],
                               detail=sorted({e['detail'] for e in entries}),
                               probes=sorted(e['probe'] for e in entries))
    _MEMORY_CACHE['forms'] = forms
    return forms


_CONTROL_CACHE = {}


def control_probe_semantics():
    """{form: reading} from the EXECUTED comparison probe, or {}.

    A TWELFTH COLUMN, on the same footing as the memory one and for the same reason: the
    authored oracle cannot reach a fused compare-select whose predicate it does not own, and the
    vendor compiler emits one for every relation in the language. `tools/g17cmpprobe.py`
    compiles `a < b ? X : Y` at each width, decodes the program to name the form that executed,
    and dispatches operands chosen so that no rival relation - and, for integers, no rival
    SIGNEDNESS - agrees with the one under test on all sixteen cases.

    WHAT IS AND IS NOT CREDITED. The relation is an operand FIELD of these forms, and that field
    is NOT decoded: six to eight relations were each measured on the same form, and a bit model
    fitted to their encodings contradicts itself on `uint <`. So what this column says is that
    the form is a fused compare-select and that N of its relation values were determined by
    execution - a fact about the form dispatched, at its own width, with the measured values
    listed. It does not say the form computes one relation, and it credits no other width.

    A form is counted only when at least four relations survived on it, each of them the only
    survivor of its own probe after every rival was excluded in advance.
    """
    if _CONTROL_CACHE:
        return _CONTROL_CACHE['forms']
    path = ISA / 'g17-comparison-semantics.json'
    forms = {}
    _CONTROL_CACHE['forms'] = forms
    if not path.exists():
        return forms
    try:
        doc = json.loads(path.read_text())
    except ValueError:
        return forms
    readings = collections.defaultdict(list)
    for name, record in (doc.get('records') or {}).items():
        if record.get('status') != 'ok':
            continue
        survivors = record.get('survivors') or []
        if len(survivors) != 1:
            continue
        if record.get('unrecognised'):
            continue
        for shape in record.get('forms') or []:
            readings[shape['form']].append(dict(relation=survivors[0], probe=name,
                                                apple_name=shape['apple_name'],
                                                bytes=shape.get('bytes')))
    for form, entries in readings.items():
        if len({e['relation'] for e in entries}) < 4:
            continue
        forms[form] = dict(
            reading=('fused compare-select; the relation is an operand field, and these %d '
                     'values of it were determined by execution: %s'
                     % (len({e['relation'] for e in entries}),
                        ', '.join(sorted({e['relation'] for e in entries})))),
            apple_name=entries[0]['apple_name'],
            relations=sorted({e['relation'] for e in entries}),
            probes=sorted(e['probe'] for e in entries),
            distinct_encodings=len({e['bytes'] for e in entries if e['bytes']}))
    return forms


def locally_witnessed_widths():
    """The counts from `isa/g17-local-witness-audit.json`, quoted rather than restated.

    D2 TAKES ITS WIDTH FROM ONE STORED WITNESS PER OPCODE, and Apple's compiler emits several
    widths for the same opcode. Every form in a program this lane compiled is audited against
    that in `tools/g17localwitness.py`; the counts are carried here so this document cannot
    report a coverage picture while the population that contradicts its width model sits in a
    file nothing reads. Nothing here moves a form into D2 - the split is published so the
    decision stays visible.
    """
    path = ISA / 'g17-local-witness-audit.json'
    if not path.exists():
        return dict(state='not built', source='tools/g17localwitness.py')
    try:
        doc = json.loads(path.read_text())
    except ValueError:
        return dict(state='unreadable', source='tools/g17localwitness.py')
    outside = doc.get('forms_outside_d2_at_the_width_they_were_emitted_at') or []
    return dict(
        source='isa/g17-local-witness-audit.json',
        counts=doc.get('counts'),
        forms_outside_d2_at_their_emitted_width=len(outside),
        forms_absent_from_the_spec_at_that_width=len(
            (doc.get('by_verdict') or {}).get('absent from the spec at this width') or []),
        forms_emitted_with_no_stored_witness_at_all=len(
            doc.get('emitted_with_no_stored_witness_at_all') or []),
        means=doc.get('the_mechanism'),
        not_moved_into_d2=doc.get('what_this_does_NOT_do'))


def lane_probe_semantics():
    """{(opcode, length): reading} from the compiled-Metal per-lane probe, or {}.

    THE EIGHTH D3 COLUMN'S SOURCE, and a different instrument from every other one here. The
    isolated-dispatch oracle stores its result at a slot indexed by the case, so only lane zero's
    store lands - and lane zero is where a shuffle offset and an XOR mask agree and where a prefix
    returns its own input, which is why all 95 prefix and 79 shuffle opcodes read as the identity
    through it. `tools/g17lanescan.py` compiles ordinary Metal with Apple's own compiler, stores
    `C[lane] = result`, and reads every lane.

    What it can get WRONG, which is why it is its own column: the COMPILER chooses the encoding, so
    the form measured is whichever one Apple emitted for that intrinsic - the probe names it by
    decoding the emitted stream rather than assuming it - and a kernel that lowers to more than one
    instruction measures the lowering rather than the opcode. The tool refuses such a probe;
    `simd_broadcast_first` is refused today for exactly that, at three instructions.

    A form is counted only when EVERY probe of it leaves exactly one surviving candidate and they
    all agree. `16873/10 simd.prefix_sum` deliberately fails that test: the inclusive and exclusive
    intrinsics emit the SAME form with different operands and compute different functions, so the
    form's semantics are not determined by the form alone and it is published as two readings with
    the operand difference named instead of being counted.
    """
    if _LANE_CACHE:
        return _LANE_CACHE['forms']
    path = ISA / 'g17-lane-semantics.json'
    forms, readings = {}, collections.defaultdict(list)
    if not path.exists():
        _LANE_CACHE['forms'] = forms
        _LANE_CACHE['readings'] = {}
        return forms
    try:
        doc = json.loads(path.read_text())
    except Exception:
        _LANE_CACHE['forms'] = forms
        _LANE_CACHE['readings'] = {}
        return forms
    for probe in (doc.get('probes') or []):
        if probe.get('status') != 'ok' or len(probe.get('survivors') or []) != 1:
            continue
        try:
            opcode, length = (int(part) for part in str(probe.get('form')).split('/'))
        except Exception:
            continue
        readings[(opcode, length)].append(dict(
            probe=probe['id'], reading=probe['survivors'][0], lanes=probe.get('lanes'),
            apple_name=probe.get('apple_name'), encoded=probe.get('encoded'),
            operands=probe.get('operands'), refuted=probe.get('refuted')))
    for key, rows in readings.items():
        distinct = {row['reading'] for row in rows}
        if len(distinct) == 1:
            forms[key] = dict(reading=rows[0]['reading'], probes=[r['probe'] for r in rows],
                              apple_name=rows[0]['apple_name'], lanes=rows[0]['lanes'],
                              refuted=rows[0]['refuted'])
    _LANE_CACHE['forms'] = forms
    _LANE_CACHE['readings'] = {'%d/%d' % k: v for k, v in readings.items()}
    return forms


_CSEL_CACHE = {}


def csel_predicate_semantics():
    """{(opcode, width): reading} for boolean-comparison forms that meet the BOTH-WAYS bar.

    THE SEVENTH D3 COLUMN'S SOURCE, and the bar is the point. The census determines which
    comparison an opcode encodes by eliminating 33 predicates against a boolean output, which
    clears the competitor bar comfortably - and a boolean output is the weakest observable this
    project has, because a predicate that is TRUE at every probed case is indistinguishable from
    the constant true. So a determination counts only when the comparison was exercised BOTH WAYS:
    true at at least MIN_CASES cases AND false at at least MIN_CASES, in every record that
    determines it. `op11481` is excluded by exactly that - its predicate is true at 15 of 18 cases,
    leaving three that could have refuted it.

    Determining which comparison an instruction computes IS its semantics, not its structural
    presence, which is why this is a column and not a candidate. It stays EXCLUSIVE of the other
    six for the usual reason: its failure mode is a predicate library that does not contain the
    right comparison, and 35 forms of the sibling select census refute all 33 offered, so that
    failure is real rather than hypothetical.
    """
    # CACHED. The first version re-read and re-parsed the census on every call, and this is
    # called once per form record - 7000 of them - which turned a two-second harvest into a
    # multi-minute one. `family_law_semantics` beside it already had the cache; the cost of
    # copying a function without its memo is invisible until the run gets slow.
    if _CSEL_CACHE:
        return _CSEL_CACHE['forms']
    try:
        doc = json.loads((ISA / 'g17-execution-fits.json').read_text())
    except Exception:
        _CSEL_CACHE['forms'] = {}
        return {}
    min_cases = int((doc.get('bar') or {}).get('min_cases') or 4)
    census = doc.get('csel_boolean_comparisons') or {}
    out = {}
    for form, entry in predicate_determined_forms().items():
        try:
            opcode, width = (int(part) for part in form.split('/'))
        except Exception:
            continue
        predicate = entry.get('predicate')
        readings = []
        for key, value in (census.get('forms') or {}).items():
            if key.split('/')[0] != str(opcode):
                continue
            for reading in (value.get('readings') or []):
                if reading.get('determined') and predicate in (reading.get('predicates') or []):
                    readings.append((int(reading.get('true_at') or 0),
                                     int(reading.get('cases') or 0)))
        if not readings:
            continue
        true_at = min(t for t, _ in readings)
        false_at = min(c - t for t, c in readings)
        if true_at < min_cases or false_at < min_cases:
            continue
        out[(opcode, width)] = dict(predicate=predicate,
                                    competitors=entry.get('competitors'),
                                    records=entry.get('records'),
                                    true_at_fewest_cases=true_at,
                                    false_at_fewest_cases=false_at,
                                    bar=dict(min_cases=min_cases))
    _CSEL_CACHE['forms'] = out
    return out


def csel_predicate_provenance():
    """Does the csel predicate census earn a D3 column? Measured, and the answer is NO.

    The question is not what KIND of instrument it is. It eliminates a library of 33 predicates
    against observed outputs and reports which SOURCE each form selects and under what condition -
    `11363/csel.mixed` returns its second source exactly when `a != 0` - so it establishes operand
    and predicate semantics, not mere structural presence. By kind it belongs in a column.

    It does not earn one because its determinations do not survive the bar every other column
    respects. The function census requires MIN_CASES distinct cases, MIN_RUNS runs and
    MIN_COMPETITORS eliminated candidates; the predicate census offers 33 predicates, which clears
    the competitor bar comfortably, and then fails the case bar almost everywhere:

      - 37 of its 38 forms have NO branch determined at all, and 35 of those carry records that
        refute every predicate offered - which is a statement that the predicate library is
        incomplete, not that the forms are explained;
      - 55 of its 139 records exercise ONE BRANCH ONLY, and a record that never takes the other
        branch measures no predicate whatever it returns;
      - of the 14 forms the candidate column publishes, 13 rest on a predicate pinned at fewer
        than MIN_CASES cases - `11372/10` is pinned at ONE of four - and one case deciding a
        predicate is the same thinness that made op3341's first rule wrong.

    So the decision is to leave it outside D3 and to publish the MEASUREMENT that decides it rather
    than the stylistic argument the candidate column carried before, which was only that boolean
    elimination is a different instrument. That argument is true and would survive a sixth column;
    the case bar is what does not. One form meets it, and one form is not a column.

    This class is counted in NOTHING. The bar is read from the census rather than restated, so the
    two cannot drift apart.
    """
    path = ISA / 'g17-execution-fits.json'
    try:
        doc = json.loads(path.read_text())
    except Exception:
        return dict(decision='undecided', note='the elimination census is unreadable')
    census = doc.get('csel_predicate_elimination') or {}
    bar = doc.get('bar') or {}
    min_cases = int(bar.get('min_cases') or 4)
    min_competitors = int(bar.get('min_competing_candidates') or 8)
    meets, below, undetermined = {}, {}, []
    for form, entry in sorted((census.get('forms') or {}).items()):
        branches = {name: value for name, value in (entry.get('per_form') or {}).items()
                    if value.get('state') == 'determined'}
        if not branches:
            undetermined.append(form)
            continue
        passing = {name: dict(predicates=value.get('predicates'),
                              pinned_at_fewest_cases=value.get('true_at_fewest_cases'),
                              pinned_at_most_cases=value.get('true_at_most_cases'),
                              records=value.get('records'))
                   for name, value in branches.items()
                   if (value.get('true_at_fewest_cases') or 0) >= min_cases}
        target = meets if passing else below
        target[form] = dict(competitors=entry.get('competitors'),
                            branches=passing or {name: dict(
                                predicates=value.get('predicates'),
                                pinned_at_fewest_cases=value.get('true_at_fewest_cases'))
                                for name, value in branches.items()})
    # THE POPULATION THE DECISION IS ACTUALLY ABOUT is the candidate column's, not the census
    # section's: they are keyed differently - the census by opcode/Apple-name, the candidate column
    # by opcode/width, which is the FORM a denominator would count - and it is the candidate
    # column's fourteen entries that a sixth column would promote. Evaluating only the census
    # section's `per_form` states reported zero forms below the bar, which is true of that keying
    # and answers the wrong question.
    # ONE BAR, APPLIED ONCE. The first version tested `true_at_fewest_cases` from the candidate
    # entry here while the column itself applied the both-ways rule to the comparison census's
    # determined readings - two different aggregates over two different populations - so
    # `11372/10` and `11462/14` appeared in BOTH the promoted list and the below-bar list of the
    # same artifact. The below-bar set is now the COMPLEMENT of what the column counts, which is
    # the only definition that cannot disagree with it.
    promoted_keys = {'%d/%d' % k for k in csel_predicate_semantics()}
    offered, offered_below = {}, {}
    for form, entry in sorted(predicate_determined_forms().items()):
        row = dict(predicate=entry.get('predicate'), competitors=entry.get('competitors'),
                   records=entry.get('records'),
                   pinned_at_fewest_cases=entry.get('true_at_fewest_cases'),
                   pinned_at_most_cases=entry.get('true_at_most_cases'))
        if form in promoted_keys:
            offered[form] = row
        else:
            offered_below[form] = dict(row, why_not=(
                'its comparison was not exercised BOTH ways at or above min_cases in every '
                'record that determines it, or no determined reading of the comparison census '
                'names this predicate at all - the three forms whose true-value the recogniser '
                'does not list are here for the second reason'))
    promoted = csel_predicate_semantics()
    return dict(
        decision=('PROMOTED to its own exclusive D3 column for the %d forms whose comparison was '
                  'exercised BOTH ways at or above the census\'s min_cases; the rest stay '
                  'candidates' % len(promoted)),
        forms_promoted_to_the_csel_column=sorted('%d/%d' % k for k in promoted),
        bar_read_from_the_census=dict(min_cases=min_cases, min_competing_candidates=min_competitors),
        candidate_forms_meeting_the_bar=offered,
        candidate_forms_below_the_bar=offered_below,
        candidate_forms_offered=len(offered) + len(offered_below),
        forms_meeting_the_bar=meets,
        forms_determined_below_the_bar=below,
        forms_with_no_branch_determined=len(undetermined),
        record_level=dict(census.get('summary') or {}),
        why=('TWO DIFFERENT INSTRUMENTS share the word csel here and the decision turns on '
             'telling them apart. `csel_boolean_comparisons` determines WHICH COMPARISON an '
             'opcode encodes by eliminating 33 predicates against a boolean output; that is the '
             'semantics of a compare instruction, not its structural presence, and 10 of its 14 '
             'determined forms survive the bar the function census uses once the bar is stated '
             'correctly for a boolean - true at min_cases cases AND false at min_cases, so a '
             'predicate true everywhere cannot pass. Those 10 are promoted. '
             '`csel_predicate_elimination` is the other instrument, over conditional SELECTS, and '
             'it is NOT promoted and should not be: 37 of its 38 forms have no branch determined, '
             '35 of those carry records refuting every predicate offered - a statement that the '
             'library is incomplete - and 55 of its records exercise one branch only, which '
             'measures no predicate whatever it returns. An earlier draft of this decision '
             'conflated the two and read the select census\'s states while the candidate column '
             'offers the comparison census\'s forms, which reported zero forms below the bar by '
             'answering a question nobody asked'),
        what_would_change_it=('cases that exercise BOTH branches of each form at least min_cases '
                              'times - the census already reports, per record, how many cases each '
                              'predicate was true at, so the shortfall is named per form rather '
                              'than estimated - and predicates the library does not yet offer, '
                              'since 35 forms refute all 33 of them'),
        counted_in=('nothing. This is a decision record, not a denominator, and a test asserts no '
                    'D3 column moves with it'))


def _never_at_this_width_flag(entry):
    """Did every record of this opcode decode at some OTHER length than this form's?"""
    return bool(entry.get('never_dispatched_at_this_width'))


def _operand_sweep_findings():
    """Per form: which non-register operand the authored sweep showed reaching the result."""
    path = ROOT / 'isa' / 'g17-operand-sweep-findings.json'
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _tail_findings():
    """What the tail batch settled per form, or {} if it has not been run."""
    path = ROOT / 'isa' / 'g17-tail-findings.json'
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


_RESULT_WIDTHS = {}


def _result_widths():
    """{opcode: {length}} from every finished ok isolated result that states its own length.

    THE FIT ROWS ARE NOT THE ONLY DISPATCHES. A whole-program record - a store, a loop, a barrier
    alone in its program - has no `cases`, so g17fitfromexecution makes no row for it, and a form
    dispatched only that way read here as "never dispatched at this width" while its hardware
    evidence said dispatched: 2190/6 on 2026-09-22 (isa/g17-execution-requestedforms-results.json).
    The result's `length` is what the runner copied from a plan whose program g17oracle REFUSES
    unless it holds the form at exactly that length, so it is a width that ran."""
    if "w" not in _RESULT_WIDTHS:
        out = {}
        for path in sorted(ISA.glob('g17-execution-*-results.json')):
            try:
                recs = json.loads(path.read_text())
            except Exception:
                continue
            for r in recs if isinstance(recs, list) else []:
                if (isinstance(r, dict) and r.get('status') == 'ok' and r.get('finished', True)
                        and isinstance(r.get('op'), int) and isinstance(r.get('length'), int)):
                    out.setdefault(r['op'], set()).add(r['length'])
        _RESULT_WIDTHS["w"] = out
    return _RESULT_WIDTHS["w"]


def untouched_form_audit(spec):
    """Every dispatched form with no semantic column, and WHY existing evidence cannot place it.

    The goal this answers: decide, for each untouched form, whether its semantics follow from a
    measured family law, are observationally equivalent at its declared width, have retained
    dispatch evidence nobody harvested, or genuinely need a new hardware experiment - and do not
    dispatch anything until those four are exhausted.

    The reasons are assigned in PRIORITY ORDER and must PARTITION the population, so a form cannot
    be counted under two of them and cannot fall out of all of them. Each carries the concrete next
    step, because "unknown" without a discriminating experiment is a restatement of the count.

    Where the census already answers a question, this reads its answer rather than recomputing it:
    `forms_a_degenerate_probe_disqualified` for probes that could not have failed,
    `forms_where_records_refute_the_fit` for forms whose own records disagree,
    `the_input_that_would_split_it` for candidates that coincide at a declared width, and
    `no_fit_rows_accounted` for which instrument already reads a no-fit record. Duplicating any of
    those here would be a second copy of a predicate this repository already owns.
    """
    path = ISA / 'g17-execution-fits.json'
    try:
        doc = json.loads(path.read_text())
    except Exception:
        return dict(forms={}, note='the elimination census is unreadable')
    rows = collections.defaultdict(list)
    for row in (doc.get('rows') or []):
        if row.get('op') is not None:
            rows[int(row['op'])].append(row)
    disqualified = doc.get('forms_a_degenerate_probe_disqualified') or {}
    refuted = doc.get('forms_where_records_refute_the_fit') or {}
    # BOTH OF THESE SETS WERE SILENTLY EMPTY. `csel_predicate_elimination` has no `determined`
    # key - its members are under `forms` - so the csel branch was matching on
    # `apple.startswith('csel')` alone, which misses every `fcsel.*` form because that starts with
    # an f. And the three `no_fit_rows_accounted` buckets below are plain COUNTS, not member
    # lists, so building a set from them yielded nothing and the branch never fired at all. A
    # reader that answers empty is the quietest kind of wrong, which is the same note
    # `_encoded_by_id` carries in the census for the same mistake.
    csel = doc.get('csel_predicate_elimination') or {}
    csel_forms = {str(k) for k in ((csel.get('forms') or {}) if isinstance(csel, dict) else {})}
    splits = ((doc.get('the_input_that_would_split_it') or {}).get('records') or {})
    coincide = set()
    for key, entry in splits.items():
        for pair in (entry.get('pairs') or []):
            state = pair.get('state') or ''
            if 'SAME FUNCTION' in state or 'cannot show apart' in state:
                coincide.add(key.split('/')[0])
    # The instruments' OWN member lists, which is where the membership actually lives.
    law_read = {str(k).split('/')[0] for k in (doc.get('transcendental_accuracy') or {})}
    law_read |= {str(k).split('/')[0] for k in csel_forms}
    # HOW LONG EACH EMITTING CONSTRUCT LOWERS TO, measured by the lane probe. A compiled-source
    # probe can only read a form that stands ALONE, so this decides whether that instrument can
    # reach a form at all - and for most of the untouched population it cannot: integer division
    # is 26 instructions and the modulus 29, while a multiply is one.
    lowering = {}
    try:
        lowering = (json.loads((ISA / 'g17-lane-semantics.json').read_text())
                    .get('construct_lowering') or {})
    except Exception:
        pass
    constructs = {}
    names, classes = {}, {}
    try:
        for line in (ISA / 'g17-contract.jsonl').read_text().splitlines():
            row = json.loads(line)
            names[row['opcode']] = row.get('name') or ''
            constructs[row['opcode']] = list((row.get('meaning') or {}).get('constructs') or [])
            classes[row['opcode']] = dict(
                reads=list((row.get('reads') or {}).get('operands') or []),
                writes=list((row.get('writes') or {}).get('operands') or []))
    except Exception:
        pass
    out = {}
    # WHAT THE AUDIT DOES NOT COVER, COUNTED. Its scope is DISPATCHED forms carrying no semantic
    # column, which its `means` has always said - but the number it publishes is easily read as
    # "all that is left", and it is not: 315 encoded forms have no semantic column and were never
    # dispatched at all, so they never reach a reason. A reader comparing 191 against the encoded
    # denominator would conclude the remainder is determined. Publishing the excluded count beside
    # the audited one costs nothing and makes the scope impossible to misread; it is NOT summed
    # with anything, for the same reason the three denominators are not.
    outside = collections.Counter()
    for name, forms in spec.items():
        for width, record in forms.items():
            axes = record.get('denominators') or {}
            if not axes.get('d2_encoded'):
                continue
            if any(v for k, v in axes.items() if k.startswith('d3_')):
                continue
            if not (record.get('hardware_evidence') or {}).get('dispatched'):
                outside['encoded, no semantic column, never dispatched'] += 1
    for name, forms in spec.items():
        opcode = int(name[2:])
        for width, record in forms.items():
            length = int(width[3:])
            axes = record.get('denominators') or {}
            hardware = record.get('hardware_evidence') or {}
            if not hardware.get('dispatched'):
                continue
            # EVERY D3 KEY, ASKED OF THE RECORD ITSELF rather than from a list that can drift.
            # A literal list here fell behind the map the moment the csel column was added, and
            # the audit then included ten forms that DO carry a column - caught by its own guard,
            # which is the only reason it did not ship. Reading the keys off the record makes a
            # new column impossible to forget.
            if any(value for key, value in axes.items() if key.startswith('d3_')):
                continue
            key = '%d/%d' % (opcode, length)
            apple = names.get(opcode) or ''
            mine = rows.get(opcode) or []
            entry = dict(apple_name=apple, records=len(mine))
            dq = disqualified.get(key) or {}
            # WHETHER THE OBSERVATION COULD HAVE IDENTIFIED ANYTHING, asked before any reason that
            # blames the instruction. `forms_a_degenerate_probe_disqualified` answers this only for
            # forms that HAVE a fit, so for a no-fit form nothing had consulted the row's own
            # degeneracy flags - and 59 of them turn out to have an output that never varied. The
            # first version of this audit called 29 of those "an operand is held constant", which
            # is true and is the wrong half of the story: a constant OUTPUT identifies nothing
            # whatever the operands did.
            _nofit = [r for r in mine if r.get('verdict') == 'no candidate fits']
            _reads = (classes.get(opcode) or {}).get('reads') or []
            # A COMPLAINT ABOUT A PROBE IS RETIRED BY A RECORD THAT DOES NOT HAVE THAT DEFECT, and
            # this audit had no way to say so. Every reason below is chosen by the FIRST branch
            # that applies, and the branches read the form's whole record set - so a form
            # dispatched afresh with a case set proved to separate every candidate group went on
            # reporting the complaint its OLDEST record earned. Thirteen forms were dispatched
            # across three families and not one changed reason.
            #
            # Two complaints are retired here, each by the observation that refutes it:
            #   * "its only fit comes from a probe that could not have failed" - refuted by any
            #     record of this form that is not degenerate.
            #   * "a declared source was fed the same value at its own width" - refuted by any
            #     single record that varied EVERY declared source at its own width. Pooling
            #     records makes this one especially misleading: a source frozen in an old record
            #     and varied in a new one lands in both lists at once.
            _is_degenerate = lambda r: bool(
                r.get('outputs_constant_across_cases')
                or r.get('outputs_equal_an_input_column')
                or r.get('outputs_equal_an_input_column_at_the_destination_width'))
            _measured = [r for r in mine if not _is_degenerate(r)]

            def _varies_every_source(row):
                plan = _audit_plans().get((row['stem'], row['id'])) or {}
                columns = list(zip(*(plan.get('cases') or [])))
                if len(columns) < len(_reads):
                    return False
                for index, cls in enumerate(_reads):
                    narrow = 0xFFFF if '16' in cls else 0xFFFFFFFF
                    if len({int(v) & narrow for v in columns[index]}) == 1:
                        return False
                return bool(_reads)

            _fully_varied = [r for r in mine if _varies_every_source(r)]
            # A DISPATCH OF THE OPCODE IS NOT A DISPATCH OF THE FORM. Every record here is keyed
            # by OPCODE, and `widths` says what its bytes actually decoded to - so a form whose
            # records all came back at some OTHER length has never been dispatched at all, and
            # telling its reader "the authored oracle already dispatches this opcode: the change
            # is CASES" points at a form the encoder cannot build.
            #
            # NO FORM IN TODAY'S CORPUS TRIPS IT, and the first version of this comment named
            # op3290 as the case on the strength of ONE record: the arithmetic batch asked for
            # length 6 and the field map authored 14. op3290 has ten OTHER records at length 6, so
            # the form has been dispatched at its own width and the claim was a single record
            # mistaken for a population. The check stays because the trap is real and cheap to
            # miss; it is exercised on a synthetic row rather than on the artifact, which by
            # construction contains no example.
            # THE MOST INFORMATIVE RECORD SHOULD PICK THE REASON, NOT THE FIRST BRANCH. A form
            # whose newest record varied every declared source and STILL returned an input column
            # has learned something the degeneracy complaint cannot express - and that complaint
            # was winning, because it is tested first and a record returning an input column is
            # "degenerate" by the same flag.
            _varied_but_column = bool(_fully_varied) and all(
                r.get('outputs_equal_an_input_column')
                or r.get('outputs_equal_an_input_column_at_the_destination_width')
                for r in _fully_varied)
            try:
                entry['field_map_authors_width'] = int(g17auth.length(opcode))
            except Exception:
                entry['field_map_authors_width'] = None
            _lengths = {w for r in mine for w in (r.get('widths') or [])} | _result_widths().get(opcode, set())
            _this_width = int(str(width).replace('len', ''))
            _never_at_this_width = bool(_lengths) and _this_width not in _lengths
            if _never_at_this_width:
                entry['dispatched_widths'] = sorted(_lengths)
                entry['never_dispatched_at_this_width'] = True
            _writes = (classes.get(opcode) or {}).get('writes') or []
            _flag_dst = any('FLAG' in x for x in _writes)
            _tuple_operand = [x for x in (_reads + _writes) if 'tup' in x]
            _all_constant = bool(_nofit) and all(
                r.get('outputs_constant_across_cases') for r in _nofit)
            _all_input_column = bool(_nofit) and not _all_constant and all(
                r.get('outputs_equal_an_input_column')
                or r.get('outputs_equal_an_input_column_at_the_destination_width')
                for r in _nofit)
            if (_operand_sweep_findings().get(key) or {}).get('settled'):
                # THE AUTHORED SWEEP RETIRES "THE OUTPUT DID NOT MOVE". It did move - under an
                # operand the cases cannot reach and `imms` can write. Holding every other free
                # operand at zero and moving one, the output column changes, so the change belongs
                # to that operand alone. The lifetime carriers are excluded because the liveness
                # pass rewrites them after encoding, which the oracle refuses outright.
                _swept = _operand_sweep_findings()[key]
                _moved = [m['operand'] for m in _swept['operands_that_move_the_output']]
                entry.update(
                    reason='an authored operand sweep shows operand%s %s reaching the result, so '
                           'the output is not constant - it is selected by a field no case set '
                           'can write'
                           % ('s' if len(_moved) != 1 else '',
                              ', '.join(str(o) for o in _moved)),
                    operands_that_move_the_output=_moved,
                    operands_that_did_not=_swept['operands_that_did_not'],
                    sweep_verdict=_swept['verdict'],
                    next_step='the carrier is located; what remains is its MEANING. Enumerate the '
                              'field\'s values against inputs that make each hypothesis about it '
                              'predict a different arm, and add a candidate of this form\'s shape '
                              'to the published library so the census can eliminate against it. '
                              'No further dispatch is needed to find the operand')
            elif 'BASELINE did not build' in (
                    (_operand_sweep_findings().get(key) or {}).get('verdict') or ''):
                # THE SWEEP COULD NOT STAND UP A BASELINE, and that is a fact about the ENCODING
                # rather than about the instruction. Two passes were tried: one holding every
                # writable free operand at zero, and one omitting `imms` entirely so the field
                # map's own defaults stand. Neither built, so at least one field of this form has
                # a value the encoder will not produce from either starting point, and the sweeps
                # measured against it would have been measured against nothing.
                entry.update(
                    reason='an authored operand sweep could not build a baseline for it from '
                           'either an all-zero or a default starting point, so at least one field '
                           'of this form carries a value the encoder will not produce',
                    sweep_verdict=_operand_sweep_findings()[key]['verdict'],
                    next_step='decode what the field map DOES author for this opcode and read the '
                              'fields it refuses to move; the encoding has to be corrected or '
                              'written by hand before any operand question can be asked of it. '
                              'This is an encoder defect to route, not a semantic unknown')
            elif (_operand_sweep_findings().get(key) or {}).get('operands_that_did_not'):
                # A SWEEP THAT LOCATED NOTHING IS STILL A RESULT, and leaving these under "the
                # output did not move although every source did" would credit the cases with a
                # limit the sweep has already pushed past. Every writable free operand was moved
                # one at a time and the output held: what selects the result is therefore neither
                # a declared source nor any field a record may write, which leaves the lifetime
                # carriers the liveness pass owns - and those need liveness to drive them, not a
                # bigger case set.
                _swept = _operand_sweep_findings()[key]
                entry.update(
                    reason='an authored operand sweep moved every writable free operand one at a '
                           'time and the output held, so the selector is neither a declared '
                           'source nor a field a record may write',
                    operands_that_did_not=_swept['operands_that_did_not'],
                    sweep_verdict=_swept['verdict'],
                    next_step='the remaining candidates are the lifetime-carrier operands the '
                              'liveness pass owns, which a record may not state - the oracle '
                              'refuses it, because the value would be rewritten after encoding '
                              'and the record would measure the default program under another '
                              'name. Driving one needs a program whose LIVENESS differs, which is '
                              'a compiler-side construction rather than a probe')
            elif (_tail_findings().get(key) or {}).get('settled'):
                # A DEDICATED PROBE READ THIS FORM, so "no other instrument reads it" is false
                # here whatever the exact-match census says. `11395/14` was dispatched with its
                # predicate operand written directly - the capability `imms` has had since
                # op9754's condition sweep - and the returned arm flips with operand 2 on half its
                # cases. The library holding no candidate of that shape is a fact about the
                # library.
                _found = _tail_findings()[key]
                entry.update(
                    reason='a dedicated probe reads it and names %s; the published library holds '
                           'no candidate of this form\'s shape, which is what the exact-match '
                           'census is reporting'
                           % ('operand %d as the carrier of its selection' % _found['carrier']
                              if _found.get('carrier') is not None else 'what it settled'),
                    tail_finding=_found.get('verdict'),
                    next_step='add a candidate of this shape to the published library and let the '
                              'census eliminate against it, or record the carrier as its own '
                              'evidence class. Neither is a dispatch: the observation is made')
            elif not mine:
                entry.update(reason='no census record reads this opcode at all',
                             next_step='the receipt exists but no plan/result pair was harvested '
                                       'for it; find the record or dispatch one')
            elif key in refuted:
                entry.update(
                    reason='a record of this form refutes the fit another record admits',
                    function=(refuted[key] or {}).get('function'),
                    next_step='read the refuting record\'s cases at THIS form\'s destination '
                              'width before dispatching: the same comparison already turned out '
                              'to be a width artifact for two other forms, and where it is real '
                              'the disagreeing cases ARE the discriminating experiment')
            elif dq and not (dq.get('causes') or []):
                entry.update(
                    reason='its degenerate probe was disqualified and every cause has since been '
                           'answered, yet no instrument claims it',
                    function=dq.get('function'),
                    next_step='the later record that answered the degeneracy is retained; either '
                              'a law must read it or it needs its own fit - this is evidence '
                              'already on disk that nothing harvests')
            elif dq and not _measured and not _varied_but_column:
                # THE NEXT STEP DEPENDS ON WHICH OBSERVABLE IS MISSING, and for a cross-lane form
                # "lane variation" is NOT enough: this oracle stores its result at a slot indexed
                # by the case rather than the thread, so only lane zero's store lands, and lane
                # zero is where a shuffle's offset and an XOR mask agree and where a prefix
                # reduction returns its own input. Those forms need a THREAD-INDEXED destination,
                # which a compiled Metal source probe has and this lane's authored-encoding path
                # does not. Saying "lane variation" for them sent a request to another lane that
                # did not need to be sent.
                _lane_bound = any(word in apple
                                  for word in ('shuffle', 'prefix', 'rotate', 'broadcast'))
                entry.update(reason='its only fit comes from a probe that could not have failed',
                             function=dq.get('function'),
                             degeneracy=dq.get('causes'),
                             next_step=('a THREAD-INDEXED destination, which this oracle does not '
                                        'build: it stores at a slot indexed by the case, so only '
                                        'lane zero lands - and lane zero is where a shuffle offset '
                                        'and an XOR mask agree and where a prefix returns its own '
                                        'input. Plain Metal source with `C[lane] = r` and a lane '
                                        'tag reads all 32 soundly; that is the vendor\'s compiler '
                                        'as an oracle, not a second authoring path'
                                        if _lane_bound else
                                        're-measure with the degeneracy removed - a second base, '
                                        'lane variation, or a case set the absorbing function '
                                        'cannot pass'))
            elif str(opcode) in coincide:
                entry.update(
                    reason='the surviving candidates coincide at this form\'s declared width',
                    next_step='no input can separate them here. The separator is a SIBLING FORM '
                              'of the same Apple name with a wider operand class; check which of '
                              'those are already dispatched before authoring anything')
            elif str(opcode) in {k.split('/')[0] for k in csel_forms}:
                # THIS REASON CLAIMED A DETERMINATION THAT DOES NOT EXIST. It read "the csel
                # predicate census determines it", and of the 34 forms it held, ZERO were
                # determined: 12 are in the select census with NO branch determined - its own
                # summary says their records refute every predicate offered - and 22 were not in
                # that census at all, matched only because their Apple name starts with "csel".
                # The comparison census's determinations are a different instrument and already
                # have their own column.
                entry.update(
                    reason='the conditional-select census reads it and its records REFUTE every '
                           'predicate offered, so nothing is determined',
                    next_step='predicates the library does not offer. 35 of the select census\'s '
                              '38 forms refute all 33, which is a statement about the library '
                              'rather than about the forms, and 55 of its records exercise ONE '
                              'BRANCH only - a record that never takes the other branch measures '
                              'no predicate whatever it returns')
            elif (apple.startswith(('csel', 'fcsel')) or 'select' in apple) and not mine:
                # "NO CENSUS READS THIS FORM" IS A CLAIM ABOUT THE RECORDS, so it now reads them.
                # It did not: the branch fired on the NAME alone, so `11395/14 select.cc` carried
                # "no census reads THIS form" beside its own `records: 4` - and after the compare
                # batch dispatched it with every source varied, the entry still said nothing reads
                # it. A reason that contradicts a field two lines above it in the same entry is
                # the cheapest kind of wrong to catch and the easiest to keep publishing.
                #
                # SPLIT BY WHETHER ANY INSTRUMENT CAN REACH IT, now that the lowering length of
                # each emitting construct is measured. `11375/14 select` is the form a source-level
                # select emits and its operand 2 carries the relation sense, so a form reachable by
                # a single-instruction construct has a concrete experiment; one whose only
                # constructs lower to twenty-odd instructions does not, and saying which is the
                # difference between a next step and a restatement.
                _emitting = constructs.get(opcode) or []
                _short = [c for c in _emitting
                          if (lowering.get(c.split('.')[0]) or {}).get('instructions') == 1]
                if _short:
                    entry.update(
                        reason='named a conditional select, no census reads THIS form, and a '
                               'single-instruction construct emits it',
                        next_step='probe %s with the compiled-source instrument and read which '
                                  'relation its operands carry - `11375/14 select` was identified '
                                  'that way, its operand 2 being 1 for `>` and 0 for `<`'
                                  % _short[0])
                else:
                    entry.update(
                        reason='named a conditional select, no census reads THIS form, and no '
                               'single-instruction construct is known to emit it',
                        next_step='the compiled-source instrument cannot isolate it; the authored '
                                  'oracle can, by mutating the relation carrier `11375/14` '
                                  'identified - operand 2 - on this form\'s own encoding')
            # ORDER MATTERS, AND THIS IS WHERE IT WENT WRONG ONCE. This check says ANOTHER
            # INSTRUMENT ALREADY READS THIS FORM, which is a near-resolution, so it must be
            # asked BEFORE any check reporting the observation as uninformative. With the
            # constant-output check first, six transcendental forms were filed under "the output
            # never varied" - true, and much less informative, because the constant IS the ULP
            # instrument's answer: log2 of a pinned zero is negative infinity, exp2 of zero is
            # one, recip of zero is infinity.
            elif str(opcode) in law_read:
                entry.update(
                    reason='a no-fit record of it is already read by another instrument',
                    next_step='its instrument\'s determinations are held out of D3; promoting '
                              'them is a classification decision, not a measurement')
            elif _flag_dst and _nofit:
                entry.update(
                    reason='the destination is a FLAG register this harness never reads back',
                    next_step='the value is not in the output buffer at all, so no choice of '
                              'inputs can reach it. It needs a second instruction that '
                              'materialises the flag into a GPR - a conditional select reading '
                              'it - which makes the probe two instructions and is why the '
                              'isolated-dispatch discipline has not covered these')
            elif _tuple_operand and _nofit:
                entry.update(
                    reason='an operand or the destination is a register TUPLE, so this harness '
                           'reads one word of a multi-word value',
                    tuple_operands=sorted(set(_tuple_operand)),
                    next_step='read the SECOND word. The oracle already accepts `read_slots`, so '
                              'this is a plan change rather than a builder change - and until it '
                              'is done, a fit against the low word alone would be a claim about '
                              'half the result')
            elif _all_constant:
                # WHY the output never varied, asked of the INPUTS at the width each source
                # actually reads. A form declaring a GPR16 source reads the LOW 16 BITS, so a case
                # set of 32-bit values with zero low halves feeds it the same operand every time
                # and no output could have moved - which is a different problem from an
                # instruction that genuinely returns a constant, and needs a different fix.
                _constant_sources, _varying_sources = [], []
                for row in _nofit:
                    plan = _audit_plans().get((row['stem'], row['id'])) or {}
                    columns = list(zip(*(plan.get('cases') or [])))
                    for index, cls in enumerate(_reads):
                        if index >= len(columns):
                            continue
                        narrow = 0xFFFF if '16' in cls else 0xFFFFFFFF
                        distinct = len({int(v) & narrow for v in columns[index]})
                        (_constant_sources if distinct == 1 else _varying_sources).append(index)
                if _fully_varied:
                    entry.update(
                        reason='a record of it varied EVERY declared source at its own width and '
                               'the output still did not, so the constancy belongs to the '
                               'instruction rather than to the probe',
                        fully_varied_records=[r['id'] for r in _fully_varied],
                        next_step='name the constant and ask whether it is the right ANSWER. A '
                                  'form that ignores its declared sources is either reading an '
                                  'operand this probe never wrote - a mode or predicate the '
                                  'field map does not place - or genuinely returns a constant. '
                                  'Asking for more inputs cannot distinguish those: every source '
                                  'this form declares has already been moved')
                elif _constant_sources and not _varying_sources:
                    entry.update(
                        reason='every declared source was fed the SAME value at its own declared '
                               'width, so no output could have varied',
                        next_step='vary each source at ITS OWN width. A GPR16 source reads the low '
                                  '16 bits, so 32-bit case values with zero low halves are one '
                                  'operand repeated - this says nothing about the instruction')
                elif _constant_sources:
                    entry.update(
                        reason='at least one declared source was fed the same value at its own '
                               'declared width, so the output could not depend on it',
                        constant_sources=sorted(set(_constant_sources)),
                        next_step='vary the named source at its own width before reading anything '
                                  'into the constant')
                else:
                    entry.update(
                        reason='the declared sources DID vary at their own widths and the output '
                               'did not, so the constancy belongs to the instruction rather than '
                               'to the probe',
                        next_step='name the constant and ask whether it is the right ANSWER. '
                                  '3817/10 `ceil` returns 0x3F80, which is 1.0 as a bfloat16, and '
                                  'every input it was given lies in (0,1) - intrinsic semantics '
                                  'with an input set that cannot show them. An exclusive prefix '
                                  'sum returns the additive identity at lane zero for the same '
                                  'kind of reason. Where the constant is NOT the right answer, a '
                                  'mode operand the probe never wrote is the next suspect')
            elif _fully_varied and all(
                    r.get('outputs_equal_an_input_column')
                    or r.get('outputs_equal_an_input_column_at_the_destination_width')
                    for r in _fully_varied):
                # THE PROBE IS NOT THE EXPLANATION ONCE EVERY SOURCE HAS MOVED. "Returns an input
                # column" reads as a degenerate probe, and for a record whose sources were frozen
                # it is one. For a record that varied EVERY declared source at its own width and
                # still returned one of them, the move is the instruction's - or it is reading an
                # operand the field map does not place. Asking for more cases cannot tell those
                # apart, because there are no more sources to move.
                entry.update(
                    reason='a record of it varied EVERY declared source at its own width and the '
                           'output still equals an input column, so the move belongs to the '
                           'instruction rather than to the probe',
                    fully_varied_records=[r['id'] for r in _fully_varied],
                    next_step='ask which operand it returns and under what condition. A '
                              'conditional select whose predicate operand the field map does not '
                              'place will return one arm for every input this harness can write, '
                              'and no case set reaches the other arm: that operand has to be '
                              'AUTHORED, not varied')
            elif _all_input_column:
                entry.update(
                    reason='every no-fit record returns an INPUT COLUMN, so the probe cannot '
                           'distinguish the instruction from a move',
                    next_step='a second base, or lane variation for a cross-lane form, or any '
                              'case set an identity cannot pass. Ten cross-lane determinations '
                              'were once retracted for exactly this')
            elif any(r.get('verdict') in ('ambiguous', 'interpretation-ambiguous') for r in mine):
                # "THE INPUTS CANNOT SEPARATE THEM" WAS WRONG FOR 44 OF THESE 49. The census's
                # split analysis excludes a record that is DEGENERATE - a constant output, or an
                # output equal to an input column - because such a record measured nothing to
                # split, and 44 of the forms here have only degenerate ambiguous records. Their
                # ambiguity is a fact about the probe, not about two functions being close. The
                # five that DO carry a separating input are answerable and are named apart.
                _amb = [r for r in mine
                        if r.get('verdict') in ('ambiguous', 'interpretation-ambiguous')]
                _splittable = str(opcode) in {k.split('/')[0] for k in splits}
                # A NON-DEGENERATE RECORD RETIRES THE DEGENERACY STORY, WHATEVER ITS VERDICT.
                # `_degenerate` was computed over the AMBIGUOUS records alone, so a form that had
                # since been dispatched with a case set proved to separate every candidate group -
                # and came back fitting NONE of them - still read "every ambiguous record of it is
                # DEGENERATE, so the survivors are an artefact of the probe". The survivors it
                # names no longer exist, and the work that retired them was invisible to its own
                # audit: thirteen forms across three families were dispatched and not one changed
                # reason. The claim "the probe measured nothing" is refuted by any record of this
                # form that measured something, so the branch reads all of `mine`.
                _degenerate = bool(_amb) and not _measured and all(
                    _is_degenerate(r) for r in _amb)
                if _measured and not _splittable:
                    _fits = [r for r in _measured if r.get('verdict') == 'unique']
                    entry.update(
                        reason=('a NON-DEGENERATE record of it exists and no candidate of the '
                                'published library fits it, so the probe is no longer the '
                                'obstacle'
                                if not _fits else
                                'a non-degenerate record of it fits a candidate uniquely but the '
                                'form has not cleared the promotion bar'),
                        next_step=('name the nearest candidate and the cases it misses. A fit '
                                   'that misses one or two cases is an instrument defect and the '
                                   'exception is the finding; a fit that misses most of them '
                                   'means the library holds no function of this shape, which is '
                                   'a gap to fill rather than a dispatch to repeat. Repeating the '
                                   'dispatch cannot help: the case set was proved to separate '
                                   'every candidate group at this destination width before it '
                                   'ran.'),
                        non_degenerate_records=[r['id'] for r in _measured])
                elif _splittable:
                    entry.update(
                        reason='an ambiguous record NAMES an input that separates the survivors, '
                               'so the ambiguity is answerable and simply was not asked',
                        next_step='add the separating input the census already computed for this '
                                  'record to the next batch. No new instrument and no new '
                                  'reasoning is needed - the case is written down')
                elif _degenerate:
                    entry.update(
                        reason='every ambiguous record of it is DEGENERATE, so nothing was '
                               'measured to split and the survivors are an artefact of the probe',
                        next_step='re-measure without the degeneracy first - a second base, lane '
                                  'variation, or a case set an absorbing function cannot pass - '
                                  'and only then ask which candidates survive. Asking for a '
                                  'separating input here is asking the wrong question: the record '
                                  'did not measure the instruction at all')
                else:
                    entry.update(
                        reason='several candidates survive and no separating input has been '
                               'computed for this form',
                        next_step='build the case where the survivors disagree; the census names '
                                  'the separating input per record where one exists')
            else:
                # WHICH OPERANDS THE RECORDS ACTUALLY VARIED. A binary function cannot be
                # identified from cases that hold one operand constant, however many there are,
                # and 37 of the no-fit forms are in exactly that state - so "no candidate fits"
                # was describing the PROBE for them and not the instruction. The frozen operand is
                # named, because the experiment is to vary that one and nothing else.
                frozen, varied, arities = None, False, set()
                for row in mine:
                    if row.get('verdict') != 'no candidate fits':
                        continue
                    arities.add(row.get('arity'))
                    cases = (_audit_plans().get((row['stem'], row['id'])) or {}).get('cases') or []
                    if not cases or (row.get('arity') or 0) < 2:
                        continue
                    columns = list(zip(*cases))
                    still = [i for i, column in enumerate(columns)
                             if len({int(x) for x in column}) == 1]
                    if still:
                        frozen = still if frozen is None else frozen
                    else:
                        varied = True
                if frozen is not None and not varied:
                    entry.update(
                        reason='every no-fit record holds an operand CONSTANT, so no function of '
                               'more than one argument is identifiable from them',
                        frozen_operands=frozen,
                        next_step='vary operand %s and nothing else. This is a statement about '
                                  'the probe, not about the instruction - the candidate set was '
                                  'eliminated against cases that could not have separated its '
                                  'members' % frozen)
                elif arities and arities <= {0, 1}:
                    entry.update(
                        reason='no candidate of the published library fits its one-argument '
                               'records, and no other instrument reads it',
                        next_step='the library lacks the function. Reading the source as its LOW '
                                  '16 BITS is what explained op9990 `msb`, and it was tested '
                                  'across this whole population: it explains nothing else here, '
                                  'so the gap is a function and not an operand width')
                else:
                    entry.update(reason='no candidate of the published library fits, and no other '
                                        'instrument reads it',
                                 next_step='the library lacks the function, or the operand '
                                           'configuration differs from what the probe supplied. '
                                           'Rank by how many cases the Apple name\'s own '
                                           'candidate MISSES: one or two is an instrument defect, '
                                           'not an unknown instruction')
            # THE LANE CAVEAT TRAVELS WITH THE FORM, NOT WITH THE BRANCH. A cross-lane form told
            # to "re-measure with lane variation" or to "name the nearest candidate" is being
            # sent to an instrument that stores at a slot indexed by the CASE, so it reads lane
            # zero alone - and lane zero is where a shuffle's offset and an XOR mask agree and
            # where a prefix reduction returns its own input. Three different branches dropped
            # the warning as they were added, which is what a per-branch fix gets you; applied
            # once here, a new branch cannot lose it.
            if any(word in apple for word in ('shuffle', 'prefix', 'rotate', 'broadcast')):
                if 'THREAD-INDEXED' not in (entry.get('next_step') or ''):
                    entry['next_step'] = (entry.get('next_step') or '') + (
                        ' AND THIS FORM IS CROSS-LANE: whatever else is tried, it needs a '
                        'THREAD-INDEXED destination, which this oracle does not build - it '
                        'stores at a slot indexed by the case, so only lane zero lands, and lane '
                        'zero is where a shuffle offset and an XOR mask agree and where a prefix '
                        'returns its own input. Plain Metal source with `C[lane] = r` and a lane '
                        'tag reads all 32 soundly.')
            out[key] = entry
    # EVERY FORM SAYS WHETHER A COMPILED-SOURCE PROBE CAN REACH IT. The construct names come from
    # the contract and the lengths from a measurement, so "no probe can isolate this" is a claim
    # with a number behind it rather than an impression.
    for key, entry in out.items():
        # THE FORM'S WIDTH COMES FROM THE KEY. This pass has no `width` in scope - the name leaks
        # from the reason loop above and holds whatever form it finished on - so reading it here
        # compares every form against one arbitrary width. The authorability branch did exactly
        # that on its first run and reclassified 164 forms instead of 6.
        opcode, form_width = int(key.split('/')[0]), int(key.split('/')[1])
        emitting = constructs.get(opcode) or []
        lengths = {}
        for name in emitting:
            stem = name.split('.')[0]
            row = lowering.get(stem)
            if row and row.get('instructions') is not None:
                lengths[stem] = row['instructions']
        entry['emitting_constructs'] = sorted(emitting)
        if not emitting:
            entry['compiled_probe'] = 'no construct is recorded for this opcode, so there is no '\
                                      'source expression known to emit it'
        elif not lengths:
            entry['compiled_probe'] = 'its constructs have not been measured for lowering length'
        elif min(lengths.values()) == 1:
            best = sorted(k for k, v in lengths.items() if v == 1)
            entry['compiled_probe'] = ('REACHABLE: %s lowers to a single instruction, so a '
                                       'compiled-source probe can isolate this form' % best[0])
        else:
            entry['compiled_probe'] = ('out of reach for a compiled-source probe: its shortest '
                                       'measured construct, %s, lowers to %d instructions, and a '
                                       'probe that does not isolate one form measures a lowering. '
                                       'The authored oracle is the instrument for these'
                                       % (min(lengths, key=lengths.get), min(lengths.values())))
    # THE EXTERNAL DEPENDENCY, one per form and never a generic unknown. A reason says what this
    # lane observed; a dependency says WHO OR WHAT would have to act next, and the two are
    # different questions. The categories are exhaustive by construction - the final branch is a
    # named category rather than a fallthrough - and each carries the cheapest experiment that
    # would settle the form.
    TENSOR_PREFIX = ('mx:', 'rt:sgmatrix', 'sp:atomic', 'tx:')
    for key, entry in out.items():
        opcode = int(key.split('/')[0])
        apple = entry.get('apple_name') or ''
        emitting = entry.get('emitting_constructs') or []
        reason = entry['reason']
        short = [c for c in emitting
                 if (lowering.get(c.split('.')[0]) or {}).get('instructions') == 1]
        axes = (classes.get(opcode) or {})
        writes = axes.get('writes') or []
        reads = axes.get('reads') or []
        if ('mma' in apple or 'simdgroup' in apple
                or any(c.startswith(TENSOR_PREFIX) for c in emitting)):
            entry['external_dependency'] = 'belongs to the TensorOps lane'
            entry['cheapest_experiment'] = (
                'the matrix and tensor constructs are that lane\'s instruments and its objects '
                'already contain these forms; ask it rather than rebuilding the machinery here')
        elif 'coincide at this form' in reason or 'SAME FUNCTION' in reason:
            entry['external_dependency'] = 'observationally equivalent at its declared width'
            entry['cheapest_experiment'] = (
                'none, and none should be designed: the surviving candidates are one function at '
                'this width and the residue is which library name to write')
        elif any('tup' in x for x in (reads + writes)) or any('FLAG' in x for x in writes):
            entry['external_dependency'] = 'requires hand-authored encoding'
            entry['cheapest_experiment'] = (
                'the destination is a register TUPLE or a flag register, so the value is not in '
                'the output buffer this harness reads. An authored record with `read_slots` '
                'naming the second word, or a second instruction materialising the flag into a '
                'GPR, is the smallest change')
        elif 'could not build a baseline' in (entry.get('reason') or ''):
            entry['external_dependency'] = 'requires hand-authored encoding'
            entry['cheapest_experiment'] = (
                'an ENCODER question, not a semantic one, and it belongs with the six forms whose '
                'width the field map cannot author: decode what it does emit for this opcode, '
                'find the field it will not move, and either correct the encoding or supply the '
                'bytes. No case set and no operand sweep can proceed until a baseline stands up')
        elif 'an authored operand sweep moved every writable' in (entry.get('reason') or ''):
            entry['external_dependency'] = 'requires hand-authored encoding'
            entry['cheapest_experiment'] = (
                'the sweep has exhausted what a record may write. The lifetime-carrier operands '
                'are the remaining candidates and the oracle refuses to state them, so this needs '
                'a program built with a different LIVENESS - two uses of the source instead of '
                'one - which is a compiler-side construction. Authoring the bytes directly, with '
                'explicit `bytes` and operand specs through the caller path, is the other way')
        elif ('a dedicated probe reads it' in (entry.get('reason') or '')
              or 'an authored operand sweep shows' in (entry.get('reason') or '')):
            # A SIXTH CATEGORY, AND IT IS THE OPPOSITE OF A GENERIC BUCKET. This form has been
            # dispatched, its predicate operand written directly, and its carrier named. Nothing
            # external blocks it: what is missing is a candidate of its shape in the published
            # library, which is this lane's own work and a different kind of task from authoring
            # bytes or finding a source construct. Filing it under "a broader targeted dispatch
            # set" would send it back to the instrument that already answered it.
            entry['external_dependency'] = 'requires a library candidate of its shape'
            entry['cheapest_experiment'] = (
                'no dispatch. Add a candidate matching this form\'s declared shape to the '
                'published library and re-run the census, which already holds the records that '
                'would eliminate against it; or record the carrier the probe named as its own '
                'evidence class, the way the lane probe\'s carrier analysis is recorded')
        elif entry.get('field_map_authors_width') not in (None, int(key.split('/')[1])):
            # CAN THE EXISTING ORACLE EVEN BUILD THIS FORM? Asked of the field map directly, and
            # it is the first question a "needs more cases" experiment should have to answer. Six
            # forms of the untouched population have a width the field map does not author, and
            # `3290/6` was one of them while being told its retained records merely needed a
            # better case set: the census names a separating input for a SIX-byte record of it,
            # and two dispatches through the field map authored FOURTEEN bytes instead - sixteen
            # cases and then six, the second built deliberately to match the shape of the existing
            # six-byte record. A form is (opcode, width), so those fourteen-byte runs determined
            # `3290/14` and said nothing about this form.
            entry['external_dependency'] = 'requires hand-authored encoding'
            entry['cheapest_experiment'] = (
                'the field map authors op%d at length %d, not %s, so no batch through the '
                'authored oracle can produce this form however its cases are chosen - the '
                'six-byte encoding has to be supplied as BYTES. Everything else is ready: %s'
                % (opcode, entry['field_map_authors_width'],
                   key.split('/')[1],
                   'the census already names an input that separates its surviving candidates'
                   if 'NAMES an input' in (entry.get('reason') or '') else
                   'its retained records name what is still open'))
        elif _never_at_this_width_flag(entry):
            entry['external_dependency'] = 'requires hand-authored encoding'
            entry['cheapest_experiment'] = (
                'no record of this opcode decoded at this form\'s width: the field map authors '
                'it at %s every time, so the authored oracle cannot produce this encoding however '
                'many cases it is given. The bytes have to be written by hand, or the form has to '
                'be found in a vendor object that contains it'
                % (', '.join(str(w) for w in sorted(entry.get('dispatched_widths') or [])) or
                   'another length'))
        elif 'THREAD-INDEXED destination' in (entry.get('next_step') or '') and not short:
            # THE REASON AND THE EXPERIMENT CONTRADICTED EACH OTHER HERE, and the experiment was
            # the one being read. The reason for a cross-lane form says this oracle CANNOT observe
            # it - it stores at a slot indexed by the case, so only lane zero lands, and lane zero
            # is where a shuffle's offset and an XOR mask agree - while the experiment underneath
            # said "the change is CASES, not instruments: the authored oracle already dispatches
            # this opcode". More cases through an instrument that reads one lane cannot settle a
            # shuffle, and filing these under "a broader targeted dispatch set" sent the work to
            # the instrument its own reason had just ruled out.
            #
            # The only instrument that can see them is a compiled-source probe with a
            # thread-indexed destination, and that needs a construct emitting this form ALONE.
            # Where none is known the dependency is the construct, not the case set.
            entry['external_dependency'] = 'requires an unavailable source construct'
            entry['cheapest_experiment'] = (
                'only a thread-indexed destination can observe this form - the authored oracle '
                'stores at a slot indexed by the case, so it reads lane zero alone, which is '
                'exactly where a shuffle offset and an XOR mask agree. The compiled lane probe '
                'has that destination but needs a Metal construct that emits this opcode ALONE, '
                'and none is known: its measured constructs lower to several instructions, so a '
                'probe built on one would measure a lowering rather than a form. Find the '
                'construct in Apple\'s corpus objects, or author the encoding')
        elif short:
            entry['external_dependency'] = 'requires a broader targeted dispatch set'
            entry['cheapest_experiment'] = (
                'a compiled-source probe CAN isolate it via %s, so it needs cases rather than '
                'machinery: add it to the lane-scan probe list with candidates that its declared '
                'operand classes can separate' % short[0])
        elif not emitting:
            entry['external_dependency'] = 'requires an unavailable source construct'
            entry['cheapest_experiment'] = (
                'no Metal expression is known to emit this opcode, so the vendor compiler cannot '
                'be pointed at it. Either find the construct in Apple\'s corpus objects that '
                'contains it, or author the encoding directly')
        elif reason.startswith(('every ambiguous record', 'its only fit comes from a probe',
                                'every declared source', 'at least one declared source',
                                'every no-fit record holds', 'every no-fit record returns',
                                'an ambiguous record NAMES')):
            entry['external_dependency'] = 'requires a broader targeted dispatch set'
            entry['cheapest_experiment'] = (
                'its retained records are degenerate or too narrow and the authored oracle '
                'already dispatches this opcode: the change is CASES, not instruments - vary each '
                'declared source at its own width, or add the separating input the census names')
        elif reason.startswith('a no-fit record of it is already read'):
            entry['external_dependency'] = 'requires a broader targeted dispatch set'
            entry['cheapest_experiment'] = (
                'another instrument reads it and its determinations are held out of D3 pending a '
                'classification decision; more cases would move it into an exact-match column')
        else:
            entry['external_dependency'] = 'requires hand-authored encoding'
            entry['cheapest_experiment'] = (
                'its shortest emitting construct lowers to %s instructions, so no compiled probe '
                'isolates it and the compiler will not produce it alone. Author the encoding with '
                'the operands this lane chooses' % (
                    min((lowering.get(c.split('.')[0]) or {}).get('instructions', 999)
                        for c in emitting) if emitting else 'an unmeasured number of'))
    tally = collections.Counter(e['reason'] for e in out.values())
    return dict(
        forms=out,
        forms_audited=len(out),
        outside_this_audit=dict(outside),
        outside_this_audit_means=(
            'encoded forms carrying no semantic column that this audit does not reach, because '
            'its scope is forms that have been DISPATCHED. They are counted here and NOT summed '
            'with anything: a reader who takes `forms_audited` for the whole unresolved '
            'population would conclude the remainder is determined, and it is not. What they need '
            'is a first dispatch, which is a different kind of work from everything the audit '
            'classifies'),
        by_reason=dict(tally),
        by_external_dependency=dict(collections.Counter(
            e['external_dependency'] for e in out.values())),
        by_compiled_probe_reachability=dict(collections.Counter(
            'reachable' if str(e.get('compiled_probe','')).startswith('REACHABLE')
            else ('out of reach: the lowering is too long'
                  if 'out of reach' in str(e.get('compiled_probe',''))
                  else 'no construct recorded or unmeasured')
            for e in out.values())),
        # THE SIZES OF THE SETS THAT DRIVE THE BRANCHES, published so a guard can see them. Two of
        # them were silently EMPTY: the csel membership read a key that does not exist, and the
        # instrument membership was built from `no_fit_rows_accounted` buckets that are plain
        # counts rather than member lists. Neither failed - the branches simply never fired, and
        # the forms fell through to a less informative reason that looked plausible. A set whose
        # emptiness cannot be seen is the same defect as a reader that answers empty.
        branch_inputs=dict(forms_the_csel_census_determines=len(csel_forms),
                           opcodes_another_instrument_reads=len(law_read),
                           forms_with_a_degenerate_probe=len(disqualified),
                           forms_whose_records_refute_their_fit=len(refuted),
                           opcodes_whose_candidates_coincide_at_a_width=len(coincide)),
        keyed_by='opcode/width - a FORM',
        means=('every dispatched form carrying no semantic column, with the reason existing '
               'structural and family evidence cannot determine it and the concrete step that '
               'would. The reasons are assigned in priority order and partition the population, '
               'so a form appears exactly once and none is dropped between them. A count without '
               'a next step is a restatement of the problem, so every reason carries one'),
        what_this_is_not=('a promise that each next step is cheap, and not a claim that the '
                          'reason is the ONLY obstacle - it is the first one that applies in a '
                          'fixed order, which is what makes the partition well defined'))


def family_law_provenance(spec):
    """For the family-law column: which forms it counts, and which it deliberately does not.

    This section replaced `d3_family_law_not_counted`, which existed while the column did not and
    argued for its promotion. The population is the same; what changed is that it is now COUNTED,
    so the useful report is provenance and residue rather than an argument.

    Three residues, each published because each is a different reason a family-law placement does
    not become coverage:

    `superseded_by_a_narrower_instrument` - the form has a tested law AND its own probe, ledger
    fact, or library fit. The narrower instrument owns the claim because its evidence is about this
    form rather than its family, and the law is then a cross-check. Counting both would inflate D3
    by exactly the overlap.

    `placed_only_by_an_untested_law` - every law placing it carries a qualifier in its own title
    saying its records do not test it. The `tested` flag is the census's; re-deriving it here would
    be a control sharing its predicate with what it controls.

    `records_whose_width_could_not_be_established` - the explaining record's bytes did not decode
    to one instruction of its own opcode, so no form can be named. Counted rather than folded in at
    the width the record's id claims, which is the fallback that once mis-attributed 89 receipts.
    """
    forms = family_law_semantics()
    counted, superseded, untested = {}, {}, {}
    for (opcode, length), entry in sorted(forms.items()):
        axes = ((spec.get('op%d' % opcode) or {}).get('len%d' % length) or {}).get('denominators') or {}
        record = dict(laws=entry['laws'], records=entry['records'])
        if not entry['tested']:
            untested['%d/%d' % (opcode, length)] = dict(
                record, laws_whose_records_do_not_test_them=entry['untested'])
        elif axes.get('d3_family_law'):
            counted['%d/%d' % (opcode, length)] = record
        else:
            owned = [key for key, value in axes.items()
                     if key.startswith('d3_') and value
                     and key not in ('d3_peer_reported', 'd3_family_law')]
            superseded['%d/%d' % (opcode, length)] = dict(record, counted_by=owned)
    return dict(
        counted=counted,
        counted_forms=len(counted),
        superseded_by_a_narrower_instrument=superseded,
        placed_only_by_an_untested_law=untested,
        records_whose_width_could_not_be_established=_FAMILY_LAW_CACHE.get('width_unknown') or {},
        keyed_by='opcode/width - a FORM, the width decoded from the record\'s own emitted bytes',
        means=('the sixth D3 column\'s provenance. A family law predicts a form\'s output from '
               'the table name, the declared precision, an immediate decoded out of the record\'s '
               'own bytes, or the lane set the record declares, and requires the retained '
               'observation to match. The VALUES are hardware observations - every record read '
               'here is a retained dispatch result - so this is not inference from structure '
               'alone; what is family-level is the ATTRIBUTION, and that is why it is a separate '
               'column rather than merged into elimination or the verified class. Its '
               'characteristic failure is being right about a family and wrong about a member '
               'whose encoding differs in something the law does not read, which is a different '
               'failure from every other column\'s and the reason none of them are summed'),
        what_it_does_not_assert=('that the form was probed in isolation to confirm its Apple '
                                 'name, and that no other function of the family could produce '
                                 'the same values on these particular cases'))


def verified_class_provenance(spec):
    """For the verified class: can its hardware evidence be NAMED from this repository?

    The strongest D3 column asserts "a probe isolating this opcode alone ran on hardware and
    returned the value the name predicts", sourced from `isa/g17-contract.jsonl`. The contract
    entry carries `meaning.evidence: verified` and does NOT carry the probe's identity or its
    returned values - op1022's entry has a witness encoding, a schedclass, reads and writes, and
    nothing that names a measurement. So for a reader asking "which record establishes this", the
    chain terminates in an assertion.

    Measured here rather than assumed: 39 of the 89 verified-class forms have at least one
    retained OK record for their opcode somewhere in `isa/g17-execution-*-results.json`, and 50
    have none at all. That is a statement about what can be POINTED AT from this repository, and
    deliberately not a stronger one: a retained record for the opcode is not proof that this
    record is what verified the name, and its absence is not proof the verification never
    happened - the class predates this branch and its probes may live elsewhere.

    Published because the mission this census serves requires every promoted semantic claim to
    identify its form, population, instrument and hardware evidence, and for 50 forms the fourth
    cannot be identified here. Nothing is retracted on that basis; the gap is named instead.
    """
    import glob as _glob
    have = set()
    for path in _glob.glob(str(ISA / 'g17-execution-*-results.json')):
        try:
            rows = json.loads(open(path).read())
        except Exception:
            continue
        rows = rows if isinstance(rows, list) else (rows.get('results') or [])
        for row in rows:
            if isinstance(row, dict) and row.get('status') == 'ok' and row.get('op') is not None:
                have.add(int(row['op']))
    forms = [(int(o[2:]), int(l[3:])) for o, f in spec.items() for l, rec in f.items()
             if ((rec.get('denominators') or {}).get('d3_semantics_checked'))]
    named = sorted('%d/%d' % (op, L) for op, L in forms if op in have)
    unnamed = sorted('%d/%d' % (op, L) for op, L in forms if op not in have)
    return dict(
        verified_forms=len(forms),
        a_retained_record_for_this_opcode_exists_here=len(named),
        no_retained_record_here_so_the_evidence_cannot_be_named=len(unnamed),
        forms_whose_evidence_cannot_be_named_here=unnamed,
        means=('the verified class is sourced from isa/g17-contract.jsonl, whose entries carry '
               '`meaning.evidence: verified` and do NOT carry the probe that established it or '
               'the values it returned. This field says how many of those forms have ANY retained '
               'record for their opcode in this repository to point at - not that such a record '
               'is what verified the name, and not that its absence means the verification never '
               'happened. The class predates this branch. Nothing is retracted on this basis; '
               'the mission requires every promoted claim to name its hardware evidence, and for '
               'these forms it cannot be named from here, so the gap is published'))


def promoted_claim_provenance(spec):
    """Does every promoted semantic claim identify its form, population, instrument and evidence?

    The mission this census serves requires exactly that, so it is measured in the artifact rather
    than by hand - and by hand is how it went wrong three times. The rule must ASK THE AXIS THAT
    OWNS THE CLAIM, not always `semantics`:

      - a fitted-by-elimination claim keeps its instrument, source and named records in
        `fitted_by_elimination`, not in `semantics`; `16806/12` has a complete fit there - `sar`,
        17 cases, 54 competitors, record `g16806` - while its semantics axis reads `unknown`,
        because that opcode is neither authorable nor in a shipped object;
      - a ledger-executed claim names its population in `ledger_references.about` and
        `ledger_claims.claims`, and its hardware evidence in `semantics.ledger_status`;
      - a probed or verified claim carries a receipt under `hardware_evidence`.

    Asking `semantics.source` of all four reported 61 incomplete when the true figure is 50. The
    50 are ENTIRELY the verified class, whose evidence `isa/g17-contract.jsonl` asserts without
    naming a probe or its values - the same gap `d3_verified_class_provenance` names form by form.
    So the answer is one clean sentence rather than a scatter: every promoted claim in this map
    identifies all four EXCEPT the verified class, and that exception is published.
    """
    complete, incomplete, by_cols = 0, [], {}
    for opname, forms in spec.items():
        for lenkey, rec in forms.items():
            d = rec.get('denominators') or {}
            cols = [k for k in ('d3_semantics_checked', 'd3_ledger_executed',
                                'd3_probed_here', 'd3_fitted_by_elimination') if d.get(k)]
            if not cols:
                continue
            sem = rec.get('semantics') or {}
            fit = rec.get('fitted_by_elimination') or {}
            receipt = (rec.get('hardware_evidence') or {}).get('receipt') or {}
            refs = rec.get('ledger_references') or {}
            claims = rec.get('ledger_claims') or {}
            instrument = bool(sem.get('evidence') or fit.get('evidence'))
            source = bool(sem.get('source') or fit.get('source') or sem.get('ledger_status'))
            population = bool(fit.get('records') or sem.get('records')
                              or receipt.get('receipt_id') or receipt.get('values')
                              or refs.get('about') or claims.get('claims'))
            hardware = bool(receipt.get('values')) or 'd3_ledger_executed' in cols \
                or bool(sem.get('ledger_status'))
            if instrument and source and population and hardware:
                complete += 1
            else:
                key = '%s/%s' % (opname[2:], lenkey[3:])
                incomplete.append(key)
                by_cols.setdefault(','.join(cols), []).append(key)
    return dict(
        forms_with_a_d3_own_instrument_column=complete + len(incomplete),
        complete_on_all_four=complete,
        incomplete=len(incomplete),
        incomplete_by_column_set={k: len(v) for k, v in sorted(by_cols.items())},
        forms_incomplete=sorted(incomplete),
        means=('form, population, instrument and hardware evidence, asked of the AXIS THAT OWNS '
               'each claim: a fitted claim in `fitted_by_elimination`, a ledger claim in '
               '`ledger_references`/`ledger_claims` and `semantics.ledger_status`, a probed or '
               'verified claim in `hardware_evidence`. Asking `semantics.source` of all four '
               'reports 61 incomplete where the truth is 50, and the 50 are entirely the verified '
               'class, whose evidence the contract asserts without naming a probe - the gap '
               '`d3_verified_class_provenance` names form by form. This field exists because '
               'measuring it by hand got the wrong answer three times'))


def _form_sort_key(form_id):
    """Sort `opcode/width` numerically where possible, lexically where not."""
    parts = str(form_id).split('/')
    out = []
    for p in parts:
        digits = ''.join(c for c in p if c.isdigit())
        out.append((0, int(digits), p) if digits else (1, 0, p))
    return tuple(out)


def coverage(src, spec, universe):
    per = defaultdict(lambda: dict(d1_structural=0, d1_apple_witnessed=0,
                                   d1_repair_walk_only=0, d2_encoded=0, d3_semantics_checked=0,
                                   d3_ledger_executed=0, d3_peer_reported=0, d3_probed_here=0,
                                   d3_fitted_by_elimination=0, d3_csel_predicate=0,
                                   d3_lane_probe=0, d3_transcendental=0, d3_memory_probe=0,
                                   d3_control_probe=0,
                                   d3_equivalence_class=0, d3_family_law=0,
                                   d3_any=0, executed_unchecked=0, dispatched=0,
                                   executed_unchecked_with_other_evidence=0,
                                   executed_with_no_semantic_column=0,
                                   executed_with_no_LOCAL_semantic_column=0))
    # THE POPULATIONS, not only their sizes.  Every count below was published on its own, and a
    # count without its population cannot be audited by anyone who did not run this generator:
    # T6 could not be finished because the 425 could not be enumerated, and T7 could not route a
    # 221-form tail it could not list.  Six small form lists were already published elsewhere in
    # this report, so the format was never the obstacle.
    pops = defaultdict(list)
    for name, forms in spec.items():
        family = universe[name]['family']
        for _form_key, record in forms.items():
            # `opcode/width`, the convention every other artifact and section uses. The spec keys
            # these `op1000` -> `len12`; publishing those raw gave populations reading ['len2',
            # 'len4', ...] with the opcode dropped, which is a list of widths and not of forms.
            form_id = '%s/%s' % (name[2:] if name.startswith('op') else name,
                                 _form_key[3:] if _form_key.startswith('len') else _form_key)
            d = record['denominators']
            row = per[family]
            row['d1_structural'] += 1
            # D1's COMPOSITION, so the total can never be quoted without its provenance.
            if record['length'].get('admissible_for_d2'):
                row['d1_apple_witnessed'] += 1
            else:
                row['d1_repair_walk_only'] += 1
            row['d2_encoded'] += int(d['d2_encoded'])
            row['d3_semantics_checked'] += int(d['d3_semantics_checked'])
            row['d3_ledger_executed'] += int(d['d3_ledger_executed'])
            row['d3_peer_reported'] += int(d['d3_peer_reported'])
            row['d3_probed_here'] += int(d['d3_probed_here'])
            row['d3_fitted_by_elimination'] += int(d['d3_fitted_by_elimination'])
            row['d3_csel_predicate'] += int(d['d3_csel_predicate'])
            row['d3_lane_probe'] += int(d['d3_lane_probe'])
            row['d3_memory_probe'] += int(d['d3_memory_probe'])
            row['d3_control_probe'] += int(d['d3_control_probe'])
            row['d3_transcendental'] += int(d['d3_transcendental'])
            row['d3_equivalence_class'] += int(d['d3_equivalence_class'])
            row['d3_family_law'] += int(d['d3_family_law'])
            row['d3_any'] += int(d['d3_semantics_checked'] or d['d3_ledger_executed']
                                 or d['d3_probed_here'] or d['d3_fitted_by_elimination']
                                 or d['d3_csel_predicate'] or d['d3_lane_probe']
                                 or d['d3_memory_probe'] or d['d3_control_probe']
                                 or d['d3_transcendental'] or d['d3_equivalence_class']
                                 or d['d3_family_law'])
            row['executed_unchecked'] += int(record['hardware_evidence']['executed_unchecked'])
            if d['d2_encoded']:
                pops['d2_encoded'].append(form_id)
            if record['hardware_evidence']['executed_unchecked']:
                pops['executed_unchecked'].append(form_id)
            # `executed_unchecked` MEANS "has a receipt and is not in the contract's verified
            # class", and the table published it as "it ran and the output was never compared".
            # That is false of 81 of the 483: 38 carry a fit from the elimination census, 25 were
            # probed here on hardware, 14 are named by an executed ledger fact, 4 by the peer
            # lane. Their outputs WERE compared, by a different instrument than the verified
            # class. A count whose predicate is narrower than its label is the shape already
            # corrected in this repository for `cases`, for the constant-output classes and for
            # the separating inputs, so all three readings are published rather than one.
            _any_col = (d['d3_semantics_checked'] or d['d3_ledger_executed']
                        or d['d3_probed_here'] or d['d3_fitted_by_elimination']
                        or d['d3_csel_predicate'] or d['d3_lane_probe']
                        or d['d3_memory_probe'] or d['d3_control_probe']
                        or d['d3_transcendental'] or d['d3_equivalence_class']
                        or d['d3_family_law'] or d['d3_peer_reported'])
            _own_col = (d['d3_semantics_checked'] or d['d3_ledger_executed']
                        or d['d3_probed_here'] or d['d3_fitted_by_elimination']
                        or d['d3_csel_predicate'] or d['d3_lane_probe']
                        or d['d3_memory_probe'] or d['d3_control_probe']
                        or d['d3_transcendental'] or d['d3_equivalence_class']
                        or d['d3_family_law'])
            if record['hardware_evidence']['dispatched']:
                if record['hardware_evidence']['executed_unchecked'] and (
                        _any_col and not d['d3_semantics_checked']):
                    # WAS THE LITERAL 81 IN THE REFERENCE TABLE. Adding the family-law column
                    # made it stale the moment it was wired, which is what a hardcoded number
                    # inside a generated document always does.
                    row['executed_unchecked_with_other_evidence'] += 1
                if not _any_col:
                    row['executed_with_no_semantic_column'] += 1
                    pops['executed_with_no_semantic_column'].append(form_id)
                if not _own_col:
                    pops['executed_with_no_LOCAL_semantic_column'].append(form_id)
                    # THE PEER COLUMN STAYS SEPARATE. A form whose only semantic evidence came
                    # from another lane has nothing THIS lane has checked, so it belongs in the
                    # local reading even though it is not evidence-free.
                    row['executed_with_no_LOCAL_semantic_column'] += 1
            # DISPATCHED is accumulated so the limits section can QUOTE it instead of restating a
            # literal. The sentence about the capability ledger said "496 dispatched" as text
            # beside a report that computes it, which is the shape that once let a published
            # artifact assert D2 = 690 next to its own field reading 628.
            row['dispatched'] += int(record['hardware_evidence']['dispatched'])
            if record['hardware_evidence']['dispatched']:
                pops['dispatched'].append(form_id)
            elif d['d2_encoded']:
                # THE NEVER-DISPATCHED TAIL, which T7 asks to be routed.  Derived here rather
                # than as `d2_encoded - dispatched` so the list and the difference cannot drift.
                pops['encoded_never_dispatched'].append(form_id)
    totals = {k: sum(r[k] for r in per.values())
              for k in ('d1_structural', 'd2_encoded', 'd3_semantics_checked',
                        'd3_ledger_executed', 'd3_peer_reported', 'd3_probed_here',
                        'd3_fitted_by_elimination', 'd3_csel_predicate',
                        'd3_lane_probe', 'd3_transcendental', 'd3_equivalence_class',
                        'd3_family_law', 'd3_memory_probe', 'd3_control_probe', 'd3_any',
                        'executed_unchecked', 'dispatched',
                        'executed_unchecked_with_other_evidence',
                        'executed_with_no_semantic_column',
                        'executed_with_no_LOCAL_semantic_column')}
    # COMPUTED HERE rather than inline in the return dict, because the `limits` section quotes
    # these numbers: a limit that restates a count it cannot see is how two of them outlived the
    # work that refuted them.
    _width_origins = Counter(
        ('receipt_unique' if (origin or '').startswith('receipt, UNIQUE') else
         'receipt_ambiguous' if (origin or '').startswith('receipt, AMBIGUOUS') else
         'forced_single_form' if count == 1 else 'inferred_could_be_wrong')
        for origin, count in (((record['semantics'].get('checked_width_origin')), len(forms))
                              for forms in spec.values() for record in forms.values()
                              if record['denominators']['d3_semantics_checked']))
    # WHY THE 89 APPLE-WITNESSED FORMS OUTSIDE D2 ARE OUTSIDE IT. D1 now covers every admitted
    # opcode, and 717 of its forms carry an Apple byte count at THAT width - the only widths D2
    # will consider. D2 is 628, so 89 are admissible and still excluded, and until now the report
    # gave the total without the reason. Each class names what would close it, which is the point:
    # the undetermined ones want more corpus instances, the five want a form-bit harvest, and the
    # unforced ones are a statement about APPLE's corpus rather than a gap in this work - no amount
    # of effort here pins a bit Apple never varies.
    d2gap = dict(admissible=0, in_d2=0, excluded=0,
                 by_state=Counter(), by_bit_kind=Counter())
    for name, forms in spec.items():
        for width_key, record in forms.items():
            if not (record['length'] or {}).get('admissible_for_d2'):
                continue
            d2gap['admissible'] += 1
            unknown = record.get('unknown_bits') or {}
            if record['denominators']['d2_encoded']:
                d2gap['in_d2'] += 1
                continue
            d2gap['excluded'] += 1
            d2gap['by_state'][str(unknown.get('state'))] += 1
            kind = ('unforced bits' if unknown.get('unforced') else
                    'undetermined bits' if unknown.get('undetermined') else
                    'no form-level bit evidence at this width'
                    if str(unknown.get('state')).startswith('no form-level') else
                    'incomplete for another reason')
            d2gap['by_bit_kind'][kind] += 1
            if unknown.get('undetermined'):
                # the width is the FORM's own key; the length record carries how the width is
                # known, not the width itself
                d2gap.setdefault('_undetermined_forms', []).append(
                    (int(str(name)[2:] if str(name).startswith('op') else name),
                     int(str(width_key)[3:] if str(width_key).startswith('len') else width_key)))
    d2gap['by_state'] = dict(d2gap['by_state'].most_common())
    d2gap['by_bit_kind'] = dict(d2gap['by_bit_kind'].most_common())
    # IS "MORE CORPUS INSTANCES" ACTUALLY A REMEDY? MEASURED - AND THE FIRST MEASUREMENT WAS OF
    # THE WRONG POPULATION. It counted these forms in the `spans` index of
    # isa/g17-corpus-programs.jsonl, called that Apple's whole shipped set, found every one a
    # singleton or absent, and concluded only execution could settle them. Over both committed
    # corpora, each distinct text once, most have two or more instances, so for those the remedy
    # IS reading. The text below is derived from the count, not written against it.
    _und_forms = [f for f in d2gap.pop('_undetermined_forms', [])]
    singletons = corpus_singletons(_und_forms)
    d2gap['undetermined_in_apples_whole_corpus'] = singletons
    fbpass = undetermined_free_bits_pass(_und_forms, src['freebits'])
    d2gap['undetermined_bits_free_bits_pass'] = fbpass
    d2gap['what_would_close_each'] = {
        'unforced bits': 'nothing in this repository - Apple never varies the bit, so its value '
                         'is not recoverable from the corpus at any effort',
        'undetermined bits':
            ('measured across both committed Apple corpora, each distinct text once: %d of %d '
             'of these forms appear once or not at all, so for them a verdict - which needs two '
             'instances of the FORM - is unreachable by reading and only execution settles '
             'them; the other %d have two or more instances, and the free-bits rule over them '
             'settles every undetermined bit of %d of those forms (over all %d forms, bits: %s). '
             'None of these is adopted, and each form carries why (%s): the GPU check section '
             '25.88 required has run (isa/g17-adoption-check-results.json) and the 13 forms it '
             'cleared were adopted and left this list.'
             % (singletons.get('singleton_or_absent', 0), singletons.get('forms_checked', 0),
                singletons.get('reachable_by_reading_more_corpus', 0),
                fbpass.get('forms_every_undetermined_bit_settled', 0),
                fbpass.get('forms_checked', 0),
                ', '.join('%d %s' % (n, k) for k, n in sorted((fbpass.get('bits') or {}).items())),
                ', '.join('%d %s' % (n, k) for k, n in sorted((fbpass.get('by_adoption') or {})
                                                            .items())))),
        'no form-level bit evidence at this width': 'a form-bit harvest at this width',
        'incomplete for another reason': 'an invisible bit that no instance settles'}

    unsep = unseparated_basis(spec)
    prereg = preregistered_expect_basis()
    _fits = fitted_semantics()
    _fit_bar = (list(_fits.values())[0]['bar'] if _fits else {})
    fitbasis = dict(
        forms_named=len(_fits),
        counted_in_d3=sum(1 for name, forms in spec.items() for record in forms.values()
                          if record['denominators']['d3_fitted_by_elimination']),
        already_counted_in_another_column=sum(
            1 for name, forms in spec.items() for record in forms.values()
            if (record.get('fitted_by_elimination') or {}).get('also_counted_elsewhere')),
        bar=_fit_bar,
        note=('a FIFTH D3 instrument, in its own column. Uniqueness holds within the published '
              'candidate library and at the operand and modifier configuration each record was '
              'dispatched with, so it is not folded into d3_semantics_checked: a function of '
              'another arity, one reading the operands differently, or one depending on a '
              'modifier is not excluded by four points. Forms already carried by another column '
              'are excluded rather than added, so the twelve the census reproduces from '
              'hand-written facts do not inflate the total - they validate the tool'))
    basis = dict(
        total=sum(_width_origins.values()),
        width_from_a_unique_receipt=_width_origins['receipt_unique'],
        width_from_receipts_but_ambiguous=_width_origins['receipt_ambiguous'],
        width_forced_one_admitted_form=_width_origins['forced_single_form'],
        width_inferred_and_could_be_wrong=_width_origins['inferred_could_be_wrong'],
        identity_not_pinned=(_width_origins['receipt_ambiguous']
                             + _width_origins['inferred_could_be_wrong']))
    axis_fill = Counter()
    for _forms in spec.values():
        for _record in _forms.values():
            for _name in ('latency_wait', 'barrier_sync', 'resources'):
                _value = _record.get(_name)
                if isinstance(_value, dict) and _value.get('evidence', 'absent') != 'absent':
                    axis_fill[_name] += 1
    return dict(
        note=('Three denominators, deliberately not combined. D1 counts forms this decoder can '
              'address; D2 requires an encoding Apple actually wrote with no unforced and no '
              'undetermined bits; D3 requires the ledger\'s `verified` class, an isolated probe '
              'that returned the value the name predicts on hardware. Dispatched-but-unchecked '
              'forms are reported apart: running is not checking.'),
        ladder=list(LADDER), axes=list(AXES), families=list(FAMILIES),
        evidence_classes={k: dict(level=v[0], means=v[1]) for k, v in EVIDENCE_CLASS.items()},
        sources_found=src['found'],
        d2_exclusion_basis=d2gap,
        unseparated_basis=unsep,
        named_unknowns=named_unknowns(src, spec, universe, totals, d2gap, unsep,
                                      field_map_width_scope(src)),
        preregistered_expect_basis=prereg,
        fitted_by_elimination_basis=fitbasis,
        d3_family_law_provenance=family_law_provenance(spec),
        untouched_form_audit=untouched_form_audit(spec),
        locally_witnessed_widths=locally_witnessed_widths(),
        csel_predicate_provenance=csel_predicate_provenance(),
        d3_candidate_not_counted=dict(
            predicate_determined_forms=predicate_determined_forms(),
            # PUBLISHED EVEN WHEN EMPTY, and it is empty: every determined entry named exactly
            # one emitted width. An absent key would leave a reader unable to tell "none were
            # ambiguous" from "nobody asked".
            forms_whose_width_is_ambiguous=predicate_forms_whose_width_is_ambiguous(),
            keyed_by=('opcode/width - a FORM. This was keyed by opcode while the reason it '
                      'stayed uncounted included "it cannot name a width"; the width was in '
                      'every row of the census underneath the aggregate that dropped it'),
            means=('a CANDIDATE sixth D3 contribution, computed and deliberately NOT counted in '
                   'd3_any or any column. The elimination census determined these forms by '
                   'eliminating a library of predicates against a BOOLEAN output, which is '
                   'elimination of the same kind as the function census with the same margins and '
                   'competitor counts. It is kept out of d3_fitted_by_elimination because that '
                   'would make one column mean two instruments with different failure modes - the '
                   'reason the five columns are exclusive - and out of a new sixth column because '
                   'that changes this repository\'s headline D3, which is root\'s call. Promoting '
                   'it is one edit on a number that is already visible here. The WIDTH objection '
                   'is discharged: each of these names one form, 12 at width 10 and 2 at 14, '
                   'measured twice - the census\'s own per-row width and an independent decode of '
                   'the retained emitted bytes - so what remains is the column question alone')),
        d3_verified_class_provenance=verified_class_provenance(spec),
        promoted_claim_provenance=promoted_claim_provenance(spec),
        d3_verified_basis=dict(basis, note=('D3 verified counts ONE form per verified opcode, because the class means "a '
                  'probe isolating this opcode alone ran on hardware" and a probe dispatches one '
                  'encoding; published uncorrected it was 140 forms over 91 opcodes. This is the '
                  'basis of the 91 - how the single width was established, per form. AMBIGUOUS is '
                  'the category that had been missing: three receipt instruments (the sweep\'s '
                  'row ids, the targeted files\' op/length fields, and the bytes those receipts '
                  'retain, decoded) place 496 forms between them, and 12 opcodes have widths from '
                  'two instruments with no overlap. Those do not conflict - they are different '
                  'dispatches of different forms, and both are true. What NO instrument records '
                  'is which dispatched width established the NAME, so where several ran the '
                  'identity of the counted form is undetermined and says so. `identity_not_pinned'
                  '` is ambiguous plus inferred: the honest exposure of this denominator. '
                  'WHERE `verified` COMES FROM, traced rather than assumed: '
                  'isa/g17-certification.jsonl carries exactly 91 verified rows and the contract '
                  'reads its class straight from there. No tool in this repository writes that '
                  'file - it is hand-maintained across separate campaigns whose commits name what '
                  'each added ("confirms 28 by value", the compare-select family across every '
                  'comparison direction, the six format conversions) - and '
                  'isa/g17-verified-opcodes.jsonl names only 54 of the 91. Per-opcode provenance '
                  'is the `constructs` list; the WIDTH the probe dispatched is recorded in '
                  'neither file, which is why this map has to choose it and label the choice.')),
        width_coverage=dict(
            admitted_opcodes=len(src['contract']),
            with_a_labelled_width=len(src['apple_width']) + len(src['repair_width']),
            from_an_apple_byte_count=len(src['apple_width']),
            from_a_repair_walk_decode=len(src['repair_width']),
            no_width=len(src['contract']) - len(src['apple_width']) - len(src['repair_width']),
            with_a_full_axis_record=len(spec),
            # DERIVED, not restated. The version of this note written at ff0b376b ended "coverage
            # rose from 727 opcodes to 6,646 while D2 stayed at 690"; the next commit to touch D2
            # moved it to 628 and left the sentence standing, so the published artifact asserted a
            # denominator its own `forms.totals.d2_encoded` field contradicted. Every live number
            # here now comes from the same values the fields above report, and the historical
            # before/after is labelled with the commit that measured it.
            note=('every admitted opcode carries a width labelled with how it is known. Only the '
                  '%d Apple byte counts are admissible for D2, so widening D1 to cover every '
                  'admitted opcode cannot raise D2: it stands at %d, against %d labelled widths '
                  'of which %d are repair-walk decodes. A repair-walk width says the decoder '
                  'accepts a form of that width, not that anything emitted one. WHAT THE WIDENING '
                  'DID, as measured at 034120f2: D1 975 -> 6,897 forms over 6,646 opcodes, and '
                  'D2 690 -> 628 - a FALL, and not a consequence of the widening. Labelling '
                  '`admissible_for_d2` per form made it checkable and exposed 62 forms that had '
                  'been in D2 at widths Apple never wrote, because the gate had been an '
                  'OPCODE-level `apple_witness` while a form is (opcode, width).'
                  % (len(src['apple_width']), totals['d2_encoded'],
                     len(src['apple_width']) + len(src['repair_width']),
                     len(src['repair_width']))),
            # A TRUNCATED LIST IS A BOUND, NOT A COUNT. This published
            # `repair_width_failed[:20]` with no total, so a reader could not tell twenty
            # failures from two thousand - the same shape as a capped scan reporting its cap.
            # The total and a per-reason census come first; the sample is explicitly a sample.
            width_failures=dict(
                total=len(src['repair_width_failed']),
                by_reason=dict(Counter(
                    str(f.get('reason')).split(':')[0].strip()
                    for f in src['repair_width_failed']).most_common()),
                by_reason_detail=dict(Counter(
                    str(f.get('reason')) for f in src['repair_width_failed']).most_common(8)),
                sample=src['repair_width_failed'][:20],
                sample_is_capped_at=20),
            failures=src['repair_width_failed'][:20]),
        decoder_cross_check=dict(
            width_instrument='the byte count of encoding.apple_witness - Apple\'s own emitted bytes',
            opcodes_with_an_apple_witness=len(src['apple_width']),
            apple_width_agrees_with_form_bits=sum(
                1 for op, w in src['apple_width'].items()
                if w in {int(k.split(',')[1]) for k in src['formbits'] if k.split(',')[0] == str(op)}),
            apple_width_contradicts_form_bits=sum(
                1 for op, w in src['apple_width'].items()
                if (d := {int(k.split(',')[1]) for k in src['formbits']
                          if k.split(',')[0] == str(op)}) and w not in d),
            apple_witness_with_no_form_bits_entry=sum(
                1 for op in src['apple_width']
                if not any(k.split(',')[0] == str(op) for k in src['formbits'])),
            identity_instrument='agxforge.g17.model.decode over the retained padded witness',
            identity_agrees=src['decoder_identity']['agree'],
            identity_mismatched=src['decoder_identity']['mismatched'],
            decoder_could_not_use_the_witness=len(src['decoder_identity']['errors']),
            errors=src['decoder_identity']['errors'][:20],
            identity_witnesses_that_are_apple_bytes=src['decoder_identity'].get(
                'apple_bytes_agree'),
            identity_means=('most of `identity_agrees` CANNOT FAIL: g17admit kept a walked '
                            'witness only when tools/agx3dis decoded it alone to that opcode, '
                            'and decode() shells out to the same binary. '
                            '`identity_witnesses_that_are_apple_bytes` counts the agreements '
                            'on bytes Apple emitted, which the decoder did not choose - and '
                            'even those carry an opcode id that decoder assigned. What the '
                            'check tests is the wrapper, not the decoder'),
            note=('opcode identity is consistent throughout - zero witnesses decode to a '
                  'different opcode. Width is a separate question and comes from Apple\'s byte '
                  'count, never from decoding the padded buffer: doing that measured the '
                  'decoder\'s reading of padding and reported a 120-opcode gap that is not real.')),
        # A curated fact whose opcode has no form in the spec is never applied to anything, and
        # silently losing evidence is worse than not having harvested it. Named here so the loss
        # is countable: an admitted opcode can have no form because nothing - form-bits,
        # free-bits, the sweep, or an Apple witness - evidences one at any width.
        curated_facts_with_no_form=dict(
            opcodes=sorted({o for o in list(LEDGER_FACTS) + list(PEER_FACTS)
                            if 'op%d' % o not in spec}),
            note=('these opcodes carry a harvested fact that reaches no form record. They are '
                  'admitted, but no source evidences a width for them, so the specification has '
                  'nowhere to put the fact')),
        # The generation direction: every form the BACKEND can actually emit must be in here, or
        # the specification is incomplete exactly where a consumer would notice first.
        backend_emittable=(lambda pairs: dict(
            source='isa/g17-template-opcodes.json - one decode per template the backend emits',
            distinct_opcode_width_pairs=len(pairs),
            covered_by_this_specification=sum(
                1 for op, w in pairs
                if 'op%d' % op in spec and 'len%d' % w in spec['op%d' % op]),
            pairs=[[op, w] for op, w in pairs],
            note=('the backend\'s 15 templates resolve to 14 distinct (opcode, width) pairs; two '
                  'templates share one. Full coverage here is what makes "g17as could be driven '
                  'from this file" a checkable claim rather than an aspiration')))(
            sorted({(op, len(h)//2) for h, op in
                    ((load(ISA/'g17-template-opcodes.json')[0] or {}).get('opcodes') or {}).items()})),
        peer_reported=dict(
            source=PEER, facts_for_opcodes=sorted(PEER_FACTS),
            opcodes_the_admission_walk_undercounts=WALK_UNDERCOUNT,
            reach_measurement=REACH_MEASUREMENT,
            # "The five opcodes named above" was a literal sitting directly beneath a list of
            # SEVEN (facts_for_opcodes), describing a different field further down. The count is
            # now read off the dict it is about and the field is named, because a hand-written
            # number beside its own data is the hiding place a regenerated report gives it.
            note=('another lane\'s reconnaissance, unmerged and unverified here. Counted in '
                  'd3_peer_reported only. The %d opcodes in '
                  'opcodes_the_admission_walk_undercounts are Apple table rows this artifact\'s '
                  'admission walk cannot reach, so D1 is an undercount by a known mechanism '
                  'rather than a decoder limit; facts_for_opcodes is a different and longer set '
                  '(%d) - the opcodes any peer-attributed fact names'
                  % (len(WALK_UNDERCOUNT), len(PEER_FACTS)))),
        field_map_width_scope=field_map_width_scope(src),
        unharvested_sources=unharvested_sources(),
        ledger_citation_index={k: v for k, v in src['ledger'].items()
                               if k not in ('by_opcode', 'ledgers')},
        mechanical_claim_extraction={k: v for k, v in src['claims'].items()
                                     if k != 'by_opcode'},
        peer_claim_extraction={k: v for k, v in src['peer_claims'].items()
                               if k != 'by_opcode'},
        isa_level_facts=GLOBAL_FACTS,
        opcode_universe=dict(
            apple_table_declares=17779,
            decoder_admits=len(src['contract']),
            with_authoritative_length=len(src['lengths']),
            apple_wrote_the_encoding=sum(1 for r in src['contract'].values()
                                         if (r.get('encoding') or {}).get('apple_witness')),
            repair_walk_witness_only=sum(1 for r in src['contract'].values()
                                         if not (r.get('encoding') or {}).get('apple_witness')),
            by_evidence_class=dict(Counter((r.get('meaning') or {}).get('evidence')
                                           for r in src['contract'].values()))),
        forms=dict(totals=totals, by_family={f: dict(per[f]) for f in FAMILIES if f in per},
                   # EVERY COUNT ABOVE, AS ITS POPULATION.  A reader can now intersect these with
                   # any other artifact; before, `dispatched` and `executed_unchecked` were sizes
                   # with nothing behind them.  `populations_agree_with_totals` is asserted rather
                   # than trusted, because a list that drifts from its own count is worse than no
                   # list: it reads as auditable and is not.
                   # Form ids are mostly `opcode/width` but some carry a `len10`-style width, so
                   # the sort is tolerant rather than assuming the format. Assuming it raised on
                   # the first regeneration.
                   populations={k: sorted(v, key=_form_sort_key)
                                for k, v in sorted(pops.items())},
                   populations_agree_with_totals={
                       'd2_encoded': len(pops['d2_encoded']) == totals['d2_encoded'],
                       'dispatched': len(pops['dispatched']) == totals['dispatched'],
                       'executed_unchecked':
                           len(pops['executed_unchecked']) == totals['executed_unchecked'],
                       'executed_with_no_semantic_column':
                           len(pops['executed_with_no_semantic_column'])
                           == totals['executed_with_no_semantic_column'],
                       'executed_with_no_LOCAL_semantic_column':
                           len(pops['executed_with_no_LOCAL_semantic_column'])
                           == totals['executed_with_no_LOCAL_semantic_column'],
                       # NOT `d2_encoded - dispatched`.  That subtraction assumes dispatched is a
                       # SUBSET of d2_encoded and it is not: many dispatched forms are not
                       # d2-encoded, so the difference (221) is smaller than the real tail (388).
                       # Section 25.65 published 221 on that arithmetic; publishing the population
                       # is what exposed it.
                       'encoded_never_dispatched':
                           len(pops['encoded_never_dispatched'])
                           == len(set(pops['d2_encoded']) - set(pops['dispatched'])),
                   }),
        family_hypothesis_calibration=src.get('family_hypothesis'),
        family_assignment=dict(
            counted_unknown=sum(1 for v in universe.values() if v['family'] == 'unknown'),
            by_family=dict(Counter(v['family'] for v in universe.values())),
            name_segments_classified=len(SEGMENT_FAMILY),
            name_segments_set_aside=UNCLASSIFIED_SEGMENTS,
            name_segments_unmapped=unmapped_name_segments(src['contract']),
            name_leads_over_declared_effects=(
                'a family describes the operation; an effect flag describes a property of it. '
                'Disagreements are listed, not hidden, and the address-file destination survives '
                'in the register_files axis.'),
            disagreements=family_disagreements(src['contract'])),
        # the two limits below quote live counts, so a harvest that moves them moves the
        # sentence. Both had outlived the work that refuted them.
        limits=[
            'D1 is the DISCOVERED, decoder-addressable surface, not the chip. No claim of absence '
            'of hidden instructions is made or implied.',
            'A bit invisible to Apple\'s decoder is not a bit the hardware ignores. Those count as '
            'unknown until independent corpus instances settle them.',
            # SUPERSEDED AND REWRITTEN, not deleted. This read "Per-width semantic attribution
            # is finer than the evidence and is not claimed" - true when written and false now:
            # the verified class IS narrowed to one form per opcode, because a probe dispatches
            # one encoding and counting every width of that opcode put 49 forms in D3 that no
            # probe reached. What replaced the old blanket disclaimer is a counted exposure.
            'D3\'s verified class is per OPCODE in its source, and this map narrows it to exactly '
            'ONE form per verified opcode, choosing the width by the strongest receipt available '
            'and LABELLING the choice. %d of the %d have an identity no instrument pins - %d '
            'dispatched at several widths with nothing recording which one established the name, '
            '%d inferred from Apple\'s witness byte count on a multi-form opcode. See '
            '`d3_verified_basis`; the per-form claim is exactly as strong as that field says.'
            % (basis['identity_not_pinned'], basis['total'],
               basis['width_from_receipts_but_ambiguous'],
               basis['width_inferred_and_could_be_wrong']),
            # A COUNTED EXPOSURE, AND THE FIRST VERSION OF THIS SENTENCE OVERSTATED IT. Audited
            # 2026-09-17 against results/g17-capability-ledger.json at sha256 439517d81dfc38d0
            # (added on main at 90a7c05f; NOT on this branch, so the figures are pinned to that
            # hash rather than recomputed - a live read here would report zero, and a zero from a
            # missing file is a limit that cannot fail).
            #
            # WHAT I GOT WRONG, corrected the same day. The sentence first published here said 54
            # of 62 forms with hardware-execution evidence "go uncounted", and leaned on
            # `forms_without_isolated_record` as the measure of how well a form's execution is
            # attributed. Both are wrong. That field is about per-form SEMANTICS records
            # (`compiler.iso.*`); the fields that speak to execution are `forms_without_execution`
            # and `forms_executed_with_retained_bytes`. And "uncounted" was not true: this map
            # already marks 496 forms `dispatched` and 455 `executed_unchecked`, harvested from
            # all 58 isa/g17-execution-*-results.json files. Those forms are in a DIFFERENT
            # COLUMN on purpose, because running is not checking - not missing from the report.
            #
            # An independent check of that harvest, run today: decoding the retained `encoded`
            # bytes of every ok record in those 58 files yields 409 forms whose bytes really do
            # contain the opcode claimed; 40 of 40 of the capability ledger's `compiler.iso.*`
            # forms agree with that decode with ZERO disagreements, and all 409 are already
            # marked dispatched here. So the execution harvest needed nothing. The gap that WAS
            # real was in the curated facts - the atomic axis sat at D3 = 0 while three of its
            # forms had GPU-execution evidence - and that is fixed rather than described.
            'The compiler capability ledger holds 128 evidence entries of kind "workload GPU '
            'execution" across 54 capabilities, reaching 62 (opcode, width) forms. Eight of those '
            'are in D3\'s executed or checked classes here. The other 54 are NOT invisible: this '
            'map marks %d forms dispatched and %d executed_unchecked from the 58 isolated-'
            'execution result files, and keeps them out of D3 deliberately, because a program '
            'that ran is not an instruction whose values were checked. What that ledger adds '
            'beyond them is per-form strength: 40 forms carry '
            '`forms_executed_with_retained_bytes`, meaning the bytes that ran were kept and the '
            'form was decoded from them rather than from a re-compilation. Promoting one of the '
            '54 into D3 requires reading its receipt and finding a checked VALUE, not counting '
            'its presence in a passing program. Audited 2026-09-17 against '
            'results/g17-capability-ledger.json sha256 439517d81dfc38d0; the 409-form decode '
            'cross-check agreed with that ledger on 40 of 40 with no disagreements.'
            % (totals['dispatched'], totals['executed_unchecked']),
            # QUOTED FROM THE FIELD BESIDE IT, never from recollection.
            'Of the %d plan/result pairs of isolated dispatches in isa/, %d forms returned '
            'exactly the words a plan\'s `expect` names, on every case and every run. Only %d '
            'are counted as checked here. The bar is not agreement - every plan was committed in '
            'the SAME COMMIT as its results, so nothing dates the expectation before the run and '
            '"observed == expect" has no way to fail - it is that the expectation was RECOMPUTED '
            'from an independent reference. The remaining %d are preservation checks whose '
            'expected value IS a prior observation ("these bits cleared, same result"); no rule '
            'regenerates them, so they stay in executed_unchecked. A further %d forms are '
            'checked with NO stored expectation at all: their records carry no `expect`, so the '
            'function was determined by computing every competing candidate and finding exactly '
            'one that reproduces the observed words, a class the caveat does not touch. See '
            '`preregistered_expect_basis`.'
            % (prereg['plan_result_pairs'], prereg['forms_matching_a_planned_expect'],
               prereg['rule_reproduced_and_promoted'],
               prereg['expectation_is_a_prior_observation'],
               prereg['function_fitted_with_no_expect']),
            'A fifth D3 instrument, reported in its own column: %d forms are named by '
            'ELIMINATION over the isolated-dispatch records - every candidate of a published '
            'library computed against the words the hardware returned, keeping only forms where '
            'exactly one survives all four cases, outputs are not equal to any input column, at '
            'least %d competitors were eliminated, and every record of the form agrees. %d of '
            'them count in D3; the other %d are already carried by another column and are '
            'excluded rather than added - those are the ones the census reproduces from '
            'hand-written facts, so they validate the instrument instead of inflating it. The '
            'claim is deliberately NOT merged into d3_semantics_checked: uniqueness holds within '
            'that library and at the configuration each record was dispatched with, so a function '
            'of another arity, reading the operands differently, or depending on a modifier is not '
            'excluded by four points. The census also reports what it could NOT name, which is '
            'most of it - see isa/g17-execution-fits.json and `fitted_by_elimination_basis`.'
            % (fitbasis['forms_named'], (fitbasis['bar'] or {}).get('min_competing_candidates', 0),
               fitbasis['counted_in_d3'], fitbasis['already_counted_in_another_column']),
            'D1 covers every admitted opcode and %d of its forms carry an Apple byte count at '
            'THAT width, which is the only width D2 considers. D2 is %d, so %d are admissible and '
            'still excluded: %s. The unforced ones are not a gap in this work - Apple never varies '
            'those bits, so no effort here recovers them - while the undetermined ones want more '
            'corpus instances and the rest want a form-bit harvest. See `d2_exclusion_basis`, '
            'which names what would close each class.'
            % (d2gap['admissible'], d2gap['in_d2'], d2gap['excluded'],
               ", ".join("%d %s" % (v, k) for k, v in d2gap['by_bit_kind'].items())),
            # THE TYPED-HOLE CENSUS BELONGS HERE AND NOT ON AN AXIS, and a guard in this
            # repository stopped me putting it there. I wrote two `semantics` facts for the only
            # two typed holes whose semantics axis was absent, at corpus level so they could not
            # reach D3 - and `test_the_isa_layer_cannot_reach_a_denominator_axis` went red,
            # because the rule is that this layer never touches `semantics` or `unknown_bits` at
            # ALL, not that it may touch them weakly. The rule is right twice over: a curated
            # fact is prose rather than a probe, and "no recovered role" written onto the
            # semantics axis makes the axis read POPULATED for an instruction whose semantics is
            # exactly what nobody knows. That is padding an axis-fill count. `absent` is the
            # correct value for a typed hole, and the census goes in prose where it cannot be
            # mistaken for evidence.
            'isa/g17-typed-holes.toml names %d opcodes with no recovered semantic ROLE, and they '
            'are not all equally out of reach: %d are FULLY DECODED with residual_bits 0, so a '
            'composer can build them from their operands alone and what is missing is a name '
            'rather than an encoding, while %d are neither named nor fully decoded - the only '
            'ones where a composer must inherit template state it cannot account for, at a stated '
            'cost of 911 instruction-bits of assumed dominant values. Their semantics axis reads '
            '`absent` deliberately: a dataflow position measured by def-use says where an opcode '
            'sits among its consumers, not what it computes, and this map does not let the isa '
            'layer write the axis D3 reads.'
            % (149, 133, 16),
            'The elimination census leaves %d records it could not separate, and that total must '
            'not be read as unknown instructions: %s. Only %d opcodes in the whole set have no '
            'semantics from any other instrument - %s - and three of those four are '
            'constant-output records, which no choice of candidate can resolve because an '
            'absorbing function fits them. The two pairs the census names most often, bitwise_A '
            'against umin and bitwise_C against rotl, span 22 opcodes of which ALL are already '
            'known, so probing them produces cross-checks and not coverage. See '
            '`unseparated_basis`.'
            % (unsep.get('records', 0),
               "; ".join("%s %s" % (v, k) for k, v in
                         sorted((unsep.get('by_kind_and_novelty') or {}).items())),
               len(unsep.get('opcodes_with_no_other_semantics') or []),
               ", ".join('op%d' % o for o in
                         (unsep.get('opcodes_with_no_other_semantics') or []))),
            'simd_shuffle_reduce, tensor_accelerator, graphics and ray_tracing have NO mechanical '
            'discriminator in Apple\'s descriptor. Forms that may belong to them sit in the '
            'counted unknown bucket. That bucket is the cartography\'s largest hole, not a tail.',
            # SUPERSEDED. This read "unpopulated on every form: no ledger establishes them per
            # form" - which was the state until the ledgers were read FOR them rather than
            # indexed. The count is derived so the sentence cannot outlive the next harvest.
            'latency_wait, barrier_sync and resources were once unpopulated on every form - "no '
            'ledger establishes them per form" - and are now filled on %d, %d and %d forms '
            'respectively, from ledgers whose content had never been cited. Every form NOT among '
            'those still reads `absent`, which means nobody has read for it, never that the '
            'machine has nothing there: the three axes were empty for exactly that reason.'
            % (axis_fill['latency_wait'], axis_fill['barrier_sync'], axis_fill['resources']),
            'isa/README.md quotes an older snapshot of the evidence classes (77 verified, 442 '
            'measured, 2724 table-only). These counts are harvested live and differ.',
            'D1 IS NOT CLOSED, and there are named counterexamples. The admitted set is sparse: '
            'of Apple\'s 17,779 declared ids only 6,718 are admitted, and the gaps are interior, '
            'not a tail. In 5090..5115 only 5106 and 5107 are admitted, yet the linker peer '
            'reports the driver compiling op5098 (f32.f32) and op5104 (f16.f32) as 16x16x16 '
            'multiply-accumulate forms at width 10 (linker/g17-tensorops-recon ce840dda, not '
            'merged, not independently verified here). An instruction absent from the admitted '
            'set is outside every number in this artifact.'],
    )


def schedclass_hypothesis(universe):
    """A CALIBRATED family hypothesis for opcodes the descriptor cannot classify.

    Apple's scheduling class is a real descriptor field and its members cluster functionally, so
    an unclassified opcode sharing a class with classified ones is evidence about its family -
    but only if that inference is actually predictive, and the honest way to know is to measure
    it. Leave-one-out over the 1,540 classified opcodes whose class supports a prediction:
    94.2% correct, abstaining on 2,173 - but 850 of the 1,540 sit in classes holding ONE family,
    where a held-out member cannot be missed. On the 690 in mixed classes it is 601 right, 87.1%,
    against 61.2% for predicting memory_addressing everywhere (and about 61% with the families
    shuffled across classes). 2026-09-23 figures; the artifact derives them. Compare the
    calibration that killed name prediction from a structural twin, which is right 0 times in 44.

    So this is published, and it is published as a HYPOTHESIS: it never sets `family`, it never
    feeds a denominator, and it carries its own accuracy and its own confusions so a reader can
    discount it correctly. The dominant confusion is atomic_sync read as memory_addressing, which
    is what a scheduling class would be expected to blur - atomics are memory operations.
    """
    known = {n: r for n, r in universe.items() if r['family'] != 'unknown'}
    by_class = defaultdict(Counter)
    for name, row in known.items():
        by_class[row['schedclass']][row['family']] += 1

    def predict(schedclass, exclude=None):
        counts = Counter(by_class.get(schedclass) or {})
        if exclude:
            counts[exclude] -= 1
        counts = Counter({k: v for k, v in counts.items() if v > 0})
        if sum(counts.values()) < 3:
            return None, counts
        family, count = counts.most_common(1)[0]
        if count/sum(counts.values()) < 0.8:
            return None, counts
        return family, counts

    right = wrong = abstained = 0
    # A HELD-OUT MEMBER OF A ONE-FAMILY CLASS CANNOT BE MISSED: every classmate is its family, so
    # leave-one-out predicts it by construction. Those predictions are counted apart, and the
    # accuracy on MIXED classes - where the gate let a prediction through and it could have been
    # wrong - is the discriminating figure.
    pure_right = mixed_right = mixed_wrong = 0
    confusions = Counter()
    for name, row in known.items():
        guess, _ = predict(row['schedclass'], exclude=row['family'])
        pure = len(by_class[row['schedclass']]) == 1
        if guess is None:
            abstained += 1
        elif guess == row['family']:
            right += 1
            if pure:
                pure_right += 1
            else:
                mixed_right += 1
        else:
            wrong += 1
            mixed_wrong += 1
            confusions['%s read as %s' % (row['family'], guess)] += 1
    base_family, base_count = (Counter(r['family'] for r in known.values()).most_common(1)
                               or [(None, 0)])[0]

    hypotheses = {}
    for name, row in universe.items():
        if row['family'] != 'unknown':
            continue
        guess, counts = predict(row['schedclass'])
        if guess is None:
            continue
        hypotheses[name] = dict(family_hypothesis=guess, schedclass=row['schedclass'],
                                classified_classmates=dict(counts))
    return dict(
        method='leave-one-out over classified opcodes; a class predicts only when at least three '
               'classified members agree at 80% or better, and abstains otherwise',
        calibrated_on=right + wrong, correct=right, wrong=wrong,
        accuracy_percent=round(100*right/(right + wrong), 1) if right + wrong else None,
        correct_in_one_family_classes=pure_right,
        calibrated_on_mixed_classes=mixed_right + mixed_wrong, correct_in_mixed_classes=mixed_right,
        accuracy_percent_mixed_classes=(round(100*mixed_right/(mixed_right + mixed_wrong), 1)
                                        if mixed_right + mixed_wrong else None),
        majority_family_baseline_percent=(round(100*base_count/len(known), 1)
                                          if known else None),
        accuracy_means=('%d of the %d calibrated predictions are members of a class holding ONE '
                        'family, which leave-one-out cannot get wrong. On the %d in mixed classes '
                        'it is %d right. Predicting %s for every opcode is right on %d of %d'
                        % (pure_right, right + wrong, mixed_right + mixed_wrong, mixed_right,
                           base_family, base_count, len(known))),
        abstained=abstained, confusions=dict(confusions.most_common(6)),
        opcodes_with_a_hypothesis=len(hypotheses),
        contrast=('name prediction from a structural twin is right 0 times in 44 on the verified '
                  'set, which is why names are only used where Apple RECORDS one. This inference '
                  'measures %s%% overall and %s%% on the classes where it could have failed, and '
                  'is still only a hypothesis: it never sets `family` and is counted in no '
                  'denominator'
                  % (round(100*right/(right + wrong), 1) if right + wrong else '-',
                     round(100*mixed_right/(mixed_right + mixed_wrong), 1)
                     if mixed_right + mixed_wrong else '-')),
        by_opcode=hypotheses)


def reference(spec, universe, report):
    """Generate the per-family reference. Documentation as OUTPUT, not as commentary.

    One section per family in investigation order, each stating what is established and - at the
    same length - what is not. The per-form detail lives in isa/g17.yaml; this is the document a
    person reads first, and its value is that nobody wrote it, so it cannot drift from the map.
    """
    totals = report['forms']['totals']
    out = ['# G17 ISA reference, by family',
           '',
           'GENERATED by `python3 tools/g17isamap.py --write` from `isa/g17.yaml`. Do not edit: '
           '`--check` fails if this file and a fresh harvest disagree. Per-form detail, every '
           'axis and every evidence level live in the YAML; this is the orientation document.',
           '',
           '| denominator | forms | what it requires |',
           '|---|---:|---|',
           '| D1 structural | %d | this decoder can address the form |' % totals['d1_structural'],
           '| D2 encoded | %d | an encoding Apple wrote, no unforced and no undetermined bits |'
           % totals['d2_encoded'],
           '| D3 verified class | %d | an isolated probe returned the value the name predicts |'
           % totals['d3_semantics_checked'],
           '| D3 ledger executed | %d | an executed-status ledger fact names the opcode |'
           % totals['d3_ledger_executed'],
           '| D3 probed here | %d | an isolated probe this lane authored returned a '
           'preregistered value |' % totals['d3_probed_here'],
           '| D3 fitted by elimination | %d | exactly one candidate of the published library '
           'survives every case of this form |' % totals['d3_fitted_by_elimination'],
           '| D3 csel predicate | %d | which comparison the opcode encodes, from a boolean '
           'output, with the comparison exercised BOTH ways above the case bar |'
           % totals['d3_csel_predicate'],
           '| D3 lane probe | %d | every lane read from a compiled Metal kernel with a '
           'thread-indexed destination (the isolated oracle reads every lane too since '
           '2026-09-23, through `per_lane` records; this row counts only the compiled probes) |'
           % totals['d3_lane_probe'],
           '| D3 transcendental | %d | reproduces the correctly-rounded value of its own '
           'Apple name to at most one ULP, with no rival operation within a ULP |'
           % totals['d3_transcendental'],
           '| D3 equivalence class | %d | its surviving candidates are ONE function at '
           'this declared width, so the residue is which library name to write |'
           % totals['d3_equivalence_class'],
           '| D3 family law | %d | a rule stated over a FAMILY reproduces this form\'s retained '
           'cases; hardware values, family-level attribution |' % totals['d3_family_law'],
           '| D3 own instruments (the nine above, never summed with the peer) | %d | any one of '
           'them |' % totals['d3_any'],
           '| D3 peer-reported | %d | another lane measured it; unmerged and unverified here |'
           % totals['d3_peer_reported'],
           '| dispatched, not in the verified class | %d | it ran and no ISOLATED one-opcode '
           'probe confirmed the name; %d of these DO carry other semantic evidence |'
           % (totals['executed_unchecked'], totals['executed_unchecked_with_other_evidence']),
           '| dispatched, no semantic evidence of any kind | %d | it ran and nothing - not this '
           'lane, not the ledger, not the peer - has read its output |'
           % totals['executed_with_no_semantic_column'],
           '| dispatched, nothing THIS lane has checked | %d | as above, but counting a '
           'peer-only form as unchecked here |'
           % totals['executed_with_no_LOCAL_semantic_column'],
           '',
           'THE D3 COLUMNS ARE TEN INSTRUMENTS AND ARE NEVER SUMMED into a completeness '
           'figure. They are mutually exclusive, so `D3 own instruments` is the count of forms '
           'carrying at least one of the nine local ones and is the only total worth quoting; '
           'the peer column is another lane\'s measurement and is excluded from it deliberately. '
           'The instruments differ in what they can get WRONG: the verified class asserts a name '
           'a probe confirmed, elimination asserts that one library candidate survived a form\'s '
           'own cases, and a family law asserts that a rule over the family reproduces them - '
           'which can be right about the family and wrong about a member whose encoding differs '
           'in something the law does not read. The csel predicate column rests on a BOOLEAN '
           'observable, the weakest here, so it counts a form only when the comparison was '
           'exercised both ways above the case bar - a predicate true at every probed case '
           'cannot be told from the constant true.',
           '']
    by_family = defaultdict(list)
    for name, forms in spec.items():
        by_family[universe[name]['family']].append((name, forms))
    for family in FAMILIES:
        rows = sorted(by_family.get(family, []))
        if not rows:
            out += ['## %s' % family, '',
                    '**No form is classified into this family.** That is a measurement, not an '
                    'omission: nothing Apple\'s descriptor declares, and no name it records, '
                    'distinguishes a member.', '']
            continue
        counts = report['forms']['by_family'].get(family, {})
        out += ['## %s' % family, '',
                '%d forms · D2 %d · D3 verified %d · D3 ledger %d · D3 peer %d · dispatched '
                'unchecked %d' % (counts.get('d1_structural', 0), counts.get('d2_encoded', 0),
                                  counts.get('d3_semantics_checked', 0),
                                  counts.get('d3_ledger_executed', 0),
                                  counts.get('d3_peer_reported', 0),
                                  counts.get('executed_unchecked', 0)), '']
        # the forms a reader should start from: strongest semantics first, then encoding
        def strength(item):
            name, forms = item
            best = max((RANK.get(f['semantics']['evidence'], 0) for f in forms.values()),
                       default=0)
            return (-best, -sum(1 for f in forms.values() if f['denominators']['d2_encoded']), name)
        out += ['| opcode | name | widths | semantics | evidence | unknown bits |',
                '|---|---|---|---|---|---:|']
        for name, forms in sorted(rows, key=strength)[:10]:
            width = 'len%d' % min(int(k[3:]) for k in forms)
            record = forms[width]
            unknown = record['unknown_bits'].get('count')
            out.append('| `%s` | %s | %s | %s | %s | %s |' % (
                name, universe[name]['name'] or '—',
                ' '.join(str(w) for w in sorted(int(k[3:]) for k in forms)),
                str(record['semantics']['state'])[:88].replace('|', '/'),
                record['semantics']['evidence'],
                'n/a' if unknown is None else unknown))
        if len(rows) > 10:
            out.append('')
            out.append('%d further forms in this family are in the YAML.' % (len(rows) - 10))
        absent = sorted({axis for _, forms in rows for f in forms.values()
                         for axis, v in f.items()
                         if isinstance(v, dict) and v.get('evidence') == 'absent'})
        if absent:
            out += ['', '**Unestablished for every form here:** %s.'
                    % ', '.join('`%s`' % a for a in absent)]
        out.append('')
    out += ['## What this document does not establish', '',
            'D1 is the surface this artifact\'s admission walk could reach, and it is an '
            'UNDERCOUNT by a known mechanism: %s are Apple table rows the decoder admits and the '
            'walk missed, because a selector bit that also changes an operand\'s register class '
            'cannot be reached one flip at a time. Nobody has counted how many other forms sit '
            'behind a two-bit selector, so this is not a bound. A family at D3 zero has no '
            'hardware-checked semantics from any instrument. Peer-reported rows are another '
            'lane\'s measurement, pinned at `%s`, unmerged and unverified in this repository.'
            % (', '.join('`op%d`' % o for o in sorted(WALK_UNDERCOUNT)), PEER_PIN[:12]), '']
    return '\n'.join(out)


try:
    from yaml import CSafeDumper as _Dumper       # libyaml, where the build has it
except ImportError:                                                      # pragma: no cover
    from yaml import SafeDumper as _Dumper


def render(value, as_yaml):
    """The spec is 34 MB of YAML now that D1 covers every admitted opcode, so the dumper is
    worth choosing rather than defaulting: CSafeDumper emits the identical document and
    `--check` regenerates it against the file on disk, which is what proves they agree."""
    return (yaml.dump(value, Dumper=_Dumper, sort_keys=True,
                      default_flow_style=False, width=100)
            if as_yaml else json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--check-committed', action='store_true',
                    help='ask the regenerability question of HEAD rather than the working tree')
    args = ap.parse_args()
    src, spec, universe = build()
    report = coverage(src, spec, universe)
    artifacts = ((SPEC, spec, True), (UNIVERSE, universe, False), (COVERAGE, report, False),
                 (LEDGER_INDEX, src['ledger'], False), (CLAIMS, src['claims'], False),
                 (PEER_CLAIMS, src['peer_claims'], False))
    generated = reference(spec, universe, report)
    if args.write:
        for path, value, is_yaml in artifacts:
            path.write_text(render(value, is_yaml))
        REFERENCE.write_text(generated)
        # Named from the list it just wrote. Three of the six were missing from this line, so a
        # regenerated artifact could go stale with the tool reporting success and not naming it.
        print('wrote %s (%d opcodes, %d forms), %s, and %s'
              % (SPEC.name, len(spec), report['forms']['totals']['d1_structural'],
                 ', '.join(path.name for path, _, _ in artifacts[1:]), REFERENCE.name))
    if args.check:
        bad = [path.name + (' absent' if not path.exists() else ' differs from a fresh harvest')
               for path, value, is_yaml in artifacts
               if not path.exists() or path.read_text() != render(value, is_yaml)]
        if not REFERENCE.exists() or REFERENCE.read_text() != generated:
            bad.append(REFERENCE.name + ' differs from a fresh harvest')
        if bad:
            raise SystemExit('specification is not regenerable: ' + '; '.join(bad))
        print('regenerable: all %d artifacts match a fresh harvest' % (len(artifacts) + 1))
    if args.check_committed:
        # THE GREEN GUARD AND THE WRONG COMMIT. `--check` reads the WORKING TREE, so a generated
        # artifact that is regenerated but never staged reports clean while the committed copy
        # still carries the old numbers - and the committed copy is what a reader of this branch
        # sees. docs/g17-isa-reference.md sat at HEAD publishing "D1 structural 7000" and
        # "dispatched, unchecked 455" while the repository's numbers were 7001 and 483, with
        # `--check` printing "all 7 artifacts match a fresh harvest" the whole time. The recurring
        # defect was "I forgot to regenerate before committing"; its real shape is that nothing
        # asked the question of the commit.
        import subprocess
        stale, unasked = [], []
        for path, value, is_yaml in list(artifacts) + [(REFERENCE, None, None)]:
            want = generated if path is REFERENCE else render(value, is_yaml)
            rel = path.relative_to(ROOT).as_posix()
            proc = subprocess.run(['git', '-C', str(ROOT), 'show', 'HEAD:' + rel],
                                  capture_output=True)
            if proc.returncode != 0:
                # Not tracked at HEAD, or not a git checkout at all. Named rather than skipped
                # silently: an unanswerable question must not read as a passing one.
                unasked.append(rel)
            elif proc.stdout.decode() != want:
                stale.append(rel)
        if unasked:
            print('not asked of HEAD (untracked there, or not a checkout): %s'
                  % ', '.join(unasked))
        if stale:
            raise SystemExit('COMMITTED artifacts are stale even though the working tree is '
                             'regenerable - commit the regeneration: ' + '; '.join(stale))
        if not unasked:
            print('committed: all %d artifacts at HEAD match a fresh harvest'
                  % (len(artifacts) + 1))
    if not (args.write or args.check):
        u = report['opcode_universe']
        print('OPCODE UNIVERSE')
        for k in ('apple_table_declares', 'decoder_admits', 'with_authoritative_length',
                  'apple_wrote_the_encoding', 'repair_walk_witness_only'):
            print('  %-28s %d' % (k, u[k]))
        print('  by evidence class            %s' % u['by_evidence_class'])
        t = report['forms']['totals']
        print('FORMS, THREE DENOMINATORS (never combined)')
        print('  D1 structural (addressable)  %d' % t['d1_structural'])
        print('  D2 encoding established      %d' % t['d2_encoded'])
        print('  D3 semantics, verified class %d' % t['d3_semantics_checked'])
        print('  D3 semantics, ledger executed %d' % t['d3_ledger_executed'])
        print('  D3 probed here on hardware %d' % t['d3_probed_here'])
        print('  D3 peer-reported (other lane) %d' % t['d3_peer_reported'])
        # Printed, never written into an artifact: it measures a branch that moves, so pinning it
        # would make the harvest un-regenerable for reasons outside this repository.
        _lag = peer_pin_lag()
        print('     peer pin %s, %s behind their pushed tip, %s unpushed beyond it'
              % (PEER_PIN[:8], _lag['behind_pushed_tip'],
                 _lag['pushed_tip_behind_their_local']))
        print('  D3 own instruments           %d' % t['d3_any'])
        _cand = ((report.get('d3_candidate_not_counted') or {})
                 .get('predicate_determined_forms') or {})
        if _cand:
            print('  D3 candidate, NOT counted    %d  (predicate elimination over a boolean '
                  'output; a sixth column is root\'s call)' % len(_cand))
        print('  dispatched, not verified class %d  (81 of these carry other D3 evidence)'
              % t['executed_unchecked'])
        print('  dispatched, no semantics at all %d  (this is the untouched population)'
              % t['executed_with_no_semantic_column'])
        print('  dispatched, none of it LOCAL   %d  (peer-only forms counted unchecked here)'
              % t['executed_with_no_LOCAL_semantic_column'])
        print('BY FAMILY (investigation order; unknown counted, not hidden)')
        print('  %-24s %5s %5s %4s %4s %4s %4s %6s'
              % ('family', 'D1', 'D2', 'D3v', 'D3l', 'D3h', 'D3p', 'exec'))
        for f in FAMILIES:
            row = report['forms']['by_family'].get(f)
            if row:
                print('  %-24s %5d %5d %4d %4d %4d %4d %6d'
                      % (f, row['d1_structural'], row['d2_encoded'],
                         row['d3_semantics_checked'], row['d3_ledger_executed'],
                         row['d3_probed_here'], row['d3_peer_reported'],
                         row['executed_unchecked']))
        fa = report['family_assignment']
        print('  opcode-level family unknown  %d of %d' % (fa['counted_unknown'], len(universe)))
        missing = [k for k, v in src['found'].items() if not v]
        un = report['unharvested_sources']
        print('SOURCES: %d harvested, %d absent' % (len(src['found']) - len(missing), len(missing)))
        print('  NOT harvested: ledger/ has %d files, %d naming an opcode; %d facts taken by hand'
              % (un['ledger_directory']['files'], un['ledger_directory']['files_naming_an_opcode'],
                 len(un['ledger_directory']['harvested_by_hand'])))
        print('  an axis reading `absent` may have evidence there - latency_wait and barrier_sync do')
    return 0


# PLACEMENT MATTERS AND I GOT IT WRONG ONCE: this block first sat at the END of the file,
# after the __main__ dispatch, so running the tool as a SCRIPT wrote the artifacts before
# reaching the append while IMPORTING it ran the append first. The committed artifact then
# disagreed with a fresh harvest, which is exactly what the regenerability test is for.
# Module-level mutation belongs above the dispatch that consumes it.

# THE ALU FORWARDING RULE IS CLASS-LEVEL, SO IT GOES HERE AND NOT ONTO EVERY ALU FORM. Applying a
# pipeline property to each of hundreds of forms is how an opcode-level fact becomes a form-level
# count, and this map has repaired that three times (D2 by 62 forms, D3's verified class by 49, the
# peer column by 2). latency_wait feeds no denominator, so nothing would have been inflated - but
# the shape is the defect, not the consequence.
GLOBAL_FACTS.append(dict(
    fact='ALU results forward immediately: there is no minimum producer-consumer separation, and '
         'no filler, spacing, scoreboard or wait encoding is needed to chain ALU operations',
    axis='latency_wait', evidence='checked',
    source='isa/g17-scalar-isa.toml [forwarding]',
    ledger_status='causal - an authored dependent pair is exact at 0, 1 and 2 intervening '
                  'instructions, adjacent included (r7 = r0 + 10 then r6 = r7 + 20 with r0 = 7 '
                  'gives 17 and 37)',
    why_unattributed='CLASS-LEVEL AND LEFT THAT WAY. The measurement is of the ALU pipeline, not '
                     'of any one opcode, and the file names no form. Attaching it to every ALU '
                     'form would state a per-form fact the source does not support - the same '
                     'shape as the opcode-level spread this map has had to undo three times. The '
                     'companion negative is recorded beside it: the file also says the COMPARE\'s '
                     'wait mechanism is unlocated, measured over eight data values all wrong in '
                     'the same way, so the compiler refuses a load operand there - forwarding '
                     'being free for the ALU does not extend to the compare'))

if __name__ == '__main__':
    raise SystemExit(main())
