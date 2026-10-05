"""Admit delivered LayerNorm instructions through the compiler owner's interpreter.

This module supplies application inputs, allocation limits and an independent
reference. Instruction semantics stay in g17normcheck; no alternate interpreter
or application-specific instruction repair is installed here.
"""
from pathlib import Path

import numpy as np

import g17layernorm
import g17layernormvalidate as validation
import g17normcheck as machine

ROOT = Path(__file__).resolve().parents[1]


def check_load_reuse(decoded):
    """Refuse unread asynchronous destinations that the interpreter cannot model.

    The retained 32x384 hardware failure contains dead original loads followed
    by another load into the same destination. Machine.step completes loads
    synchronously, so its correct arithmetic does not validate this sequence.
    Until the compiler supplies a measured ordering rule, this admission path
    requires a load's destination to be read before another instruction writes
    it. This is a conservative domain restriction, not an ISA-wide claim that
    every such overwrite is invalid. Nor does a read prove all timing is valid.
    """
    pending = {}
    loads = 0
    store_ops = machine.STORES32 | {machine.TG_STORE} | set(machine.RANGE_STORES)
    load_ops = {machine.LOAD32, machine.TG_LOAD}
    for offset, _size, opcode, tokens in decoded:
        if opcode not in machine.ARITH_FP | machine.STRUCTURAL:
            raise ValueError(f"LayerNorm load-reuse check cannot classify op{opcode} at +{offset:#x}")
        registers = machine._regs(tokens)
        uses = registers if opcode in store_ops else registers[1:]
        if opcode == machine.READ_SR:
            uses = []  # Its second printed register names SR state, not a GPR read.
        for register in uses:
            pending.pop(register, None)
        if registers and opcode not in store_ops:
            destination = registers[0]
            if destination in pending:
                raise ValueError(
                    "LayerNorm has unresolved asynchronous load reuse: "
                    f"load at +{pending[destination]:#x} writes {destination}, "
                    f"then op{opcode} at +{offset:#x} overwrites it before a read; "
                    "synchronous delivered-code interpretation does not establish write ordering")
            if opcode in load_ops:
                pending[destination] = offset
                loads += 1
        if opcode == machine.END:
            break
    return dict(loads=loads, unread_destination_overwrites=0,
                scope="conservative admission restriction; not a complete asynchronous timing model")


def check(decoded, manifest):
    if not callable(getattr(machine, "simulate_rows", None)):
        raise ValueError("LayerNorm image structure passed, but the committed instruction checker "
                         "does not yet interpret whole matrices (g17normcheck.simulate_rows)")
    load_reuse = check_load_reuse(decoded)
    rows, columns = manifest.shape.rows, manifest.shape.columns
    bindings = [(b.index, b.offset, b.written) for b in manifest.abi.bindings]
    sizes = [rows*columns, columns, columns, rows*columns]
    memory = machine.check_operands(decoded, bindings, sizes, rows=rows)
    if memory.get("stores") != rows*columns:
        raise ValueError("delivered LayerNorm must store every output exactly once")
    refusals = []
    for rank in range(4):
        short = sizes.copy()
        short[rank] -= 1
        try:
            machine.check_operands(decoded, bindings, short, rows=rows)
        except ValueError as error:
            refusals.append(dict(rank=rank,words=short[rank],error=str(error)))
        else:
            raise ValueError(f"LayerNorm checker accepts an undersized buffer at rank {rank}")

    # Read the same repository-owned MiniLM parameters used by hardware
    # validation. Smaller stages select a prefix; larger staging row counts
    # repeat real token rows, without changing the compiled dimensions.
    with np.load(ROOT/"results/g17-layernorm-fixtures-v1/minilm_embeddings.npz",allow_pickle=False) as data:
        source = data["source"][np.arange(rows) % len(data["source"]),:columns].copy()
        gamma,beta = (data[k][:columns].copy() for k in ("gamma","beta"))
    cases = [(name,x,gamma,beta) for name,x in validation.cases(source)]
    cases += [("zero_gamma_changed_beta",source,np.zeros_like(gamma),beta+np.float32(.25)),
              ("changed_gamma_zero_beta",source,-gamma,np.zeros_like(beta))]
    reports, first = [], None
    for name,x,g,b in cases:
        output,confidence,notes = machine.simulate_rows(decoded,x,g,b,g17layernorm.EPSILON,
                                                       bindings=bindings)
        report,_ = validation.compare(output,x,g,b)
        if not report["ok"]:
            raise ValueError(f"delivered LayerNorm arithmetic differs from FP64 reference: {name}: {report}")
        # UNVERIFIED IS NOT REFUTED, AND A BOUND IS EVIDENCE. "bounded" means every off-domain
        # opcode carries a measured magnitude (g17normcheck.MEASURED_BOUND) and the numerical
        # comparison above holds within the budget; it is admitted and named in the report.
        # "isolation" and "unverified" - an opcode with no measured meaning or bound - still refuse.
        if confidence not in ("executed", "bounded"):
            raise ValueError(f"delivered LayerNorm has unresolved instruction evidence: {confidence}: {notes}")
        if name == "real":
            first = output.tobytes()
        elif name == "repeated_real" and output.tobytes() != first:
            raise ValueError("delivered LayerNorm interpretation is not repeatable")
        reports.append(dict(case=name,reference=report,instruction_evidence=confidence,
                            instruction_notes=notes))
    return memory, dict(cases=reports,undersized_buffer_controls=refusals,load_reuse=load_reuse,
        scope="delivered instruction interpretation against FP64; not GPU execution")
