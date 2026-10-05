"""Interpret the query's delivered instructions on CPU, with explicit limits.

The decoded operands supply addresses, arithmetic and branch displacements.
This is a synchronous, uniform-loop model. It does not establish GPU mask-stack
behavior, lifetime semantics, asynchronous completion or loader compatibility.
"""
import ctypes
from pathlib import Path

import numpy as np


def check(decoded, manifest):
    """Check the full delivered query with real parameters and discriminating inputs."""
    import g17minilmquery as application
    import g17minilmquerycheck as validation
    rows, columns = manifest.shape.rows, manifest.shape.columns
    if columns != 384 or not 1 <= rows <= 128:
        raise ValueError("unsupported query launch dimensions")
    if tuple(manifest.abi.system_registers) != (160, 161):
        raise ValueError("query requires both delivered launch coordinates")
    bindings = [(b.index, b.offset, b.written) for b in manifest.abi.bindings]
    fixture = Path(__file__).resolve().parents[1] / "results/g17-minilm-query-admission/fixture.npz"
    with np.load(fixture, allow_pickle=False) as data:
        source = data["source"][np.arange(rows) % len(data["source"])].copy()
        weight, bias = data["weight"].copy(), data["bias"].copy()
    basis = np.zeros_like(source)
    basis[:, -1] = 2
    # Non-symmetric weights distinguish W from W.T; the last term distinguishes
    # the full reduction from a shortened loop or a duplicated body using old indices.
    control_weight = np.zeros_like(weight)
    control_weight[:, -1] = np.arange(1, 385, dtype=np.float32)
    cases = [("real", source, weight, bias),
             ("negative_source", -source, weight, bias),
             ("last_term", basis, control_weight, np.full_like(bias, .25)),
             ("bias_only", np.zeros_like(source), weight, -bias),
             ("repeated_real", source, weight, bias)]
    reports, first, memory = [], None, None
    for name, x, w, b in cases:
        output, execution = simulate(decoded, x, w, b, bindings)
        comparison = validation.compare(output, application.reference(x, w, b))
        if comparison["status"] != "passed":
            raise ValueError(f"delivered query arithmetic failed {name}: {comparison}")
        if name == "real":
            first, memory = output.tobytes(), execution
        elif name == "repeated_real" and output.tobytes() != first:
            raise ValueError("delivered query interpretation is not repeatable")
        reports.append(dict(case=name, reference=comparison, execution=execution))
    return memory, dict(cases=reports, gpu_dispatched=False,
                        scope="decoded CPU execution; hardware mask and timing behavior remain unproven")


def simulate(decoded, source, weight, bias, bindings):
    source, weight, bias = map(np.asarray, (source, weight, bias))
    if (source.ndim != 2 or weight.ndim != 2 or
            source.shape[1] != weight.shape[1] or bias.shape != (weight.shape[0],) or
            min(*source.shape, *weight.shape) < 1):
        raise ValueError("dense allocation shapes disagree")
    rows, input_width = source.shape
    columns = weight.shape[0]
    if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in (source, weight, bias)):
        raise ValueError("query requires finite FP32 inputs")
    if list(bindings) != [(1, 0, False), (2, 2, False), (3, 4, False), (4, 6, True)]:
        raise ValueError("unsupported query binding contract")
    instructions = {}
    cursor = 0
    for off, size, opcode, tokens in decoded:
        if off != cursor or off in instructions or size <= 0:
            raise ValueError("nonconsecutive delivered instruction boundaries")
        instructions[off] = (size, opcode, tokens)
        cursor += size
    if not decoded or decoded[-1][2] != 684:
        raise ValueError("query must end with END")
    count = rows * columns
    ids = np.arange(count, dtype=np.uint32)
    memory = [np.array(v, dtype=np.float32, order="C", copy=True).ravel().view(np.uint32)
              for v in (source, weight, bias)]
    memory.append(np.full(count, 0x7fc00001, np.uint32))
    writes = np.zeros(count, np.uint32)
    registers, pending = {}, set()
    condition = None
    pc, steps, loads, branches = 0, 0, 0, 0
    fmaf = ctypes.CDLL(None).fmaf
    fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    fmaf.restype = ctypes.c_float

    def read(token):
        if token not in registers:
            raise ValueError(f"uninitialized delivered register {token}")
        pending.discard(token)
        return registers[token]

    def assign(token, value, is_load=False):
        if token in pending:
            raise ValueError(f"unread load destination overwritten: {token} at +{pc:#x}")
        registers[token] = np.broadcast_to(np.asarray(value, np.uint32), (count,)).copy()
        if is_load:
            pending.add(token)

    # Input-derived host limit, not a proof that an arbitrary GPU loop terminates.
    while steps < len(decoded) * (input_width + 1):
        if pc not in instructions:
            raise ValueError("branch does not target a delivered instruction boundary")
        size, op, tokens = instructions[pc]
        regs = [t for t in tokens if t.startswith("reg:")]
        imms = [int(t[4:]) for t in tokens if t.startswith("imm:")]
        next_pc = pc + size
        steps += 1
        if (op, size) == (14059, 4):
            if len(regs) != 2 or regs[1] not in ("reg:61", "reg:62") or imms != [1048576, 0]:
                raise ValueError("unsupported delivered system-register read")
            assign(regs[0], ids % columns if regs[1] == "reg:61" else ids // columns)
        elif (op, size) == (11842, 8):
            if len(regs) != 1 or len(imms) != 2 or imms[0] != 68736253952:
                raise ValueError("unsupported delivered constant")
            assign(regs[0], imms[-1] & 0xffffffff)
        elif op in (10279, 10282) and size == 12:
            if op == 10279 and len(regs) == 2 and len(imms) == 3 and imms[0] == 0:
                value = read(regs[1]) + np.uint32(imms[1])
            elif op == 10282 and len(regs) == 3 and len(imms) == 3 and imms[0] == 0:
                value = read(regs[1]) + read(regs[2])
            else:
                raise ValueError("unsupported delivered integer add")
            assign(regs[0], value)
        elif (op, size) == (10825, 14):
            if len(regs) != 3 or len(imms) != 4 or imms[0] != 0 or imms[-1] != 0:
                raise ValueError("unsupported delivered integer multiply")
            assign(regs[0], read(regs[1]) * read(regs[2]))
        elif (op, size) in ((12682, 14), (17229, 8)):
            if len(tokens) != 9 or len(regs) != 2 or tokens[2] != "imm:2066" or tokens[4] != "imm:0" or tokens[8] != "imm:4":
                raise ValueError("unsupported delivered memory form")
            expressions = [f"expr:bin(op0,const({4*i}),8)" for i in range(4)]
            if tokens[3] not in expressions or not tokens[7].startswith("imm:"):
                raise ValueError("unmapped delivered binding or displacement")
            rank = expressions.index(tokens[3])
            displacement = int(tokens[7][4:])
            if displacement % 4:
                raise ValueError("unaligned delivered memory displacement")
            index = read(regs[1]).astype(np.int64) + displacement // 4
            if np.any(index < 0) or np.any(index >= len(memory[rank])):
                raise ValueError("delivered memory access exceeds allocation")
            if op == 12682:
                if rank == 3:
                    raise ValueError("query reads an uninitialized output")
                assign(regs[0], memory[rank][index], is_load=True)
                loads += count
            else:
                if not bindings[rank][2]:
                    raise ValueError("delivered store writes read-only input")
                memory[rank][index] = read(regs[0])
                np.add.at(writes, index, 1)
        elif (op, size) == (2190, 16):
            if len(regs) != 4 or len(imms) != 4 or imms[0] != 10737418240:
                raise ValueError("unsupported delivered FMA modifiers")
            a, b, c = (read(r).view(np.float32) for r in regs[1:])
            value = np.fromiter((fmaf(float(x), float(y), float(z)) for x, y, z in zip(a, b, c)),
                                np.float32, count=count)
            assign(regs[0], value.view(np.uint32))
        elif (op, size) == (998, 12):
            if len(regs) != 3 or len(imms) != 3 or imms[0] != 148176371712:
                raise ValueError("unsupported delivered float add modifiers")
            value = read(regs[1]).view(np.float32) + read(regs[2]).view(np.float32)
            assign(regs[0], value.view(np.uint32))
        elif (op, size) == (10369, 6):
            if len(regs) != 2 or len(imms) != 4 or imms[:3] != [0, 9, 0]:
                raise ValueError("unsupported delivered loop compare")
            condition = read(regs[1]) < imms[-1]
            registers[regs[0]] = condition.astype(np.uint32)
        elif (op, size) == (582, 4):
            if regs != ["reg:74"] or imms != [0, 1] or condition is None:
                raise ValueError("unsupported delivered loop mask")
            if not (np.all(condition) or not np.any(condition)):
                raise ValueError("divergent loop is outside this interpreter")
        elif (op, size) == (577, 4):
            if tokens != ["imm:2345052143616", "imm:1"]:
                raise ValueError("unsupported delivered mask restore")
        elif (op, size) == (458, 10):
            if len(imms) != 2 or imms[0] != 0 or imms[1] >= 0 or condition is None:
                raise ValueError("unsupported delivered back edge")
            branches += 1
            if np.all(condition):
                next_pc = pc + imms[1]
            condition = None
        elif (op, size) == (684, 4):
            if next_pc != cursor or tokens != ["imm:0"] or not np.all(writes == 1):
                raise ValueError("delivered program did not write every output exactly once then end")
            return memory[3].view(np.float32).reshape(rows, columns).copy(), dict(
                steps=steps, loads=loads, stores=int(writes.sum()), branches=branches,
                scope="synchronous CPU model of decoded operands and uniform loop; not GPU ordering proof")
        else:
            raise ValueError(f"unsupported delivered instruction op{op}/{size} at +{pc:#x}")
        pc = next_pc
    raise ValueError("delivered execution exceeds bounded interpreter steps")
