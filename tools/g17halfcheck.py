#!/usr/bin/env python3
"""Check delivered FP16 scan addressing before any Metal call.

Structural metadata validity alone cannot establish the addresses a shader uses.
This gate checks decoded memory operands and never repairs or rewrites the image.
The CPU interpreter consumes decoded operands, not compiler SSA or allocation
tables. It does not model silicon scheduling or establish hardware correctness.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import g17packedcheck
import g17scalarabi
import g17scanlink
import g17verify
import g17scan


def check_memory(instructions, *, store_bytes=2, memory_layout="packed"):
    if memory_layout not in ("packed", "separate", "affine"):
        raise ValueError("unsupported memory layout")
    if store_bytes not in (2, 4):
        raise ValueError("unsupported control store width")
    loads = stores = 0
    for offset, size, opcode, operands in instructions:
        if opcode == 12646:
            loads += 1
            if size != 14 or len(operands) != 9:
                raise ValueError(f"half load at +{offset:#x}: unsupported instruction form")
            bases = {"expr:bin(op0,const(0),8)"}
            if memory_layout == "separate":
                bases.add("expr:bin(op0,const(4),8)")
            if operands[2] != "imm:2065" or operands[3] not in bases or operands[4] != "imm:0":
                raise ValueError(f"half load at +{offset:#x}: wrong width, binding, or displacement")
            if operands[6:] != ["imm:0", "imm:0", "imm:2"]:
                raise ValueError(f"half load at +{offset:#x}: scan requires zero offsets and two-byte elements")
            reg = int(operands[0].removeprefix("reg:"))
            if not operands[0].startswith("reg:") or not 425 <= reg < 553:
                raise ValueError("half load destination must be in the low half-register file")
        elif opcode == 17193 or (opcode == 17229 and store_bytes == 4):
            stores += 1
            if (opcode, size) != ((17193, 14) if store_bytes == 2 else (17229, 8)) or len(operands) != 9:
                raise ValueError(f"half store at +{offset:#x}: unsupported instruction form")
            if operands[7] != "imm:0":
                raise ValueError(f"half store at +{offset:#x} carries byte displacement {operands[7]}; "
                                 "scan requires zero, not the template's h[400 + row] displacement")
            base = 8 if memory_layout == "separate" else 4
            if operands[2:5] != [f"imm:{2065 if store_bytes == 2 else 2066}", f"expr:bin(op0,const({base}),8)", "imm:0"]:
                raise ValueError(f"half store at +{offset:#x}: wrong width, binding, or displacement")
            if operands[8] != f"imm:{store_bytes}":
                raise ValueError("store element size differs from the selected control width")
            reg = int(operands[0].removeprefix("reg:"))
            base = 425 if store_bytes == 2 else 105
            if not operands[0].startswith("reg:") or not base <= reg < base + 128:
                raise ValueError("store value is in the wrong register file")
        elif opcode in (12682, 17229, 17235):
            raise ValueError("FP32 device memory form cannot satisfy direct FP16 storage")
    if not loads or not stores:
        raise ValueError("scan must contain both half device loads and half device stores")
    return {"half_loads": loads, "half_stores": stores if store_bytes == 2 else 0,
            "word_stores": stores if store_bytes == 4 else 0, "memory_operand_check": "passed"}


class Registers:
    """Physical 32-bit storage with separately initialized low/high half lanes."""
    def __init__(self, count):
        self.count, self.values, self.defined = count, {}, {}

    @staticmethod
    def location(token):
        if not token.startswith("reg:"):
            raise ValueError("expected a register: " + token)
        number = int(token[4:])
        if 105 <= number < 233:
            return number - 105, 0, 0xFFFFFFFF
        if 425 <= number < 553:
            return number - 425, 0, 0xFFFF
        if 281 <= number < 409:
            return number - 281, 16, 0xFFFF
        raise ValueError("unmodeled register bank: " + token)

    def read(self, token):
        index, shift, mask = self.location(token)
        required = mask << shift
        if self.defined.get(index, 0) & required != required:
            raise ValueError("uninitialized register bits: " + token)
        return (self.values[index] >> np.uint32(shift)) & np.uint32(mask)

    def write(self, token, value):
        index, shift, mask = self.location(token)
        bits = np.broadcast_to(np.asarray(value, dtype=np.uint32), (self.count,)).copy()
        old = self.values.get(index, np.zeros(self.count, np.uint32))
        self.values[index] = ((old & np.uint32(0xFFFFFFFF ^ (mask << shift))) |
                              ((bits & np.uint32(mask)) << np.uint32(shift)))
        self.defined[index] = self.defined.get(index, 0) | (mask << shift)

    def float32(self, token):
        if self.location(token)[2] != 0xFFFFFFFF:
            raise ValueError("FP32 arithmetic requested a half-register operand")
        return self.read(token).view(np.float32)

    def half(self, token):
        if self.location(token)[2] != 0xFFFF:
            raise ValueError("FP16 operation requested a whole-register operand")
        return self.read(token).astype(np.uint16).view(np.float16)


def simulate(instructions, matrix, vector, *, rows, row_ids, store_bytes=2, memory_layout="packed"):
    """Interpret the peer's explicit widen/multiply/add/narrow scan on sampled rows."""
    check_memory(instructions, store_bytes=store_bytes, memory_layout=memory_layout)
    if store_bytes == 4 and (rows != 1 or list(row_ids) != [0] or np.shape(matrix) != (1, 1)):
        raise ValueError("word-store control is limited to exactly one thread and one column")
    a, x = np.asarray(matrix), np.asarray(vector)
    ids = np.asarray(row_ids, dtype=np.uint32)
    if (a.dtype != np.float16 or x.dtype != np.float16 or a.shape != (len(ids), len(x)) or
            not len(ids) or not len(x) or np.any(ids >= rows) or np.any(ids[:-1] >= ids[1:])):
        raise ValueError("expected sorted unique sample rows and matching half input arrays")
    if not np.isfinite(a).all() or not np.isfinite(x).all():
        raise ValueError("nonfinite inputs are outside the scan contract")
    regs = Registers(len(ids))
    result = None
    vector_base, columns = rows * len(x), len(x)
    cursor = 0

    def imm(token):
        if not token.startswith("imm:"):
            raise ValueError("expected an immediate: " + token)
        return int(token[4:])

    def load(indices, binding):
        offsets = indices.astype(np.uint64)
        if memory_layout == "separate" and binding == "expr:bin(op0,const(4),8)":
            if np.any(offsets >= columns):
                raise ValueError("query load addresses outside its separate allocation")
            return x[offsets.astype(int)].view(np.uint16).astype(np.uint32)
        if memory_layout != "packed" and np.any(offsets >= vector_base):
            raise ValueError("matrix load addresses outside its separate allocation")
        if np.any(offsets >= vector_base + columns):
            raise ValueError("half load addresses outside the packed allocation")
        values = np.empty(len(ids), np.float16)
        is_query = offsets >= vector_base
        values[is_query] = x[(offsets[is_query] - vector_base).astype(int)]
        physical_rows, ks = offsets[~is_query] // columns, offsets[~is_query] % columns
        local_rows = np.searchsorted(ids, physical_rows)
        if np.any(local_rows >= len(ids)) or not np.array_equal(ids[local_rows], physical_rows):
            raise ValueError("half load addresses a different row's matrix")
        values[~is_query] = a[local_rows, ks.astype(int)]
        return values.view(np.uint16).astype(np.uint32)

    for offset, size, opcode, t in instructions:
        if offset != cursor:
            raise ValueError("instruction boundaries are not consecutive")
        cursor += size
        if (opcode, size) == (14059, 4):
            if t[1:] != ["imm:1048576", "reg:61", "imm:0"]:
                raise ValueError("unsupported thread-id special register")
            regs.write(t[0], ids)
        elif (opcode, size) == (11842, 8):
            if t[1] != "imm:68736253952":
                raise ValueError("unsupported constant width")
            regs.write(t[0], imm(t[2]) & 0xFFFFFFFF)
        elif (opcode, size) == (10825, 14):
            if t[1] != "imm:0" or t[6] != "imm:0":
                raise ValueError("unsupported integer multiply modifiers")
            regs.write(t[0], regs.read(t[2]) * regs.read(t[4]))
        elif (opcode, size) == (10279, 12):
            if t[1] != "imm:0":
                raise ValueError("unsupported integer addition modifiers")
            regs.write(t[0], regs.read(t[3]) + np.uint32(imm(t[2]) & 0xFFFFFFFF))
        elif (opcode, size) == (12646, 14):
            if t[1] != "imm:137447342080":
                raise ValueError("unsupported half-load modifiers")
            regs.write(t[0], load(regs.read(t[5]), t[3]))
        elif (opcode, size) == (1004, 12):
            if t[1] != "imm:2147483648" or t[3] not in ("imm:16", "imm:32") or t[4] != "imm:128":
                raise ValueError("widening must add floating-point zero to a half source")
            value = regs.half(t[2]).astype(np.float32) + np.float32(0)
            regs.write(t[0], value.view(np.uint32))
        elif (opcode, size) in ((3290, 14), (998, 12)):
            modes = {"imm:137438953472", "imm:139586437120"} if opcode == 3290 else {"imm:146028888064"}
            if t[1] not in modes or t[3] not in ("imm:16", "imm:32") or t[5] not in ("imm:16", "imm:32"):
                raise ValueError("unsupported FP32 arithmetic modifiers")
            left, right = regs.float32(t[2]), regs.float32(t[4])
            value = left * right if opcode == 3290 else left + right
            regs.write(t[0], value.view(np.uint32))
        elif (opcode, size) == (1016, 12):
            if t[1] != "imm:2147483648" or t[3] not in ("imm:16", "imm:32") or t[4] != "imm:128":
                raise ValueError("unsupported half-narrowing modifiers")
            if regs.location(t[0])[2] != 0xFFFF:
                raise ValueError("narrowing must write a half register")
            with np.errstate(over="ignore"):
                value = regs.float32(t[2]).astype(np.float16)
            regs.write(t[0], value.view(np.uint16).astype(np.uint32))
        elif (opcode, size) == (17193, 14):
            if result is not None or not np.array_equal(regs.read(t[5]), ids):
                raise ValueError("half scan must store exactly once at each thread's row")
            result = regs.half(t[0]).copy()
        elif (opcode, size) == (17229, 8) and store_bytes == 4:
            if result is not None or not np.array_equal(regs.read(t[5]), ids):
                raise ValueError("word control must store exactly once at index zero")
            result = regs.float32(t[0]).copy()
        elif (opcode, size) == (684, 4) and t == ["imm:0"]:
            if offset != instructions[-1][0]:
                raise ValueError("early end instruction")
        else:
            raise ValueError(f"unmodeled half-scan instruction +{offset:#x}: op{opcode}/{size}")
    if result is None or instructions[-1][2] != 684 or not np.isfinite(result).all():
        raise ValueError("scan did not finish with finite half scores")
    return result


def check_arithmetic(instructions, rows, columns):
    ids = sorted({i for i in (0, 1, 2, 3, 7, 15, 16, 31, 32, rows - 2, rows - 1) if 0 <= i < rows})
    rng = np.random.default_rng(1729)
    checks = []
    for case in ("random", "cancellation", "subnormal", "basis", "signed_zero"):
        a = (rng.standard_normal((len(ids), columns)) * 0.2).astype(np.float16)
        x = (rng.standard_normal(columns) * 0.2).astype(np.float16)
        if case == "cancellation":
            a[:] = np.where(np.arange(columns) % 2, -0.125, 0.125)
            x[:] = 1
        elif case == "subnormal":
            a[:] = np.nextafter(np.float16(0), np.float16(1)); x[:] = 0.5
        elif case == "basis":
            a[:] = 0; a[:, -1] = 1
        elif case == "signed_zero":
            a[:] = np.float16(-0.0); x[:] = 1
        got = simulate(instructions, a, x, rows=rows, row_ids=ids)
        sequential = np.zeros(len(ids), np.float32)
        for k in range(columns):
            sequential = sequential + a[:, k].astype(np.float32) * np.float32(x[k])
        if not np.array_equal(got.view(np.uint16), sequential.astype(np.float16).view(np.uint16)):
            raise ValueError(f"{case}: interpreted half scores differ from sequential FP32 accumulation")
        numeric = g17scan.check_output(a, x, got)
        if not numeric["ok"]:
            raise ValueError(f"{case}: interpreted half scores violate the independent error bound")
        checks.append({"case": case, "half_bit_exact": True, **numeric})
    return {"arithmetic_check": "passed", "sampled_row_ids": ids, "cases": checks}


def check_bundle(directory, *, arch_flag=False):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    archive = (directory / "scan.arc.metallib").read_bytes()
    library = (directory / "scan.lib.metallib").read_bytes()
    if hashlib.sha256(archive).hexdigest() != manifest["sha256"]["archive"]:
        raise ValueError("archive differs from the recorded half image")
    obj = g17scanlink.verify_contract(archive, library, [(1, 0, False), (2, 2, True)])
    if hashlib.sha256(obj).hexdigest() != manifest["sha256"]["object"]:
        raise ValueError("delivered object differs from the recorded half image")
    code = g17verify._objsect(obj, "__text")[manifest["abi"]["entry"]:]
    while code.endswith(bytes.fromhex("0600")):
        code = code[:-2]
    if hashlib.sha256(code).hexdigest() != manifest["sha256"]["code"]:
        raise ValueError("delivered program differs from the recorded half image")
    # This compares the metadata bytes and delivered entry, not the scalar
    # compiler's restricted instruction set and not FP16 hardware semantics.
    g17scalarabi.verify_object(obj, code, arch_flag=arch_flag)
    instructions = g17packedcheck.decode(code)
    memory = check_memory(instructions)
    shape = manifest["shape"]
    return {"scope": "delivered operands interpreted on CPU; no GPU execution or scheduling model",
            "archive_sha256": manifest["sha256"]["archive"],
            **memory, **check_arithmetic(instructions, shape["rows"], shape["columns"])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    args = p.parse_args()
    try:
        report = {"status": "passed", **check_bundle(args.bundle)}
    except (ValueError, KeyError) as exc:
        report = {"status": "refused", "gpu_executed": False, "reason": str(exc)}
        print(json.dumps(report, indent=2))
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
