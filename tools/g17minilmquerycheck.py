"""CPU execution of MiniLM query application IR; no native-code or GPU claim."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

import g17ir as ir
import g17minilmquery as query

ERROR_FACTOR = 2e-5


def evaluate(function, source, weight, bias):
    """Execute the actual CFG over all grid points, including simultaneous phis.

    Host fmaf supplies one FP32 rounding for each fused operation. This models
    application semantics, not GPU scheduling, denormal modes or emitted bytes.
    The finite step cap is an interpreter guard, not a GPU termination proof.
    """
    source, weight, bias = map(np.asarray, (source, weight, bias))
    if (source.ndim != 2 or weight.ndim != 2 or
            source.shape[1] != weight.shape[1] or bias.shape != (weight.shape[0],) or
            min(*source.shape, *weight.shape) < 1):
        raise ValueError("dense application shapes disagree")
    if any(v.dtype != np.float32 or not np.isfinite(v).all()
           for v in (source, weight, bias)):
        raise ValueError("dense application requires finite FP32 inputs")
    ir.verify(function)
    rows, input_width = source.shape
    width = weight.shape[0]
    count = rows * width
    if [(b.name, b.slot) for b in function.buffers] != [
            ("source", 1), ("weight", 2), ("bias", 3), ("output", 4)]:
        raise ValueError("unexpected application binding list")
    memory = {b: np.array(v, dtype=np.float32, order="C", copy=True).ravel().view(np.uint32)
              for b, v in zip(function.buffers[:3], (source, weight, bias))}
    output = np.full(count, 0x7fc00001, np.uint32)
    written = np.zeros(count, np.uint32)
    values = {}
    visits = {}
    block = function.blocks[0]
    fmaf = ctypes.CDLL(None).fmaf
    fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    fmaf.restype = ctypes.c_float

    def bits(value):
        if isinstance(value, ir.Imm):
            return np.full(count, value.v, np.uint32)
        return values[value]

    def index(value, size):
        got = bits(value)
        if np.any(got >= size):
            raise ValueError("application IR accesses outside its declared buffer")
        return got

    for _ in range(input_width + 4):
        seen = visits.get(block, 0)
        visits[block] = seen + 1
        # All incoming edges observe the old values, even when phis cross-reference.
        incoming = {op.dest: bits(op.args[1 if seen else 0]).copy()
                    for op in block.ops if op.kind == "phi"}
        values.update(incoming)
        for op in block.ops:
            args, kind = op.args, op.kind
            if kind == "phi":
                continue
            if kind == "const":
                result = bits(args[0])
            elif kind == "builtin":
                if op.attrs.get("which") != "thread_position_in_grid":
                    raise ValueError("unsupported builtin")
                axis = op.attrs.get("axis")
                if axis not in ("x", "y"):
                    raise ValueError("unsupported grid axis")
                ids = np.arange(count, dtype=np.uint32)
                result = ids % width if axis == "x" else ids // width
            elif kind in ("add", "mul"):
                a, b = map(bits, args)
                result = a + b if kind == "add" else a * b
            elif kind == "load":
                # the addressing keys exactly, an unknown key still refuses, and `form_length` (integration's
                # op12682/8 lowering) admitted at its witnessed values: it selects the instruction's LENGTH,
                # not its address, so this reference's arithmetic is unaffected
                attrs = dict(op.attrs); length = attrs.pop("form_length", None)
                if attrs != dict(offset=0, disp=0, scale=1, width="word", shift16=False) or length not in (None, 8, 14):
                    raise ValueError("unsupported application load")
                if args[0] not in memory:
                    raise ValueError("unexpected application load buffer")
                data = memory[args[0]]
                result = data[index(args[1], len(data))].copy()
            elif kind == "fma":
                a, b, c = (bits(v).view(np.float32) for v in args)
                result = np.fromiter((fmaf(float(x), float(y), float(z))
                                      for x, y, z in zip(a, b, c)),
                                     dtype=np.float32, count=count).view(np.uint32)
            elif kind == "fadd":
                a, b = (bits(v).view(np.float32) for v in args)
                result = (a + b).view(np.uint32)
            elif kind == "cmp":
                if op.attrs.get("pred") != "lt":
                    raise ValueError("unsupported comparison")
                result = (bits(args[0]) < bits(args[1])).astype(np.uint32)
            elif kind == "br":
                block = args[0]
                break
            elif kind == "br_cond":
                condition = bits(args[0]) != 0
                if not (np.all(condition) or not np.any(condition)):
                    raise ValueError("application interpreter requires uniform loop control")
                block = args[1] if np.all(condition) else args[2]
                break
            elif kind == "store_at":
                if args[0] is not function.buffers[3] or op.attrs != dict(width="word"):
                    raise ValueError("application store targets a read-only buffer or unsupported width")
                destination = index(args[1], count)
                output[destination] = bits(args[2])
                np.add.at(written, destination, 1)
                continue
            elif kind == "ret":
                if not np.all(written == 1):
                    raise ValueError("application IR must write every output exactly once")
                return output.view(np.float32).reshape(rows, width).copy()
            else:
                raise ValueError(f"unsupported application operation {kind}")
            values[op.dest] = np.array(result, dtype=np.uint32, copy=True)
        else:
            raise ValueError("application block lacks a terminator")
    raise ValueError("application interpreter exceeded its block-visit limit")


def compare(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("result shape differs from reference")
    error = np.abs(actual.astype(np.float64) - expected)
    finite = np.isfinite(actual) & np.isfinite(expected)
    failures = int(np.count_nonzero(~finite | (error > ERROR_FACTOR * (1 + np.abs(expected)))))
    return dict(status="passed" if failures == 0 else "failed", failures=failures,
                outputs_checked=int(actual.size), max_absolute_error=float(np.max(error)),
                error_budget="2e-5 * (1 + abs(FP64 reference))")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).resolve().parents[1] /
                        "results/g17-minilm-query-admission/fixture.npz")
    parser.add_argument("--campaign", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    with np.load(args.fixture, allow_pickle=False) as data:
        x, w, b = (data[k] for k in ("source", "weight", "bias"))
    if args.out is not None and args.out.exists():
        raise FileExistsError(args.out)
    cases = [("real", x)]
    if args.campaign:
        basis = np.zeros_like(x)
        basis[:, -1] = 1
        cases += [("negated", -x), ("repeated_real", x.copy()),
                  ("zeros", np.zeros_like(x)), ("last_element_basis", basis),
                  ("reversed_rows", x[::-1].copy())]
    reports = []
    first = None
    for name, value in cases:
        result = evaluate(query.query_projection_ir(*value.shape), value, w, b)
        measured = compare(result, query.reference(value, w, b))
        measured.update(case=name, output_sha256=hashlib.sha256(result.tobytes()).hexdigest())
        if first is None:
            first = result.copy()
        if name == "repeated_real":
            measured["repeat_bits_equal"] = bool(np.array_equal(result.view(np.uint32), first.view(np.uint32)))
            if not measured["repeat_bits_equal"]:
                measured["status"] = "failed"
        reports.append(measured)
    report = dict(status="passed" if all(r["status"] == "passed" for r in reports) else "failed",
                  cases=reports, outputs_checked=sum(r["outputs_checked"] for r in reports),
                  fixture_sha256=query._file_sha(args.fixture),
                  source_sha256={str(Path("tools") / p.name): query._file_sha(p) for p in
                                 (Path(__file__), Path(query.__file__), Path(ir.__file__))})
    report.update(scope="application IR interpreted on CPU, not emitted G17 or GPU execution",
                  gpu_dispatched=False)
    print(json.dumps(report, indent=2))
    if args.out is not None:
        with args.out.open("x") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
