"""MiniLM LayerNorm workload and CPU-only compiler/linker admission probe.

The target is the FP32, width-384 LayerNorm in spike/serve/mix_mlx.py.
This module owns application IR and reference math, not compiler or image rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import traceback

EPSILON = 1e-12


def float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def layernorm_ir(rows=32, columns=384, epsilon=EPSILON):
    """One thread per row; explicit FP32 operations and four real bindings.

    Keep each loaded value through mean subtraction, as the straightforward IR
    requires. Register-pressure failures belong to the backend and are retained.
    """
    import g17ir as ir
    if type(rows) is not int or type(columns) is not int or not 1 <= rows <= 128 or not 1 <= columns <= 384:
        raise ValueError("LayerNorm probe dimensions must fit 128 x 384")
    if not 0 < epsilon < 1:
        raise ValueError("LayerNorm requires a positive epsilon below one")
    source, gamma, beta, output = [ir.Buffer(name, slot, elem=ir.F32) for slot, name
                                   in enumerate(("source", "gamma", "beta", "output"), 1)]
    function = ir.Function("minilm_layernorm", [source, gamma, beta, output])
    b = ir.Builder(function, function.block("entry"))
    row = b.builtin("thread_position_in_grid")
    base = b.mul(row, b.const(columns))
    indices = [base if k == 0 else b.add(base, ir.Imm(k)) for k in range(columns)]
    values = [b.load(source, index, width="word") for index in indices]
    # LayerNorm is invariant under a common row offset. Center around a real
    # input before accumulating the mean so large offsets do not erase the
    # small variations in FP32. This is application math, not an encoding fix.
    negative_anchor = b.fmul(values[0], b.const(float_bits(-1.0)))
    shifted = [b.fadd(value, negative_anchor) for value in values]
    total = b.const(float_bits(0.0))
    for value in shifted:
        total = b.fadd(total, value)
    reciprocal_width = b.const(float_bits(1.0 / columns))
    mean = b.fmul(total, reciprocal_width)
    negative_mean = b.fmul(mean, b.const(float_bits(-1.0)))
    centered = [b.fadd(value, negative_mean) for value in shifted]
    squared_sum = b.const(float_bits(0.0))
    for value in centered:
        squared_sum = b.fadd(squared_sum, b.fmul(value, value))
    variance = b.fmul(squared_sum, reciprocal_width)
    inverse_std = b.rsqrt(b.fadd(variance, b.const(float_bits(epsilon))))
    for k, (index, value) in enumerate(zip(indices, centered)):
        parameter_index = b.const(k)
        scale = b.load(gamma, parameter_index, width="word")
        bias = b.load(beta, parameter_index, width="word")
        normalized = b.fmul(value, inverse_std)
        result = b.fadd(b.fmul(normalized, scale), bias)
        b.store_at(output, index, result, width="word")
    b.ret()
    return function


def reference(source, gamma, beta, epsilon=EPSILON):
    """Independent FP64 LayerNorm over exact FP32 inputs; returns every output."""
    import numpy as np
    x, g, b = (np.asarray(v) for v in (source, gamma, beta))
    if x.ndim != 2 or g.shape != (x.shape[1],) or b.shape != g.shape:
        raise ValueError("LayerNorm input/parameter shapes disagree")
    if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in (x, g, b)):
        raise ValueError("LayerNorm inputs must be finite FP32 arrays")
    if not 0 < epsilon < 1:
        raise ValueError("LayerNorm requires a positive epsilon below one")
    wide = x.astype(np.float64)
    centered = wide - wide.mean(axis=1, keepdims=True)
    variance = np.mean(centered * centered, axis=1, keepdims=True)
    return centered / np.sqrt(variance + float(epsilon)) * g.astype(np.float64) + b.astype(np.float64)


def immediate_ir(offset, *, materialized=False):
    """Minimal selector reproducer; the materialized variant is a diagnostic control.

    The real LayerNorm continues to use its original IR until its owner fixes
    lowering. No workaround is substituted into the application.
    """
    import g17ir as ir
    output = ir.Buffer("output", 1, elem=ir.I32)
    function = ir.Function("layernorm_index_literal", [output])
    b = ir.Builder(function, function.block("entry"))
    row = b.builtin("thread_position_in_grid")
    rhs = b.const(offset) if materialized else ir.Imm(offset)
    value = b.add(row, rhs)
    b.store_at(output, row, value, width="word")
    b.ret()
    return function


def probe_immediate(offset, materialized=False):
    report = dict(workload="layernorm_index_literal", offset=offset,
                  materialized_control=materialized, gpu_dispatched=False,
                  pipeline_created=False, stage="compiler", status="pending")
    try:
        import g17cc
        program = g17cc.compile_function(immediate_ir(offset, materialized=materialized))
        report.update(status="compiled_unvalidated", code_size=len(program.code),
                      code_sha256=hashlib.sha256(program.code).hexdigest(),
                      abi=program.abi_plain(program.abi()))
    except Exception as error:
        report.update(status="refused", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    return report


def probe(rows, columns):
    """Attempt real compilation and image authoring; retain the first refusal."""
    report = dict(workload="minilm_layernorm", rows=rows, columns=columns,
                  storage="float32", epsilon=EPSILON, gpu_dispatched=False,
                  pipeline_created=False, status="pending", stage="application_ir")
    try:
        function = layernorm_ir(rows, columns)
        report["ir_operations"] = sum(len(block.ops) for block in function.blocks)
        import g17cc
        report["stage"] = "compiler"
        program = g17cc.compile_function(function)
        report.update(code_size=len(program.code), code_sha256=hashlib.sha256(program.code).hexdigest(),
                      code_hex=program.code.hex(), instructions=program.contract().to_dict()["instructions"])
        report["stage"] = "compiler_abi"
        abi = program.abi_plain(program.abi())
        report["abi"] = abi
        import g17link
        import g17scanlink
        report["stage"] = "image_authoring"
        kernel = g17link.Kernel(code=program.code, name=program.name,
            entry=abi["entry"], prologue=bytes.fromhex(abi["prologue"]),
            bindings=[g17link.Binding(index=b["index"], readonly=not b["written"],
                                      element_type=b["element_type"]) for b in abi["bindings"]])
        image = g17scanlink.link(kernel, abi, binding_offsets=[b["offset"] for b in abi["bindings"]])
        report.update(status="authored_unvalidated", field_ledger=image.field_ledger,
                      archive_sha256=hashlib.sha256(image.archive).hexdigest())
    except Exception as error:
        report.update(status="refused", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--columns", type=int, default=384)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--index-repro", type=int)
    parser.add_argument("--materialized-control", action="store_true")
    parser.add_argument("--verify-source", action="store_true")
    args = parser.parse_args()
    # This probe must expose decoder dependence too, rather than silently use a
    # fallback to choose executable bytes for a newly exercised form.
    def no_process(event, values):
        if event in ("subprocess.Popen", "os.exec", "os.posix_spawn", "os.system"):
            raise RuntimeError("LayerNorm construction attempted an external process")
    if args.materialized_control and args.index_repro is None:
        parser.error("--materialized-control requires --index-repro")
    def attempt():
        return (probe(args.rows, args.columns) if args.index_repro is None else
                probe_immediate(args.index_repro, args.materialized_control))
    if args.verify_source:
        import g17buildaudit
        report, source = g17buildaudit.verified_build(Path(__file__).resolve().parents[1], attempt)
        report["source"] = source
    else:
        sys.addaudithook(no_process)
        report = attempt()
    encoded = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return 2 if report["status"] == "refused" else 0


if __name__ == "__main__":
    raise SystemExit(main())
