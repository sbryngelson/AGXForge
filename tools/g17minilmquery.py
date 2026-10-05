#!/usr/bin/env python3
"""MiniLM layer-0 query projection and native-G17 admission probe.

This is the application stage immediately following embeddings.LayerNorm in
spike/serve/mix_mlx.py::embed_fn.  It owns the unchanged application operation

    query = x @ attention.self.query.weight.T + attention.self.query.bias

for 32 token rows of width 384.  Compiler, ABI, image and runtime rules remain
in their owning modules; this file exposes the real stage to each boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import traceback

import numpy as np

ROWS = 32
WIDTH = 384
WEIGHT_NAME = "encoder.layer.0.attention.self.query.weight"
BIAS_NAME = "encoder.layer.0.attention.self.query.bias"


def query_projection_ir(rows=ROWS, width=WIDTH, packed_tile_columns=None):
    """One output per (column,row) grid point and a bounded K reduction."""
    import g17ir as ir

    if type(rows) is not int or type(width) is not int or not 1 <= rows <= 128 or width != WIDTH:
        raise ValueError("MiniLM query projection requires 1..128 rows and width 384")
    if packed_tile_columns is not None:
        if type(packed_tile_columns) is not int or not 1 <= packed_tile_columns <= width:
            raise ValueError("packed query tile requires 1..384 output columns")
        source, weight, output = [ir.Buffer(name, slot, elem=ir.F32)
                                  for slot, name in enumerate(
                                      ("source", "packed_weight_bias", "output"), 1)]
        bias = weight
        function = ir.Function("minilm_layer0_query_packed_tile", [source, weight, output])
    else:
        source, weight, bias, output = [ir.Buffer(name, slot, elem=ir.F32)
                                        for slot, name in enumerate(
                                            ("source", "weight", "bias", "output"), 1)]
        function = ir.Function("minilm_layer0_query", [source, weight, bias, output])
    pre, loop, end = (function.block(name) for name in ("pre", "loop", "exit"))
    b = ir.Builder(function, pre)

    column = b.builtin("thread_position_in_grid", axis="x", name="column")
    row = b.builtin("thread_position_in_grid", axis="y", name="row")
    stride = b.const(width, name="stride")
    source_base = b.mul(row, stride, name="source_base")
    weight_base = b.mul(column, stride, name="weight_base")
    zero = b.const(0, name="k0")
    zero_float = b.const(_float_bits(0.0), name="sum0")
    b.br(loop)

    b.at(loop)
    k = b.phi(zero, name="k")
    total = b.phi(zero_float, type=ir.F32, name="total")
    x_index = b.add(source_base, k, name="x_index")
    w_index = b.add(weight_base, k, name="w_index")
    x = b.load(source, x_index, width="word", name="x")
    w = b.load(weight, w_index, width="word", name="w")
    next_total = b.fma(x, w, total, name="next_total")
    next_k = b.add(k, ir.Imm(1), name="next_k")
    ir.Builder.phi_latch(k, next_k)
    ir.Builder.phi_latch(total, next_total)
    b.br_cond(b.cmp(next_k, width, "lt", name="more"), loop, end)

    b.at(end)
    offset = b.add(source_base, column, name="output_index")
    bias_index = (b.add(column, ir.Imm(packed_tile_columns * width))
                  if packed_tile_columns is not None else column)
    shift = b.load(bias, bias_index, width="word", name="bias")
    result = b.fadd(next_total, shift, name="result")
    b.store_at(output, offset, result, width="word")
    b.ret()
    return function


def loop_bound_ir(bound):
    """Minimal form of the query reduction's first compiler boundary."""
    import g17ir as ir

    if type(bound) is not int or bound < 1:
        raise ValueError("loop bound must be a positive integer")
    output = ir.Buffer("output", 1, elem=ir.I32)
    function = ir.Function("minilm_query_loop_bound", [output])
    pre, loop, end = (function.block(name) for name in ("pre", "loop", "exit"))
    b = ir.Builder(function, pre)
    zero = b.const(0, name="zero")
    b.br(loop)
    b.at(loop)
    k = b.phi(zero, name="k")
    next_k = b.add(k, ir.Imm(1), name="next_k")
    ir.Builder.phi_latch(k, next_k)
    b.br_cond(b.cmp(next_k, bound, "lt", name="more"), loop, end)
    b.at(end)
    b.store(output, ir.Imm(4), next_k)
    b.ret()
    return function


def reference(source, weight, bias):
    """Independent FP64 evaluation of the exact FP32 inputs and parameters."""
    x, w, b = (np.asarray(value) for value in (source, weight, bias))
    if x.ndim != 2 or x.shape[1] != WIDTH or w.shape != (WIDTH, WIDTH) or b.shape != (WIDTH,):
        raise ValueError("MiniLM query projection shapes disagree")
    if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in (x, w, b)):
        raise ValueError("MiniLM query projection requires finite FP32 inputs")
    return x.astype(np.float64) @ w.astype(np.float64).T + b.astype(np.float64)


def read_model_parameters(model_path):
    """Read only layer-0 query weight and bias from a local safetensors file."""
    model_path = Path(model_path)
    with model_path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if header_size > 16 * 1024 * 1024:
            raise ValueError("unexpected safetensors header size")
        header = json.loads(stream.read(header_size))
        data_base = 8 + header_size

        def read(name, shape):
            spec = header[name]
            if spec["dtype"] != "F32" or tuple(spec["shape"]) != shape:
                raise ValueError(f"unexpected MiniLM tensor {name}: {spec}")
            begin, finish = spec["data_offsets"]
            stream.seek(data_base + begin)
            raw = stream.read(finish - begin)
            if len(raw) != int(np.prod(shape)) * 4:
                raise ValueError(f"truncated MiniLM tensor {name}")
            return np.frombuffer(raw, dtype="<f4").copy().reshape(shape)

        return read(WEIGHT_NAME, (WIDTH, WIDTH)), read(BIAS_NAME, (WIDTH,))


def build_fixture(layernorm_result, model_path, destination):
    """Freeze the hardware-returned LayerNorm output and matching model parameters."""
    layernorm_result, destination = Path(layernorm_result), Path(destination)
    with np.load(layernorm_result, allow_pickle=False) as data:
        source = np.asarray(data["result"], dtype=np.float32)
    if source.shape != (ROWS, WIDTH):
        raise ValueError("the application fixture must be the validated 32x384 LayerNorm output")
    weight, bias = read_model_parameters(model_path)
    expected = reference(source, weight, bias)
    if destination.exists():
        raise FileExistsError(destination)
    np.savez(destination, source=source, weight=weight, bias=bias, reference=expected)
    return {
        "fixture": str(destination),
        "fixture_sha256": _file_sha(destination),
        "layernorm_result": str(layernorm_result),
        "layernorm_result_sha256": _file_sha(layernorm_result),
        "model": str(model_path),
        "model_sha256": _file_sha(Path(model_path)),
        "outputs": int(expected.size),
    }


def compile_probe(rows=ROWS, width=WIDTH):
    report = {
        "application": "MiniLM layer-0 attention query projection",
        "operation": "x @ weight.T + bias",
        "shape": {"rows": rows, "input": width, "output": width},
        "gpu_dispatched": False,
        "pipeline_created": False,
        "stage": "compiler",
    }
    try:
        import g17cc
        program = g17cc.compile_function(query_projection_ir(rows, width))
        report.update(status="compiled_unvalidated", code_size=len(program.code),
                      code_sha256=hashlib.sha256(program.code).hexdigest(),
                      abi=program.abi_plain(program.abi()))
    except Exception as error:
        report.update(status="refused", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    return report


def loop_bound_probe(bound):
    report = {
        "reproducer": "MiniLM query reduction loop bound",
        "bound": bound,
        "gpu_dispatched": False,
        "pipeline_created": False,
        "stage": "compiler",
    }
    try:
        import g17cc
        program = g17cc.compile_function(loop_bound_ir(bound))
        report.update(status="compiled_unvalidated", code_size=len(program.code),
                      code_sha256=hashlib.sha256(program.code).hexdigest())
    except Exception as error:
        report.update(status="refused", error_type=type(error).__name__, error=str(error))
    return report


def _float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--layernorm-result", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--loop-bound-repro", type=int)
    args = parser.parse_args(argv)
    if args.fixture:
        if not args.layernorm_result or not args.model:
            parser.error("--fixture requires --layernorm-result and --model")
        result = build_fixture(args.layernorm_result, args.model, args.fixture)
    elif args.layernorm_result or args.model:
        parser.error("--layernorm-result and --model require --fixture")
    elif args.loop_bound_repro is not None:
        result = loop_bound_probe(args.loop_bound_repro)
    else:
        result = compile_probe(args.rows, args.width)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status", "prepared") != "refused" else 2


if __name__ == "__main__":
    raise SystemExit(main())
