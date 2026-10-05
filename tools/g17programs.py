"""Common-ABI application programs expressed as repository IR."""
import struct

KINDS = ("packed_scan", "separate_scan", "affine", "layernorm", "minilm_query")


def affine_ir(rows):
    """FP16 y[i] = round_half(float(x[i]) * 2 + 1), with separate FP32 operations."""
    import g17ir as ir
    if type(rows) is not int or not 1 <= rows <= 500_000:
        raise ValueError("affine row count must be in [1, 500000]")
    source, output = ir.Buffer("source", 1, elem=ir.F16), ir.Buffer("output", 2, elem=ir.F16)
    f = ir.Function("half_affine", [source, output])
    b = ir.Builder(f, f.block("entry"))
    row = b.builtin("thread_position_in_grid")
    x = b.f16_to_f32(b.load(source, row, width="half"))
    alpha = b.const(struct.unpack("<I", struct.pack("<f", 2.0))[0])
    beta = b.const(struct.unpack("<I", struct.pack("<f", 1.0))[0])
    value = b.fadd(b.fmul(x, alpha), beta)
    b.store_at(output, row, b.f32_to_f16_rte(value), width="half")
    # The exact affine image has execution evidence in the retained common-pipeline records. Keep
    # its source-owned long shape separate from arbitrary half-load consumers.
    f.allow_validated_half_store = True
    b.ret()
    return f


def separate_scan_ir(rows, columns, *, indices=(1, 2, 3)):
    """Matrix, query and output have distinct bindings; one exact-grid thread per row."""
    import g17ir as ir
    if type(rows) is not int or type(columns) is not int or not 1 <= rows <= 500_000 or not 1 <= columns <= 384:
        raise ValueError("scan dimensions must fit 500000 x 384")
    if len(indices) != 3 or list(indices) != sorted(set(indices)):
        raise ValueError("separate scan requires three ascending distinct indices")
    matrix, query, scores = [ir.Buffer(name, index, elem=ir.F16)
                             for name, index in zip(("matrix", "query", "scores"), indices)]
    f = ir.Function("half_scan_separate", [matrix, query, scores])
    b = ir.Builder(f, f.block("entry"))
    row = b.builtin("thread_position_in_grid")
    ai = b.mul(row, b.const(columns))
    xi, acc = b.const(0), b.const(0)
    for k in range(columns):
        a = b.f16_to_f32(b.load(matrix, ai, width="half"))
        x = b.f16_to_f32(b.load(query, xi, width="half"))
        acc = b.fadd(acc, b.fmul(a, x))
        if k + 1 < columns:
            ai, xi = b.add(ai, ir.Imm(1)), b.add(xi, ir.Imm(1))
    b.store_at(scores, row, b.f32_to_f16_rte(acc), width="half")
    # This delivery is compiler/linker evidence only; it has no retained GPU receipt. Preserve the
    # explicit compile-only opt-in so the common ABI can carry the program without making the
    # unvalidated half-load dependency look generally supported.
    f.allow_unvalidated_half_store = True
    b.ret()
    return f


def compile_program(kind, rows, columns):
    import g17cc
    if kind == "packed_scan":
        import g17halfscan
        function = g17halfscan.scan_ir(rows, columns)
    elif kind == "separate_scan":
        function = separate_scan_ir(rows, columns)
    elif kind == "affine":
        if columns != 1:
            raise ValueError("affine is a one-dimensional program; columns must be 1")
        function = affine_ir(rows)
    elif kind == "layernorm":
        import g17layernorm
        function = g17layernorm.layernorm_ir(rows, columns)
    elif kind == "minilm_query":
        import g17minilmquery
        function = g17minilmquery.query_projection_ir(rows, columns)
    else:
        raise ValueError("unknown common-ABI program")
    try:
        program = g17cc.compile_function(function)
    except g17cc.Unsupported as error:
        raise ValueError(f"compiler refused {kind}: {error}") from error
    return program, program.contract()


def main():
    """Deliver concrete native code and the versioned compiler ABI for the linker owner."""
    import argparse
    import hashlib
    import json
    from pathlib import Path
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=KINDS)
    parser.add_argument("--rows", type=int, default=33)
    parser.add_argument("--columns", type=int, default=384)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-source", action="store_true",
                        help="require every observed build input to match one committed revision")
    args = parser.parse_args()
    # Install before compiler imports so a warm process cannot hide a fork.
    def guard(event, values):
        if event in ("subprocess.Popen", "os.exec", "os.posix_spawn", "os.system"):
            raise RuntimeError("program delivery must not start an external process")
    source = None
    if args.verify_source:
        import g17buildaudit
        (program, checked), source = g17buildaudit.verified_build(
            Path(__file__).resolve().parents[1],
            lambda: compile_program(args.kind, args.rows, args.columns))
    else:
        sys.addaudithook(guard)
        program, checked = compile_program(args.kind, args.rows, args.columns)
    abi = program.abi()
    if (tuple((b["index"], b["offset"], b["written"]) for b in abi["bindings"])
            != tuple((b.index, b.offset, b.written) for b in checked.bindings)
            or abi["forms"] != checked.forms
            or abi["prologue"] != checked.prologue
            or abi["arch_flag"] != checked.arch_flag):
        raise ValueError("compiler ABI and captured instruction facts disagree")
    checked.check_code(program.code)
    report = dict(kind=args.kind, rows=args.rows, columns=args.columns,
                  name=program.name, code_sha256=hashlib.sha256(program.code).hexdigest(),
                  code_size=len(program.code), abi=program.abi_plain(abi),
                  instructions=[[i.offset, i.length, i.opcode] for i in checked.instructions],
                  external_processes=0, image_authored=False, gpu_dispatched=False)
    if source is not None:
        report["source"] = source
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "program.bin").write_bytes(program.code)
    (args.out / "abi.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("abi", "instructions", "source")}))


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        import sys
        print(f"refused: {error}", file=sys.stderr)
        raise SystemExit(2)
