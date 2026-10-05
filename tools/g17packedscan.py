#!/usr/bin/env python3
"""Native scan with persistent FP32 packing around the half-valued public API."""
from __future__ import annotations

import g17ir as ir


def scan_ir(rows, columns):
    if not (1 <= rows and 1 <= columns and (rows + 1) * columns < 2**30):
        raise ValueError("positive dimensions and a packed buffer below 4 GiB required")
    packed, scores = ir.Buffer("packed", 1), ir.Buffer("scores", 2)
    f = ir.Function("packed_scan", [packed, scores])
    b = ir.Builder(f, f.block("entry"))
    row = b.builtin("thread_position_in_grid", name="row")
    stride = b.const(columns, name="stride")
    ai = b.mul(row, stride, name="ai0")
    xi = b.const(rows * columns, name="xi0")
    acc = b.const(0, name="zero")
    for k in range(columns):
        a = b.load(packed, ai, name=f"a{k}")
        x = b.load(packed, xi, name=f"x{k}")
        p = b.fmul(a, x, name=f"p{k}")
        acc = b.fadd(acc, p, name=f"sum{k}")
        if k + 1 < columns:
            ai = b.add(ai, ir.Imm(1), name=f"ai{k+1}")
            xi = b.add(xi, ir.Imm(1), name=f"xi{k+1}")
    b.store_at(scores, row, acc)
    # Re-read the thread ID so neither operand of the score store needs to be
    # preserved by the currently uncertified eight-byte store lifetime form.
    done_row = b.builtin("thread_position_in_grid", name="done_row")
    count = b.const(rows, name="row_count")
    done_idx = b.add(done_row, count, name="done_idx")
    marker = b.const(0x5A17C0DE, name="done_marker")
    b.store_at(scores, done_idx, marker)
    b.ret()
    return f


def compile_scan(rows=500_000, columns=384):
    """Compile and link without Metal, a donor object, or a compiler subprocess.

    The named compatibility ABI reproduces every metadata byte in the measured
    scalar profile. Compiler semantic facts remain separate from class constants.
    """
    import g17cc as cc
    import g17link as link
    import g17scalarabi
    fn = scan_ir(rows, columns)
    program = cc.compile_function(fn)
    facts = program.abi_inputs()
    if facts["uses_threadgroup"] or not facts["arch_flag"] or not facts["writes_buffer"]:
        raise ValueError("program is outside the independent scalar buffer ABI")
    abi = {"measured_class": g17scalarabi.PROFILE, "entry": 64,
           "bindings": g17scalarabi.BINDINGS, "compiler_facts": facts}
    kernel = link.Kernel(
        program.code,
        [link.Binding(1, readonly=True, element_type="float"),
         link.Binding(2, readonly=False, element_type="float")],
        name="packed_scan", entry=64,
        prologue=link.PROLOGUE_WORD + link.FILLER * 30)
    image = g17scalarabi.link(kernel, program)
    manifest = {
        "format": "g17-packed-scan-v1", "rows": rows, "columns": columns,
        "entry": 64, "function": "packed_scan", "abi": abi,
        "binding_order": [1, 2], "binding_offsets": [0, 2],
        "packed_elements": (rows + 1) * columns,
        "query_offset_elements": rows * columns,
        "output_elements": 2 * rows, "completion_offset_elements": rows,
        "completion_value": 0x5A17C0DE,
        "instructions": len(program.layout), "code_bytes": len(program.code),
        "archive_bytes": len(image.archive),
        "launch": {"threads": rows, "exact_grid_required": True},
        "precision": "half inputs expanded exactly on host; sequential FP32 products/adds; host rounds scores to half",
        "profile": {
            "name": g17scalarabi.PROFILE,
            "loader_validation": "pending for this scan; dispatch blocked until loader-only validation",
            "metadata_validation": g17scalarabi.verify_object(image.object, program.code),
            "evidence": "isa/g17-scalar-abi-v2.json: all 844 metadata bytes match the measured scalar class",
            "limitation": "fixed measured class compatibility; not general semantic metadata synthesis or proof of scan execution",
        },
        "gpu_executed": False,
        "field_ledger": image.field_ledger,
    }
    return program, image, manifest


class PackedIndex:
    """Persistent host packing; query updates reuse the same FP32 allocation."""

    def __init__(self, matrix):
        import numpy as np
        a = np.asarray(matrix)
        if a.dtype != np.float16 or a.ndim != 2 or not all(a.shape):
            raise ValueError("index must be a nonempty float16 matrix")
        self.rows, self.columns = a.shape
        if (self.rows + 1) * self.columns >= 2**30 or not np.isfinite(a).all():
            raise ValueError("finite index below the packed 4 GiB limit required")
        self.data = np.empty((self.rows + 1) * self.columns, dtype=np.float32)
        self.query_offset = self.rows * self.columns
        self.data[:self.query_offset] = a.reshape(-1)
        self.data[self.query_offset:] = 0

    def set_query(self, vector):
        import numpy as np
        x = np.asarray(vector)
        if x.dtype != np.float16 or x.shape != (self.columns,) or not np.isfinite(x).all():
            raise ValueError("query must be a finite float16 vector of the index width")
        self.data[self.query_offset:] = x
        return self.data

    def finish(self, output, status):
        """Accept results only after successful completion and every row's marker."""
        import numpy as np
        y = np.asarray(output)
        if status != 0 or y.dtype != np.float32 or y.shape != (2 * self.rows,):
            raise ValueError("unsuccessful command or invalid output buffer")
        if not np.all(y.view(np.uint32)[self.rows:] == 0x5A17C0DE):
            raise ValueError("scan did not complete every row")
        if not np.isfinite(y[:self.rows]).all():
            raise ValueError("scan produced nonfinite scores")
        with np.errstate(over="ignore"):
            result = y[:self.rows].astype(np.float16)
        if not np.isfinite(result).all():
            raise ValueError("scores exceed the finite half-output contract")
        return result


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    import time
    import g17scan
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--columns", type=int, default=384)
    parser.add_argument("--out", type=Path, default=g17scan.ROOT / "results/g17-packed-scan-v2")
    parser.add_argument("--fixture", action="store_true", help="prepare a deterministic small hardware-test input")
    args = parser.parse_args()
    if args.fixture and (args.rows > 128 or args.columns > 384):
        parser.error("initial hardware fixtures are limited to 128 rows and 384 columns")
    events = []
    g17scan.install_dependency_guard(events)
    # Hash only inputs this build actually opens, rather than walking the corpus.
    import importlib.util
    import os
    import sys
    inputs = {}
    hashing = False
    def record_input(event, arguments):
        nonlocal hashing
        if event != "open" or hashing or not isinstance(arguments[0], (str, bytes)):
            return
        path = Path(os.fsdecode(arguments[0])).resolve()
        if not path.is_relative_to(g17scan.ROOT):
            return
        if path.suffix == ".pyc":
            path = Path(importlib.util.source_from_cache(str(path)))
        if path.suffix not in (".py", ".json", ".jsonl", ".toml") or not path.is_file():
            return
        key = str(path.relative_to(g17scan.ROOT))
        if key not in inputs:
            hashing = True
            try:
                inputs[key] = hashlib.sha256(path.read_bytes()).hexdigest()
            finally:
                hashing = False
    sys.addaudithook(record_input)
    for module in list(sys.modules.values()):
        if getattr(module, "__file__", None):
            record_input("open", (module.__file__,))
    started = time.monotonic()
    program, image, manifest = compile_scan(args.rows, args.columns)
    if events:
        raise RuntimeError("compiler attempted forbidden dependencies: " + repr(events))
    manifest["build_seconds"] = time.monotonic() - started
    hashing = True
    for name, expected in inputs.items():
        if hashlib.sha256((g17scan.ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("compiler input changed during build: " + name)
    manifest["source_sha256"] = inputs
    blobs = {"scan.bin": program.code, "scan.arc.metallib": image.archive,
             "scan.lib.metallib": image.library, "scan.o": image.object}
    manifest["sha256"] = {n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()}
    manifest["dependency_events"] = events
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in blobs.items():
        (args.out / name).write_bytes(data)
    if args.fixture:
        import numpy as np
        rng = np.random.default_rng(1729)
        a = (rng.standard_normal((args.rows, args.columns)) * 0.2).astype(np.float16)
        x = (rng.standard_normal(args.columns) * 0.2).astype(np.float16)
        if args.rows > 1:
            a[0] = 0
            a[1] = 0
            a[1, -1] = 1
        if args.rows > 2:
            a[2] = np.where(np.arange(args.columns) % 2, -0.125, 0.125)
        packed = PackedIndex(a)
        data = packed.set_query(x).tobytes()
        (args.out / "packed.f32").write_bytes(data)
        np.savez(args.out / "fixture.npz", index=a, embedding=x)
        for name in ("packed.f32", "fixture.npz"):
            manifest["sha256"][name] = hashlib.sha256((args.out / name).read_bytes()).hexdigest()
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "instructions": manifest["instructions"],
                      "archive_bytes": len(image.archive), "build_seconds": manifest["build_seconds"],
                      "gpu_executed": False}))


if __name__ == "__main__":
    main()
