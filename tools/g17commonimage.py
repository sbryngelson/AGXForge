"""One CPU-only compiler -> author -> native image path for application programs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = ("manifest.json", "scan.arc.metallib", "scan.lib.metallib", "scan.o", "program.bin")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _with_register_count(metadata, count):
    """The retained class's bytes with per-kernel slot 0 set to a delivered register count.

    Used only to build the comparison target. The value comes from the compiler's ABI, so this
    cannot absorb a difference in the authored section: any other byte that moved still fails the
    comparison, and a wrong register count fails it too.
    """
    import struct
    import g17gpumd
    data = bytearray(metadata)
    table = g17gpumd.kernel_table(bytes(data))
    slots, _size = g17gpumd.table_at(bytes(data), table)
    if not slots or not slots[0]:
        raise ValueError("the retained class carries no per-kernel slot 0 to compare against")
    struct.pack_into("<I", data, table + slots[0], count)
    return bytes(data)


def build(kind, rows, columns):
    """Pure build: no process, decoder, cache, runtime library or GPU operation."""
    import g17commonruntime as runtime
    import g17link
    import g17programs
    import g17scanlink
    program, checked = g17programs.compile_program(kind, rows, columns)
    plain = program.abi_plain(program.abi())
    runtime_contract = runtime.RuntimeContract.read(dict(
        format=runtime.manifest_format(kind),
        kind=kind, name=checked.name,
        shape=dict(rows=rows, columns=columns), abi=plain))
    abi = runtime_contract.abi
    checked.check_code(program.code)
    if checked.forms != abi.forms:
        raise ValueError("typed instruction declarations disagree with the compiler ABI")
    kernel = g17link.Kernel(code=program.code, name=runtime_contract.name,
        entry=abi.entry, prologue=abi.prologue,
        bindings=[g17link.Binding(index=b.index, readonly=not b.written, element_type=b.element_type)
                  for b in abi.bindings])
    image = g17scanlink.link(kernel, abi.model_dump(mode="python"),
                            binding_offsets=[b.offset for b in abi.bindings])
    manifest = runtime.ImageContract.read(dict(**runtime_contract.model_dump(mode="json"),
        code_size=len(program.code), instructions=checked.to_dict()["instructions"],
        sha256=dict(archive=sha(image.archive), library=sha(image.library), object=sha(image.object),
                    code=sha(program.code)), field_ledger=image.field_ledger))
    return image, manifest, program.code


def assumptions(ledger):
    return {key: value for key, value in ledger.items()
            if re.search(r"\bASSUMED\b|\bNOT VERIFIED\b", value)}


def arithmetic(decoded, kind, rows, columns):
    """Interpret delivered operands on endpoint and interior rows against CPU math."""
    import numpy as np
    import g17commonruntime as runtime
    import g17halfcheck
    ids = sorted({i for i in (0, 1, 2, 7, 15, 16, 31, 32, rows-2, rows-1) if 0 <= i < rows})
    mode = {"packed_scan": "packed", "separate_scan": "separate", "affine": "affine"}[kind]
    rng = np.random.default_rng(1733)
    results = []
    for case in ("random", "cancellation", "basis", "subnormal", "signed_zero"):
        matrix = (rng.standard_normal((len(ids), columns))*0.1).astype(np.float16)
        query = (rng.standard_normal(columns)*0.1).astype(np.float16)
        if case == "cancellation":
            matrix[:] = np.where(np.arange(columns) % 2, -0.125, 0.125)
            query[:] = 1
        elif case == "basis":
            matrix[:] = 0
            matrix[:, -1] = np.arange(len(ids), dtype=np.float16)
            query[:] = 0.5
        elif case == "subnormal":
            matrix[:] = np.nextafter(np.float16(0), np.float16(1))
            query[:] = 0.5
        elif case == "signed_zero":
            matrix[:] = -0.0
            query[:] = 1
        got = g17halfcheck.simulate(decoded, matrix, query, rows=rows, row_ids=ids, memory_layout=mode)
        expected = runtime.reference(kind, matrix, matrix[:, 0] if kind == "affine" else query)
        if not np.array_equal(got.view(np.uint16), expected.view(np.uint16)):
            raise ValueError(f"delivered arithmetic differs from CPU reference: {case}")
        results.append(dict(case=case, row_ids=ids, half_bits=got.view(np.uint16).tolist()))
    return results


def verify(bundle, *, require_explicit_class=True):
    """Read and check delivered bytes. This may use the CPU decoder, never Metal."""
    import g17archcheck
    import g17gpumd
    import g17commonruntime as runtime
    import g17halfcheck
    import g17packedcheck
    import g17scalarabi
    import g17scanlink
    bundle = Path(bundle)
    manifest = runtime.ImageContract.read(json.loads((bundle / "manifest.json").read_text()))
    expected = [(b.index, b.offset, b.written) for b in manifest.abi.bindings]
    archive = (bundle / "scan.arc.metallib").read_bytes()
    library = (bundle / "scan.lib.metallib").read_bytes()
    obj = g17scanlink.verify_contract(archive, library, expected)
    if obj != (bundle / "scan.o").read_bytes():
        raise ValueError("archive contains a different object than the delivered object file")
    sections, symbols = g17archcheck.object_contents(obj)
    entry = manifest.abi.entry
    if [(s[0], s[4]) for s in symbols if s[0] == "_agc.main"] != [("_agc.main", entry)]:
        raise ValueError("delivered main entry differs from compiler ABI")
    if [(s[0], s[4]) for s in symbols if s[0] == "_agc.main.constant_program"] != [("_agc.main.constant_program", 0)]:
        raise ValueError("delivered constant-program symbol differs from compiler prologue")
    text = sections.get("__TEXT,__text", b"")
    code_end = entry + manifest.code_size
    aligned_end = (code_end + 15) & ~15
    if text[:entry] != manifest.abi.prologue or len(text) != aligned_end:
        raise ValueError("delivered prologue or code size differs from compiler ABI")
    if text[code_end:] != bytes.fromhex("0600") * ((aligned_end-code_end)//2):
        raise ValueError("delivered text alignment contains something other than filler")
    code = text[entry:code_end]
    if code != (bundle / "program.bin").read_bytes():
        raise ValueError("delivered object code differs from the program file")
    for name, data in dict(archive=archive, library=library, object=obj, code=code).items():
        if sha(data) != manifest.sha256[name]:
            raise ValueError(f"{name} hash differs from manifest")
    arch = g17archcheck.decode_arch(sections["__GPU_ARCH_LD_MD,__compute"])
    if arch["serialized_flag"] != manifest.abi.arch_flag:
        raise ValueError("delivered ARCH differs from compiler semantics")
    # Entry=64 and device-buffer-only launches use the separately retained LD
    # measurement. This gate does not ask the author to verify its own output.
    references = g17scalarabi.reference_sections()
    if manifest.kind == "minilm_query":
        measured_metadata = (ROOT/"results/g17-two-coordinate-controls/measured/sr-xy-nolit.metadata.bin").read_bytes()
    elif manifest.kind == "layernorm":
        measured = ROOT/"results/g17-four-user-buffer-witness/measured/indexed.o"
        measured_metadata = g17archcheck.object_contents(measured.read_bytes())[0]["__GPU_METADATA,__compute"]
    elif len(manifest.abi.bindings) == 2:
        measured_metadata = references["__GPU_METADATA"]
    else:
        measured = ROOT/"results/g17-abi-class-probes-indexed/separate-indexed.o"
        measured_metadata = g17archcheck.object_contents(measured.read_bytes())[0]["__GPU_METADATA,__compute"]
    # THE WITNESS IS A DIFFERENT PROGRAM, AND ONE FIELD OF THIS SECTION IS PER-PROGRAM.
    # Per-kernel slot 0 is the register count - the highest 32-bit register index named in
    # _agc.main plus one - so byte equality with a witness compiled from other code can only hold
    # while nothing in the section depends on this program, which is exactly the defect that had
    # every class emitting its witness's count. The witness declares 5 and its own main names 5,
    # so the law holds there too; what must not happen is inheriting that 5.
    #
    # The gate keeps its strength. Slot 0 is set from the DELIVERED register_count - an
    # independent compiler fact, never from the bytes just authored - and every other byte must
    # still match the retained class exactly. A section that differs anywhere else still fails.
    expected = measured_metadata
    delivered_count = manifest.abi.register_count
    # A class with no slot 0 - SCALAR and TENSOR - declares no count, so there is nothing to
    # align and the comparison stays byte-for-byte.
    retained_count = g17gpumd.register_count(expected)
    if (delivered_count is not None and retained_count is not None
            and retained_count != delivered_count):
        expected = _with_register_count(expected, delivered_count)
    if sections["__GPU_METADATA,__compute"] != expected:
        raise ValueError("delivered metadata differs from the independently retained measured class")
    for name in ("__GPU_LD_MD", "__GPU_STATS_MD", "__GPU_REMARKS_MD"):
        if sections[name+",__compute"] != references[name]:
            raise ValueError(f"{name} differs from the measured device-buffer launch contract")
    decoded = g17packedcheck.decode(code)
    actual = [(offset, length, opcode) for offset, length, opcode, _ in decoded]
    declared = [(i.offset, i.length, i.opcode) for i in manifest.instructions]
    if actual != declared:
        raise ValueError("compiler instruction declarations differ from the delivered code")
    if manifest.kind == "minilm_query":
        import g17queryimagecheck
        memory, cases = g17queryimagecheck.check(decoded, manifest)
    elif manifest.kind == "layernorm":
        import g17layernormimagecheck
        memory, cases = g17layernormimagecheck.check(decoded, manifest)
    else:
        mode = {"packed_scan": "packed", "separate_scan": "separate", "affine": "affine"}[manifest.kind]
        memory = g17halfcheck.check_memory(decoded, memory_layout=mode)
        cases = arithmetic(decoded, manifest.kind, manifest.shape.rows, manifest.shape.columns)
    unresolved = assumptions(manifest.field_ledger)
    report = dict(status="passed" if not unresolved else "class facts unresolved",
        kind=manifest.kind, sha256=manifest.sha256, instructions=len(actual), memory=memory,
        arithmetic=cases, serialized_arch=arch["serialized_flag"],
        class_assumptions=unresolved, gpu_dispatched=False, loader_eligible=not unresolved)
    if require_explicit_class and unresolved:
        raise ValueError("metadata class still assumes: " + ", ".join(unresolved))
    return report


def prepare(bundle, kind, rows, columns, *, verify_source=True):
    """Write a new CPU-checked bundle; class assumptions remain an explicit refusal."""
    import g17buildaudit
    bundle = Path(bundle).resolve()
    if bundle.exists():
        raise ValueError("refusing to overwrite an existing bundle")
    if verify_source:
        (image, manifest, code), source = g17buildaudit.verified_build(ROOT, lambda: build(kind, rows, columns))
        manifest = manifest.model_copy(update=dict(source=source))
    else:
        image, manifest, code = build(kind, rows, columns)
    bundle.mkdir(parents=True)
    try:
        (bundle / "manifest.json").write_text(manifest.model_dump_json(indent=2)+"\n")
        for name, data in (("scan.arc.metallib", image.archive), ("scan.lib.metallib", image.library),
                           ("scan.o", image.object), ("program.bin", code)):
            (bundle / name).write_bytes(data)
        report = verify(bundle, require_explicit_class=False)
        (bundle / "offline-check.json").write_text(json.dumps(report, indent=2)+"\n")
    except BaseException:
        shutil.rmtree(bundle)
        raise
    return report


def rebuild(bundle):
    """Fresh-process rebuild, including the author and packagers in the input audit."""
    import g17buildaudit
    import g17commonruntime as runtime
    bundle = Path(bundle)
    original = runtime.ImageContract.read(json.loads((bundle / "manifest.json").read_text()))
    (image, manifest, code), source = g17buildaudit.verified_build(ROOT,
        lambda: build(original.kind, original.shape.rows, original.shape.columns))
    if manifest.sha256 != original.sha256:
        raise ValueError("delivered image differs from the committed-source rebuild")
    if manifest.model_dump(exclude={"source"}) != original.model_dump(exclude={"source"}):
        raise ValueError("delivered compiler or metadata contract differs from the rebuild")
    for name, data in (("scan.arc.metallib", image.archive), ("scan.lib.metallib", image.library),
                       ("scan.o", image.object), ("program.bin", code)):
        if (bundle / name).read_bytes() != data:
            raise ValueError(f"delivered {name} differs from the committed-source rebuild")
    return dict(status="passed", sha256=manifest.sha256, source=source,
                class_assumptions=assumptions(manifest.field_ledger), gpu_dispatched=False)


def main():
    import g17programs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--prepare", choices=g17programs.KINDS)
    parser.add_argument("--rows", type=int, default=33)
    parser.add_argument("--columns", type=int, default=384)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--inspect", action="store_true", help="report CPU checks and unresolved class facts")
    args = parser.parse_args()
    if sum((args.prepare is not None, args.rebuild, args.inspect)) > 1:
        parser.error("choose one operation")
    if args.prepare:
        report = prepare(args.bundle, args.prepare, args.rows, args.columns)
    elif args.rebuild:
        report = rebuild(args.bundle)
    else:
        report = verify(args.bundle, require_explicit_class=not args.inspect)
    print(json.dumps(report, indent=2))
    return 2 if report.get("class_assumptions") else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, RuntimeError) as error:
        print(f"refused: {error}", file=sys.stderr)
        raise SystemExit(2)
