"""One-thread GPU control: a word store must change the neighboring halfword.

The four-byte write stays inside a 130-byte output allocation. Normal half-scan
validation rejects this program. Only the separate control worker mode accepts
the expected first-guard change; all subsequent guard bytes must be preserved.
Default operation is CPU-only. --execute-approved performs loader then dispatch
stages, serialized against other GPU experiments, stopping on the first failure.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "halfword-preservation-word-control-v1"


def program():
    """The control's IR, on its own, so a reproduction check can rebuild exactly what build() compiles.

    tools/g17capcompiler.py's receipt table carried an EMPTY factory for this receipt, which compiles
    nothing and so reported `does NOT reproduce` forever, whatever the compiler did."""
    import g17ir as ir
    packed, output = ir.Buffer("packed", 1, elem=ir.F16), ir.Buffer("output", 2, elem=ir.F32)
    function = ir.Function("half_scan", [packed, output])
    b = ir.Builder(function, function.block("entry"))
    row = b.builtin("thread_position_in_grid", name="row")
    query = b.const(1)
    a = b.f16_to_f32(b.load(packed, row, width="half"))
    x = b.f16_to_f32(b.load(packed, query, width="half"))
    value = b.fadd(b.const(0), b.fmul(a, x))
    b.store_at(output, row, value, width="word")
    b.ret()
    return function


def build():
    import g17cc
    import g17authorobj
    import g17link
    import g17scanlink
    compiled = g17cc.compile_function(program())
    abi = compiled.abi()
    bindings = [(1, 0, False), (2, 2, True)]
    sections, ledger = g17authorobj.author(b"", abi["entry"], bindings,
        {"measured_class": "scalar-buffer-two-bindings-measured-v2", "arch_flag": False})
    kernel = g17link.Kernel(code=compiled.code, name="half_scan", entry=abi["entry"],
                           prologue=abi["prologue"], bindings=[
                               g17link.Binding(1, readonly=True, element_type="half"),
                               g17link.Binding(2, element_type="float")])
    return g17scanlink.package_sections(kernel, sections, ledger, bindings), abi, compiled


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(directory):
    import g17halfimage
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        image, abi, program = build()
        manifest = {"experiment": EXPERIMENT, "profile": "half-buffer-two-bindings-1x1",
                    "shape": {"rows": 1, "columns": 1}, "gpu_executed": False,
                    "abi": g17halfimage.serialisable(abi),
                    "sha256": {"archive": digest(image.archive), "object": digest(image.object),
                               "code": digest(program.code)}}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (directory / "scan.arc.metallib").write_bytes(image.archive)
        (directory / "scan.lib.metallib").write_bytes(image.library)
        (directory / "scan.o").write_bytes(image.object)
        report = verify(directory)
        (directory / "offline-check.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    except BaseException:
        shutil.rmtree(directory)
        raise


def verify(directory):
    import numpy as np
    import g17halfcheck
    import g17scalarabi
    import g17scanlink
    import g17verify
    import g17packedcheck
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest.get("experiment") != EXPERIMENT or
            manifest.get("shape") != {"rows": 1, "columns": 1} or
            manifest.get("profile") != "half-buffer-two-bindings-1x1"):
        raise ValueError("word control requires its named one-thread contract")
    archive = (directory / "scan.arc.metallib").read_bytes()
    obj = g17scanlink.verify_contract(archive, (directory / "scan.lib.metallib").read_bytes(),
                                     [(1, 0, False), (2, 2, True)])
    if manifest["abi"]["entry"] != 64:
        raise ValueError("word control requires entry 64")
    code = g17verify._objsect(obj, "__text")[64:]
    while code.endswith(bytes.fromhex("0600")):
        code = code[:-2]
    for name, data in (("archive", archive), ("object", obj), ("code", code)):
        if digest(data) != manifest["sha256"][name]:
            raise ValueError(f"word control {name} differs from its manifest")
    g17scalarabi.verify_object(obj, code)
    ins = g17packedcheck.decode(code)
    memory = g17halfcheck.check_memory(ins, store_bytes=4)
    if memory["half_loads"] != 2 or memory["word_stores"] != 1:
        raise ValueError("control needs exactly two half loads and one word store")
    cases = []
    for query in (1.0, -2.0, 1.0):
        a, x = np.ones((1, 1), np.float16), np.array([query], np.float16)
        got = g17halfcheck.simulate(ins, a, x, rows=1, row_ids=[0], store_bytes=4)
        expected = np.float32(0) + np.float32(a[0, 0]) * np.float32(x[0])
        word = int(got.view(np.uint32)[0])
        if word != int(expected.view(np.uint32)) or word >> 16 == 0xA55A:
            raise ValueError("word control dataflow does not produce the expected overwrite")
        cases.append({"query": query, "word_bits": word, "adjacent_halfword": word >> 16})
    return {"status": "passed", "gpu_executed": False, "sha256": manifest["sha256"],
            "allocation_bytes": 130, "write_byte_range": [0, 4],
            "preserved_guard_byte_range": [4, 130], "initial_adjacent_halfword": 0xA55A,
            "instructions": len(ins), **memory, "cases": cases}


def frames(data):
    result = []
    while data:
        if len(data) < 4:
            raise ValueError("truncated control frame")
        n = struct.unpack_from("<I", data)[0]
        if not 1 <= n <= 4096 or len(data) < 4 + n:
            raise ValueError("invalid control frame length")
        header = json.loads(data[4:4 + n])
        payload_size = header.get("bytes")
        if type(payload_size) is not int or payload_size not in (0, 4) or len(data) < 4 + n + payload_size:
            raise ValueError("invalid control payload length")
        result.append((header, data[4 + n:4 + n + payload_size]))
        data = data[4 + n + payload_size:]
    return result


def execute(directory):
    import g17halfruntime
    import g17packeddispatch
    directory = Path(directory).resolve()
    with (directory / "execution.json").open("x") as evidence:
        report = {"status": "pending", "experiment": EXPERIMENT, "attempts": []}
        try:
            with tempfile.TemporaryDirectory(prefix="g17-word-control-") as temp:
                frozen = Path(temp)
                for name in g17halfruntime.FILES:
                    shutil.copyfile(directory / name, frozen / name)
                report["offline"] = verify(frozen)
                rebuilt = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(frozen), "--rebuild"],
                                         capture_output=True, text=True, timeout=30)
                if rebuilt.returncode:
                    raise ValueError("word control rebuild refused: " + rebuilt.stderr[-2000:])
                report["rebuild"] = json.loads(rebuilt.stdout)
                runner = ROOT / "tools/g17scanworker"
                report["worker_sha256"] = digest(runner.read_bytes())
                import g17gpulock
                with g17gpulock.acquire("exclusive"):
                    before = g17packeddispatch.gpu_events()

                    def run(command, input=None):
                        p = subprocess.run(command, input=input, capture_output=True, timeout=30)
                        if p.returncode or before != g17packeddispatch.gpu_events():
                            raise RuntimeError(f"control stopped: returncode={p.returncode}; {p.stderr[-2000:]!r}")
                        if digest(runner.read_bytes()) != report["worker_sha256"]:
                            raise RuntimeError("control worker changed during execution")
                        return p

                    loaded = run([str(runner), str(frozen), "--half-load-approved"])
                    report["loader"] = json.loads(loaded.stdout)
                    if report["loader"] != {"status": 0, "load_only": True, "gpu_dispatched": False}:
                        raise ValueError("unexpected control loader response")
                    packed = frozen / "packed.f16"
                    packed.write_bytes(struct.pack("<ee", 1.0, 0.0))
                    for case in report["offline"]["cases"]:
                        attempt = {"query": case["query"], "status": "pending"}
                        report["attempts"].append(attempt)
                        p = run([str(runner), str(frozen), str(packed), "1", "--half-word-control-approved"],
                                struct.pack("<Ie", 2, case["query"]))
                        reply = frames(p.stdout)
                        if len(reply) != 2 or reply[0][0].get("sequence") != 0:
                            raise ValueError("invalid control handshake")
                        header, data = reply[1]
                        if (header.get("status") != 0 or header.get("sequence") != 1 or
                                header.get("expected_control") is not True or
                                header.get("boundary_guard") is not False or header.get("tail_guard") is not True or
                                data != struct.pack("<I", case["word_bits"]) or
                                header.get("word_bits") != case["word_bits"] or
                                header.get("adjacent_halfword") != case["adjacent_halfword"]):
                            raise ValueError("GPU word control did not produce exactly the predicted overwrite")
                        attempt.update(status="passed", native=header, stderr=p.stderr.decode())
                    report.update(status="passed", gpu_dispatches=3, new_gpu_diagnostics=False,
                                  repeated_word_bit_exact=True)
        except BaseException as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            evidence.write(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    stage = p.add_mutually_exclusive_group()
    stage.add_argument("--prepare", action="store_true")
    stage.add_argument("--rebuild", action="store_true")
    stage.add_argument("--execute-approved", action="store_true")
    args = p.parse_args()
    if args.prepare:
        report = prepare(args.directory)
    elif args.rebuild:
        import g17halfruntime
        report = g17halfruntime.rebuild_probe(args.directory)
    elif args.execute_approved:
        report = execute(args.directory)
    else:
        report = verify(args.directory)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
