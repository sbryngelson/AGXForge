"""Acceptance and loader-only staging for the narrow half scan runtime.

The elided ARCH layout is an explicit validation hypothesis, not a general ABI
agreement. This module never dispatches a kernel. Its default CLI is CPU only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HYPOTHESIS = "half-scan-semantic-arch-true-serialized-elided-v1"
SET_HYPOTHESIS = "half-scan-semantic-arch-true-serialized-set-experiment-v1"
FILES = ("manifest.json", "scan.arc.metallib", "scan.lib.metallib")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serialized_arch(manifest):
    flag = manifest.get("serialized_arch_flag", False)
    if type(flag) is not bool:
        raise ValueError("serialized ARCH flag must be an explicit boolean")
    return flag


def hypothesis(manifest):
    return SET_HYPOTHESIS if serialized_arch(manifest) else HYPOTHESIS


def build_image(rows, columns, arch_flag):
    if arch_flag:
        import g17archexperiment
        return g17archexperiment.build(rows, columns)
    import g17halfimage
    return g17halfimage.build(rows, columns)


def prepare(bundle, rows, columns, *, arch_flag=False):
    """Author a new bounded fixture from the integrated compiler; never load it."""
    import g17halfimage
    import numpy as np
    bundle = Path(bundle).resolve()
    manifest = {"profile": f"half-buffer-two-bindings-{rows}x{columns}",
                "shape": {"rows": rows, "columns": columns}, "gpu_executed": False,
                "status": "CPU checked; hardware validation not run", "serialized_arch_flag": arch_flag}
    manifest["abi_hypothesis"] = hypothesis(manifest)
    shape(manifest)
    bundle.mkdir(parents=True, exist_ok=False)
    try:
        image, abi, program = build_image(rows, columns, arch_flag)
        manifest.update(abi=g17halfimage.serialisable(abi), sha256={
            "archive": hashlib.sha256(image.archive).hexdigest(),
            "object": hashlib.sha256(image.object).hexdigest(),
            "code": hashlib.sha256(program.code).hexdigest()})
        (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (bundle / "scan.arc.metallib").write_bytes(image.archive)
        (bundle / "scan.lib.metallib").write_bytes(image.library)
        (bundle / "scan.o").write_bytes(image.object)
        report = verify_bundle(bundle)
        rng = np.random.default_rng(731)
        if rows > 128:
            matrix = np.lib.format.open_memmap(bundle / "matrix.f16.npy", mode="w+", dtype=np.float16,
                                               shape=(rows, columns))
            for start in range(0, rows, 1024):
                matrix[start:start + 1024] = (rng.standard_normal((min(1024, rows-start), columns)) * 0.05).astype(np.float16)
            matrix.flush()
        else:
            matrix = (rng.standard_normal((rows, columns)) * 0.05).astype(np.float16)
        query = (rng.standard_normal(columns) * 0.1).astype(np.float16)
        if rows > 128:
            np.save(bundle / "query.f16.npy", query)
            del matrix
        else:
            np.savez(bundle / "fixture.npz", index=matrix, embedding=query)
        (bundle / "offline-check.json").write_text(json.dumps(report, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(bundle)
        raise
    return report


def shape(manifest):
    dims = manifest.get("shape", {})
    rows, columns = dims.get("rows"), dims.get("columns")
    if (type(rows) is not int or type(columns) is not int or
            not 1 <= rows <= 500_000 or not 1 <= columns <= 384):
        raise ValueError("half layout must fit the 500000 x 384 target")
    if manifest.get("profile") != f"half-buffer-two-bindings-{rows}x{columns}":
        raise ValueError("unsupported half runtime profile")
    return rows, columns


def snapshot(source, target, *, loader=False):
    source, target = Path(source), Path(target)
    manifest = json.loads((source / "manifest.json").read_text())
    shape(manifest)
    for name in FILES + (("half-loader-check.json",) if loader else ()):
        shutil.copyfile(source / name, target / name)
    if json.loads((target / "manifest.json").read_text()) != manifest:
        raise ValueError("half manifest changed while taking its snapshot")
    return manifest


def verify_bundle(directory):
    """Check delivered instructions and the exact ABI subset this worker implements."""
    import g17halfcheck
    import g17packedcheck
    import g17scanlink
    import g17verify
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    rows, columns = shape(manifest)
    checked = g17halfcheck.check_bundle(directory, arch_flag=serialized_arch(manifest))
    abi = manifest["abi"]
    expected = [(1, 0, False, "half", 2), (2, 2, True, "half", 2)]
    got = [(b.get("index"), b.get("offset"), b.get("written"), b.get("element_type"),
            b.get("element_bytes")) for b in abi.get("bindings", [])]
    if (got != expected or abi.get("entry") != 64 or abi.get("arch_flag") is not True or
            abi.get("uses_threadgroup") is not False or abi.get("writes_buffer") is not True or
            abi.get("writes_texture") is not False):
        raise ValueError("half compiler ABI differs from the runtime contract")
    obj = g17scanlink.verify_contract((directory / "scan.arc.metallib").read_bytes(),
                                     (directory / "scan.lib.metallib").read_bytes(),
                                     [(1, 0, False), (2, 2, True)])
    code = g17verify._objsect(obj, "__text")[64:]
    while code.endswith(bytes.fromhex("0600")):
        code = code[:-2]
    instructions = g17packedcheck.decode(code)
    actual = sorted({(opcode, size) for _, size, opcode, _ in instructions})
    declared = sorted(tuple(f) for f in abi.get("forms", []))
    if declared != actual:
        raise ValueError("half compiler form set differs from delivered instructions")
    # check_bundle checks all metadata and the selected measured ARCH layout.
    return {"status": "passed", "rows": rows, "columns": columns,
            "abi_hypothesis": hypothesis(manifest), "manifest_sha256": digest(directory / "manifest.json"),
            "archive_sha256": digest(directory / "scan.arc.metallib"),
            "library_sha256": digest(directory / "scan.lib.metallib"), "code_check": checked}


def rebuild_probe(directory):
    """Run in a fresh CPU child: hash actual build inputs and reproduce the archive.

    The Apple decoder is permitted for form enumeration and recorded separately.
    External compilers, caches, and hardware libraries are refused here.
    """
    import importlib.util
    import os
    tracked = {str(Path(__file__).relative_to(ROOT)): digest(__file__)}
    decoder = ROOT / "tools/agx3dis"
    active, reading = True, False

    def audit(event, values):
        nonlocal reading
        if not active:
            return
        if event in ("subprocess.Popen", "os.exec", "os.posix_spawn", "os.system"):
            if event != "subprocess.Popen" or Path(os.fsdecode(values[0])).resolve() != decoder:
                raise RuntimeError("half rebuild attempted a process other than the CPU decoder")
        if event == "ctypes.dlopen" and any(x in str(values[0]) for x in ("Metal", "AGX", "libaccel")):
            raise RuntimeError("half rebuild attempted a hardware library")
        if event != "open" or reading or not isinstance(values[0], (str, bytes)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if "/.cache/agxforge/" in str(path):
            raise RuntimeError("half rebuild attempted an external cache")
        if path.suffix == ".pyc":
            path = Path(importlib.util.source_from_cache(str(path)))
        if path.is_relative_to(ROOT) and path.is_file() and path.suffix in (".py", ".json", ".jsonl", ".toml"):
            reading = True
            try:
                tracked.setdefault(str(path.relative_to(ROOT)), digest(path))
            finally:
                reading = False

    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    rows, columns = shape(manifest)
    sys.addaudithook(audit)
    try:
        import g17halfimage
        if manifest.get("experiment") == "halfword-preservation-word-control-v1":
            import g17wordcontrol
            image, abi, program = g17wordcontrol.build()
        else:
            image, abi, program = build_image(rows, columns, serialized_arch(manifest))
    finally:
        active = False
    # cat-file checks the actual bytes read against HEAD, including imported
    # serializers and operand maps, rather than checking three source names only.
    for rel, wanted in tracked.items():
        proc = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:" + rel],
                              capture_output=True, timeout=5)
        if proc.returncode or hashlib.sha256(proc.stdout).hexdigest() != wanted or digest(ROOT / rel) != wanted:
            raise ValueError(f"rebuild input is uncommitted or changed: {rel}")
    hashes = {"archive": hashlib.sha256(image.archive).hexdigest(),
              "object": hashlib.sha256(image.object).hexdigest(),
              "code": hashlib.sha256(program.code).hexdigest()}
    if any(manifest["sha256"].get(k) != v for k, v in hashes.items()):
        raise ValueError("half bundle differs from the committed-source rebuild")
    if image.library != (Path(directory) / "scan.lib.metallib").read_bytes():
        raise ValueError("half library differs from the committed-source rebuild")
    return {"status": "passed", "sha256": hashes, "inputs": tracked,
            "decoder_sha256": digest(decoder), "cpu_decoder_dependency": True,
            "commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}


def loader_verified(directory):
    try:
        directory = Path(directory)
        report = json.loads((directory / "half-loader-check.json").read_text())
        return (report.get("status") == "passed" and report.get("returncode") == 0 and
                report.get("abi_hypothesis") == hypothesis(json.loads((directory / "manifest.json").read_text())) and
                report.get("manifest_sha256") == digest(directory / "manifest.json") and
                report.get("archive_sha256") == digest(directory / "scan.arc.metallib") and
                report.get("library_sha256") == digest(directory / "scan.lib.metallib") and
                report.get("worker_sha256") == digest(ROOT / "tools/g17scanworker") and
                report.get("rebuild", {}).get("status") == "passed" and
                report.get("native") == {"status": 0, "load_only": True, "gpu_dispatched": False})
    except (OSError, ValueError, TypeError):
        return False


def load_only(bundle, *, half_validation_approved=False, full_size_approved=False):
    if not half_validation_approved:
        raise ValueError("half loader validation requires explicit approval of the ARCH hypothesis")
    import g17packeddispatch
    bundle = Path(bundle).resolve()
    rows, columns = shape(json.loads((bundle / "manifest.json").read_text()))
    if rows > 128 and (not full_size_approved or (rows, columns) != (500_000, 384)):
        raise ValueError("full-size loader requires separate approval for exactly 500000 x 384")
    # Reserve the evidence path before any Metal call. A failed stage is retained
    # and cannot be implicitly retried by invoking this command again.
    with (bundle / "half-loader-check.json").open("x") as evidence:
        report = {"status": "pending", "gpu_dispatched": False}
        try:
            with tempfile.TemporaryDirectory(prefix="g17-half-loader-") as temp:
                frozen = Path(temp)
                snapshot(bundle, frozen)
                report.update(verify_bundle(frozen))
                report["status"] = "pending"
                rebuild = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(frozen),
                                          "--rebuild-only"], capture_output=True, text=True, timeout=30)
                if rebuild.returncode:
                    raise ValueError("half rebuild refused: " + rebuild.stderr[-2000:])
                report["rebuild"] = json.loads(rebuild.stdout)
                import g17gpulock
                with g17gpulock.acquire("exclusive"):
                    before = g17packeddispatch.gpu_events()
                    runner = ROOT / "tools/g17scanworker"
                    report["worker_sha256"] = digest(runner)
                    result = subprocess.run([str(runner), str(frozen),
                                             "--half-full-load-approved" if rows > 128 else "--half-load-approved"],
                                            capture_output=True, text=True, timeout=30)
                    report.update(returncode=result.returncode, stderr=result.stderr[-4000:])
                    if result.returncode or before != g17packeddispatch.gpu_events():
                        raise RuntimeError("half loader failed or GPU diagnostics changed; stop")
                    report["native"] = json.loads(result.stdout)
                    if report["native"] != {"status": 0, "load_only": True, "gpu_dispatched": False}:
                        raise RuntimeError("unexpected half loader reply")
                    if digest(runner) != report["worker_sha256"]:
                        raise RuntimeError("worker changed during loader validation")
                report["status"] = "passed"
        except BaseException as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            evidence.write(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("--rebuild-only", action="store_true")
    p.add_argument("--half-load-approved", action="store_true")
    p.add_argument("--full-size-approved", action="store_true")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--arch-set-experiment", action="store_true", help="prepare the single-variable set ARCH control")
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--columns", type=int, default=1)
    args = p.parse_args()
    if sum((args.rebuild_only, args.half_load_approved, args.prepare)) > 1:
        p.error("choose only one stage")
    if args.arch_set_experiment and not args.prepare:
        p.error("--arch-set-experiment only applies to --prepare")
    if args.prepare:
        report = prepare(args.bundle, args.rows, args.columns, arch_flag=args.arch_set_experiment)
    elif args.rebuild_only:
        report = rebuild_probe(args.bundle)
    elif args.half_load_approved:
        report = load_only(args.bundle, half_validation_approved=True, full_size_approved=args.full_size_approved)
    else:
        report = verify_bundle(args.bundle)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
