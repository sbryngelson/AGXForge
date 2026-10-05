"""Staged loader and persistent runtime for CPU-verified common images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np

import g17commonimage as image
import g17commonruntime as runtime
from g17native import Worker

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = "common-loader-check.json"
RUNNER = "common-worker"


def digest(path):
    return image.sha(Path(path).read_bytes())


def snapshot(source, target, *, loaded=False):
    """Own all delivered inputs before checking them or launching a process."""
    source, target = Path(source), Path(target)
    names = image.FILES + ((RECEIPT, RUNNER) if loaded else ())
    before = {name: digest(source/name) for name in names}
    for name in names:
        shutil.copy2(source/name, target/name)
    if before != {name: digest(target/name) for name in names} or \
            before != {name: digest(source/name) for name in names}:
        raise ValueError("bundle changed while taking its snapshot")
    return runtime.ImageContract.read(json.loads((target/"manifest.json").read_text()))


def source_check(bundle):
    result = subprocess.run([sys.executable, str(ROOT/"tools/g17commonimage.py"),
                             str(bundle), "--rebuild"], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError("common image rebuild refused: " + (result.stderr or result.stdout)[-2000:])
    report = json.loads(result.stdout)
    if report.get("status") != "passed" or report.get("class_assumptions"):
        raise ValueError("common image rebuild did not establish the delivered contract")
    return report


def stage_shape(contract, full_size):
    full = contract.kind == "packed_scan" and (contract.shape.rows, contract.shape.columns) == (500_000, 384)
    if full_size and not full:
        raise ValueError("full stage requires exactly the 500000 x 384 packed scan")
    if contract.shape.rows > 128 and not (full and full_size):
        raise ValueError("small stages are bounded to 128 rows; full scan is a separate stage")
    return full


def lock_gpu():
    # the machine-wide lock (tools/g17gpulock.py): waits rather than raising when another
    # dispatcher holds it, and is inherited from a holding parent such as the suite
    import g17gpulock
    return g17gpulock.acquire("exclusive")


def build_runner(destination):
    """Compile a private worker from verified, copied C sources before loading."""
    import g17buildaudit
    paths = ("tools/g17commonworker.m", "tools/g17scanstorage.h", "tools/g17gpulock.h")
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    expected = g17buildaudit.committed_hashes(ROOT, revision, paths)
    with tempfile.TemporaryDirectory(prefix="g17-common-worker-build-") as tmp:
        tmp = Path(tmp)
        for name in paths:
            data = (ROOT/name).read_bytes()
            if image.sha(data) != expected[name]:
                raise ValueError(f"worker source is not committed: {name}")
            (tmp/Path(name).name).write_bytes(data)
        result = subprocess.run(["clang", "-fobjc-arc", "-O2", "-Wall", "-Wextra", "-Werror",
            "-framework", "Foundation", "-framework", "Metal", "-o", str(tmp/RUNNER),
            str(tmp/"g17commonworker.m")], capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError("worker compilation failed: " + result.stderr[-2000:])
        with Path(destination).open("xb") as out:
            out.write((tmp/RUNNER).read_bytes())
        Path(destination).chmod(0o700)
    return dict(commit=revision, inputs=expected, sha256=digest(destination))


def receipt_matches(bundle):
    try:
        bundle = Path(bundle)
        receipt = json.loads((bundle/RECEIPT).read_text())
        return (receipt["status"] == "passed" and receipt["returncode"] == 0 and
            receipt["native"] == dict(status=0, load_only=True, gpu_dispatched=False) and
            receipt["rebuild"]["status"] == "passed" and
            receipt["files"] == {name: digest(bundle/name) for name in image.FILES} and
            receipt["worker"]["sha256"] == digest(bundle/RUNNER))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def load_only(bundle, *, approved=False, full_size=False):
    if not approved:
        raise ValueError("loader stage requires explicit selection")
    import g17packeddispatch
    bundle = Path(bundle).resolve()
    # Reserve evidence before any Metal call. A failed stage cannot silently retry.
    with (bundle/RECEIPT).open("x") as evidence:
        report = dict(status="pending", gpu_dispatched=False)
        try:
            with tempfile.TemporaryDirectory(prefix="g17-common-load-") as tmp:
                frozen = Path(tmp)
                contract = snapshot(bundle, frozen)
                full = stage_shape(contract, full_size)
                report["offline"] = image.verify(frozen)
                report["rebuild"] = source_check(frozen)
                report["worker"] = build_runner(bundle/RUNNER)
                shutil.copy2(bundle/RUNNER, frozen/RUNNER)
                report["files"] = {name: digest(frozen/name) for name in image.FILES}
                with lock_gpu():
                    before = g17packeddispatch.gpu_events()
                    result = subprocess.run([str(frozen/RUNNER), str(frozen),
                        "--full-load-approved" if full else "--load-approved"],
                        capture_output=True, text=True, timeout=30)
                    report.update(returncode=result.returncode, stderr=result.stderr[-4000:])
                    if result.returncode or before != g17packeddispatch.gpu_events():
                        raise RuntimeError("loader failed or GPU diagnostics changed; stop")
                    report["native"] = json.loads(result.stdout)
                    if report["native"] != dict(status=0, load_only=True, gpu_dispatched=False):
                        raise RuntimeError("unexpected loader response")
                    if digest(frozen/RUNNER) != report["worker"]["sha256"]:
                        raise RuntimeError("loader executable changed")
                report["status"] = "passed"
        except BaseException as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            evidence.write(json.dumps(report, indent=2)+"\n")
    return report


class NativeProgram:
    """One owned input snapshot and one worker; a failed query closes the session."""
    def __init__(self, matrix, bundle, *, approved=False, full_size=False, queries=3, timeout=30,
                 parameters=None):
        if not approved:
            raise ValueError("dispatch stage requires explicit selection")
        if type(queries) is not int or not 1 <= queries <= 16:
            raise ValueError("validation stages permit one to sixteen queries")
        self.lock = threading.RLock()
        self.closed, self.calls, self.queries = False, 0, queries
        self.worker = self.gpu_lock = None
        self.directory = tempfile.TemporaryDirectory(prefix="g17-common-runtime-")
        try:
            import g17packeddispatch
            frozen = Path(self.directory.name)
            self.contract = snapshot(bundle, frozen, loaded=True)
            full = stage_shape(self.contract, full_size)
            self.fp32 = self.contract.kind in ("layernorm", "minilm_query")
            if not self.fp32 and queries not in (1, 3, 4):
                raise ValueError("scan and affine stages permit one, three or four queries")
            if not self.fp32 and queries == 4 and not full:
                raise ValueError("four-query stage is reserved for the full packed scan")
            if not self.fp32 and parameters is not None:
                raise ValueError("separate parameters require an FP32 four-buffer program")
            self.offline = image.verify(frozen)
            if not receipt_matches(frozen):
                raise ValueError("no successful loader receipt for this exact image and executable")
            self.rebuild = source_check(frozen)
            layout = self.contract.layout()
            if self.fp32:
                writer = (runtime.write_query_inputs if self.contract.kind == "minilm_query"
                          else runtime.write_layernorm_inputs)
                self.input_snapshots = writer(matrix, parameters, frozen, self.contract)
                matrix_file, storage_dtype, width = frozen, "float32", 4
                result_options = dict(reply_elements=layout["reply_elements"], completion_markers=False)
            else:
                matrix_file, storage_dtype, width = frozen/"matrix.f16", "float16", 2
                self.maxima = runtime.write_matrix(matrix, matrix_file, self.contract)
                result_options = {}
            self.gpu_lock = lock_gpu()
            self.events = g17packeddispatch.gpu_events()
            self.worker = Worker([str(frozen/RUNNER), str(frozen), str(matrix_file), str(queries),
                "--full-dispatch-approved" if full else "--dispatch-approved"],
                rows=self.contract.shape.rows, columns=self.contract.shape.columns,
                request_elements=layout["request_bytes"]//width,
                storage_dtype=storage_dtype, timeout=timeout, **result_options)
            self.check_events()
            self.identity = dict(self.worker.identity)
            expected = dict(matrix_bytes=layout["matrix_bytes"], output_bytes=layout["output_bytes"],
                buffer_allocations=layout["buffer_allocations"], bindings=layout["bindings"],
                pipeline_builds=1, matrix_uploads=1, storage_dtype=storage_dtype)
            if self.fp32:
                expected.update(parameter_uploads=2, buffer_offsets=layout["buffer_offsets"],
                    buffer_payload_bytes=layout["buffer_payload_bytes"],
                    buffer_allocation_bytes=layout["buffer_allocation_bytes"])
            if any(self.identity.get(key) != value for key, value in expected.items()):
                raise RuntimeError("worker resource identity differs from the common contract")
        except BaseException:
            self.close()
            raise

    def check_events(self):
        import g17packeddispatch
        if g17packeddispatch.gpu_events() != self.events:
            raise RuntimeError("GPU diagnostics changed; session stopped")

    def __call__(self, request):
        with self.lock:
            if self.closed:
                raise RuntimeError("native program is closed or failed")
            if self.calls >= self.queries:
                raise RuntimeError("validation query budget exhausted")
            # Own the request before validating it; caller mutation cannot change
            # the bytes sent after the finite/domain checks.
            dtype, width = (np.float32, 4) if self.fp32 else (np.float16, 2)
            if not isinstance(request, np.ndarray) or request.dtype != dtype:
                raise ValueError(f"request must be a {np.dtype(dtype).name} array")
            x = request.copy(order="C")
            count = self.contract.layout()["request_bytes"]//width
            shape = (self.contract.shape.rows, self.contract.shape.columns) if self.fp32 else (count,)
            if x.shape != shape or not np.isfinite(x).all():
                raise ValueError("request must be finite and match the program shape")
            if not self.fp32:
                bound = (2*float(np.max(np.abs(x.astype(np.float64))))+1
                         if self.contract.kind == "affine" else
                         float(self.maxima @ np.abs(x.astype(np.float64))))
                if bound > 60000:
                    raise ValueError("request exceeds conservative finite FP16 output domain")
            try:
                self.check_events()
                result = self.worker.query(x.reshape(-1))
                self.calls += 1
                self.check_events()
                self.last_report = dict(self.worker.last_header)
                if self.fp32 and self.last_report.get("readonly_inputs") is not True:
                    raise RuntimeError("worker did not verify all readonly FP32 inputs")
                if self.fp32:
                    self.last_request_bytes = x.tobytes()
                return result.reshape(shape) if self.fp32 else result
            except BaseException:
                self.close()
                raise

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.worker is not None:
                self.worker.close()
            if self.gpu_lock is not None:
                self.gpu_lock.close()
            self.directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--load-approved", action="store_true")
    parser.add_argument("--full-size", action="store_true")
    args = parser.parse_args()
    report = load_only(args.bundle, approved=args.load_approved, full_size=args.full_size)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
