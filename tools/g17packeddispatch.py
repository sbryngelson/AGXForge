#!/usr/bin/env python3
"""Dispatch one prepared small scan three times in isolated processes, after explicit GPU approval."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import g17packedcheck
import g17packedscan
import g17scan

# Do not replay a profile whose first loader check failed. A future corrected
# contract needs its own version and offline evidence before hardware validation.
FAILED_PROFILES = {"scalar-buffer-two-bindings-v1"}


def loader_verified(directory, manifest):
    """A loader result authorizes only its exact archive, not every image in a class."""
    try:
        result = json.loads((directory / "loader-check.json").read_text())
    except (OSError, ValueError):
        return False
    return (result.get("status") == "passed" and result.get("returncode") == 0
            and result.get("archive_sha256") == manifest.get("sha256", {}).get("scan.arc.metallib")
            and result.get("native") == {"status": 0, "load_only": True, "gpu_dispatched": False})


def gpu_events():
    roots = (Path.home() / "Library/Logs/DiagnosticReports", Path("/Library/Logs/DiagnosticReports"))
    return {str(p): p.stat().st_mtime_ns for root in roots for p in root.glob("*")
            if p.is_file() and any(s in p.name.lower() for s in ("gpu", "agx", "windowserver"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--dispatch-approved", action="store_true")
    args = parser.parse_args()
    if not args.dispatch_approved:
        parser.error("requires Spencer's explicit OK under docs/execution-validation.md")
    directory = args.bundle.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("profile", {}).get("name") in FAILED_PROFILES:
        raise ValueError("this profile failed pipeline validation; hardware attempts are blocked "
                         "pending a corrected ABI contract (ledger/g17-packed-scan-pipeline-failure.toml)")
    if not loader_verified(directory, manifest):
        raise ValueError("profile has no completed loader-only validation; hardware attempts are blocked")
    g17packedcheck.check_bundle(directory)
    if manifest["rows"] > 128 or manifest["columns"] > 384:
        raise ValueError("only the prepared small validation fixtures may be dispatched")
    with np.load(directory / "fixture.npz", allow_pickle=False) as f:
        a, x = f["index"], f["embedding"]
    packed = g17packedscan.PackedIndex(a)
    if packed.set_query(x).tobytes() != (directory / "packed.f32").read_bytes():
        raise ValueError("raw input differs from the reference fixture")
    fp32_reference = np.zeros(a.shape[0], dtype=np.float32)
    for k in range(a.shape[1]):
        fp32_reference = fp32_reference + a[:, k].astype(np.float32) * np.float32(x[k])
    runner = Path(__file__).with_name("g17packedrun")
    if not runner.exists():
        raise FileNotFoundError("compile tools/g17packedrun.m before requesting hardware validation")
    attempts, results = [], []
    report = {"archive_sha256": manifest["sha256"]["scan.arc.metallib"], "attempts": attempts,
              "status": "pending", "scope": "small fixture GPU validation; not a performance measurement"}
    before = gpu_events()
    try:
        for i in range(3):
            # A stale file from a prior success can never stand in for a failed run.
            (directory / "output.f32").unlink(missing_ok=True)
            started = time.monotonic()
            proc = subprocess.run([str(runner), str(directory), "--dispatch-approved"],
                                  capture_output=True, text=True, timeout=30)
            attempt = {"run": i + 1, "returncode": proc.returncode,
                       "seconds": time.monotonic() - started, "stderr": proc.stderr[-2000:]}
            stages = [line.removeprefix("phase=") for line in proc.stderr.splitlines()
                      if line.startswith("phase=")]
            attempt["last_stage"] = stages[-1] if stages else "unknown"
            attempt["gpu_submission_attempted"] = (
                any(s in stages for s in ("submitting", "submitted", "command_finished"))
                if stages else None)
            attempts.append(attempt)
            if proc.returncode or gpu_events() != before:
                raise RuntimeError("runner failed at %s or a new GPU diagnostic appeared; batch stopped"
                                   % attempt["last_stage"])
            native = json.loads(proc.stdout)
            output = np.fromfile(directory / "output.f32", dtype=np.float32)
            scores = packed.finish(output, native["status"])
            fp32_exact = np.array_equal(output[:a.shape[0]].view(np.uint32), fp32_reference.view(np.uint32))
            if not fp32_exact:
                raise ValueError("GPU FP32 scores differ from the sequential reference; batch stopped")
            valid = g17scan.check_output(a, x, scores)
            attempt.update(native=native, numerical_check=valid, fp32_bit_exact=True)
            if not valid["ok"]:
                raise ValueError("GPU result disagrees with the independent reference; batch stopped")
            results.append(output.copy())
        if any(not np.array_equal(results[0].view(np.uint32), y.view(np.uint32)) for y in results[1:]):
            raise ValueError("repeated GPU results disagree")
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (directory / "gpu-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
