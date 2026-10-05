"""Compatibility entry point for native scan admission and library worker transport.

The reusable framing implementation lives in agxforge.g17.transport. NativeScan
retains workload-specific image admission here until the image API migration.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import select
import shutil
import struct
import subprocess
import tempfile
import threading
import time

import numpy as np


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agxforge.g17.transport import StorageLayout, BufferTransportLayout, Worker


class NativeScan:
    """Owned matrix snapshot and synchronous FP16 API for checked small images.

    The current runtime deliberately retains the small-fixture launch gate.
    Half validation requires explicit acceptance of the named ARCH hypothesis.
    A timeout kills the host worker; it cannot guarantee cancellation of GPU work.
    """

    def __init__(self, matrix, bundle, *, dispatch_approved=False, max_queries=3, timeout=30,
                 half_validation_approved=False, full_size_approved=False):
        if not dispatch_approved:
            raise ValueError("native execution requires explicit dispatch approval")
        a = np.asarray(matrix)
        if a.dtype != np.float16 or a.ndim != 2 or not all(a.shape):
            raise ValueError("matrix must be a nonempty finite FP16 array")
        full = full_size_approved and half_validation_approved and a.shape == (500_000, 384)
        if (a.shape[0] > 128 or a.shape[1] > 384) and not full:
            raise ValueError("native execution remains limited to small fixtures (128 x 384)")
        if any(not np.isfinite(a[start:start + 1024]).all() for start in range(0, len(a), 1024)):
            raise ValueError("matrix must be finite")
        if isinstance(max_queries, bool) or not isinstance(max_queries, int) or not 1 <= max_queries <= 100:
            raise ValueError("max_queries must be an explicit integer in 1..100")
        self.rows, self.columns = a.shape
        self.max_queries, self.calls = max_queries, 0
        self.lock = threading.RLock()
        self.closed = False
        self.worker = None
        self.gpu_lock = None
        self.directory = tempfile.TemporaryDirectory(prefix="g17-native-")
        try:
            import g17packedcheck
            import g17packeddispatch
            import g17scalarabi
            source, target = Path(bundle).resolve(), Path(self.directory.name)
            manifest = json.loads((source / "manifest.json").read_text())
            half = isinstance(manifest.get("profile"), str)
            self.storage_dtype = "float16" if half else "float32"
            self.abi_hypothesis = None
            if half:
                import g17halfruntime
                if not half_validation_approved:
                    raise ValueError("half validation requires explicit approval of the ARCH hypothesis")
                if g17halfruntime.shape(manifest) != a.shape:
                    raise ValueError("matrix shape differs from the compiled image")
                g17halfruntime.snapshot(source, target, loader=True)
                self.half_check = g17halfruntime.verify_bundle(target)
                if not g17halfruntime.loader_verified(target):
                    raise ValueError("runtime requires half loader-only evidence for the exact bundle and worker")
                self.archive_sha256 = manifest["sha256"]["archive"]
                self.abi_hypothesis = self.half_check["abi_hypothesis"]
            else:
                if full:
                    raise ValueError("full-size validation requires direct half storage")
                if manifest.get("profile", {}).get("name") != g17scalarabi.PROFILE:
                    raise ValueError("runtime requires the measured scalar-v2 ABI or checked half validation image")
                if (manifest.get("rows"), manifest.get("columns")) != a.shape:
                    raise ValueError("matrix shape differs from the compiled image")
            # Freeze artifacts before checking: concurrent builds cannot replace a
            # verified archive between verification and worker pipeline creation.
            names = set() if half else set(manifest["sha256"]) | {"manifest.json", "loader-check.json"}
            for name in names:
                if Path(name).name != name:
                    raise ValueError("bundle members must be plain file names")
                shutil.copyfile(source / name, target / name)
            frozen = json.loads((target / "manifest.json").read_text())
            if frozen != manifest:
                raise ValueError("bundle manifest changed while taking its snapshot")
            if not half:
                g17packedcheck.check_bundle(target)
                if not g17packeddispatch.loader_verified(target, frozen):
                    raise ValueError("runtime requires loader-only evidence for the exact archive")
                self.archive_sha256 = frozen["sha256"]["scan.arc.metallib"]
            # A per-column absolute maximum provides a conservative domain check
            # per query without rescanning or retaining another matrix-sized array.
            packed_file = target / ("runtime-input.f16" if half else "runtime-input.f32")
            self.column_max = StorageLayout(self.rows, self.columns, self.storage_dtype).write_snapshot(a, packed_file)
            # Shared across worktrees. Other GPU tools must cooperate with this
            # advisory lock; it does not claim to exclude unrelated GPU clients.
            import g17gpulock
            self.gpu_lock = g17gpulock.acquire("exclusive")
            self.events = g17packeddispatch.gpu_events()
            self.worker = Worker([str(Path(__file__).with_name("g17scanworker")), str(target),
                                  str(packed_file), str(max_queries),
                                  "--half-full-validation-approved" if full else
                                  "--half-validation-approved" if half else "--dispatch-approved"],
                                 rows=self.rows, columns=self.columns, timeout=timeout,
                                 storage_dtype=self.storage_dtype)
            self._check_events()
            self.identity = dict(self.worker.identity)
        except BaseException:
            self.close()
            raise

    def _check_events(self):
        import g17packeddispatch
        if g17packeddispatch.gpu_events() != self.events:
            raise RuntimeError("GPU diagnostics changed; native session stopped")

    def __call__(self, vector):
        with self.lock:
            if self.closed:
                raise RuntimeError("native scan is closed or failed")
            x = np.asarray(vector)
            if x.dtype != np.float16 or x.shape != (self.columns,) or not np.isfinite(x).all():
                raise ValueError("query must be a finite FP16 vector of the index width")
            if float(self.column_max @ np.abs(x.astype(np.float64))) > 60000:
                raise ValueError("query exceeds conservative finite-output domain")
            if self.calls >= self.max_queries:
                raise RuntimeError("approved query budget exhausted")
            try:
                self._check_events()
                raw = self.worker.query(x)
                self.calls += 1
                self._check_events()
                with np.errstate(over="ignore"):
                    result = raw.astype(np.float16)
                if not np.isfinite(result).all():
                    raise RuntimeError("native scores exceed the FP16 output contract")
                if self.storage_dtype == "float32":
                    self.last_fp32 = raw
                self.last_scores = result.copy()
                self.last_report = dict(self.worker.last_header)
                return result
            except BaseException:
                self.__dict__.pop("last_fp32", None)
                self.__dict__.pop("last_scores", None)
                self.__dict__.pop("last_report", None)
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

    def __exit__(self, *exc):
        self.close()
