"""Persistent native scan service. Importing this module never loads Metal.

One worker owns one pipeline, queue, and pair of device buffers. Calls serialize
under a lock; each returned array owns its storage. Any transport/command/result
failure poisons the service, with no implicit retry or worker restart.
"""
from __future__ import annotations

import fcntl
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


@dataclass(frozen=True)
class StorageLayout:
    """Host storage contract; describing a layout grants no dispatch eligibility.

    Half programs return scores only. Their completion check is a finite score
    replacing each NaN sentinel, command completion, and intact boundary guards.
    Scalar programs additionally return one explicit completion word per row.
    """

    rows: int
    columns: int
    dtype: str = "float32"

    def __post_init__(self):
        if (type(self.rows) is not int or type(self.columns) is not int or
                not 1 <= self.rows <= 500_000 or not 1 <= self.columns <= 384):
            raise ValueError("storage shape must be within 500000 x 384")
        if self.dtype not in ("float16", "float32", "uint32"):
            raise ValueError("unsupported native storage dtype")

    @property
    def numpy_dtype(self):
        return np.dtype({"float16":"<f2", "float32":"<f4", "uint32":"<u4"}[self.dtype])

    @property
    def matrix_bytes(self):
        return self.rows * self.columns * self.numpy_dtype.itemsize

    @property
    def query_bytes(self):
        return self.columns * self.numpy_dtype.itemsize

    @property
    def reply_bytes(self):
        return self.rows * {"float16":2,"float32":8,"uint32":4}[self.dtype]

    @property
    def output_bytes(self):
        return self.reply_bytes + 128

    def write_snapshot(self, matrix, path):
        """Write the owned device input once, using at most 1024 rows of scratch.

        Keep half storage half, including noncontiguous caller input. Return only
        per-column bounds; no reference to the caller's matrix survives here.
        """
        a = np.asarray(matrix)
        if a.dtype != np.float16 or a.shape != (self.rows, self.columns):
            raise ValueError("snapshot requires a matching FP16 matrix")
        maxima = np.zeros(self.columns, dtype=np.float64)
        with Path(path).open("wb") as f:
            for start in range(0, self.rows, 1024):
                chunk = np.array(a[start:start + 1024], dtype=self.numpy_dtype, order="C", copy=True)
                if not np.isfinite(chunk).all():
                    raise ValueError("matrix must be finite")
                maxima = np.maximum(maxima, np.max(np.abs(chunk), axis=0))
                chunk.tofile(f)
            np.zeros(self.columns, dtype=self.numpy_dtype).tofile(f)
        return maxima


@dataclass(frozen=True)
class BufferTransportLayout:
    """Protocol-2 framing dimensions, independent of the scan's width limit.

    Keep the previous maximum matrix population. Per-graph allocation limits
    and actual request/reply lengths are checked separately before dispatch.
    """
    rows: int
    columns: int
    dtype: str = 'float32'

    def __post_init__(self):
        if (type(self.rows) is not int or type(self.columns) is not int or
                not 1 <= self.rows <= 500_000 or self.columns < 1 or
                self.rows * self.columns > 500_000 * 384):
            raise ValueError('transport dimensions exceed the bounded matrix population')
        if self.dtype not in ('float16', 'float32', 'uint32', 'uint16'):
            raise ValueError('unsupported native storage dtype')

    @property
    def numpy_dtype(self):
        return np.dtype({'float16':'<f2', 'float32':'<f4', 'uint32':'<u4', 'uint16':'<u2'}[self.dtype])

    @property
    def matrix_bytes(self):
        return self.rows * self.columns * self.numpy_dtype.itemsize


class Worker:
    """Bounded binary framing over a private subprocess; stdout is never a log.

    Legacy protocol 1 returns one result per row. Explicit result counts or
    marker settings require protocol 2 and an exact layout in the handshake.
    Protocol 2 may also declare a distinct reply dtype. A mixed-dtype worker
    must name that dtype in its handshake; equal byte counts are not enough
    to distinguish, for example, 2048 halves from 1024 floats.
    """

    def __init__(self, command, *, rows, columns, timeout=30, storage_dtype="float32",
                 request_elements=None, reply_elements=None, completion_markers=None, reply_dtype=None, header_limit=4096, reply_value_policy="finite-v1"):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if type(header_limit) is not int or header_limit not in (4096,65536,1048576):
            raise ValueError("unsupported native header limit")
        self.header_limit=header_limit
        self.rows, self.columns, self.timeout = rows, columns, timeout
        self.protocol = 2 if reply_elements is not None or completion_markers is not None or reply_dtype is not None else 1
        if header_limit!=4096 and self.protocol!=2:
            raise ValueError("expanded headers require explicit protocol 2 layout")
        if reply_value_policy not in ("finite-v1", "raw-bits-v1"):
            raise ValueError("unsupported reply value policy")
        if reply_value_policy=="raw-bits-v1" and (self.protocol!=2 or completion_markers is not False):
            raise ValueError("raw reply bits require explicit protocol 2 without completion markers")
        self.reply_value_policy=reply_value_policy
        layout = BufferTransportLayout if self.protocol == 2 else StorageLayout
        self.storage = layout(rows, columns, storage_dtype)
        self.reply_dtype = storage_dtype if reply_dtype is None else reply_dtype
        self.reply_numpy_dtype = layout(rows, columns, self.reply_dtype).numpy_dtype
        self.request_elements = columns if request_elements is None else request_elements
        if type(self.request_elements) is not int or not 1 <= self.request_elements <= 500_000:
            raise ValueError("native request must contain 1 to 500000 elements")
        if storage_dtype in ("uint32", "uint16") and self.protocol != 2:
            raise ValueError(storage_dtype+" storage requires an explicit protocol 2 layout")
        self.reply_elements = rows if reply_elements is None else reply_elements
        if type(self.reply_elements) is not int or not 1 <= self.reply_elements <= 500_000:
            raise ValueError("native reply must contain 1 to 500000 elements")
        self.completion_markers = (storage_dtype == self.reply_dtype == "float32" if completion_markers is None
                                   else completion_markers)
        if type(self.completion_markers) is not bool or (self.completion_markers and
                (storage_dtype != "float32" or self.reply_dtype != "float32" or self.reply_elements != rows)):
            raise ValueError("row completion markers require one FP32 result per row")
        self.reply_bytes = (self.reply_elements*self.reply_numpy_dtype.itemsize +
                            (4*rows if self.completion_markers else 0))
        self.sequence = 0
        self.closed = False
        self.lock = threading.RLock()
        self.log = tempfile.TemporaryFile()
        self.proc = None
        try:
            self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=self.log, bufsize=0)
            header, payload = self._receive(0)
            if (header.get("rows"), header.get("columns")) != (rows, columns) or payload:
                raise RuntimeError("worker shape differs from the verified image")
            self.identity = header.get("identity")
            if not isinstance(self.identity, dict):
                raise RuntimeError("missing worker resource identity")
            if storage_dtype == "float16" and (
                    self.identity.get("storage_dtype") != storage_dtype or
                    self.identity.get("matrix_bytes") != self.storage.matrix_bytes):
                raise RuntimeError("worker half storage differs from the requested layout")
            if self.protocol == 2:
                expected = dict(storage_dtype=storage_dtype, matrix_bytes=self.storage.matrix_bytes,
                    request_bytes=self.request_elements*self.storage.numpy_dtype.itemsize,
                    reply_bytes=self.reply_bytes, reply_elements=self.reply_elements,
                    completion_markers=self.completion_markers)
                if header_limit!=4096 or 'header_limit' in self.identity:
                    expected['header_limit']=header_limit
                if header_limit!=4096:
                    expected['reply_selection']='final-output-only'
                if self.reply_dtype != storage_dtype or 'reply_dtype' in self.identity:
                    expected['reply_dtype'] = self.reply_dtype
                if any(type(self.identity.get(k)) is not type(v) or self.identity[k] != v
                       for k,v in expected.items()):
                    raise RuntimeError("worker explicit result layout differs from the requested layout")
            self.last_header = header
        except BaseException:
            self.close()
            raise

    def _read(self, size, deadline):
        result = bytearray()
        while len(result) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.proc.stdout], [], [], remaining)[0]:
                raise TimeoutError("native worker timed out; no retry")
            data = os.read(self.proc.stdout.fileno(), size - len(result))
            if not data:
                raise RuntimeError("native worker exited or truncated its reply; no scores accepted")
            result.extend(data)
        return bytes(result)

    def _receive(self, sequence):
        deadline = time.monotonic() + self.timeout
        size, = struct.unpack("<I", self._read(4, deadline))
        if not 1 <= size <= self.header_limit:
            raise RuntimeError("invalid native header size")
        header = json.loads(self._read(size, deadline))
        expected = 0 if sequence == 0 else self.reply_bytes
        if (not isinstance(header, dict) or
                any(type(header.get(k)) is not int for k in ("protocol", "sequence", "bytes")) or
                header.get("protocol") != self.protocol or
                header.get("sequence") != sequence or header.get("bytes") != expected):
            raise RuntimeError("invalid native reply sequence or length")
        return header, self._read(expected, deadline)

    def query(self, vector):
        with self.lock:
            if self.closed:
                raise RuntimeError("native worker is closed or failed")
            try:
                if self.storage.dtype in ("uint32", "uint16") and np.asarray(vector).dtype != self.storage.numpy_dtype:
                    raise ValueError(self.storage.dtype+" queries require exact "+self.storage.dtype+" storage; implicit conversion refused")
                vector = np.asarray(vector, dtype=self.storage.numpy_dtype)
                if vector.shape != (self.request_elements,) or not np.isfinite(vector).all():
                    raise ValueError("wrong query width")
                data = vector.tobytes()
                # Scan requests contain one query; affine requests contain an
                # input vector. Both have a fixed size bounded at construction.
                frame = struct.pack("<I", len(data)) + data
                deadline = time.monotonic() + self.timeout
                while frame:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not select.select([], [self.proc.stdin], [], remaining)[1]:
                        raise TimeoutError("native request timed out")
                    count = os.write(self.proc.stdin.fileno(), frame)
                    if count <= 0:
                        raise RuntimeError("native request pipe closed")
                    frame = frame[count:]
                self.sequence += 1
                self.last_raw_reply = None
                header, payload = self._receive(self.sequence)
                # Keep the complete bounded frame even if numerical, guard or
                # identity validation below rejects it. Diagnostic owners can
                # persist evidence; this never admits a rejected reply.
                self.last_raw_reply = (header, payload)
                if (header.get("status") != 0 or header.get("boundary_guard") is not True or
                        header.get("identity") != self.identity):
                    raise RuntimeError("native command, boundary guard, or resource identity failed")
                result = np.frombuffer(payload, dtype=self.reply_numpy_dtype).copy()
                if (self.completion_markers and
                        not np.all(result.view(np.uint32)[self.reply_elements:] == 0x5A17C0DE)):
                    raise RuntimeError("native query did not complete every row")
                if self.reply_value_policy=="finite-v1" and not np.isfinite(result[:self.reply_elements]).all():
                    raise RuntimeError("native query returned nonfinite values")
                self.last_header = header
                return result[:self.reply_elements].copy()
            except BaseException:
                self.close()
                raise

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.proc is not None:
                self.proc.stdin.close()
                if self.proc.poll() is None:
                    try:
                        self.proc.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                        self.proc.wait(timeout=5)
                self.proc.stdout.close()
            self.log.seek(0)
            self.stderr = self.log.read().decode("utf-8", errors="replace")[-8000:]
            self.log.close()

