#!/usr/bin/env python3
"""Prepare and exercise the persistent below-Metal runtime's tensor-tile control.

This first runtime control is 32x32x64 half GEMM, not the complete FFN or
attention workload. Those application schedules will use this resident runner.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import select
import struct
import subprocess
import tempfile
import time

import numpy as np
import g17authoredtensorffntile as graph
import g17gpulock
from g17purecommandprobe import events, recovery

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'spike/agxsub/g17pure3_preflight.c'
HEADER = ROOT / 'spike/agxsub/g17workload_tile.h'
MAGIC = 0x47575231
PROGRAM_SHA = '472570faa7d227ee02179ac3dc83e0f81155913b6b3e6f7d7210f5fc8df18283'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def control_operand(value, shape):
    value = np.asarray(value)
    if value.shape != shape or not np.isfinite(value).all() or not np.equal(value, np.rint(value)).all() or np.any(np.abs(value) > 4):
        raise ValueError('refused: runtime control requires integer-valued operands in -4..4 with shape ' + str(shape))
    return value.astype('<f2')


def compile_control():
    import g17cc
    import g17tensordelivery
    compiled = g17cc.compile_function(g17tensordelivery.program())
    if compiled.abi_plain(compiled.abi())['system_registers'] != [130]:
        raise ValueError('refused: control compiler metadata class changed')
    return compiled.code


def prepare(destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('refused: bundle destination already exists')
    code = compile_control()
    c = struct.pack('<I', 0x7fc01234) * 1024
    if sha(code) != PROGRAM_SHA:
        raise ValueError('refused: tensor-tile program identity changed')
    # Small signed integers make all products and all FP32 sums exact, so an
    # independent logical matrix product is a bit-exact oracle for this control.
    rng = np.random.default_rng(292)
    a = rng.integers(-4, 5, (32, 64)).astype('<f2').tobytes()
    b = rng.integers(-4, 5, (64, 32)).astype('<f2').tobytes()
    blobs = {'program.bin': code, 'a.f16': a, 'b.f16': b,
             'ordered.bin': graph.requests(), 'pages.bin': graph.pages(),
             'physical.bin': graph.payload(code, a, b, c)}
    destination.mkdir(parents=True)
    for name, raw in blobs.items():
        (destination / name).write_bytes(raw)
    manifest = {'schema': 1, 'workload': 'tensor-tile-runtime-control',
                'shape': [32, 32, 64], 'operand_type': 'half',
                'graph': 'measured FFN tile, 29 registrations',
                'native_program_sha256': PROGRAM_SHA,
                'program_origin': 'ordinary g17cc.compile_function(g17tensordelivery.program())',
                'files': {name: {'bytes': len(raw), 'sha256': sha(raw)}
                          for name, raw in blobs.items()}}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('schema') != 1 or manifest.get('workload') != 'tensor-tile-runtime-control':
        raise ValueError('refused: unmeasured workload bundle')
    names = {'program.bin', 'a.f16', 'b.f16', 'ordered.bin', 'pages.bin', 'physical.bin'}
    if set(manifest['files']) != names:
        raise ValueError('refused: bundle inventory differs')
    for name, identity in manifest['files'].items():
        raw = (directory / name).read_bytes()
        if len(raw) != identity['bytes'] or sha(raw) != identity['sha256']:
            raise ValueError('refused: bundle identity mismatch: ' + name)
    if sha((directory / 'program.bin').read_bytes()) != PROGRAM_SHA:
        raise ValueError('refused: program outside measured runtime class')
    # Reconstruct the launch surfaces rather than trusting a self-signed hash
    # inventory to admit a changed packet, pointer, or resource graph.
    a, b = ((directory / name).read_bytes() for name in ('a.f16', 'b.f16'))
    if len(a) != 4096 or len(b) != 4096:
        raise ValueError('refused: operand size outside tensor-tile class')
    control_operand(np.frombuffer(a, dtype='<f2').reshape(32, 64), (32, 64))
    control_operand(np.frombuffer(b, dtype='<f2').reshape(64, 32), (64, 32))
    c = struct.pack('<I', 0x7fc01234) * 1024
    expected = {'ordered.bin': graph.requests(), 'pages.bin': graph.pages(),
                'physical.bin': graph.payload((directory / 'program.bin').read_bytes(), a, b, c)}
    for name, raw in expected.items():
        if (directory / name).read_bytes() != raw:
            raise ValueError('refused: launch surface outside measured class: ' + name)
    return manifest


def read_exact(stream, length, seconds=30):
    deadline = time.monotonic() + seconds
    chunks = bytearray()
    while len(chunks) < length:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([stream], [], [], remaining)[0]:
            raise TimeoutError('below-Metal runtime response deadline')
        raw = os.read(stream.fileno(), length - len(chunks))
        if not raw:
            raise RuntimeError('below-Metal runtime closed before reply')
        chunks.extend(raw)
    return bytes(chunks)


def write_all(stream, raw):
    view = memoryview(raw)
    while view:
        written = stream.write(view)
        if not written:
            raise BrokenPipeError('below-Metal runtime request stream closed')
        view = view[written:]


class Session:
    """One native process owns one graph, program, and weight upload."""
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = verify_bundle(self.directory)
        self.stack = ExitStack()
        self.process = None
        self.calls = 0

    def launch_environment(self):
        return dict(ORDERED_ALLOC_PREFLIGHT=str(self.directory / 'ordered.bin'),
                    STAGE_CAPTURED_PAYLOAD=str(self.directory / 'physical.bin'),
                    STAGE_COMMAND_PAGES=str(self.directory / 'pages.bin'),
                    ORDERED_TENSOR_GRAPH='1', ORDERED_QUEUE_PREFLIGHT='1',
                    ORDERED_FFN_TILE='1', ORDERED_WORKLOAD_TILE='1')

    def expected_ready(self):
        return (MAGIC, 1, 4096, 0)

    def request_selector(self):
        return 0

    def __enter__(self):
        try:
            self.stack.enter_context(g17gpulock.acquire('exclusive', timeout=30))
            self.before_recovery, self.before_events = recovery(), events()
            started = time.perf_counter()
            # The measured queue descriptor admits executable paths <29 bytes.
            tmp = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix='gw', dir='/tmp')))
            binary = tmp / 'p'
            subprocess.run(['xcrun', 'clang', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-Wno-unused-function', '-fblocks', '-DG17_BLOCK_PREFLIGHT',
                            str(SOURCE), '-framework', 'IOKit', '-framework', 'CoreFoundation',
                            '-o', str(binary)], check=True, capture_output=True, timeout=30)
            self.build_seconds = time.perf_counter() - started
            self.binary_sha256 = sha(binary.read_bytes())
            # Remove every legacy trial switch so a caller's environment cannot
            # silently turn this session into a different preflight experiment.
            runtime_sources = SOURCE.read_text() + ''.join(p.read_text() for p in HEADER.parent.glob('g17workload_*.h'))
            switches = set(re.findall(r'getenv\("([^"\n]+)"\)', runtime_sources))
            env = {k: v for k, v in os.environ.items() if k not in switches}
            env.update(self.launch_environment())
            self.log = self.stack.enter_context((tmp / 'native.log').open('w+b'))
            started = time.perf_counter()
            self.process = subprocess.Popen([str(binary)], env=env, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
            ready = struct.unpack('<4I', read_exact(self.process.stdout, 16))
            if ready != self.expected_ready():
                raise RuntimeError('below-Metal runtime prepared a different class')
            self.prepare_seconds = time.perf_counter() - started
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def execute(self, a, readback=True):
        a = control_operand(a, (32, 64))
        if self.calls >= 64:
            raise ValueError('refused: bounded session admits at most 64 executions')
        started = time.perf_counter_ns()
        self.calls += 1
        write_all(self.process.stdin, struct.pack('<4I', MAGIC, 1 if readback else 2, 4096, self.request_selector()) + a.tobytes())
        response = read_exact(self.process.stdout, 48, seconds=10)
        words = struct.unpack('<8I2Q', response)
        self.last_reply = words
        output_raw = read_exact(self.process.stdout, words[6]) if words[6] == 4096 else b''
        self.last_output_sha256 = sha(output_raw) if output_raw else None
        if words[:6] != (MAGIC, 0, 0, 63, 1, 1) or words[6] != (4096 if readback else 0):
            raise RuntimeError('below-Metal execution failed status/callback/readonly/guard checks: ' + str(words))
        if words[7] != self.calls:
            raise RuntimeError('below-Metal execution sequence mismatch')
        output = np.frombuffer(output_raw, dtype='<f4').reshape(32, 32).copy() if readback else None
        return output, {'submit_call_ns': words[8], 'submit_to_completion_ns': words[9],
                        'client_roundtrip_ns': time.perf_counter_ns() - started,
                        'output_readback_bytes': words[6], 'input_upload_bytes': 4096,
                        'input_sha256': sha(a.tobytes()), 'output_sha256': self.last_output_sha256,
                        'submits': 1, 'callbacks': 2, 'guards_and_readonly': True}

    def __exit__(self, *exc):
        try:
            if self.process is not None:
                if self.process.poll() is None:
                    try:
                        write_all(self.process.stdin, struct.pack('<4I', MAGIC, 0, 0, 0))
                        self.process.wait(timeout=5)
                    except (BrokenPipeError, subprocess.TimeoutExpired):
                        self.process.kill(); self.process.wait()
                self.native_returncode = self.process.returncode
                self.process.stdin.close(); self.process.stdout.close()
                self.log.seek(0); self.native_log = self.log.read().decode(errors='replace')
                self.after_recovery = recovery()
                self.new_events = sorted(events() - self.before_events)
        finally:
            self.stack.close()


def run(directory, mode, iterations=3):
    if not 1 <= iterations <= 16:
        raise ValueError('refused: iterations must be 1..16')
    b = np.frombuffer((Path(directory) / 'b.f16').read_bytes(), dtype='<f2').reshape(64, 32).astype(np.float64)
    rng = np.random.default_rng(293)
    rows = []
    session = Session(directory)
    error = None
    try:
        with session:
            if mode != 'prepare':
                for i in range(iterations):
                    a = rng.integers(-4, 5, (32, 64)).astype('<f2')
                    # Benchmark one validated warm-up, then avoid intermediate readback.
                    output, timing = session.execute(a, readback=mode != 'benchmark' or i in (0, iterations - 1))
                    exact = None if output is None else bool(np.array_equal(output, (a.astype(np.float64) @ b).astype(np.float32)))
                    rows.append({'iteration': i, 'bit_exact': exact, **timing})
                    if exact is False:
                        raise RuntimeError('independent logical GEMM comparison failed')
    except (RuntimeError, TimeoutError, subprocess.CalledProcessError) as exc:
        # Preserve failure evidence and return a failed report; main exits nonzero.
        # Programming errors and unsupported bundle errors still propagate.
        error = type(exc).__name__ + ': ' + str(exc)
    passed = (error is None and getattr(session, 'native_returncode', None) == 0 and
              not getattr(session, 'new_events', None) and
              getattr(session, 'before_recovery', None) == getattr(session, 'after_recovery', None))
    report = {'scope': 'persistent runtime control, not full FFN/attention',
              'workload': '32x32x64 half TensorOps GEMM', 'mode': mode, 'passed': passed,
              'processes': 1, 'graph_initializations': 1, 'weight_uploads': 1, 'program_uploads': 1,
              'program_sha256': PROGRAM_SHA, 'native_binary_sha256': getattr(session, 'binary_sha256', None),
              'native_source_sha256': sha(SOURCE.read_bytes()), 'runtime_header_sha256': sha(HEADER.read_bytes()),
              'native_build_seconds': getattr(session, 'build_seconds', None),
              'driver_prepare_seconds': getattr(session, 'prepare_seconds', None),
              'executions': rows, 'completed_executions': len(rows), 'issued_requests': session.calls,
              'numerically_checked_executions': sum(row['bit_exact'] is True for row in rows),
              'before_recovery': getattr(session, 'before_recovery', None),
              'after_recovery': getattr(session, 'after_recovery', None),
              'new_gpu_events': getattr(session, 'new_events', None),
              'native_returncode': getattr(session, 'native_returncode', None),
              'native_log': getattr(session, 'native_log', None), 'error': error,
              'last_reply': getattr(session, 'last_reply', None),
              'timing_scope': 'host clocks: submit call, submit-to-callback, client roundtrip; not GPU timestamps'}
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('prepare', 'execute', 'validate', 'benchmark'))
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--report', type=Path)
    ap.add_argument('--iterations', type=int, default=3)
    args = ap.parse_args()
    if args.report and args.report.exists():
        ap.error('refusing to overwrite report')
    if args.command == 'prepare' and not args.bundle.exists():
        prepare(args.bundle)
    result = run(args.bundle, args.command, args.iterations)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'native_log'}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
