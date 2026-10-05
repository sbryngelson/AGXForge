"""Deliver a real 32x32x64 matmul to the common compiler, without loading Metal.

Inputs are distinct row-major FP16 matrices; FP64 is the independent reference.
Compilation is attempted, but no fragment or successful compile is dispatchable
without the missing complete tensor resource/image contract.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np


def program():
    import g17ir as ir
    buffers = [ir.Buffer('a', 1, elem=ir.F16), ir.Buffer('b', 2, elem=ir.F16),
               ir.Buffer('output', 3, elem=ir.F32)]
    fn = ir.Function('tensor_matmul_32_32_64', buffers)
    builder = ir.Builder(fn, fn.block('entry'))
    builder.tensor_matmul(*buffers, M=32, N=32, K=64)
    builder.ret()
    return fn


def arrays():
    rng = np.random.default_rng(193)
    a = rng.integers(-4, 5, (32, 64)).astype(np.float16)
    b = rng.integers(-4, 5, (64, 32)).astype(np.float16)
    reference = a.astype(np.float64) @ b.astype(np.float64)
    if not np.array_equal(reference, reference.astype(np.float32).astype(np.float64)):
        raise ValueError('integer reference is not exact in FP32')
    return dict(a=a, b=b, reference=reference)


def deliver(destination):
    import g17cc
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    fn = program()
    files = {'program.ir.txt': (str(fn) + '\n').encode()}
    for name, value in arrays().items():
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        files[name + '.npy'] = stream.getvalue()
    report = dict(status='pending', gpu_dispatched=False, loader_eligible=False, dispatch_eligible=False,
        provenance_scope='Application inputs and reference are repository-owned. The diagnostic compiler '
                         'attempt may consult the legacy local tensor cache; this is not a source-verified native image.',
        shape=dict(M=32, N=32, K=64), operation='C = A @ B, no initial C accumulation',
        buffers=[dict(name='a', index=1, dtype='float16', shape=[32, 64], written=False),
                 dict(name='b', index=2, dtype='float16', shape=[64, 32], written=False),
                 dict(name='output', index=3, dtype='float32', shape=[32, 32], written=True)],
        layout='Independent contiguous row-major A, B, and C allocations.',
        reference='Small integer inputs; FP64 A@B is exactly representable in FP32. '
                  'Different A and B separate transpose, operand substitution and self-product errors.',
        remaining_requirements=['Complete lowering of A/B loads, accumulator initialization, MACs and C readout',
            'Tensor allocation, load/MAC dependency and lifetime rules',
            'Semantic compiler/linker contract for descriptors, relocations, constant program and launch state',
            'Repository-owned native image followed by staged hardware validation'],
        limitations='No tensor launch geometry is asserted by this application delivery. '
                    'The old compiler tensor.seq fragment is not a complete matrix multiply.')
    try:
        compiled = g17cc.compile_function(fn)
        abi = compiled.abi_plain(compiled.abi())
        files['program.bin'] = compiled.code
        files['abi.json'] = (json.dumps(abi, indent=2) + '\n').encode()
        report.update(status='compiled_unvalidated', code_bytes=len(compiled.code))
    except (KeyError, ValueError, RuntimeError, NotImplementedError, g17cc.Unsupported) as error:   # Unsupported: the compiler's precise refusal
        report.update(status='compiler_refused', error=type(error).__name__ + ': ' + str(error))
    registry = getattr(g17cc, '_TREG', None) or {}
    if (32, 32, 64) in registry:
        seq = registry[(32, 32, 64)]
        report['legacy_fragment'] = dict(source=seq.source, mac_units=len(seq.macs), bound_units=len(seq.bounds),
            scope='Cache-derived fragment selection, not a complete program or native-image provenance.')
    for name, data in files.items():
        (destination / name).write_bytes(data)
    report['files'] = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    (destination / 'delivery.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination', type=Path)
    a = p.parse_args()
    r = deliver(a.destination)
    print(json.dumps(r, indent=2))
    p.exit(2 if r['status'] == 'compiler_refused' else 0)
