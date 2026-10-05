"""Committed-source image delivery for the actual MiniLM rectangular projections."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHAPES = {'expand': (384, 1536), 'contract': (1536, 384)}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(rows=32):
    """Pure compiler/common-author path: no decoder, cache or Metal calls."""
    import g17cc
    import g17minilmffn
    import g17link
    import g17scanlink
    result = {}
    for name, (ni, no) in SHAPES.items():
        program = g17cc.compile_function(g17minilmffn.dense_ir(ni, no, rows))
        abi, contract = program.abi(), program.contract()
        kernel = g17link.Kernel(code=program.code, name=contract.name,
            entry=contract.entry, prologue=contract.prologue,
            bindings=[g17link.Binding(index=b.index, readonly=not b.written,
                        element_type=b.element_type) for b in contract.bindings])
        image = g17scanlink.link(kernel, abi,
            binding_offsets=[b.offset for b in contract.bindings], program_contract=contract)
        result[name] = dict(files={'program.bin': bytes(program.code), 'program.o': image.object,
            'program.lib.metallib': image.library, 'program.arc.metallib': image.archive},
            abi=program.abi_plain(abi), contract=contract.to_dict(),
            field_ledger=image.field_ledger, grid=[no, rows, 1],
            shape=dict(rows=rows, input_width=ni, output_width=no))
    return result


def prepare(destination, rows=32):
    import g17buildaudit
    import g17abi
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    programs, audit = g17buildaudit.verified_build(ROOT, lambda: build(rows))
    # The earlier compiler delivery is a committed reference, not an arbitrary
    # file whose accidental mutation may redefine what "unchanged" means.
    baseline = 'results/g17-minilm-ffn-compiler-v1/'
    paths = [baseline + n + suffix for n in SHAPES for suffix in ('.program.bin', '.abi.json')]
    expected = g17buildaudit.committed_hashes(ROOT, audit['commit'], paths)
    for name, program in programs.items():
        old_code, old_abi = baseline + name + '.program.bin', baseline + name + '.abi.json'
        for path in (old_code, old_abi):
            if sha((ROOT / path).read_bytes()) != expected[path]:
                raise ValueError('retained compiler baseline is dirty: ' + path)
        if sha(program['files']['program.bin']) != expected[old_code]:
            raise ValueError('projection code moved since the retained compiler delivery: ' + name)
        expected_abi = g17abi.with_instruction_count(
            json.loads((ROOT / old_abi).read_text()),
            instructions=program['contract']['instructions'],
            code=(ROOT / old_code).read_bytes())
        if json.loads(json.dumps(program['abi'])) != expected_abi:
            raise ValueError('projection ABI moved since the retained compiler delivery: ' + name)
    manifest = dict(status='authored_unvalidated', source=audit,
        retained_compiler_inputs=expected, programs={}, gpu_dispatched=False,
        pipeline_created=False, loader_eligible=False, dispatch_eligible=False,
        blockers=['integration admission and native worker preparation',
                  'staged loader and hardware correctness validation'],
        scope='Two rectangular projections only; not a complete feed-forward block. '
              'GELU and residual/normalization composition remain separate deliveries.')
    destination.mkdir(parents=True, exist_ok=False)
    for name, program in programs.items():
        folder = destination / 'programs' / name
        folder.mkdir(parents=True)
        for filename, data in program['files'].items():
            (folder / filename).write_bytes(data)
        for filename, data in [('abi.json', program['abi']), ('contract.json', program['contract'])]:
            (folder / filename).write_text(json.dumps(data, indent=2) + '\n')
        manifest['programs'][name] = {k: v for k, v in program.items() if k != 'files'}
        manifest['programs'][name]['sha256'] = {p.name: sha(p.read_bytes()) for p in folder.iterdir()}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--rows', type=int, choices=(1, 32), default=32)
    args = parser.parse_args()
    report = prepare(args.destination, args.rows)
    print(json.dumps(dict(status=report['status'], source_commit=report['source']['commit'],
        verified_inputs=len(report['source']['inputs']), external_processes=0,
        programs={k: v['sha256'] for k, v in report['programs'].items()}, gpu_dispatched=False)))
