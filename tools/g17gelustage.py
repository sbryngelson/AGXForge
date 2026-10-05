"""Prepare the retained MiniLM GELU for staged validation; no Metal calls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import g17gelugraph as G

ROOT = Path(__file__).resolve().parents[1]


def sha(data): return hashlib.sha256(data).hexdigest()


def cases(source, campaign=True):
    if not campaign: return [('real', source.copy())]
    near = np.linspace(-0.01, 0.01, source.size, dtype=np.float32).reshape(source.shape)
    tails = np.resize(np.array([-40., -20., -9.4, -9.2, 0., 9.2, 9.4, 20., 40.],
                              np.float32), source.size).reshape(source.shape)
    return [('real', source.copy()), ('negative', -np.abs(source)),
            ('near_zero', near), ('tails', tails), ('repeat_real', source.copy())]


def source(rows):
    G.elements(rows)
    with np.load(ROOT/'results/g17-minilm-ffn-fixture-v1/fp64-reference.npz',
                 allow_pickle=False) as data:
        return data['expand'][:rows].astype(np.float32)


def build(rows=1):
    import g17geluimage as I
    import g17buffergraph
    image, compiled, abi, _ = I.author()
    native = {'program.bin': bytes(compiled.code), 'program.o': image.object,
              'program.lib.metallib': image.library, 'program.arc.metallib': image.archive}
    for name, data in native.items():
        retained = ROOT/'results/g17-minilm-gelu-image-v1/programs/gelu'/name
        if retained.read_bytes() != data:
            raise ValueError('reviewed GELU image moved: '+name)
    plain = compiled.abi_plain(abi)
    graph = G.graph(rows); requirements = G.requirements(rows, plain, sha(compiled.code))
    g17buffergraph.require(graph, requirements)
    manifest = dict(format='g17-attention-images-v1', graph=graph, programs={'gelu': dict(
        name=compiled.contract().name, abi=plain,
        instructions=compiled.contract().to_dict()['instructions'],
        sha256={n: sha(b) for n, b in native.items()}, field_ledger=image.field_ledger)})
    files = {'programs/gelu/'+n: b for n, b in native.items()}
    for name, value in [('manifest', manifest), ('graph', graph), ('requirements', requirements)]:
        files[name+'.json'] = (json.dumps(value, indent=2)+'\n').encode()
    x = source(rows)
    if x.shape != (rows, G.WIDTH) or not np.isfinite(x).all():
        raise ValueError('invalid GELU source fixture')
    files['inputs/source.f32'] = x.astype('<f4').tobytes()
    return files


def prepare(destination, rows=1):
    import g17buildaudit
    destination = Path(destination)
    if destination.exists(): raise FileExistsError(destination)
    files, audit = g17buildaudit.verified_build(ROOT, lambda: build(rows))
    import g17attentionadmit as A, g17attentionstage as W
    import g17packedcheck as D, g17normcheck as N
    import g17minilmffn as F, g17minilmquerycheck as C
    destination.mkdir(parents=True)
    report = dict(status='pending', rows=rows, source=audit, gpu_dispatched=False,
                  loader_eligible=False, dispatch_eligible=False,
                  predictions={}, prediction_files={})
    try:
        for name, data in files.items():
            path = destination/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        report['image'] = A.inspect(destination, expected_graph=G.graph(rows),
                                    requirements=json.loads(files['requirements.json']))
        if report['image']['image_blockers']: raise ValueError('structural GELU admission refused')
        decoded = D.decode(files['programs/gelu/program.bin'])
        x = np.frombuffer(files['inputs/source.f32'], dtype='<f4').reshape(rows, G.WIDTH)
        (destination/'predictions').mkdir()
        for name, value in cases(x):
            expected = F.gelu_reference(value)
            output, confidence, notes = N.simulate_threads(decoded,
                {0: value.reshape(-1).tolist(), 1: [float('nan')]*value.size},
                [(1, 0, False), (2, 2, True)], value.size)
            predicted = np.asarray(output[1], np.float32).reshape(value.shape)
            comparison = C.compare(predicted, expected)
            if comparison['failures']: raise ValueError('GELU prediction exceeds fixed budget: '+name)
            path = destination/'predictions'/(name+'.npz')
            np.savez(path, source=value, prediction=predicted, reference=expected)
            report['prediction_files'][str(path.relative_to(destination))] = sha(path.read_bytes())
            report['predictions'][name] = dict(comparison=comparison, confidence=confidence,
                model_notes=notes, input_sha256=sha(value.tobytes()),
                reference_sha256=sha(expected.tobytes()), prediction_sha256=sha(predicted.tobytes()))
        report['worker'] = W.build_worker(destination/'attention-worker')
        run = subprocess.run([str((destination/'attention-worker').resolve()),
            str((destination/'graph.json').resolve()), '--describe-schedule'],
            capture_output=True, text=True, timeout=10)
        if run.returncode: raise ValueError('GELU native schedule refused: '+run.stderr)
        report['native_schedule'] = json.loads(run.stdout)
        if report['native_schedule'].get('gpu_dispatched') is not False:
            raise ValueError('unexpected planning reply')
        report['files'] = {n: sha(b) for n, b in files.items()}
        report['files']['attention-worker'] = report['worker']['sha256']
        report.update(status='prepared_for_staged_validation', loader_eligible=True, dispatch_eligible=True,
            limitations='CPU prediction and bounded experiment only. Hardware approximation/dependency '
            'behavior and the 49-instruction image with slot 32 absent are not yet established. '
            'Input is the FP64 projection reference rounded to FP32; no resident FFN composition.')
    except BaseException as error:
        report.update(status='refused', error=str(error)); raise
    finally: (destination/'preparation.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--rows', type=int, choices=[1, 32], default=1)
    args = parser.parse_args(); result = prepare(args.destination, args.rows)
    print(json.dumps({k: result[k] for k in ('status', 'rows', 'loader_eligible', 'dispatch_eligible', 'gpu_dispatched')}))
