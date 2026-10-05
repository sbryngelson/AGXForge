"""Prepare the five-buffer residual norm for the common native executor; CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROBE=ROOT/'results/g17-residualnorm-probe-v1'


def sha(data):return hashlib.sha256(data).hexdigest()


def graph():
    allocations={}
    for name,shape,role in [('source',[32,384],'input'),('residual',[32,384],'parameter'),
                           ('gamma',[384],'parameter'),('beta',[384],'parameter'),('output',[32,384],'output')]:
        size=4*int(np.prod(shape))
        allocations[name]=dict(shape=shape,element_type='float',payload_bytes=size,
                               allocation_bytes=size+256,offset=128,role=role)
    bindings=[dict(index=i+1,allocation=name,offset=128,length=allocations[name]['payload_bytes'],written=name=='output')
              for i,name in enumerate(allocations)]
    return dict(format='g17-attention-graph-v1',status='proposed_not_executed',allocations=allocations,
                stages=[dict(name='output',program='residualnorm',grid=[32,1,1],bindings=bindings)])


def build():
    import g17residualnorm as R
    import g17link as L
    import g17scanlink as S
    import g17buffergraph as G
    p,a=R.compile_program(32,384)
    abi=p.abi();bindings=abi['bindings']
    kernel=L.Kernel(code=p.code,name=p.name,entry=abi['entry'],prologue=abi['prologue'],bindings=[
        L.Binding(index=b['index'],readonly=not b['written'],element_type=b['element_type']) for b in bindings])
    image=S.link(kernel,abi,binding_offsets=[b['offset'] for b in bindings])
    blobs={'program.bin':p.code,'program.o':image.object,'program.lib.metallib':image.library,'program.arc.metallib':image.archive}
    original=json.loads((PROBE/'manifest.json').read_text())
    for key,file in [('code','program.bin'),('object','program.o'),('library','program.lib.metallib'),('archive','program.arc.metallib')]:
        if sha(blobs[file])!=original['sha256'][key]:raise ValueError(f'rebuild differs from specialist {key}')
    plain=json.loads(json.dumps(a))
    requirements={'residualnorm':dict(abi=plain,code_sha256=sha(p.code),exact_grid=[32,1,1],
                                     binding_elements=[12288,12288,384,384,12288])}
    G.require(graph(),requirements)
    manifest=dict(format='g17-attention-images-v1',graph=graph(),programs={'residualnorm':dict(
        name=p.name,abi=plain,instructions=p.contract().to_dict()['instructions'],
        sha256={f:sha(b) for f,b in blobs.items()},field_ledger=image.field_ledger)})
    files={f'programs/residualnorm/{name}':data for name,data in blobs.items()}
    for name,key in [('inputs.npz','inputs'),('fp64-reference.npz','reference')]:
        data=(PROBE/name).read_bytes()
        if sha(data)!=original['sha256'][key]:raise ValueError(f'{name}: specialist hash differs')
        files[name]=data
    files['manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    files['requirements.json']=(json.dumps(requirements,indent=2)+'\n').encode()
    files['graph.json']=(json.dumps(graph(),indent=2)+'\n').encode()
    with np.load(PROBE/'inputs.npz',allow_pickle=False) as data:
        arrays={name:data[name].copy() for name in ('source','residual','gamma','beta')}
    expected=R.reference(*(arrays[name] for name in ('source','residual','gamma','beta')))
    with np.load(PROBE/'fp64-reference.npz',allow_pickle=False) as data:
        if not np.array_equal(expected,data['output']):raise ValueError('independent reference differs')
    for name,array in arrays.items():
        if array.dtype!=np.float32 or not np.isfinite(array).all():raise ValueError('invalid input array')
        files[f'inputs/{name}.f32']=array.astype('<f4').tobytes()
    return files


def prepare(destination):
    import g17buildaudit
    import g17attentionstage
    import g17attentionadmit
    import g17packedcheck
    import g17layernormimagecheck
    import g17residualnormcheck
    destination=Path(destination)
    if destination.exists():raise ValueError('refusing to overwrite a preparation')
    files,source=g17buildaudit.verified_build(ROOT,build)
    destination.mkdir(parents=True)
    report=dict(status='pending',gpu_dispatched=False,loader_eligible=False,source=source)
    try:
        for name,data in files.items():
            path=destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        requirements=json.loads(files['requirements.json'])
        report['image']=g17attentionadmit.inspect(destination,expected_graph=graph(),requirements=requirements)
        if any('omits register' in b for b in report['image']['blockers']):raise ValueError('register declaration missing')
        decoded=g17packedcheck.decode(files['programs/residualnorm/program.bin'])
        report['load_reuse']=g17layernormimagecheck.check_load_reuse(decoded)
        with np.load(destination/'inputs.npz',allow_pickle=False) as data:
            arrays=[data[name].copy() for name in ('source','residual','gamma','beta')]
        bindings=[(b['index'],b['offset'],b['written']) for b in requirements['residualnorm']['abi']['bindings']]
        got,model=g17residualnormcheck.simulate(decoded,bindings,*arrays)
        with np.load(destination/'fp64-reference.npz',allow_pickle=False) as data:expected=data['output'].copy()
        error=np.abs(got.astype(np.float64)-expected);limit=2e-5*(1+np.abs(expected))
        failures=int(np.count_nonzero(error>limit))
        report['prediction']=dict(outputs=int(got.size),failed_outputs=failures,max_abs_error=float(error.max()),
                                  max_budget_fraction=float((error/limit).max()),model=model)
        if failures:raise ValueError('delivered arithmetic exceeds the fixed budget')
        report['worker']=g17attentionstage.build_worker(destination/'attention-worker')
        native=subprocess.run([str((destination/'attention-worker').resolve()),str((destination/'graph.json').resolve()),
                               '--describe-schedule'],capture_output=True,text=True,timeout=10)
        if native.returncode:raise ValueError('native schedule refused: '+native.stderr)
        report['native_schedule']=json.loads(native.stdout)
        if report['native_schedule']['gpu_dispatched'] is not False:raise ValueError('unexpected host planning receipt')
        report['files']={name:sha(data) for name,data in files.items()}
        report['files']['attention-worker']=sha((destination/'attention-worker').read_bytes())
        report.update(status='prepared_for_staged_validation',loader_eligible=True,
            limitations='Admission to a bounded experiment, not validation of GPU results. rsqrt is a bounded prediction outside measured exact inputs. Load-only must pass before dispatch.')
        return report
    except BaseException as error:
        report.update(status='refused',error=str(error));raise
    finally:(destination/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    try:
        r=prepare(args.destination)
        print(json.dumps({k:r[k] for k in ('status','gpu_dispatched','loader_eligible','prediction')},indent=2))
    except Exception as error:parser.exit(2,f'preparation refused: {type(error).__name__}: {error}\n')
