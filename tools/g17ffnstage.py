"""Prepare bounded projection experiments with the shared worker; no GPU calls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import g17ffngraph as G

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'results/g17-minilm-ffn-fixture-v1/fixture.npz'


def sha(data):return hashlib.sha256(data).hexdigest()


def cases(x,campaign=True):
    return [('real',x.copy()),('negative',-x),('repeat_real',x.copy())] if campaign else [('real',x.copy())]


def arrays(kind,rows):
    import g17minilmffn as F
    G.dimensions(kind,rows)
    with np.load(FIXTURE,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    if kind=='expand':x=data['source'][:rows];prefix='intermediate.dense'
    else:
        x=F.gelu_reference(F.dense_reference(data['source'][:rows],
            data['intermediate.dense.weight'],data['intermediate.dense.bias'])).astype(np.float32)
        prefix='output.dense'
    return dict(source=x.copy(),weight=data[prefix+'.weight'],bias=data[prefix+'.bias'])


def build(kind,rows=1):
    import g17ffnimage as I,g17buffergraph
    p=I.build(rows)[kind];code=p['files']['program.bin'];graph=G.graph(kind,rows)
    # Pin the reviewed image identities, not merely the compiler's latest bytes.
    for name,data in p['files'].items():
        retained=ROOT/'results/g17-minilm-ffn-images-v1/programs'/kind/name
        if data!=retained.read_bytes():raise ValueError('reviewed projection image moved: '+name)
    requirements=G.requirements(kind,rows,p['abi'],sha(code));g17buffergraph.require(graph,requirements)
    manifest=dict(format='g17-attention-images-v1',graph=graph,programs={kind:dict(
        name=p['contract']['name'],abi=p['abi'],instructions=p['contract']['instructions'],
        sha256={n:sha(b) for n,b in p['files'].items()},field_ledger=p['field_ledger'])})
    files={'programs/'+kind+'/'+n:b for n,b in p['files'].items()}
    for name,value in [('manifest',manifest),('requirements',requirements),('graph',graph)]:
        files[name+'.json']=(json.dumps(value,indent=2)+'\n').encode()
    for name,a in arrays(kind,rows).items():
        if a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('invalid projection input')
        files['inputs/'+name+'.f32']=a.astype('<f4').tobytes()
    return files


def prepare(destination,kind,rows=1):
    import g17buildaudit
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    files,source=g17buildaudit.verified_build(ROOT,lambda:build(kind,rows))
    import g17attentionadmit as A,g17attentionstage as W,g17packedcheck as D
    import g17queryimagecheck as M,g17minilmffn as F,g17minilmquerycheck as C
    ni,no=G.dimensions(kind,rows);destination.mkdir(parents=True)
    report=dict(status='pending',kind=kind,rows=rows,source=source,gpu_dispatched=False,
        loader_eligible=False,dispatch_eligible=False,predictions={},prediction_files={})
    try:
        for name,data in files.items():
            path=destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        report['image']=A.inspect(destination,expected_graph=G.graph(kind,rows),
                                 requirements=json.loads(files['requirements.json']))
        if report['image']['image_blockers']:raise ValueError('structural image admission refused')
        decoded=D.decode(files['programs/'+kind+'/program.bin'])
        x=np.frombuffer(files['inputs/source.f32'],dtype='<f4').reshape(rows,ni)
        w=np.frombuffer(files['inputs/weight.f32'],dtype='<f4').reshape(no,ni)
        b=np.frombuffer(files['inputs/bias.f32'],dtype='<f4')
        (destination/'predictions').mkdir()
        for name,value in cases(x):
            expected=F.dense_reference(value,w,b)
            predicted,execution=M.simulate(decoded,value,w,b,[(1,0,False),(2,2,False),(3,4,False),(4,6,True)])
            result=C.compare(predicted,expected)
            if result['failures']:raise ValueError('delivered arithmetic exceeded fixed projection budget')
            path=destination/'predictions'/(name+'.npz')
            np.savez(path,source=value,prediction=predicted,reference=expected)
            report['prediction_files'][str(path.relative_to(destination))]=sha(path.read_bytes())
            report['predictions'][name]=dict(comparison=result,execution=execution,
                input_sha256=sha(value.tobytes()),reference_sha256=sha(expected.tobytes()),
                prediction_sha256=sha(predicted.tobytes()))
        report['worker']=W.build_worker(destination/'attention-worker')
        result=subprocess.run([str((destination/'attention-worker').resolve()),
            str((destination/'graph.json').resolve()),'--describe-schedule'],
            capture_output=True,text=True,timeout=10)
        if result.returncode:raise ValueError('native schedule refused: '+result.stderr)
        report['native_schedule']=json.loads(result.stdout)
        if report['native_schedule'].get('gpu_dispatched') is not False:raise ValueError('unexpected planning reply')
        report['files']={n:sha(b) for n,b in files.items()}
        report['files']['attention-worker']=report['worker']['sha256']
        report.update(status='prepared_for_staged_validation',loader_eligible=True,dispatch_eligible=True,
            limitations='Bounded projection experiment, not validated execution. Synchronous prediction '
            'does not establish waits, lifetimes or mask behavior. Contraction input is FP64 GELU '
            'reference rounded to FP32, not output of a native GELU stage.')
    except BaseException as error:report.update(status='refused',error=str(error));raise
    finally:(destination/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('kind',choices=['expand','contract'])
    parser.add_argument('destination',type=Path);parser.add_argument('--rows',type=int,choices=[1,32],default=1)
    args=parser.parse_args();r=prepare(args.destination,args.kind,args.rows)
    print(json.dumps({k:r[k] for k in ('status','kind','rows','loader_eligible','dispatch_eligible','gpu_dispatched')}))
