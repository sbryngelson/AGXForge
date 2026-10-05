"""The actual four-stage MiniLM FFN, using individually executed native images."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ROWS=32
OUTPUTS=('expand','gelu','contract','output')
SHAPES={'expand':(32,1536),'gelu':(32,1536),'contract':(32,384),'output':(32,384)}
PARAMETERS={'expand_weight':'intermediate.dense.weight','expand_bias':'intermediate.dense.bias',
            'contract_weight':'output.dense.weight','contract_bias':'output.dense.bias',
            'gamma':'output.LayerNorm.weight','beta':'output.LayerNorm.bias'}


def sha(data):return hashlib.sha256(data).hexdigest()


def arrays():
    with np.load(ROOT/'results/g17-minilm-ffn-fixture-v1/fixture.npz',allow_pickle=False) as f:
        return {'source':f['source'].copy(),**{k:f[v].copy() for k,v in PARAMETERS.items()}}


def graph():
    allocations={}
    shapes={'source':(32,384),'expand_weight':(1536,384),'expand_bias':(1536,),
            'contract_weight':(384,1536),'contract_bias':(384,),'gamma':(384,),'beta':(384,),**SHAPES}
    for name,shape in shapes.items():
        size=4*int(np.prod(shape))
        role='input' if name=='source' else 'output' if name=='output' else 'intermediate' if name in OUTPUTS else 'parameter'
        allocations[name]=dict(shape=list(shape),role=role,element_type='float',payload_bytes=size,
                               allocation_bytes=size+256,offset=128)
    stages=[]
    for name,program,grid,inputs in [
        ('expand','expand',[1536,32,1],['source','expand_weight','expand_bias']),
        ('gelu','gelu',[49152,1,1],['expand']),
        ('contract','contract',[384,32,1],['gelu','contract_weight','contract_bias']),
        ('output','residualnorm',[32,1,1],['contract','source','gamma','beta'])]:
        bindings=[dict(index=i+1,allocation=a,offset=128,length=allocations[a]['payload_bytes'],written=a==name)
                  for i,a in enumerate(inputs+[name])]
        stages.append(dict(name=name,program=program,grid=grid,bindings=bindings))
    return dict(format='g17-attention-graph-v1',status='proposed_not_executed',allocations=allocations,stages=stages)


def requirements(programs):
    specs={'expand':([1536,32,1],[12288,589824,1536,49152],[160,161]),
           'gelu':([49152,1,1],[49152,49152],[160]),
           'contract':([384,32,1],[49152,589824,384,12288],[160,161]),
           'residualnorm':([32,1,1],[12288,12288,384,384,12288],[160])}
    out={}
    for name,(grid,elements,registers) in specs.items():
        p=programs[name]
        if p['abi']['system_registers']!=registers:raise ValueError('FFN coordinate contract changed: '+name)
        out[name]=dict(abi=p['abi'],code_sha256=p['sha256']['program.bin'],exact_grid=grid,binding_elements=elements)
    return out


def build():
    import g17ffnstage as P,g17gelustage as A,g17residualnormstage as N,g17buffergraph
    files={};programs={}
    # Each existing builder pins all native files to its reviewed delivery.
    for name,part in [('expand',P.build('expand',32)),('gelu',A.build(32)),
                      ('contract',P.build('contract',32)),('residualnorm',N.build())]:
        manifest=json.loads(part['manifest.json']);programs[name]=manifest['programs'][name]
        files.update({k:v for k,v in part.items() if k.startswith('programs/'+name+'/')})
    g=graph();req=requirements(programs);g17buffergraph.require(g,req)
    manifest=dict(format='g17-attention-images-v1',graph=g,programs=programs)
    for name,value in [('graph',g),('requirements',req),('manifest',manifest)]:
        files[name+'.json']=(json.dumps(value,indent=2)+'\n').encode()
    for name,value in arrays().items():
        allocation=g['allocations'][name]
        if value.dtype!=np.float32 or list(value.shape)!=allocation['shape'] or not np.isfinite(value).all():
            raise ValueError('FFN checkpoint allocation differs: '+name)
        files['inputs/'+name+'.f32']=value.astype('<f4').tobytes()
    return files


def read_arrays(files):
    return {name:np.frombuffer(files['inputs/'+name+'.f32'],dtype='<f4').reshape(a['shape']).copy()
            for name,a in graph()['allocations'].items() if a['role'] in ('input','parameter')}


def reference(source,parameters):
    import g17minilmffn as F
    x=np.asarray(source)
    if x.dtype!=np.float32 or x.shape!=(32,384) or not np.isfinite(x).all():raise ValueError('invalid FFN source')
    p={k:v.astype(np.float64) for k,v in parameters.items()}
    expand=x.astype(np.float64)@p['expand_weight'].T+p['expand_bias']
    gelu=F.gelu_reference(expand)
    contract=gelu@p['contract_weight'].T+p['contract_bias']
    residual=contract+x.astype(np.float64)
    centered=residual-residual.mean(axis=1,keepdims=True)
    output=centered/np.sqrt((centered*centered).mean(axis=1,keepdims=True)+1e-12)*p['gamma']+p['beta']
    return dict(expand=expand,gelu=gelu,contract=contract,output=output)


def check(source,parameters,outputs):
    import g17minilmquerycheck as C
    if tuple(outputs)!=OUTPUTS:raise ValueError('expected four FFN stage readbacks')
    refs=reference(source,parameters);checks={}
    for name in OUTPUTS:
        if outputs[name].dtype!=np.float32:raise ValueError('FFN readback must be FP32')
        checks[name]=C.compare(outputs[name],refs[name])
    return dict(status='passed' if all(c['failures']==0 for c in checks.values()) else 'failed',
        stages=checks,complete_block=True,
        scope='Every actual stage against the full FP64 chain from original input; no intermediate FP32 rounding in the reference.')


def simulate(files,source,parameters):
    import g17packedcheck as D,g17queryimagecheck as M,g17normcheck as N,g17residualnormcheck as R
    decoded={n:D.decode(files['programs/'+n+'/program.bin']) for n in ('expand','gelu','contract','residualnorm')}
    bindings=[(1,0,False),(2,2,False),(3,4,False),(4,6,True)]
    expand,_=M.simulate(decoded['expand'],source,parameters['expand_weight'],parameters['expand_bias'],bindings)
    got,confidence,notes=N.simulate_threads(decoded['gelu'],{0:expand.reshape(-1).tolist(),1:[float('nan')]*expand.size},
                                           [(1,0,False),(2,2,True)],expand.size)
    gelu=np.asarray(got[1],np.float32).reshape(32,1536)
    contract,_=M.simulate(decoded['contract'],gelu,parameters['contract_weight'],parameters['contract_bias'],bindings)
    output,model=R.simulate(decoded['residualnorm'],[(1,0,False),(2,2,False),(3,4,False),(4,6,False),(5,8,True)],
                            contract,source,parameters['gamma'],parameters['beta'])
    return dict(expand=expand,gelu=gelu,contract=contract,output=output),dict(gelu_confidence=confidence,
        gelu_notes=notes,residualnorm_model=model,scope='Sequential delivered-byte composition; hardware dependencies remain a separate check.')
