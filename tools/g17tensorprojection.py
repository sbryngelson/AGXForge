"""CPU-only delivery for a full FP16-operand MiniLM query projection.

GPU packing -> 72 unchanged K64 tensor tiles -> FP32 reduction and bias.
This is explicitly a quantized-operand variant of the retained FP32 application.
The proposed graph requires disjoint buffer views and remains non-admissible
until the common validators, worker and delivered-byte checks support it.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE='results/g17-minilm-query-admission/fixture.npz'


def pack_ir(depth=384):
    depth=_depth(depth)
    import g17ir as ir
    source=ir.Buffer('source',1,elem=ir.F32)
    packed=ir.Buffer('packed_a',2,elem=ir.F16)
    f=ir.Function('tensor_projection_pack',[source,packed]);b=ir.Builder(f,f.block('entry'))
    t=b.builtin('thread_position_in_grid')
    band=getattr(b,'and')
    if depth==384:
        # Preserve the already executed pack image byte for byte.
        row=band(b.shr(t,ir.Imm(6)),ir.Imm(31))
        block=b.shr(t,ir.Imm(11));k=band(t,ir.Imm(63))
        index=b.add(b.add(b.mul(row,b.const(384)),b.mul(block,b.const(64))),k)
        target=t
    else:
        # 2-D source coordinates keep the mask operand inside its measured
        # domain, instead of AND-ing a flattened index up to 49151.
        row=b.builtin('thread_position_in_grid',axis='y')
        block=b.shr(t,ir.Imm(6));k=band(t,ir.Imm(63))
        index=b.add(b.mul(row,b.const(depth)),t)
        target=b.add(b.mul(b.add(b.mul(block,b.const(32)),row),b.const(64)),k)
    value=b.load(source,index,width='word')
    b.store_at(packed,target,b.f32_to_f16_rte(value),width='half');b.ret()
    return f


def _columns(columns):
    if type(columns) is not int or columns<32 or columns>12288 or columns%32:
        raise ValueError('columns must be a multiple of 32 within the measured index domain 0..12287')
    return columns


def _depth(depth):
    if type(depth) is not int or depth not in (384,1536):
        raise ValueError('projection depth must be 384 or 1536')
    return depth


def finish_ir(columns=384,depth=384):
    columns=_columns(columns);blocks=_depth(depth)//64
    import g17ir as ir
    partial,bias,out=[ir.Buffer(n,i,elem=ir.F32) for i,n in enumerate(('partials','bias','output'),1)]
    f=ir.Function('tensor_projection_finish',[partial,bias,out]);b=ir.Builder(f,f.block('entry'))
    column=b.builtin('thread_position_in_grid');row=b.builtin('thread_position_in_grid',axis='y')
    block=b.shr(column,ir.Imm(5));local=getattr(b,'and')(column,ir.Imm(31))
    base=b.add(b.add(b.mul(block,b.const(blocks*1024)),b.mul(row,b.const(32))),local)
    total=b.load(partial,base,width='word')
    for k in range(1,blocks):
        total=b.fadd(total,b.load(partial,b.add(base,b.const(k*1024)),width='word'))
    value=b.fadd(total,b.load(bias,column,width='word'))
    b.store_at(out,b.add(b.mul(row,b.const(columns)),column),value,width='word');b.ret()
    return f


def programs(columns=384,depth=384):
    columns=_columns(columns);depth=_depth(depth)
    import g17cc,g17tensordelivery
    return {name:g17cc.compile_function(f()) for name,f in
            (('pack',lambda:pack_ir(depth)),('matmul',g17tensordelivery.program),('finish',lambda:finish_ir(columns,depth)))}


def requirements(compiled,columns=384,depth=384):
    columns=_columns(columns);depth=_depth(depth);blocks=depth//64
    sizes={'pack':([12288,1,1] if depth==384 else [depth,32,1],[32*depth,32*depth]),
           'matmul':([32,1,1],[2048,2048,1024]),
           'finish':([columns,32,1],[columns*blocks*32,columns,columns*32])}
    return {name:dict(code_sha256=hashlib.sha256(p.code).hexdigest(),abi=p.abi_plain(p.abi()),
                     exact_grid=sizes[name][0],binding_elements=sizes[name][1]) for name,p in compiled.items()}


def graph(req,columns=384,depth=384):
    columns=_columns(columns);nblocks=columns//32;depth=_depth(depth);kblocks=depth//64
    allocations={}
    def alloc(name,shape,kind,role):
        size=int(np.prod(shape))*({'float':4,'half':2}[kind])
        allocations[name]=dict(shape=list(shape),element_type=kind,role=role,
                              payload_bytes=size,offset=128,allocation_bytes=size+256)
    alloc('source',(32,depth),'float','input');alloc('packed_a',(kblocks,32,64),'half','intermediate')
    alloc('packed_b',(nblocks,kblocks,64,32),'half','parameter');alloc('partials',(nblocks,kblocks,32,32),'float','intermediate')
    alloc('bias',(columns,),'float','parameter');alloc('output',(32,columns),'float','output')
    stages=[]
    def stage(name,program,windows):
        r=req[program];bindings=[]
        for index,(allocation,start,length) in enumerate(windows,1):
            bindings.append(dict(index=index,allocation=allocation,offset=128+start,length=length,
                                 written=index==len(windows)))
        s=dict(name=name,program=program,grid=r['exact_grid'],bindings=bindings)
        if r['abi'].get('execution') is not None:s['execution']=r['abi']['execution']
        stages.append(s)
    stage('pack','pack',[('source',0,32*depth*4),('packed_a',0,32*depth*2)])
    for n in range(nblocks):
        for k in range(kblocks):
            tile=n*kblocks+k
            stage('matmul_n%d_k%d'%(n,k),'matmul',[
                ('packed_a',k*4096,4096),('packed_b',tile*4096,4096),('partials',tile*4096,4096)])
    stage('finish','finish',[('partials',0,columns*kblocks*32*4),('bias',0,columns*4),('output',0,columns*32*4)])
    return dict(format='g17-attention-graph-v1',status='proposed_not_executed',binding_windows='disjoint-v1',allocations=allocations,stages=stages,
                uploads='FP32 source once per request; packed FP16 weights and FP32 bias once per session; no host intermediate transfers',
                execution=f'{2+nblocks*kblocks} sequential encoders; each tensor tile uses the unchanged one-SIMD-group K64 image',
                view_requirements='Disjoint writes publish only their byte ranges; reads require full coverage from earlier stages; no in-place aliasing')


def packed_inputs(source,weight):
    source=np.asarray(source);weight=np.asarray(weight)
    if source.ndim!=2 or source.shape[0]!=32 or weight.ndim!=2 or weight.shape[1]!=source.shape[1] or source.dtype!=np.float32 or weight.dtype!=np.float32:
        raise ValueError('projection requires FP32 source[32,K] and weight[N,K]')
    columns=_columns(weight.shape[0]);depth=_depth(source.shape[1]);blocks=depth//64
    with np.errstate(over='ignore'):
        a=source.astype('<f2').reshape(32,blocks,64).transpose(1,0,2).copy()
        b=weight.astype('<f2').reshape(columns//32,32,blocks,64).transpose(0,2,3,1).copy()
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite quantized operand')
    return a,b


def finish_reference(partials,bias):
    if partials.ndim!=4 or partials.shape[2:]!=(32,32) or partials.dtype!=np.float32 or bias.shape!=(partials.shape[0]*32,) or bias.dtype!=np.float32:
        raise ValueError('partial/bias layout differs')
    columns=_columns(bias.size);blocks=_depth(partials.shape[1]*64)//64
    total=partials[:,0].copy()
    for k in range(1,blocks):total=np.add(total,partials[:,k],dtype=np.float32)
    return np.add(total.transpose(1,0,2).reshape(32,columns),bias,dtype=np.float32)


def fixture():
    with np.load(ROOT/SOURCE,allow_pickle=False) as z:
        source,weight,bias=[z[k].copy() for k in ('source','weight','bias')]
        original=source.astype(np.float64)@weight.astype(np.float64).T+bias.astype(np.float64)
        if not np.array_equal(original,z['reference']):raise ValueError('application fixture reference changed')
    a,b=packed_inputs(source,weight)
    partials=np.empty((12,6,32,32),np.float32)
    for n in range(12):
        for k in range(6):partials[n,k]=(a[k].astype(np.float64)@b[n,k].astype(np.float64)).astype(np.float32)
    quantized=source.astype(np.float16).astype(np.float64)@weight.astype(np.float16).astype(np.float64).T+bias.astype(np.float64)
    composed=finish_reference(partials,bias)
    return dict(source=source,packed_a=a,packed_b=b,bias=bias,ideal_partials=partials,
                quantized_reference=quantized,original_fp32_reference=original,ideal_composed=composed)


def build():
    import g17buffergraph
    compiled=programs();req=requirements(compiled);candidate=graph(req);values=fixture()
    native=ROOT/'results/g17-tensor-compiled-runtime-v1/programs/matmul/program.bin'
    if compiled['matmul'].code!=native.read_bytes():raise ValueError('K64 compiler program changed')
    files={}
    for name,p in compiled.items():
        files['programs/'+name+'/program.bin']=p.code
        files['programs/'+name+'/abi.json']=(json.dumps(req[name]['abi'],indent=2)+'\n').encode()
    for name,value in values.items():
        stream=io.BytesIO();np.save(stream,value,allow_pickle=False);files[name+'.npy']=stream.getvalue()
    findings=g17buffergraph.validate(candidate,req)
    report=dict(status='cpu_plan_not_admitted',gpu_dispatched=False,loader_eligible=False,dispatch_eligible=False,
        operation='FP16(source) @ FP16(weight).T + FP32(bias), output FP32; distinct from the original FP32-operand projection',
        source=dict(path=SOURCE,sha256=hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest()),
        shape=dict(M=32,N=384,K=384),tensor_tiles=72,stages=74,
        graph_findings=findings,
        quantization_max_abs_difference=float(np.max(np.abs(values['quantized_reference']-values['original_fp32_reference']))),
        ideal_composition_max_abs_difference=float(np.max(np.abs(values['ideal_composed'].astype(np.float64)-values['quantized_reference']))),
        limits='Ideal tile results are FP64 dot products rounded once to FP32, not a tensor arithmetic prediction. No tolerance or dispatch admission assigned. Need common buffer-view/schedule support and delivered-byte checks for pack/finish.')
    for name,value in (('graph.json',candidate),('requirements.json',req),('plan.json',report)):
        files[name]=(json.dumps(value,indent=2)+'\n').encode()
    return files


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);p.add_argument('--verify-source',action='store_true');args=p.parse_args()
    if args.destination.exists():raise ValueError('refusing to overwrite projection delivery')
    if args.verify_source:
        import g17buildaudit
        files,audit=g17buildaudit.verified_build(ROOT,build)
        files['source-audit.json']=(json.dumps(audit,indent=2)+'\n').encode()
    else:files=build()
    args.destination.mkdir(parents=True)
    for name,raw in files.items():
        path=args.destination/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    print(json.dumps(json.loads(files['plan.json']),indent=2))
