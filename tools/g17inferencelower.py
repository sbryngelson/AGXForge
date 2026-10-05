"""Compile complete bounded MiniLM graphs to native stage programs; no dispatch."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g17modelimport import import_model
from g17inferencekernels import gather_rows_ir
from agxforge.g17 import ir, cc
from agxforge.g17.inferencegraph import Unsupported

POLICY = 'half_transport_fp32_accumulate_v1'


def function_key(fn):
    """Exact inference IR structure, independent of process-global SSA names.

    This cache lives only inside one graph compilation. Types and declared
    buffer types are included: Function.__repr__ omits both. Unknown objects
    refuse rather than becoming an ambiguous string cache key.
    """
    values={}
    def encode(x):
        if isinstance(x,ir.Value):
            if x not in values: values[x]=len(values)
            return ('value',values[x],x.type)
        if isinstance(x,ir.Imm): return ('immediate',x.v)
        if isinstance(x,ir.Buffer):
            return ('buffer',x.name,x.slot,x.elem,x.declared_element)
        if isinstance(x,ir.Block): return ('block',x.label)
        if isinstance(x,dict): return ('dict',tuple((k,encode(v)) for k,v in sorted(x.items())))
        if isinstance(x,(list,tuple)): return (type(x).__name__,tuple(map(encode,x)))
        if x is None or type(x) in (str,int,float,bool,bytes): return (type(x).__name__,x)
        raise TypeError('unsupported inference IR cache field: '+type(x).__name__)
    return (fn.name,encode(fn.buffers),encode(fn.threadgroup),
            tuple((block.label,tuple((op.kind,encode(op.dest),encode(op.args),encode(op.attrs))
                                    for op in block.ops)) for block in fn.blocks))


def bits(x):
    return struct.unpack('<I',struct.pack('<f',x))[0]


def function(name, names, types=None):
    buffers=[ir.Buffer(n,i+1,elem=t) for i,(n,t) in enumerate(zip(names,types or [ir.F32]*len(names)))]
    fn=ir.Function(name,buffers)
    return fn,ir.Builder(fn,fn.block('entry')),buffers


def element_ir(op, width):
    names=['source','rhs','output'] if op in ('add','multiply','bias') else ['source','output']
    fn,b,buf=function('inference_'+op,names)
    col=b.builtin('thread_position_in_grid',axis='x')
    row=b.builtin('thread_position_in_grid',axis='y')
    index=b.add(b.mul(row,b.const(width)),col)
    x=b.load(buf[0],index,type=ir.F32)
    if op in ('add','multiply','bias'):
        y=b.load(buf[1],col if op=='bias' else index,type=ir.F32)
        value=b.fmul(x,y) if op=='multiply' else b.fadd(x,y)
    elif op=='gelu_erf':
        e=b._def('erf',[b.fmul(x,b.const(bits(1/math.sqrt(2))))],ir.F32)
        value=b.fmul(b.fmul(x,b.const(bits(.5))),b.fadd(e,b.const(bits(1))))
    else:
        raise Unsupported('refused: elementwise operation '+op)
    b.store_at(buf[-1],index,value);b.ret()
    return fn


def pack_ir():
    fn,b,(source,out)=function('inference_pack',['source','output'],[ir.F32,ir.F16])
    index=b.builtin('thread_position_in_grid')
    b.store_at(out,index,b.f32_to_f16_rte(b.load(source,index)),width='half')
    b.ret();return fn


def tensor_ir(rows,n,k):
    if rows!=32 or (n,k) not in ((384,384),(1536,384),(384,1536)):
        raise Unsupported('refused: native projection shape outside resident MiniLM domain')
    fn,b,(a,w,c)=function('inference_projection',['source','weight','output'],[ir.F16,ir.F16,ir.F32])
    b.tensor_matmul(a,w,c,M=rows,N=n,K=k,kloop=True,kloop_chunk=64,grid_n=n//32)
    b.ret();return fn


def score_mask_ir():
    fn,b,(scores,mask)=function('inference_score_mask',['scores','mask'],[ir.F32,ir.I32])
    index=b.builtin('thread_position_in_grid')
    key=b._def('and',[index,b.const(31)])
    valid=b.u32_to_f32(b.load(mask,key))
    invalid=b.fadd(b.const(bits(1)),b.fmul(valid,b.const(bits(-1))))
    bias=b.fmul(invalid,b.const(bits(-3.4028234663852886e38)))
    value=b.fadd(b.load(scores,index,type=ir.F32),bias)
    b.store_at(scores,index,value);b.ret();return fn


def mean_ir():
    fn,b,(source,mask,out)=function('inference_masked_mean',['source','mask','output'],[ir.F32,ir.I32,ir.F32])
    col=b.builtin('thread_position_in_grid')
    total=b.const(bits(0));count=b.const(bits(0))
    for row in range(32):
        valid=b.u32_to_f32(b.load(mask,b.const(row)))
        count=b.fadd(count,valid)
        index=b.add(col,b.const(row*384))
        total=b.fadd(total,b.fmul(b.load(source,index,type=ir.F32),valid))
    value=b.fmul(total,b.recip(b.fmax(count,b.const(bits(1e-9)))))
    b.store_at(out,col,value);b.ret();return fn


def normalize_ir():
    fn,b,(source,out)=function('inference_l2_normalize',['source','output'])
    lane=b.builtin('thread_position_in_grid')
    partial=b.const(bits(0))
    for i in range(12):
        index=b.add(lane,b.const(i*32))
        x=b.load(source,index,type=ir.F32)
        partial=b.fadd(partial,b.fmul(x,x))
    for mask in (1,2,4,8,16):
        partial=b.fadd(partial,b.simd_shuffle_xor(partial,mask))
    inverse=b.rsqrt(b.fmax(partial,b.const(bits(1e-24))))
    for i in range(12):
        index=b.add(lane,b.const(i*32))
        b.store_at(out,index,b.fmul(b.load(source,index,type=ir.F32),inverse))
    b.ret();return fn


def compile_graph(graph, *, policy):
    if graph.get('profile', {}).get('family') == 'qwen2':
        from g17decoderlower import compile_graph as decoder_compile
        return decoder_compile(graph, policy=policy)
    if policy!=POLICY:
        raise Unsupported('refused: explicit half-transport numerical policy required')
    profile=graph['profile']
    if graph.get('format')!='g17-inference-graph-v1' or graph.get('gpu_admitted') is not False:
        raise Unsupported('refused: expected logical model graph')
    if (profile['family'],profile['width'],profile['hidden'],profile['heads'],graph['tokens'])!=('bert',384,1536,12,32):
        raise Unsupported('refused: complete native graph lowering currently admits MiniLM shape at 32 tokens only')
    tensors=graph['tensors']
    stages=[];programs={};cache={};prepared={};temporaries={};coverage=[]
    def emit(node,suffix,fn,bindings,grid):
        key=function_key(fn)
        if key not in cache:
            p=cc.compile_function(fn)
            cache[key]=p
        p=cache[key];digest=hashlib.sha256(p.code).hexdigest();programs[digest]=p.code
        abi=p.abi_plain(p.abi())
        if len(bindings)!=len(abi['bindings']) or len(bindings)>3:
            raise Unsupported('refused: native stage binding inventory')
        stages.append(dict(name=node['name']+'.'+suffix,node=node['name'],program_sha256=digest,
                           code_bytes=len(p.code),bindings=[dict(name=n,**a) for n,a in zip(bindings,abi['bindings'])],
                           abi=abi,grid=grid,threadgroup=[32,1,1]))
    for node in graph['nodes']:
        op=node['op'];inputs=node['inputs'];out=node['outputs'][0] if node['outputs'] else None
        if out is None or tensors[out]['dtype']!='F32':
            raise Unsupported('refused: state effect/non-FP32 output requires separate lowering')
        shape=tensors[out]['shape'];before=len(stages)
        if op=='gather_rows':
            table=tensors[inputs[0]]
            emit(node,'gather',gather_rows_ir(rows=32,width=384,vocabulary=table['shape'][0],dtype=table['dtype']),[inputs[1],inputs[0],out],[384,32,1])
        elif op in ('add','multiply','gelu_erf'):
            if len(shape)!=2 or shape[0]!=32 or shape[1] not in (384,1536):
                raise Unsupported('refused: scalar model shape')
            if op=='add' and len(inputs)==3:
                emit(node,'add_first',element_ir(op,shape[1]),inputs[:2]+[out],[shape[1],32,1])
                emit(node,'add_second',element_ir(op,shape[1]),[out,inputs[2],out],[shape[1],32,1])
            else:
                emit(node,op,element_ir(op,shape[1]),inputs+[out],[shape[1],32,1])
        elif op=='linear':
            source,weight=inputs[:2];k=tensors[source]['shape'][1];n=shape[1]
            if node['attrs']['weight_layout']!='out_in' or tensors[weight]['dtype']!='F32':
                raise Unsupported('refused: checkpoint projection layout/dtype')
            half=node['name']+'.half_input'
            temporaries[half]=dict(shape=[32,k],dtype='F16',producer=node['name'],purpose='GPU input narrowing before MMA')
            packed='prepared:'+weight
            prepared[packed]=dict(source=weight,shape=[k,n],dtype='F16',transformation='transpose out-in to in-out; IEEE FP16 RNE once during checkpoint preparation',bytes=k*n*2)
            emit(node,'pack',pack_ir(),[source,half],[32*k,1,1])
            emit(node,'gemm',tensor_ir(32,n,k),[half,packed,out],[32*(n//32),1,1])
            if len(inputs)==3:
                emit(node,'bias',element_ir('bias',n),[out,inputs[2],out],[n,32,1])
        elif op=='layer_norm':
            import g17cooplayernorm as norm
            if shape!=[32,384] or node['attrs']['axis']!=-1 or node['attrs']['epsilon']!=1e-12:
                raise Unsupported('refused: normalization outside measured MiniLM configuration')
            packed='prepared:'+node['name']+'.gamma_beta'
            prepared[packed]=dict(sources=inputs[1:],shape=[768],dtype='F32',transformation='concatenate gamma then beta without arithmetic',bytes=3072)
            emit(node,'norm',norm.cooperative_layernorm_ir(32,packed_parameters=True),[inputs[0],packed,out],[32,32,1])
        elif op=='attention':
            import g17attention as attention
            at=node['attrs']
            if at['causal'] or at['heads']!=12 or at['kv_heads']!=12 or at['head_width']!=32 or at['dropout']!=0 or at['scale']!=1/math.sqrt(32) or at['mask_semantics']!='one_valid_zero_padding' or at['all_masked_policy']!='refuse' or at['evaluation_only'] is not True:
                raise Unsupported('refused: causal/GQA/non-MiniLM attention pending native lowering')
            q,k,v,mask=inputs
            scores=node['name']+'.scores';prob=node['name']+'.probabilities'
            for name in (scores,prob):temporaries[name]=dict(shape=[12,32,32],dtype='F32',producer=node['name'],purpose='GPU attention temporary')
            emit(node,'scores',attention.scores_ir(),[q,k,scores],[32,384,1])
            emit(node,'mask',score_mask_ir(),[scores,mask],[12288,1,1])
            emit(node,'softmax',attention.softmax_ir(),[scores,prob],[384,1,1])
            emit(node,'context',attention.context_ir(),[prob,v,out],[384,32,1])
        elif op=='masked_mean':
            if shape!=[384] or node['attrs']['denominator_min']!=1e-9:
                raise Unsupported('refused: sentence pooling configuration')
            emit(node,'mean',mean_ir(),inputs+[out],[384,1,1])
        elif op=='l2_normalize':
            if shape!=[384] or node['attrs']['epsilon']!=1e-12:
                raise Unsupported('refused: sentence normalization configuration')
            emit(node,'normalize',normalize_ir(),inputs+[out],[32,1,1])
        else:
            raise Unsupported('refused: native graph operation '+op)
        coverage.append(dict(node=node['name'],op=op,stages=[s['name'] for s in stages[before:]]))
    for stage in stages:
        for binding in stage['bindings']:
            name=binding['name']
            descriptor=tensors.get(name) or temporaries.get(name) or prepared.get(name)
            if descriptor is None:
                raise Unsupported('refused: unresolved native tensor binding '+name)
            size=math.prod(descriptor['shape'])*{'F32':4,'F16':2,'BF16':2,'I32':4}[descriptor['dtype']]
            binding['bytes']=size
    report=dict(format='g17-native-model-lowering-v1',status='compiled_not_executed',gpu_admitted=False,
                graph_sha256=hashlib.sha256(json.dumps(graph,sort_keys=True).encode()).hexdigest(),
                numerical_policy=policy,stages=stages,coverage=coverage,prepared_parameters=prepared,
                temporary_tensors=temporaries,unique_programs=len(programs),unique_code_bytes=sum(map(len,programs.values())),
                preconditions=['all IDs valid','padding mask values 0/1 with at least one valid token',
                               'finite checkpoint and activation domain; intermediate overflow must be checked',
                               'each stage executes in order; no host activation reconstruction'],
                pending=['checkpoint conversion validation','native temporary lifetimes/placement','resource/launch admission',
                         'hardware numerical references and preregistered bounds','persistent complete-model execution'])
    return programs,report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata',type=Path)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--policy',required=True)
    args=parser.parse_args()
    graph=import_model(args.metadata,sentence_pooling=True)
    programs,report=compile_graph(graph,policy=args.policy)
    args.destination.mkdir(parents=True,exist_ok=False)
    for digest,code in programs.items():(args.destination/(digest+'.bin')).write_bytes(code)
    (args.destination/'lowering.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(nodes=len(report['coverage']),stages=len(report['stages']),unique_programs=len(programs),
                         unique_code_bytes=report['unique_code_bytes'],gpu_admitted=False)))
