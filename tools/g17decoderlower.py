"""Complete bounded Qwen stage lowering; native execution remains unvalidated."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import g17decoderkernels as D
from g17inferencelower import function, element_ir, bits, function_key
from g17inferencekernels import gather_rows_ir
from g17modelimport import import_model
from agxforge.g17 import ir, cc
from agxforge.g17.inferencegraph import Unsupported

POLICY = 'qwen_bf16_checkpoint_half_mma_fp32_v1'
PACK_REUSE = 'reuse_projection_packs_v1'
CROP_BIAS = 'reuse_projection_packs_and_crop_bias_v1'


def crop_bias_ir():
    fn,b,(source,bias,out)=function('qwen_decode_crop_bias',['source','bias','output'])
    index=b.builtin('thread_position_in_grid')
    value=b.fadd(b.load(source,index,type=ir.F32),b.load(bias,index,type=ir.F32))
    b.store_at(out,index,value);b.ret();return fn


def fuse_crop_bias(lowering,programs):
    """Row-zero FP32 transport plus bias: same fadd, one store/submission."""
    if lowering.get('optimization',{}).get('name')!=PACK_REUSE:
        raise Unsupported('refused: crop-bias fusion requires pack-reuse stage graph')
    result=copy.deepcopy(lowering);stages=result['stages'];kept=[];fused=[];program=None;i=0
    while i<len(stages):
        crop=stages[i]
        if crop['name'].endswith('.crop') and i+1<len(stages) and stages[i+1]['name']==crop['node']+'.bias':
            bias=stages[i+1];cb=crop['bindings'];bb=bias['bindings']
            if (len(cb)!=2 or len(bb)!=3 or cb[0]['written'] or not cb[1]['written']
                or bb[0]['written'] or bb[1]['written'] or not bb[2]['written']
                or cb[1]['name']!=bb[0]['name'] or bb[0]['name']!=bb[2]['name']
                or crop['grid']!=bias['grid'] or crop['grid'][1:]!=[1,1]
                or crop['grid'][0] not in (128,896)):
                raise Unsupported('refused: crop-bias fusion binding/shape domain')
            if program is None:
                program=cc.compile_function(crop_bias_ir())
                sha=hashlib.sha256(program.code).hexdigest();programs[sha]=program.code
            abi=program.abi_plain(program.abi())
            if [b['written'] for b in abi['bindings']]!=[False,False,True]:
                raise Unsupported('refused: crop-bias compiler ABI')
            bindings=[dict(a,name=b['name'],bytes=b['bytes']) for a,b in zip(abi['bindings'],(cb[0],bb[1],bb[2]))]
            kept.append(dict(crop,name=crop['node']+'.crop_bias',program_sha256=sha,
                             code_bytes=len(program.code),abi=abi,bindings=bindings))
            fused.append(dict(crop=crop['name'],bias=bias['name'],replacement=crop['node']+'.crop_bias'))
            i+=2
        else:kept.append(crop);i+=1
    result['stages']=kept
    for row in result['coverage']:
        for pair in fused:
            if pair['crop'] in row['stages']:
                row['stages']=[pair['replacement'] if n==pair['crop'] else n for n in row['stages'] if n!=pair['bias']]
    result['optimization'].update(name=CROP_BIAS,fused_stages=fused,stage_count=len(kept))
    result['unique_programs']=len(programs);result['unique_code_bytes']=sum(map(len,programs.values()))
    return result


def reuse_projection_packs(lowering):
    """Reuse identical narrowing of the same unchanged source, then replan.

    The result remains a stage graph, never physical-address aliasing. Its
    ordinary resource planner must extend the retained pack's lifetime through
    every consumer. Program bytes and numerical operations are unchanged.
    """
    if lowering['numerical_policy']!=POLICY or lowering.get('optimization'):
        raise Unsupported('refused: projection pack optimization domain')
    result=copy.deepcopy(lowering);cache={};epochs={};aliases={};removed=[];kept=[]
    writers={}
    for stage in result['stages']:
        for b in stage['bindings']:
            if b['written']:writers[b['name']]=writers.get(b['name'],0)+1
    for stage in result['stages']:
        if stage['name'].endswith('.pack'):
            bindings=stage['bindings']
            if len(bindings)!=2 or bindings[0]['written'] or not bindings[1]['written']:
                raise Unsupported('refused: projection pack binding structure')
            source,out=[b['name'] for b in bindings]
            if writers[out]!=1:raise Unsupported('refused: mutable packed projection input')
            key=(source,epochs.get(source,0),stage['program_sha256'],tuple(stage['grid']),
                 bindings[0]['bytes'],bindings[1]['bytes'])
            if key in cache:
                aliases[out]=cache[key]
                removed.append(dict(stage=stage['name'],source=source,removed_output=out,reused_output=cache[key]))
                continue
            cache[key]=out
        for b in stage['bindings']:
            if b['name'] in aliases:
                if b['written']:raise Unsupported('refused: write to reused projection pack')
                b['name']=aliases[b['name']]
            if b['written']:epochs[b['name']]=epochs.get(b['name'],0)+1
        kept.append(stage)
    result['stages']=kept
    for name in aliases:del result['temporary_tensors'][name]
    names={s['name'] for s in kept}
    for row in result['coverage']:row['stages']=[n for n in row['stages'] if n in names]
    result['optimization']=dict(name=PACK_REUSE,removed_stages=removed,
                                baseline_stage_count=len(lowering['stages']),stage_count=len(kept))
    return result


def pack_ir(rows, width):
    fn, b, (source, out) = function('qwen_projection_pack', ['source', 'output'], [ir.F32, ir.F16])
    col = b.builtin('thread_position_in_grid', axis='x'); row = b.builtin('thread_position_in_grid', axis='y')
    # Decode uses the existing 32-row accelerator tile. Never read beyond the
    # single source row: extra tile rows load row zero and are zeroed on the GPU.
    source_row = b._def('and', [row, b.const(rows-1)])
    value = b.load(source, b.add(b.mul(source_row, b.const(width)), col))
    if rows == 1:
        value = b.fmul(value, b.u32_to_f32(b.icmp(row, b.const(0), rel='eq')))
    b.store_at(out, b.add(b.mul(row, b.const(width)), col), b.f32_to_f16_rte(value), width='half')
    b.ret(); return fn


def crop_ir(width):
    fn, b, (source, out) = function('qwen_projection_crop', ['source', 'output'])
    col = b.builtin('thread_position_in_grid')
    b.store_at(out, col, b.load(source, col, type=ir.F32)); b.ret(); return fn


def tensor_ir(n, k):
    if (n, k) not in ((896,896),(128,896),(4864,896),(896,4864)):
        raise Unsupported('refused: Qwen accelerator projection shape')
    fn, b, (a, w, c) = function('qwen_projection', ['source','weight','output'], [ir.F16,ir.F16,ir.F32])
    b.tensor_matmul(a, w, c, M=32, N=n, K=k, kloop=True, kloop_chunk=64, grid_n=n//32)
    b.ret(); return fn


def compile_graph(graph, *, policy):
    p = graph['profile']; rows = graph['tokens']
    if policy != POLICY:
        raise Unsupported('refused: explicit Qwen half-MMA and FP32 tied-logits policy required')
    if graph['format'] != 'g17-inference-graph-v1' or graph.get('gpu_admitted') is not False:
        raise Unsupported('refused: expected unadmitted logical Qwen graph')
    if (p['family'],p['layers'],p['width'],p['hidden'],p['heads'],p['kv_heads'],p['width']//p['heads'],p['vocabulary'],p['epsilon'],p['rope_theta']) != ('qwen2',24,896,4864,14,2,64,151936,1e-6,1e6):
        raise Unsupported('refused: Qwen profile outside bounded checkpoint domain')
    D.domain(rows)
    tensors = graph['tensors']
    if any(t['shape'] != [2,256,64] for t in tensors.values() if t['role'] == 'state'):
        raise Unsupported('refused: decoder KV capacity must be 256')
    stages=[]; programs={}; cache={}; prepared={}; temporary={}; coverage=[]
    phase='rotary.phase'; temporary[phase]=dict(shape=[rows,64],dtype='F32',purpose='GPU-gathered position phases')
    table='prepared:rotary.cos_sin'
    prepared[table]=dict(shape=[256,64],dtype='F32',transformation='fixed split-half RoPE cos32/sin32 table: theta=1e6, FP64 calculation rounded to FP32',theta=1e6)
    def fp32(name):
        if tensors[name]['dtype'] != 'BF16': raise Unsupported('refused: Qwen checkpoint parameter must be BF16')
        result='prepared:'+name+'.fp32'
        prepared[result]=dict(source=name,shape=tensors[name]['shape'],dtype='F32',transformation='exact BF16 bit widening')
        return result
    def emit(node,suffix,fn,names,grid):
        key=function_key(fn)
        if key not in cache: cache[key]=cc.compile_function(fn)
        program=cache[key]; sha=hashlib.sha256(program.code).hexdigest(); programs[sha]=program.code
        abi=program.abi_plain(program.abi())
        if len(names)!=len(abi['bindings']) or len(names)>3: raise Unsupported('refused: decoder native binding count')
        stages.append(dict(name=node['name']+'.'+suffix,node=node['name'],program_sha256=sha,
                           code_bytes=len(program.code),abi=abi,grid=grid,threadgroup=[32,1,1],
                           bindings=[dict(name=name,**binding) for name,binding in zip(names,abi['bindings'])]))
    for node in graph['nodes']:
        before=len(stages); op=node['op']; inputs=node['inputs']; out=node['outputs'][0] if node['outputs'] else None
        if op=='gather_rows':
            if inputs[0]!='model.embed_tokens.weight' or tensors[inputs[0]]['dtype']!='BF16':
                raise Unsupported('refused: Qwen embedding source')
            emit(node,'gather',gather_rows_ir(rows=rows,width=896,vocabulary=151936,dtype='BF16'),[inputs[1],inputs[0],out],[896,rows,1])
            emit(node,'phase',gather_rows_ir(rows=rows,width=64,vocabulary=256),['position_ids',table,phase],[64,rows,1])
        elif op=='rms_norm':
            if node['attrs']!={'axis':-1,'epsilon':1e-6}: raise Unsupported('refused: Qwen RMS configuration')
            emit(node,'rms',D.rms_ir(rows),[inputs[0],fp32(inputs[1]),out],[32,rows,1])
        elif op=='linear':
            source,weight=inputs[:2]; n=tensors[out]['shape'][1]; k=tensors[source]['shape'][1]
            if node['name']=='logits':
                if weight!='model.embed_tokens.weight' or len(inputs)!=2: raise Unsupported('refused: tied logits source')
                emit(node,'logits',D.logits_ir(rows),[source,weight,out],[151936,rows,1])
            else:
                half=node['name']+'.half_input'; temporary[half]=dict(shape=[32,k],dtype='F16',purpose='GPU narrowing and decode zero padding')
                packed='prepared:'+weight
                prepared[packed]=dict(source=weight,shape=[k,n],dtype='F16',transformation='BF16 widening then transpose out-in to in-out and FP16 RNE')
                emit(node,'pack',pack_ir(rows,k),[source,half],[k,32,1])
                target=out
                if rows==1:
                    target=node['name']+'.tile_output'; temporary[target]=dict(shape=[32,n],dtype='F32',purpose='32-row accelerator output; row zero retained')
                emit(node,'gemm',tensor_ir(n,k),[half,packed,target],[32*(n//32),1,1])
                if rows==1: emit(node,'crop',crop_ir(n),[target,out],[n,1,1])
                if len(inputs)==3: emit(node,'bias',element_ir('bias',n),[out,fp32(inputs[2]),out],[n,rows,1])
        elif op=='rotary':
            at=node['attrs']; heads=at['heads']
            if at!={'theta':1e6,'heads':heads,'head_width':64,'convention':'split_half'}:
                raise Unsupported('refused: Qwen rotary convention')
            emit(node,'rotary',D.rotary_ir(rows,heads),[inputs[0],phase,out],[heads*64,rows,1])
        elif op=='kv_cache_update':
            if node['attrs']!={'writes':inputs[4:],'cache_layout':'head_position_element','position_semantics':'absolute','bounds_check_required':True}:
                raise Unsupported('refused: Qwen cache effect contract')
            emit(node,'append_k',D.cache_append_ir(rows),[inputs[0],inputs[2],inputs[4]],[128,rows,1])
            emit(node,'append_v',D.cache_append_ir(rows),[inputs[1],inputs[2],inputs[5]],[128,rows,1])
        elif op=='attention':
            at=node['attrs']
            if (at['heads'],at['kv_heads'],at['head_width'],at['scale'],at['causal'],at['dropout'],at['mask_semantics'],at['all_masked_policy'],at['evaluation_only'])!=(14,2,64,.125,True,0.,'one_valid_zero_padding','refuse',True):
                raise Unsupported('refused: Qwen causal GQA contract')
            q,k,v,mask,positions,valid=inputs
            scores=node['name']+'.scores'; prob=node['name']+'.probabilities'
            for name in (scores,prob): temporary[name]=dict(shape=[14,rows,256],dtype='F32',purpose='GPU causal grouped attention')
            emit(node,'scores',D.scores_ir(rows),[q,k,scores],[256,14*rows,1])
            emit(node,'mask',D.mask_ir(rows),[scores,positions,mask],[256,14*rows,1])
            emit(node,'softmax',D.softmax_ir(rows),[scores,prob],[32,14*rows,1])
            emit(node,'context',D.context_ir(rows),[prob,v,out],[896,rows,1])
        elif op=='silu': emit(node,'silu',D.silu_ir(rows),inputs+[out],[4864,rows,1])
        elif op in ('add','multiply'):
            emit(node,op,element_ir(op,tensors[out]['shape'][1]),inputs+[out],[tensors[out]['shape'][1],rows,1])
        else: raise Unsupported('refused: decoder operation '+op)
        coverage.append(dict(node=node['name'],op=op,stages=[s['name'] for s in stages[before:]]))
    descriptors=dict(tensors,**temporary,**prepared)
    for stage in stages:
        for binding in stage['bindings']:
            d=descriptors[binding['name']]; binding['bytes']=math.prod(d['shape'])*{'F32':4,'BF16':2,'F16':2,'I32':4}[d['dtype']]
    return programs,dict(format='g17-native-model-lowering-v1',status='compiled_not_executed',gpu_admitted=False,
                         graph_sha256=hashlib.sha256(json.dumps(graph,sort_keys=True).encode()).hexdigest(),
                         numerical_policy=policy,stages=stages,coverage=coverage,prepared_parameters=prepared,
                         temporary_tensors=temporary,unique_programs=len(programs),unique_code_bytes=sum(map(len,programs.values())),
                         state_effects=[n for n in graph['nodes'] if n['op']=='kv_cache_update'],
                         host_control_inputs=['cache_valid_length'],
                         preconditions=['256-position cache; reset zeros all state', 'positions unique/increasing and in bounds',
                                        'cache_valid_length/attention_mask must describe only populated causal keys',
                                        'finite activations within FP16 transport range',
                                        'BF16 checkpoint; FP16 projection transport; FP32 scalar tied logits reuses original BF16 table'],
                         pending=['state-aware native resource planner','checkpoint conversion validation',
                                  'complete prefill/decode independent references','native persistent decoder execution'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('metadata',type=Path); parser.add_argument('destination',type=Path)
    parser.add_argument('--tokens',type=int,choices=(1,32),default=32); args=parser.parse_args()
    programs,report=compile_graph(import_model(args.metadata,tokens=args.tokens,cache_capacity=256),policy=POLICY)
    args.destination.mkdir(parents=True,exist_ok=False)
    for sha,code in programs.items(): (args.destination/(sha+'.bin')).write_bytes(code)
    (args.destination/'lowering.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(nodes=len(report['coverage']),stages=len(report['stages']),unique_programs=len(programs),unique_code_bytes=report['unique_code_bytes'],gpu_admitted=False)))
