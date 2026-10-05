"""Place shared model stages in exact measured 128 MiB or 1 GiB native graphs."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g17inferenceresourcegraph import recover
import g17authoredblock640payload as block
import g17structuredbasepayload as base
import g17authoredtensorphysical as physical
import g17authoredtensorsg4h as heads
from g17modelimport import import_model
from g17inferencelower import compile_graph,POLICY
from agxforge.g17.inferenceresources import plan

ROOT=Path(__file__).resolve().parents[1]
STAGE=struct.Struct('<14I4Q');REGION=struct.Struct('<8I')
INPUTS=('token_ids','position_ids','token_type_ids','attention_mask')
DECODER_INPUTS=('token_ids','position_ids','attention_mask','cache_valid_length')
DECODER_HEADER=struct.Struct('<10I')


def decoder_contract(prepared,*,optimization=None):
    """Compiler-derived schedules sharing parameters, code and persistent state.

    This does not admit a GPU launch. GIN2 is a host wire protocol; every GPU
    coordinate below comes from recover(1024), not the protocol's dimensions.
    """
    from g17decoderlower import POLICY as decoder_policy,PACK_REUSE,CROP_BIAS,reuse_projection_packs,fuse_crop_bias
    if optimization not in (None,PACK_REUSE,CROP_BIAS):raise ValueError('refused: decoder optimization')
    prepared=Path(prepared);manifest_bytes=(prepared/'manifest.json').read_bytes()
    m=json.loads(manifest_bytes)
    if m['policy']!=decoder_policy:raise ValueError('refused: decoder preparation policy')
    request,pages,resource=recover(1024);rows={r['index']:r for r in resource['rows']}
    compiled={};programs={};baseline_prefill=None;optimization_reports={}
    for mode,tokens in (('prefill',32),('decode',1)):
        graph=import_model(ROOT/'evidence/g17-inference-models-v1/qwen',tokens=tokens,cache_capacity=256)
        native,lowering=compile_graph(graph,policy=decoder_policy)
        baseline=plan(graph,lowering)
        if mode=='prefill':baseline_prefill=baseline
        if optimization:
            lowering=reuse_projection_packs(lowering)
            if optimization==CROP_BIAS:lowering=fuse_crop_bias(lowering,native)
            optimization_reports[mode]=lowering['optimization']
        layout=plan(graph,lowering)
        for name,r in baseline['regions'].items():
            if r['arena'] not in ('parameters','state'):continue
            other=layout['regions'][name]
            if any(r[k]!=other[k] for k in ('offset','bytes','capacity','readonly','dtype')):
                raise ValueError('optimization changes persistent storage')
        compiled[mode]=(graph,layout);programs.update(native)
    if m['layout']!=json.loads(json.dumps(baseline_prefill)):
        raise ValueError('prepared decoder storage differs from compiler')
    if m['parameter_file']['bytes']!=m['layout']['arenas']['parameters']:
        raise ValueError('prepared decoder parameter extent')
    for sha,extent in m['programs'].items():
        raw=(prepared/(sha+'.bin')).read_bytes()
        if len(raw)!=extent or raw!=programs.get(sha):raise ValueError('prepared decoder native identity')
    parameter_bytes=m['parameter_file']['bytes'];activity_base=(parameter_bytes+255)//256*256
    activity_bytes=max(layout['arenas']['activations'] for _,layout in compiled.values())
    state_base=activity_base+activity_bytes;state_bytes=compiled['prefill'][1]['arenas']['state']
    if state_base+state_bytes>rows[20]['bytes']:raise ValueError('decoder arenas exceed measured 1 GiB allocation')
    placements={};cursor=0x6c0
    for sha,code in programs.items():
        cursor=(cursor+63)//64*64
        if cursor+len(code)>rows[0]['bytes']:raise ValueError('decoder native code pool overflow')
        placements[sha]=cursor;cursor+=len(code)
    schedules={};wires={}
    for mode,(graph,layout) in compiled.items():
        regions={}
        for name,r in layout['regions'].items():
            arena=r['arena'];index=19 if arena=='inputs' else 20
            if arena not in ('inputs','parameters','activations','state'):raise ValueError('decoder arena')
            offset=r['offset']+({'activations':activity_base,'state':state_base}.get(arena,0))
            if offset+r['capacity']+256>rows[index]['bytes']:raise ValueError('decoder region exceeds measured allocation')
            regions[name]=dict(r,allocation=index,offset=offset,gpu_address=rows[index]['gpu_address']+offset,
                              guard_before=offset-256,guard_after=offset+r['capacity'])
        stages=[];stage_bytes=[]
        for stage in layout['stages']:
            bindings=[regions[b['name']] for b in stage['bindings']]
            written=[r for r,b in zip(bindings,stage['bindings']) if b['written']]
            if len(written)!=1 or written[0]['readonly'] or not 1<=len(bindings)<=3:
                raise ValueError('decoder stage write/binding inventory')
            out=written[0];state=0x0e58 if stage['name'].endswith('.gemm') else 0x0e40
            scratch=0x0c00100f if state==0x0e58 else 0x0c000007
            args=[r['gpu_address'] for r in bindings]+[0]*(4-len(bindings))
            stage_bytes.append(STAGE.pack(placements[stage['program_sha256']],stage['code_bytes'],state,
                *stage['grid'],32,scratch,len(bindings),out['allocation'],out['offset'],out['bytes'],0,0,*args))
            stages.append(dict(stage,resident_code_offset=placements[stage['program_sha256']],state=state,scratch=scratch,
                               physical_bindings=bindings,output_region=out))
        region_bytes=[]
        for name,r in regions.items():
            flags=int(r['readonly'])|(2 if name in DECODER_INPUTS else 0)|(4 if r['arena']=='state' else 0)
            slot=DECODER_INPUTS.index(name)+1 if name in DECODER_INPUTS else 0
            region_bytes.append(REGION.pack(r['allocation'],r['offset'],r['bytes'],r['capacity'],flags,
                                            max(0,r['birth']),r['last_use'],slot))
        input_bytes=sum(regions[n]['bytes'] for n in DECODER_INPUTS)
        wires[mode]=DECODER_HEADER.pack(0x47494e32,len(stages),len(regions),input_bytes,
            parameter_bytes,activity_base,activity_bytes,state_base,state_bytes,graph['tokens'])+b''.join(stage_bytes)+b''.join(region_bytes)
        schedules[mode]=dict(regions=regions,stages=stages,input_bytes=input_bytes,tokens=graph['tokens'],output_name=graph['outputs'][0])
    for name,r in schedules['prefill']['regions'].items():
        if r['arena'] not in ('parameters','state'):continue
        other=schedules['decode']['regions'][name]
        if any(r[k]!=other[k] for k in ('allocation','offset','bytes','capacity','readonly','dtype')):
            raise ValueError('prefill/decode persistent storage identity differs')
    report=dict(format='g17-native-inference-bundle-v2',status='prepared_not_dispatched',gpu_dispatched=False,
        resource=resource,policy=decoder_policy,schedules=schedules,input_names=DECODER_INPUTS,
        parameter_bytes=parameter_bytes,parameter_file=m['parameter_file'],activity_base=activity_base,activity_bytes=activity_bytes,
        state_base=state_base,state_bytes=state_bytes,code_end=cursor,code_placements=placements,
        prepared_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
    if optimization:report['optimization']=dict(name=optimization,schedules=optimization_reports)
    return report,programs,dict(ordered=request,pages=pages,**{mode+'-schedule':wire for mode,wire in wires.items()})


def decoder_allocations(report,programs):
    """Small surfaces only: allocation 20 is deliberately streamed separately."""
    rows={r['index']:r for r in report['resource']['rows']}
    allocation={0:bytearray(block.allocation_zero(prefix_bytes=0x400)),1:bytearray(base.allocation(1)),
                2:bytearray(heads.allocation2()),15:bytearray(physical.allocation(15)),17:bytearray(physical.allocation(17))}
    for i in (19,21):allocation[i]=bytearray(b'\xa5'*rows[i]['bytes'])
    allocation.update({i:bytearray(physical.allocation(i)) for i in range(22,29)})
    for sha,code in programs.items():
        offset=report['code_placements'][sha];allocation[0][offset:offset+len(code)]=code
    struct.pack_into('<H',allocation[23],6,0x002c)
    struct.pack_into('<H',allocation[25],9,0x001a)
    allocation[17][:0xc000]=allocation[28]
    # The proven 1 GiB selector reads this low shader-record mirror. Patching
    # allocation 23 after this copy would not change the selected record.
    allocation[17][0x10000:0x18000]=allocation[23]
    return allocation


def prepare_decoder(prepared,destination,*,optimization=None):
    report,programs,files=decoder_contract(prepared,optimization=optimization)
    rows={r['index']:r for r in report['resource']['rows']}
    allocation=decoder_allocations(report,programs)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    identities={}
    for name,raw in files.items():
        (destination/(name+'.bin')).write_bytes(raw)
        identities[name+'.bin']=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    digest=hashlib.sha256();count=0;fill=b'\xa5'*(1024*1024)
    with (destination/'physical.bin').open('xb') as output:
        def write(raw):
            nonlocal count
            output.write(raw);digest.update(raw);count+=len(raw)
        def pad(size):
            if size<0:raise ValueError('decoder physical intervals overlap')
            while size:
                n=min(size,len(fill));write(fill[:n]);size-=n
        for i in sorted(i for i,r in rows.items() if r['kind']!=1):
            if i!=20:
                raw=allocation[i]
                if len(raw)!=rows[i]['bytes']:raise ValueError('decoder physical allocation extent')
                write(raw);continue
            parameter_digest=hashlib.sha256();position=0
            with (Path(prepared)/'parameters.bin').open('rb') as source:
                while raw:=source.read(len(fill)):
                    write(raw);parameter_digest.update(raw);position+=len(raw)
            if position!=report['parameter_bytes'] or parameter_digest.hexdigest()!=report['parameter_file']['sha256']:
                raise ValueError('decoder checkpoint parameter identity')
            states=sorted((r for r in report['schedules']['prefill']['regions'].values() if r['arena']=='state'),key=lambda r:r['offset'])
            for r in states:
                pad(r['offset']-position);write(b'\0'*r['bytes']);position=r['offset']+r['bytes']
            pad(rows[20]['bytes']-position)
    identities['physical.bin']=dict(bytes=count,sha256=digest.hexdigest())
    report['files']=identities
    (destination/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def prepare(prepared,destination):
    prepared=Path(prepared);m=json.loads((prepared/'manifest.json').read_text())
    graph=import_model(ROOT/'evidence/g17-inference-models-v1/minilm',sentence_pooling=True)
    programs,lowering=compile_graph(graph,policy=POLICY)
    layout=plan(graph,lowering)
    if m['layout']!=json.loads(json.dumps(layout)) or m['policy']!=POLICY:raise ValueError('prepared model contract differs from compiler')
    parameters=(prepared/'parameters.bin').read_bytes()
    if hashlib.sha256(parameters).hexdigest()!=m['parameter_file']['sha256']:raise ValueError('parameter identity mismatch')
    request,pages,resource=recover(128);rows={r['index']:r for r in resource['rows']}
    allocation={0:bytearray(block.allocation_zero(prefix_bytes=0x400)),1:bytearray(base.allocation(1)),
                2:bytearray(heads.allocation2()),15:bytearray(physical.allocation(15)),17:bytearray(physical.allocation(17))}
    for i in (19,20,21):allocation[i]=bytearray(b'\xa5'*rows[i]['bytes'])
    allocation.update({i:bytearray(physical.allocation(i)) for i in range(22,29)})
    allocation[20][:len(parameters)]=parameters
    activity_base=(len(parameters)+255)//256*256
    if activity_base+layout['arenas']['activations']>rows[20]['bytes']:raise ValueError('model arenas exceed measured allocation')
    regions={}
    for name,r in layout['regions'].items():
        arena=r['arena'];index=19 if arena=='inputs' else 20
        offset=r['offset']+(activity_base if arena=='activations' else 0)
        regions[name]=dict(r,allocation=index,offset=offset,gpu_address=rows[index]['gpu_address']+offset,
                          guard_before=offset-256,guard_after=offset+r['capacity'])
    placements={};cursor=0x6c0
    for sha,code in programs.items():
        raw=(prepared/(sha+'.bin')).read_bytes()
        if raw!=code:raise ValueError('prepared native program differs from compiler')
        cursor=(cursor+63)//64*64
        if cursor+len(code)>0x10000:raise ValueError('native code pool overflow')
        allocation[0][cursor:cursor+len(code)]=code;placements[sha]=cursor;cursor+=len(code)
    stage_bytes=[];stages=[]
    for stage in layout['stages']:
        bindings=[regions[b['name']] for b in stage['bindings']]
        written=[r for r,b in zip(bindings,stage['bindings']) if b['written']]
        if len(written)!=1 or written[0]['readonly']:raise ValueError('unmeasured native stage write inventory')
        out=written[0];tensor=stage['name'].endswith('.gemm')
        state=0x0e58 if tensor or stage['name'].endswith('.norm') else 0x0e40
        # Cooperative LayerNorm already uses the measured tensor-state packet.
        if stage['name'].endswith('.normalize'):state=0x0e40
        scratch=0x0c00100f if state==0x0e58 else 0x0c000007
        args=[r['gpu_address'] for r in bindings]+[0]*(4-len(bindings))
        stage_bytes.append(STAGE.pack(placements[stage['program_sha256']],stage['code_bytes'],state,
                                      *stage['grid'],32,scratch,len(bindings),out['allocation'],out['offset'],out['bytes'],0,0,*args))
        stages.append(dict(stage,resident_code_offset=placements[stage['program_sha256']],state=state,scratch=scratch,
                           physical_bindings=bindings,output_region=out))
    region_bytes=[]
    for name,r in regions.items():
        flags=int(r['readonly'])|(2 if name in INPUTS else 0)
        slot=INPUTS.index(name)+1 if name in INPUTS else 0
        region_bytes.append(REGION.pack(r['allocation'],r['offset'],r['bytes'],r['capacity'],flags,
                                        max(0,r['birth']),r['last_use'],slot))
    schedule=struct.pack('<4I',0x47494e31,len(stages),len(regions),512)+b''.join(stage_bytes)+b''.join(region_bytes)
    # Complete binding metadata is selected from its proven low carrier.
    struct.pack_into('<H',allocation[23],6,0x002c);struct.pack_into('<H',allocation[25],9,0x2048)
    allocation[17][:0xc000]=allocation[28]
    indices=sorted(i for i,r in rows.items() if r['kind']!=1)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    files=dict(ordered=request,pages=pages,schedule=schedule)
    identities={}
    for name,raw in files.items():
        (destination/(name+'.bin')).write_bytes(raw);identities[name+'.bin']=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    h=hashlib.sha256();count=0
    with (destination/'physical.bin').open('xb') as f:
        for i in indices:
            raw=allocation[i]
            if len(raw)!=rows[i]['bytes']:raise ValueError('physical allocation extent mismatch')
            f.write(raw);h.update(raw);count+=len(raw)
    identities['physical.bin']=dict(bytes=count,sha256=h.hexdigest())
    report=dict(format='g17-native-inference-bundle-v1',status='prepared_not_dispatched',gpu_dispatched=False,
                resource=resource,files=identities,regions=regions,stages=stages,
                activity_base=activity_base,activity_bytes=layout['arenas']['activations'],parameter_bytes=len(parameters),
                input_names=INPUTS,output_name=graph['outputs'][0],policy=POLICY,prepared_manifest_sha256=hashlib.sha256((prepared/'manifest.json').read_bytes()).hexdigest())
    (destination/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('prepared',type=Path);p.add_argument('destination',type=Path)
    p.add_argument('--model',choices=('minilm','qwen'),default='minilm')
    p.add_argument('--optimization',choices=('reuse_projection_packs_v1','reuse_projection_packs_and_crop_bias_v1'))
    a=p.parse_args()
    if a.model=='minilm' and a.optimization:p.error('decoder optimization requires --model qwen')
    r=prepare(a.prepared,a.destination) if a.model=='minilm' else prepare_decoder(a.prepared,a.destination,optimization=a.optimization)
    print(json.dumps(dict(stages={n:len(s['stages']) for n,s in r['schedules'].items()} if 'schedules' in r else len(r['stages']),
        physical_bytes=r['files']['physical.bin']['bytes'],gpu_dispatched=False)))
