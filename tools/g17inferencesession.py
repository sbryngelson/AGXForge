"""Persistent below-Metal shared-stage executor; complete-model validation pending."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import struct
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import g17belowmetalworkload as R
from g17inferenceresourcegraph import recover
from g17inferencebundle import STAGE,REGION,INPUTS,DECODER_INPUTS,decoder_contract,decoder_allocations
from g17inferencelower import compile_graph,POLICY
from g17modelimport import import_model
from agxforge.g17.inferenceresources import plan

ROOT=Path(__file__).resolve().parents[1]


def verify_decoder(directory,prepared):
    """Recompute the native contract and verify streamed bytes before launch."""
    import hashlib
    directory=Path(directory);m=json.loads((directory/'manifest.json').read_text())
    optimization=m.get('optimization',{}).get('name')
    expected,programs,files=decoder_contract(prepared,optimization=optimization)
    if {k:v for k,v in m.items() if k!='files'}!=json.loads(json.dumps(expected)):
        raise ValueError('decoder bundle differs from ordinary compiler contract')
    pin=json.loads((ROOT/'evidence/g17-native-qwen-preparation.json').read_text())
    if m['prepared_manifest_sha256']!=pin['prepared_manifest_sha256'] or m['parameter_file']!=pin['parameter_file']:
        raise ValueError('decoder prepared checkpoint provenance')
    if set(m['files'])!={*(name+'.bin' for name in files),'physical.bin'}:raise ValueError('decoder bundle inventory')
    for name,raw in files.items():
        if (directory/(name+'.bin')).read_bytes()!=raw:raise ValueError('decoder wire publication: '+name)
    for name,identity in m['files'].items():
        digest=hashlib.sha256();count=0
        with (directory/name).open('rb') as f:
            while raw:=f.read(1024*1024):digest.update(raw);count+=len(raw)
        if count!=identity['bytes'] or digest.hexdigest()!=identity['sha256']:raise ValueError('decoder member identity: '+name)
    small=decoder_allocations(expected,programs)
    rows={r['index']:r for r in m['resource']['rows']}
    with (directory/'physical.bin').open('rb') as f:
        for i in sorted(i for i,r in rows.items() if r['kind']!=1):
            if i!=20:
                if f.read(rows[i]['bytes'])!=small[i]:raise ValueError('decoder measured physical surface: '+str(i))
                continue
            digest=hashlib.sha256();remaining=m['parameter_bytes']
            while remaining:
                raw=f.read(min(remaining,1024*1024))
                if not raw:raise ValueError('truncated decoder parameter prefix')
                digest.update(raw);remaining-=len(raw)
            if digest.hexdigest()!=m['parameter_file']['sha256']:raise ValueError('decoder actual checkpoint bytes')
            position=m['parameter_bytes']
            states=sorted((r for r in m['schedules']['prefill']['regions'].values() if r['arena']=='state'),key=lambda r:r['offset'])
            def check_fill(size,value):
                if size<0:raise ValueError('decoder initial arena overlap')
                while size:
                    n=min(size,1024*1024)
                    if f.read(n)!=bytes([value])*n:raise ValueError('decoder initial cache/guard/padding bytes')
                    size-=n
            for region in states:
                check_fill(region['offset']-position,0xa5);check_fill(region['bytes'],0)
                position=region['offset']+region['bytes']
            check_fill(rows[20]['bytes']-position,0xa5)
        if f.read(1):raise ValueError('decoder trailing physical bytes')
    return m


def decoder_inputs(inputs,rows,current_valid):
    """Host admission protects cache append kernels that have no bounds checks."""
    if rows not in (1,32) or type(current_valid) is not int or not 0<=current_valid<=256:
        raise ValueError('decoder row/cache domain')
    if set(inputs)!=set(DECODER_INPUTS):raise ValueError('decoder input inventory')
    shapes=((rows,),(rows,),(256,),())
    for name,shape in zip(DECODER_INPUTS,shapes):
        a=np.asarray(inputs[name])
        if a.shape!=shape or a.dtype!=np.int32:raise ValueError('decoder input shape/dtype: '+name)
    tokens=inputs['token_ids'];positions=inputs['position_ids'];mask=inputs['attention_mask']
    valid=int(inputs['cache_valid_length'])
    if np.any(tokens<0) or np.any(tokens>=151936):raise ValueError('decoder token domain')
    if current_valid+rows>256 or not np.array_equal(positions,np.arange(current_valid,current_valid+rows)):
        raise ValueError('decoder cache position/footprint')
    if not current_valid<valid<=current_valid+rows or (rows==1 and not current_valid):
        raise ValueError('decoder cache valid length/prefill required')
    if not np.array_equal(mask,(np.arange(256)<valid).astype(np.int32)):
        raise ValueError('decoder attention mask must match valid cache prefix')
    return b''.join(np.asarray(inputs[n],dtype='<i4').tobytes() for n in DECODER_INPUTS),valid


class DecoderSession(R.Session):
    """Same persistent AGX process and queue; two model schedules and live KV."""
    def __init__(self,directory,prepared):
        self.directory=Path(directory).resolve();self.manifest=verify_decoder(self.directory,prepared)
        self.stack=ExitStack();self.process=None;self.calls=0;self.valid=0;self.dirty=False

    def launch_environment(self):
        env=super().launch_environment();env.pop('ORDERED_WORKLOAD_TILE')
        env.update(ORDERED_WORKLOAD_DECODER='1',ORDERED_DECODER_PREFILL=str(self.directory/'prefill-schedule.bin'),
                   ORDERED_DECODER_DECODE=str(self.directory/'decode-schedule.bin'))
        return env

    def expected_ready(self):return (R.MAGIC,5,2,256)

    def execute(self,inputs,*,reset=False,stages=None,readback=True):
        tokens=np.asarray(inputs.get('token_ids',[]))
        mode='prefill' if tokens.shape==(32,) else 'decode'
        schedule=self.manifest['schedules'][mode]
        count=len(schedule['stages']) if stages is None else stages
        if type(count) is not int or not 1<=count<=len(schedule['stages']) or self.calls>=256:
            raise ValueError('decoder stage/request bound')
        if self.dirty and not reset:raise ValueError('partial decoder execution requires cache reset')
        raw,valid=decoder_inputs(inputs,schedule['tokens'],0 if reset else self.valid)
        flags=(1 if readback else 2)|(16 if mode=='decode' else 0)|(32 if reset else 0)
        self.calls+=1;self.dirty=True;started=time.perf_counter_ns()
        R.write_all(self.process.stdin,struct.pack('<4I',R.MAGIC,flags,len(raw),count)+raw)
        outputs=[]
        for s in range(count):
            words=struct.unpack('<8I2Q',R.read_exact(self.process.stdout,48,seconds=30))
            stage=schedule['stages'][s];expected_bytes=stage['output_region']['bytes'] if readback or s+1==count else 0
            if words[:8]!=(R.MAGIC,0,0,63 if s+1==count else 21,1,1,expected_bytes,self.calls*1024+s):
                raise RuntimeError('decoder stage status/preservation/callback/sequence failed: '+repr(words))
            data=R.read_exact(self.process.stdout,expected_bytes,seconds=30) if expected_bytes else b''
            outputs.append(dict(name=stage['name'],data=data,submit_ns=words[8],completion_ns=words[9],flags=words[3]))
        self.roundtrip_ns=time.perf_counter_ns()-started
        if count==len(schedule['stages']):self.valid=valid;self.dirty=False
        return outputs


def verify(directory):
    directory=Path(directory);m=json.loads((directory/'manifest.json').read_text())
    if m['format']!='g17-native-inference-bundle-v1' or m['policy']!=POLICY:raise ValueError('unmeasured model bundle')
    prep=json.loads((ROOT/'evidence/g17-native-encoder-preparation.json').read_text())
    if m['prepared_manifest_sha256']!=prep['prepared_manifest_sha256']:raise ValueError('prepared checkpoint provenance mismatch')
    graph=import_model(ROOT/'evidence/g17-inference-models-v1/minilm',sentence_pooling=True)
    programs,lowering=compile_graph(graph,policy=POLICY);layout=plan(graph,lowering)
    request,pages,resource=recover(128)
    if m['resource']!=resource:raise ValueError('publication graph contract mismatch')
    if (directory/'ordered.bin').read_bytes()!=request or (directory/'pages.bin').read_bytes()!=pages:
        raise ValueError('publication surfaces differ from measured graph')
    expected_regions={};rows={r['index']:r for r in resource['rows']}
    for name,r in layout['regions'].items():
        index=19 if r['arena']=='inputs' else 20
        offset=r['offset']+(prep['arenas']['parameters'] if r['arena']=='activations' else 0)
        expected_regions[name]=dict(r,allocation=index,offset=offset,gpu_address=rows[index]['gpu_address']+offset,
                                    guard_before=offset-256,guard_after=offset+r['capacity'])
    if m['regions']!=expected_regions or m['input_names']!=list(INPUTS) or m['output_name']!=graph['outputs'][0]:
        raise ValueError('native physical regions differ from compiler lifetimes')
    if m['parameter_bytes']!=prep['arenas']['parameters'] or m['activity_base']!=m['parameter_bytes'] or m['activity_bytes']!=layout['arenas']['activations']:
        raise ValueError('native arena extents differ')
    placements={};cursor=0x6c0
    for sha,code in programs.items():cursor=(cursor+63)//64*64;placements[sha]=cursor;cursor+=len(code)
    stage_bytes=[];expected_stages=[]
    for s in layout['stages']:
        bindings=[expected_regions[b['name']] for b in s['bindings']]
        out=next(r for r,b in zip(bindings,s['bindings']) if b['written'])
        state=0x0e58 if s['name'].endswith(('.gemm','.norm')) else 0x0e40
        scratch=0x0c00100f if state==0x0e58 else 0x0c000007
        args=[r['gpu_address'] for r in bindings]+[0]*(4-len(bindings))
        stage_bytes.append(STAGE.pack(placements[s['program_sha256']],s['code_bytes'],state,*s['grid'],32,scratch,
                                      len(bindings),out['allocation'],out['offset'],out['bytes'],0,0,*args))
        expected_stages.append(dict(s,resident_code_offset=placements[s['program_sha256']],state=state,scratch=scratch,
                                    physical_bindings=bindings,output_region=out))
    if m['stages']!=json.loads(json.dumps(expected_stages)):raise ValueError('native launch schedule differs from compiler')
    region_bytes=[]
    for name,r in expected_regions.items():
        flags=int(r['readonly'])|(2 if name in INPUTS else 0);slot=INPUTS.index(name)+1 if name in INPUTS else 0
        region_bytes.append(REGION.pack(r['allocation'],r['offset'],r['bytes'],r['capacity'],flags,max(0,r['birth']),r['last_use'],slot))
    wire=struct.pack('<4I',0x47494e31,len(expected_stages),len(expected_regions),512)+b''.join(stage_bytes)+b''.join(region_bytes)
    if (directory/'schedule.bin').read_bytes()!=wire:raise ValueError('wire schedule differs from compiler')
    if set(m['files'])!={'ordered.bin','pages.bin','physical.bin','schedule.bin'}:raise ValueError('bundle member inventory')
    import hashlib
    for name,pin in m['files'].items():
        h=hashlib.sha256();count=0
        with (directory/name).open('rb') as f:
            while data:=f.read(1024*1024):h.update(data);count+=len(data)
        if count!=pin['bytes'] or h.hexdigest()!=pin['sha256']:raise ValueError('bundle member identity: '+name)
    with (directory/'physical.bin').open('rb') as f:
        # Payload order is the measured row order excluding views.
        offsets={};cursor=0
        for i,r in rows.items():
            if r['kind']!=1:offsets[i]=cursor;cursor+=r['bytes']
        f.seek(offsets[20]);remaining=m['parameter_bytes'];h=hashlib.sha256()
        while remaining:
            block=f.read(min(remaining,1024*1024));h.update(block);remaining-=len(block)
        if h.hexdigest()!=prep['parameter_file']['sha256']:raise ValueError('actual parameter bytes differ from verified checkpoint preparation')
        for sha,offset in placements.items():
            f.seek(offsets[0]+offset)
            if f.read(len(programs[sha]))!=programs[sha]:raise ValueError('actual native bytes differ from compiler')
    return m


class Session(R.Session):
    def __init__(self,directory):
        self.directory=Path(directory).resolve();self.manifest=verify(self.directory)
        self.stack=ExitStack();self.process=None;self.calls=0

    def launch_environment(self):
        env=super().launch_environment();env.pop('ORDERED_WORKLOAD_TILE')
        env.update(ORDERED_WORKLOAD_INFERENCE='1',ORDERED_INFERENCE_SCHEDULE=str(self.directory/'schedule.bin'))
        return env

    def expected_ready(self):return (R.MAGIC,4,512,len(self.manifest['stages']))

    def execute(self,inputs,*,stages=None,readback=True):
        count=len(self.manifest['stages']) if stages is None else stages
        if type(count) is not int or not 1<=count<=len(self.manifest['stages']) or self.calls>=16:raise ValueError('bounded stage/request count')
        if set(inputs)!=set(INPUTS):raise ValueError('input inventory')
        arrays=[]
        for name in INPUTS:
            array=np.asarray(inputs[name])
            if array.shape!=(32,) or array.dtype!=np.int32:raise ValueError('input shape/dtype')
            arrays.append(array.astype('<u4').tobytes())
        limits=(30522,512,2,2)
        if any(np.any(inputs[n]<0) or np.any(inputs[n]>=limit) for n,limit in zip(INPUTS,limits)) or not inputs['attention_mask'].any():raise ValueError('input ID/mask domain')
        self.calls+=1;started=time.perf_counter_ns()
        R.write_all(self.process.stdin,struct.pack('<4I',R.MAGIC,1 if readback else 2,512,count)+b''.join(arrays))
        rows=[]
        for s in range(count):
            words=struct.unpack('<8I2Q',R.read_exact(self.process.stdout,48,seconds=15))
            stage=self.manifest['stages'][s];expected_bytes=stage['output_region']['bytes'] if readback or s+1==count else 0
            expected_flags=63 if s+1==count else 21
            if words[:8]!=(R.MAGIC,0,0,expected_flags,1,1,expected_bytes,self.calls*1024+s):raise RuntimeError('stage status/preservation/callback/sequence failed: '+repr(words))
            data=R.read_exact(self.process.stdout,expected_bytes) if expected_bytes else b''
            rows.append(dict(name=stage['name'],data=data,submit_ns=words[8],completion_ns=words[9],flags=words[3]))
        self.roundtrip_ns=time.perf_counter_ns()-started
        return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path);p.add_argument('receipt',type=Path)
    p.add_argument('--decoder-prepared',type=Path,help='Verified Qwen preparation; selects the shared 1 GiB decoder executor')
    p.add_argument('--reference',type=Path,help='Case NPZ; explicit --run-first-gather dispatches only stage 0')
    p.add_argument('--run-first-gather',action='store_true')
    p.add_argument('--trials',type=int,choices=(1,2,3),default=1)
    a=p.parse_args();session=DecoderSession(a.bundle,a.decoder_prepared) if a.decoder_prepared else Session(a.bundle);numerical=None;error=None
    try:
        with session:
            if a.run_first_gather:
                if not a.reference:raise ValueError('independent reference required')
                with np.load(a.reference,allow_pickle=False) as z:
                    inputs={n:z[n].astype(np.int32) for n in (DECODER_INPUTS if a.decoder_prepared else INPUTS)}
                    expected=z['embedding.word' if a.decoder_prepared else 'native_schedule_estimate:embedding.word'].astype('<f4')
                trials=[]
                for trial in range(a.trials):
                    rows=session.execute(inputs,stages=1,**({'reset':True} if a.decoder_prepared else {}))
                    got=np.frombuffer(rows[0]['data'],dtype='<f4').reshape(expected.shape)
                    row=dict(exact=bool(np.array_equal(got.view('<u4'),expected.view('<u4'))),output_sha256=R.sha(rows[0]['data']))
                    trials.append(row)
                    if not row['exact']:raise RuntimeError('independent embedding gather comparison failed')
                numerical=dict(exact=all(t['exact'] for t in trials),trials=trials,output_sha256=trials[-1]['output_sha256'])
    except (RuntimeError,TimeoutError,EOFError) as e:error=type(e).__name__+': '+str(e)
    passed=error is None and getattr(session,'native_returncode',None)==0 and getattr(session,'before_recovery',None)==getattr(session,'after_recovery',None) and not getattr(session,'new_events',None)
    report=dict(format='g17-native-inference-session-v1',passed=passed,error=error,gpu_dispatched=session.calls>0,
                scope='first embedding gather only' if a.run_first_gather else 'complete model graph preparation; zero Submits',
                numerical=numerical,requests=session.calls,binary_sha256=getattr(session,'binary_sha256',None),
                source_sha256=R.sha(R.SOURCE.read_bytes()),header_sha256=R.sha((ROOT/'spike/agxsub/g17workload_inference.h').read_bytes()),
                decoder_header_sha256=R.sha((ROOT/'spike/agxsub/g17workload_decoder.h').read_bytes()) if a.decoder_prepared else None,
                before_recovery=getattr(session,'before_recovery',None),after_recovery=getattr(session,'after_recovery',None),new_events=getattr(session,'new_events',None),
                native_log=getattr(session,'native_log',''),native_returncode=getattr(session,'native_returncode',None))
    a.receipt.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('passed','scope','gpu_dispatched','error')}))
    raise SystemExit(0 if passed else 1)
