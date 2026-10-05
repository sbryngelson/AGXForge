"""Offline independent checkpoint-preparation check; no GPU or model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from g17modelimport import load
from g17checkpoint import CheckpointReader


def check(metadata, checkpoint, prepared, lowering):
    manifest,_,parameters,blobs=load(metadata)
    prepared=Path(prepared)
    report=json.loads((prepared/'manifest.json').read_text())
    lowered=json.loads(Path(lowering).read_text())
    torch.set_num_threads(1)
    identity=report['parameter_file']
    digest=hashlib.sha256()
    with (prepared/'parameters.bin').open('rb') as stream:
        while block:=stream.read(1024*1024):digest.update(block)
    if (prepared/'parameters.bin').stat().st_size!=identity['bytes'] or digest.hexdigest()!=identity['sha256']:
        raise ValueError('prepared parameter file identity mismatch')
    if (prepared/'parameters.bin').stat().st_mode & 0o222:
        raise ValueError('prepared parameter file is writable')
    rows=[]
    with CheckpointReader(checkpoint,manifest,blobs['safetensors-header.json']) as reader, (prepared/'parameters.bin').open('rb') as data:
        if reader.receipt!=report['checkpoint']:raise ValueError('checkpoint identity mismatch')
        for name,region in report['layout']['regions'].items():
            if region['arena']!='parameters':continue
            spec=lowered['prepared_parameters'].get(name)
            if spec and 'source' in spec:
                source=spec['source'];p=parameters[source]
                if p.dtype=='BF16':
                    tensor=torch.frombuffer(bytearray(reader.small_tensor(source,limit=16*1024*1024)),dtype=torch.bfloat16).reshape(p.shape)
                elif p.dtype=='F32':
                    array=np.frombuffer(reader.small_tensor(source),dtype='<f4').copy().reshape(p.shape)
                    tensor=torch.from_numpy(array)
                else:raise ValueError('unsupported independent checkpoint dtype')
                if spec['dtype']=='F16':
                    expected=tensor.T.to(torch.float16).contiguous().numpy().tobytes()
                    method='torch CPU transpose and half conversion'
                elif spec['dtype']=='F32' and p.dtype=='BF16':
                    expected=tensor.to(torch.float32).numpy().tobytes()
                    method='torch CPU BF16 to FP32 conversion'
                else:raise ValueError('unsupported independent prepared conversion')
                chunks=[expected]
            elif spec and 'sources' in spec:
                expected=b''.join(reader.small_tensor(n) for n in spec['sources'])
                method='raw concatenation without arithmetic'
                chunks=[expected]
            elif spec and name=='prepared:rotary.cos_sin':
                inv=torch.pow(torch.tensor(1e6,dtype=torch.float64),-torch.arange(32,dtype=torch.float64)/32)
                phase=torch.arange(256,dtype=torch.float64)[:,None]*inv[None,:]
                expected=torch.cat([phase.cos(),phase.sin()],dim=1).float().numpy().tobytes()
                chunks=[expected];method='independent Torch FP64 rotary table then FP32 rounding'
            else:
                chunks=reader.chunks(name);method='streamed checkpoint raw bytes'
            data.seek(region['offset']);count=0;digest=hashlib.sha256()
            for expected in chunks:
                actual=data.read(len(expected))
                if actual!=expected:raise ValueError('prepared bytes differ from independent reference: '+name)
                count+=len(actual);digest.update(actual)
            if count!=region['bytes'] or digest.hexdigest()!=report['parameters'][name]['sha256']:
                raise ValueError('parameter digest mismatch: '+name)
            for start,end in [(region['guard_before'],region['offset']),
                              (region['offset']+region['bytes'],region['guard_after']+region['guard_bytes'])]:
                data.seek(start)
                if data.read(end-start)!=b'\xa5'*(end-start):raise ValueError('parameter guard/padding changed: '+name)
            rows.append(dict(name=name,bytes=count,method=method,exact=True))
    for sha,size in report['programs'].items():
        code=(prepared/(sha+'.bin')).read_bytes()
        if len(code)!=size or hashlib.sha256(code).hexdigest()!=sha:raise ValueError('program identity mismatch')
    return dict(format='g17-native-preparation-validation-v1',status='all_prepared_parameters_exact',
                gpu_dispatched=False,torch_version=torch.__version__,checkpoint=report['checkpoint'],
                parameter_file=identity,arenas=report['layout']['arenas'],stage_count=len(report['layout']['stages']),
                activation_slots=report['layout']['activation_slots'],program_count=len(report['programs']),
                parameters=rows,scope='checkpoint conversion, raw bytes, parameter guards, readonly file and code identities only',
                prepared_manifest_sha256=hashlib.sha256((prepared/'manifest.json').read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('metadata','checkpoint','prepared','lowering','receipt'):p.add_argument(name,type=Path)
    a=p.parse_args();r=check(a.metadata,a.checkpoint,a.prepared,a.lowering)
    a.receipt.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('status','program_count','stage_count','arenas','gpu_dispatched')}))
