"""Prepare immutable native checkpoint bytes and stage lifetimes; never dispatch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from g17modelimport import load, import_model
from g17checkpoint import CheckpointReader, CHUNK
from g17inferencelower import compile_graph, POLICY
from agxforge.g17.inferenceresources import plan


def widen_bfloat(raw):
    """Exact BF16 bit widening, not a floating conversion or multiplication."""
    return (np.frombuffer(raw,dtype='<u2').astype('<u4') << 16).view('<f4')


def rotary_table():
    """Fixed Qwen model constants, not host activation computation."""
    inverse=np.power(np.float64(1e6),-np.arange(32,dtype=np.float64)/32)
    phase=np.arange(256,dtype=np.float64)[:,None]*inverse[None,:]
    return np.concatenate([np.cos(phase),np.sin(phase)],axis=1).astype('<f4').tobytes()


def convert_weight(raw, shape, dtype='F32'):
    out,inside=shape
    if dtype not in ('F32','BF16'):raise ValueError('refused: projection checkpoint dtype')
    source=(np.frombuffer(raw,dtype='<f4') if dtype=='F32' else widen_bfloat(raw)).reshape(out,inside)
    if not np.isfinite(source).all():raise ValueError('refused: nonfinite checkpoint weight')
    with np.errstate(over='raise',invalid='raise'):
        result=source.T.astype('<f2').copy()
    return result.tobytes()


def prepare(metadata,checkpoint,destination,*,tokens=32):
    manifest,config,parameters,blobs=load(metadata)
    policy=POLICY
    if config['model_type']=='qwen2':
        from g17decoderlower import POLICY as decoder_policy
        policy=decoder_policy
    graph=import_model(metadata,tokens=tokens,sentence_pooling=config['model_type']=='bert')
    programs,lowering=compile_graph(graph,policy=policy)
    layout=plan(graph,lowering)
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    temporary=destination/'.parameters.tmp';final=destination/'parameters.bin'
    identities={}
    try:
        with CheckpointReader(checkpoint,manifest,blobs['safetensors-header.json']) as reader, temporary.open('xb') as output:
            remaining=layout['arenas']['parameters']
            while remaining:
                n=min(remaining,CHUNK);output.write(b'\xa5'*n);remaining-=n
            for name,region in layout['regions'].items():
                if region['arena']!='parameters':continue
                output.seek(region['offset'])
                digest=hashlib.sha256();count=0
                if name in lowering['prepared_parameters']:
                    spec=lowering['prepared_parameters'][name]
                    if 'source' in spec:
                        original=spec['source'];p=parameters[original]
                        source=reader.small_tensor(original,limit=16*CHUNK if p.dtype=='BF16' else 4*CHUNK)
                        if spec['dtype']=='F16':raw=convert_weight(source,p.shape,p.dtype)
                        elif spec['dtype']=='F32' and p.dtype=='BF16':raw=widen_bfloat(source).tobytes()
                        else:raise ValueError('refused: prepared checkpoint conversion')
                    elif 'sources' in spec:
                        raw=b''.join(reader.small_tensor(n) for n in spec['sources'])
                    elif spec==dict(shape=[256,64],dtype='F32',transformation='fixed split-half RoPE cos32/sin32 table: theta=1e6, FP64 calculation rounded to FP32',theta=1e6):
                        raw=rotary_table()
                    else:raise ValueError('refused: unrecognized derived model parameter')
                    chunks=[raw]
                else:
                    chunks=reader.chunks(name)
                for block in chunks:
                    if region['dtype']=='F32' and not np.isfinite(np.frombuffer(block,dtype='<f4')).all():
                        raise ValueError('refused: nonfinite raw checkpoint parameter')
                    if region['dtype']=='BF16' and np.any((np.frombuffer(block,dtype='<u2') & 0x7f80)==0x7f80):
                        raise ValueError('refused: nonfinite BF16 checkpoint parameter')
                    output.write(block);digest.update(block);count+=len(block)
                if count!=region['bytes']:raise ValueError('native parameter byte extent mismatch')
                identities[name]=dict(bytes=count,sha256=digest.hexdigest())
            reader.check();output.flush();os.fsync(output.fileno())
            checkpoint_receipt=reader.receipt
        os.chmod(temporary,0o444);os.link(temporary,final)
    finally:
        temporary.unlink(missing_ok=True)
    digest=hashlib.sha256()
    with final.open('rb') as stream:
        while block:=stream.read(CHUNK):digest.update(block)
    for sha,code in programs.items():(destination/(sha+'.bin')).write_bytes(code)
    result=dict(format='g17-native-inference-preparation-v1',status='checkpoint_prepared_not_executed',gpu_dispatched=False,
                checkpoint=checkpoint_receipt,policy=policy,layout=layout,parameters=identities,
                parameter_file=dict(bytes=final.stat().st_size,sha256=digest.hexdigest()),
                programs={sha:len(code) for sha,code in programs.items()},
                pending=lowering['pending'])
    (destination/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('metadata',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('destination',type=Path)
    p.add_argument('--tokens',type=int,choices=(1,32),default=32)
    a=p.parse_args();r=prepare(a.metadata,a.checkpoint,a.destination,tokens=a.tokens)
    print(json.dumps(dict(status=r['status'],arenas=r['layout']['arenas'],prepared_parameters=len(r['parameters']),
                          activation_slots=r['layout']['activation_slots'],gpu_dispatched=False)))
