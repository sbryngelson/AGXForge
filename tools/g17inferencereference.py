"""Offline complete encoder references; never a production inference fallback."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from g17checkpoint import CheckpointReader
from g17modelimport import load, import_model
from g17textinput import prepare as tokenize
import g17residentffnreference as F
import g17residentattentionreference as A

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'evidence/g17-native-encoder-validation-contract.json'


def graph_reference(graph, parameters, inputs, *, policy):
    if (graph['profile']['family'],graph['profile']['width'],graph['profile']['heads'],graph['tokens'])!=('bert',384,12,32):
        raise ValueError('refused: reference supports complete 32-token encoder only')
    if policy not in ('original_fp64','half_transport_fp64','native_schedule_estimate'):
        raise ValueError('refused: reference policy')
    native=policy=='native_schedule_estimate';half=policy!='original_fp64'
    if set(inputs)!={'token_ids','position_ids','token_type_ids','attention_mask'}:
        raise ValueError('refused: encoder reference input inventory')
    for name,array in inputs.items():
        if np.asarray(array).shape!=(32,) or np.asarray(array).dtype!=np.int32:
            raise ValueError('refused: reference input shape/dtype: '+name)
    mask=inputs['attention_mask']
    if not np.isin(mask,[0,1]).all() or not mask.any():
        raise ValueError('refused: 0/1 mask with at least one valid token required')
    for name,limit in [('token_ids',graph['profile']['vocabulary']),('position_ids',512),('token_type_ids',2)]:
        if np.any(inputs[name]<0) or np.any(inputs[name]>=limit):
            raise ValueError('refused: reference index outside domain: '+name)
    dtype=np.float32 if native else np.float64
    values={n:np.asarray(v,dtype=dtype) for n,v in parameters.items()}
    values.update(inputs)
    outputs={}
    def matmul(a,b):
        return (torch.from_numpy(np.ascontiguousarray(a)) @ torch.from_numpy(np.ascontiguousarray(b))).numpy()
    for node in graph['nodes']:
        args=[values[n] for n in node['inputs']];op=node['op']
        if op=='gather_rows':y=args[0][args[1]]
        elif op=='add':
            y=args[0]
            for value in args[1:]:y=F._add(y,value) if native else y+value
        elif op=='linear':
            x,w=args[:2]
            if native:y=F.half_mma(x,w)
            else:
                if half:x=x.astype(np.float16).astype(np.float64);w=w.astype(np.float16).astype(np.float64)
                y=matmul(x,w.T)
            if len(args)==3:y=F._add(y,args[2]) if native else y+args[2]
        elif op=='layer_norm':
            x,g,b=args
            if native:y=F.layernorm_model(x,g,b)
            else:
                centered=x-x.mean(axis=-1,keepdims=True)
                y=centered/np.sqrt(np.mean(centered*centered,axis=-1,keepdims=True)+node['attrs']['epsilon'])*g+b
        elif op=='gelu_erf':
            x=args[0]
            if native:
                # The existing sign approximation deliberately saturates z*1e30.
                with np.errstate(over='ignore'):y=F.gelu_model(x)
            else:y=(torch.nn.functional.gelu(torch.from_numpy(x),approximate='none')).numpy()
        elif op=='attention':
            q,k,v,mask=args
            if native:
                scores=A.scalar_scores(q,k)
                scores=F._add(scores,np.where(mask[None,None,:]==1,np.float32(0),np.float32(-np.finfo(np.float32).max)))
                probability=A.scalar_softmax(scores);y=A.scalar_context(probability,v)
            else:
                q,k,v=(a.reshape(32,12,32).transpose(1,0,2) for a in (q,k,v))
                scores=matmul(q,k.transpose(0,2,1))*node['attrs']['scale']
                scores=np.where(mask[None,None,:]==1,scores,-np.inf)
                e=np.exp(scores-scores.max(axis=-1,keepdims=True));p=e/e.sum(axis=-1,keepdims=True)
                y=matmul(p,v).transpose(1,0,2).reshape(32,384)
        elif op=='masked_mean':
            x,mask=args
            if native:
                y=np.zeros(384,np.float32);count=np.float32(0)
                for row in range(32):
                    count=F._add(count,np.float32(mask[row]));y=F._add(y,F._mul(x[row],np.float32(mask[row])))
                y=F._mul(y,F._f32(np.float32(1)/count))
            else:y=(x*mask[:,None]).sum(axis=0)/max(mask.sum(),node['attrs']['denominator_min'])
        elif op=='l2_normalize':
            x=args[0]
            if native:
                partial=np.zeros(32,np.float32)
                for i in range(12):partial=F._add(partial,F._mul(x[i*32:(i+1)*32],x[i*32:(i+1)*32]))
                for bit in (1,2,4,8,16):partial=F._add(partial,partial[np.arange(32)^bit])
                inverse=F._f32(np.float32(1)/np.sqrt(np.maximum(partial,np.float32(1e-24))))
                y=F._mul(x,np.tile(inverse,12))
            else:y=x/max(float(np.linalg.norm(x)),node['attrs']['epsilon'])
        else:raise ValueError('refused: unimplemented reference op '+op)
        if not np.isfinite(y).all():raise ValueError('nonfinite reference at '+node['name'])
        values[node['outputs'][0]]=y;outputs[node['name']]=y
    return outputs


def compare(got,expected):
    got,expected=np.asarray(got,np.float64),np.asarray(expected,np.float64)
    if got.shape!=expected.shape or not np.isfinite(got).all() or not np.isfinite(expected).all():
        raise ValueError('reference comparison shape/finiteness')
    return dict(max_abs=float(np.max(np.abs(got-expected))),
                cosine=float(np.dot(got.reshape(-1),expected.reshape(-1))/(np.linalg.norm(got)*np.linalg.norm(expected))))


def run(metadata,checkpoint,tokenizer,destination):
    from transformers import BertConfig,BertModel
    torch.set_num_threads(1)
    manifest,config,specs,blobs=load(metadata)
    graph=import_model(metadata,sentence_pooling=True)
    contract=json.loads(CONTRACT.read_text())
    if manifest['revision']!=contract['model_revision']:raise ValueError('contract model revision mismatch')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    parameters={};state={}
    with CheckpointReader(checkpoint,manifest,blobs['safetensors-header.json']) as reader:
        for name,p in specs.items():
            raw=b''.join(reader.chunks(name))
            if p.dtype=='F32':array=np.frombuffer(raw,dtype='<f4').copy().reshape(p.shape)
            elif p.dtype=='I64':array=np.frombuffer(raw,dtype='<i8').copy().reshape(p.shape)
            else:raise ValueError('refused: encoder reference checkpoint dtype')
            state[name]=torch.from_numpy(array)
            if name in graph['tensors'] and graph['tensors'][name]['role']=='parameter':parameters[name]=array
        checkpoint_receipt=reader.receipt
    cfg=BertConfig.from_dict(config);cfg._attn_implementation='eager'
    model=BertModel(cfg).double().eval()
    # position_ids is a generated, nonpersistent Transformers buffer in this version.
    filtered={n:v for n,v in state.items() if n in model.state_dict()}
    extra=set(state)-set(filtered)
    if extra-{'embeddings.position_ids'}:raise ValueError('unexpected reference checkpoint keys: '+repr(extra))
    model.load_state_dict(filtered,strict=True)
    rows=[]
    for case,text in enumerate(contract['cases']):
        encoded=tokenize(metadata,tokenizer,text,limit=32)
        length=len(encoded['token_ids'])
        inputs=dict(token_ids=np.array(encoded['token_ids']+[0]*(32-length),np.int32),
                    position_ids=np.arange(32,dtype=np.int32),token_type_ids=np.zeros(32,np.int32),
                    attention_mask=np.array([1]*length+[0]*(32-length),np.int32))
        predictions={policy:graph_reference(graph,parameters,inputs,policy=policy)
                     for policy in ('original_fp64','half_transport_fp64','native_schedule_estimate')}
        with torch.inference_mode():
            original=model(input_ids=torch.from_numpy(inputs['token_ids'].astype(np.int64))[None],
                           position_ids=torch.from_numpy(inputs['position_ids'].astype(np.int64))[None],
                           token_type_ids=torch.from_numpy(inputs['token_type_ids'].astype(np.int64))[None],
                           attention_mask=torch.from_numpy(inputs['attention_mask'].astype(np.int64))[None]).last_hidden_state[0].numpy()
        control=float(np.max(np.abs(original-predictions['original_fp64']['layer.5.ffn_norm'])))
        if control>1e-10:raise ValueError('independent framework control failed: '+str(control))
        native=predictions['native_schedule_estimate']['sentence.embedding']
        quality=compare(native,predictions['original_fp64']['sentence.embedding'])
        scheduled=compare(native,predictions['half_transport_fp64']['sentence.embedding'])
        reference=predictions['half_transport_fp64']['sentence.embedding']
        policy_pass=bool(np.all(np.abs(native-reference)<=2e-4*(1+np.abs(reference))))
        quality_pass=quality['max_abs']<=contract['original_model_quality']['max_absolute_embedding_error'] and quality['cosine']>=contract['original_model_quality']['minimum_cosine_similarity']
        arrays=dict(inputs)
        for policy,values in predictions.items():arrays.update({policy+':'+n:v for n,v in values.items()})
        path=destination/f'case-{case}.npz';np.savez(path,**arrays)
        rows.append(dict(text=text,tokens=length,input=encoded,framework_control_max_abs=control,
                         native_estimate_vs_original=quality,native_estimate_vs_policy=scheduled,
                         cpu_policy_bound_passed=policy_pass,cpu_original_quality_passed=quality_pass,
                         reference_file=dict(name=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                         node_outputs_per_policy={k:len(v) for k,v in predictions.items()}))
    result=dict(format='g17-complete-encoder-reference-v1',status='cpu_references_generated_not_gpu_validated',
                gpu_dispatched=False,checkpoint=checkpoint_receipt,contract_sha256=hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
                graph_sha256=hashlib.sha256(json.dumps(graph,sort_keys=True).encode()).hexdigest(),
                source_sha256={n:hashlib.sha256((ROOT/'tools'/n).read_bytes()).hexdigest() for n in
                    ('g17inferencereference.py','g17residentffnreference.py','g17residentattentionreference.py')},
                libraries=dict(torch=torch.__version__,numpy=np.__version__,transformers=__import__('transformers').__version__),
                cases=rows,scope='Independent original FP64 framework control and half-transport FP64 model; native arithmetic estimate is not an exact transcendental silicon oracle')
    (destination/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('metadata','checkpoint','tokenizer','destination'):p.add_argument(n,type=Path)
    a=p.parse_args();r=run(a.metadata,a.checkpoint,a.tokenizer,a.destination)
    print(json.dumps(dict(status=r['status'],cases=len(r['cases']),gpu_dispatched=False,
                         quality=[c['native_estimate_vs_original'] for c in r['cases']])))
