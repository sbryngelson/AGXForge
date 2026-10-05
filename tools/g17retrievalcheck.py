"""Offline independent checkpoint/FP64 checks of captured native retrieval.

No compilation or GPU work. Bound values come from the original encoder
contract; token IDs and FP32 embedding hashes are checked before comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from g17checkpoint import CheckpointReader
from g17inferencereference import graph_reference, compare, CONTRACT
from g17modelimport import load, import_model
from g17textinput import prepare

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def captured_embedding(request,values):
    actual=np.asarray(values,dtype='<f4')
    if actual.shape!=(384,) or not np.isfinite(actual).all():
        raise ValueError('retrieval capture embedding shape/finiteness')
    if hashlib.sha256(actual.tobytes()).hexdigest()!=request['output_sha256']:
        raise ValueError('retrieval capture embedding hash')
    return actual


def run(retrieval,checkpoint,tokenizer,receipt):
    from transformers import BertConfig,BertModel
    retrieval=Path(retrieval);native=json.loads(retrieval.read_text())
    if native.get('format')!='g17-native-retrieval-v1' or not native.get('passed') or not native.get('repeat_exact'):
        raise ValueError('successful retained retrieval required')
    requests=native['requests'];embeddings=native['embeddings']
    if not 3<=len(requests)<=16 or len(requests)!=len(embeddings):
        raise ValueError('retrieval request/embedding coverage')
    metadata=ROOT/'evidence/g17-inference-models-v1/minilm'
    manifest,config,specs,blobs=load(metadata);contract=json.loads(CONTRACT.read_text())
    if native['model_revision']!=manifest['revision'] or manifest['revision']!=contract['model_revision']:
        raise ValueError('retrieval checkpoint revision')
    graph=import_model(metadata,sentence_pooling=True)
    torch.set_num_threads(1)
    parameters={};state={}
    with CheckpointReader(checkpoint,manifest,blobs['safetensors-header.json']) as reader:
        for name,p in specs.items():
            if p.dtype not in ('F32','I64'):raise ValueError('encoder reference checkpoint dtype')
            raw=b''.join(reader.chunks(name))
            array=np.frombuffer(raw,dtype='<f4' if p.dtype=='F32' else '<i8').copy().reshape(p.shape)
            state[name]=torch.from_numpy(array)
            if name in graph['tensors'] and graph['tensors'][name]['role']=='parameter':parameters[name]=array
        identity=reader.receipt
    cfg=BertConfig.from_dict(config);cfg._attn_implementation='eager'
    model=BertModel(cfg).double().eval()
    filtered={n:v for n,v in state.items() if n in model.state_dict()}
    if set(state)-set(filtered)-{'embeddings.position_ids'}:raise ValueError('unexpected encoder checkpoint keys')
    model.load_state_dict(filtered,strict=True)
    checks=[];actuals=[]
    with torch.inference_mode():
        for request,embedding in zip(requests,embeddings,strict=True):
            encoded=prepare(metadata,tokenizer,request['text'],limit=32)
            if encoded['tokenizer']!=native['tokenizer'] or request.get('token_ids')!=encoded['token_ids']:
                raise ValueError('retrieval requires captured token IDs matching pinned tokenizer')
            ids=encoded['token_ids'];count=len(ids)
            inputs=dict(token_ids=np.array(ids+[0]*(32-count),np.int32),position_ids=np.arange(32,dtype=np.int32),
                        token_type_ids=np.zeros(32,np.int32),attention_mask=np.array([1]*count+[0]*(32-count),np.int32))
            actual=captured_embedding(request,embedding);actuals.append(actual)
            original=graph_reference(graph,parameters,inputs,policy='original_fp64')
            policy=graph_reference(graph,parameters,inputs,policy='half_transport_fp64')['sentence.embedding']
            hidden=model(input_ids=torch.from_numpy(inputs['token_ids'].astype(np.int64))[None],
                         position_ids=torch.from_numpy(inputs['position_ids'].astype(np.int64))[None],
                         token_type_ids=torch.from_numpy(inputs['token_type_ids'].astype(np.int64))[None],
                         attention_mask=torch.from_numpy(inputs['attention_mask'].astype(np.int64))[None]).last_hidden_state[0].numpy()
            control=float(np.max(np.abs(hidden-original['layer.5.ffn_norm'])))
            pooled=hidden[:count].mean(axis=0);expected=pooled/max(float(np.linalg.norm(pooled)),1e-12)
            quality=compare(actual,expected);error=np.abs(actual.astype(np.float64)-policy)
            budget=2e-4*(1+np.abs(policy));limits=contract['original_model_quality']
            passed=(control<=1e-10 and quality['max_abs']<=limits['max_absolute_embedding_error']
                    and quality['cosine']>=limits['minimum_cosine_similarity'] and np.all(error<=budget))
            checks.append(dict(text=request['text'],token_ids=ids,passed=bool(passed),framework_graph_max_error=control,
                               original_model_quality=quality,half_policy_max_error=float(error.max()),
                               half_policy_max_budget_fraction=float((error/budget).max())))
    repeat=actuals[0].tobytes()==actuals[-1].tobytes() and requests[0]['token_ids']==requests[-1]['token_ids']
    ranking=[dict(document_index=i,cosine_similarity=float(np.dot(actuals[0].astype(np.float64),x.astype(np.float64))/
                         (np.linalg.norm(actuals[0].astype(np.float64))*np.linalg.norm(x.astype(np.float64)))))
             for i,x in enumerate(actuals[1:-1])]
    ranking.sort(key=lambda x:x['cosine_similarity'],reverse=True)
    ranking_exact=ranking==[{k:r[k] for k in ('document_index','cosine_similarity')} for r in native['ranking']]
    result=dict(format='g17-native-retrieval-independent-check-v1',passed=all(c['passed'] for c in checks) and repeat and ranking_exact,
                gpu_dispatched=False,checkpoint=identity,checks=checks,repeat_exact=repeat,ranking_exact=ranking_exact,
                retrieval_sha256=sha(retrieval),contract_sha256=sha(CONTRACT),source_sha256=sha(__file__),
                graph_reference_sha256=sha(ROOT/'tools/g17inferencereference.py'),
                scope='Every captured FP32 embedding hash and token sequence checked; independent original-checkpoint eager FP64 BertModel plus original FP64 graph control and half-transport FP64 graph. Original preregistered max-absolute/cosine and half-policy bounds unchanged. CPU-only validation, not a production fallback.')
    Path(receipt).write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('retrieval','checkpoint','tokenizer','receipt'):p.add_argument(name,type=Path)
    a=p.parse_args();r=run(a.retrieval,a.checkpoint,a.tokenizer,a.receipt)
    print(json.dumps(dict(passed=r['passed'],requests=len(r['checks']),gpu_dispatched=False)))
    raise SystemExit(0 if r['passed'] else 1)
