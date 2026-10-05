"""Offline eager Transformers control for independent Qwen graph references."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from g17checkpoint import CheckpointReader
from g17modelimport import load


def comparison(actual,expected):
    actual=np.asarray(actual,dtype=np.float64);expected=np.asarray(expected,dtype=np.float64)
    if actual.shape!=expected.shape:raise ValueError('framework/reference shape mismatch')
    error=np.abs(actual-expected)
    return dict(passed=bool(np.isfinite(actual).all() and np.isfinite(expected).all() and np.all(error<=2e-4*(1+np.abs(expected)))),
                max_absolute_error=float(error.max()),max_budget_fraction=float(np.max(error/(2e-4*(1+np.abs(expected))))),
                elements=actual.size)


def load_eager_model(metadata,checkpoint,*,precision='float64'):
    if precision not in ('float32','float64'):raise ValueError('framework precision')
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.set_num_threads(1)
    manifest,config,parameters,blobs=load(metadata)
    cfg=Qwen2Config.from_dict(config);cfg._attn_implementation='eager'
    # Meta construction avoids randomly initializing half a billion parameters.
    with torch.device('meta'):model=Qwen2ForCausalLM(cfg)
    state={}
    with CheckpointReader(checkpoint,manifest,blobs['safetensors-header.json']) as reader:
        for name,p in parameters.items():
            if p.dtype!='BF16':raise ValueError('framework control checkpoint dtype')
            state[name]=torch.frombuffer(bytearray(b''.join(reader.chunks(name))),dtype=torch.bfloat16).reshape(p.shape).float()
        identity=reader.receipt
    missing=model.load_state_dict(state,strict=False,assign=True)
    if missing.unexpected_keys or missing.missing_keys!=['lm_head.weight']:
        raise ValueError('unexpected framework checkpoint inventory: '+str(missing))
    model.tie_weights();del state
    rotary=model.model.rotary_emb
    inv,scale=rotary.compute_default_rope_parameters(cfg,'cpu')
    rotary.inv_freq=inv;rotary.original_inv_freq=inv.clone();rotary.attention_scaling=scale
    if any(p.is_meta for p in model.parameters()) or any(b.is_meta for b in model.buffers()):
        raise ValueError('framework control has unmaterialized model tensors')
    if precision=='float64':model.double()
    model.eval()
    return model,identity


def run(metadata,checkpoint,references,receipt,*,precision='float64'):
    import transformers
    model,identity=load_eager_model(metadata,checkpoint,precision=precision)
    references=Path(references);reference_report=json.loads((references/'report.json').read_text())
    rows=[]
    with torch.inference_mode():
        for case,row in enumerate(reference_report['cases']):
            ids=torch.tensor([row['token_ids']],dtype=torch.long)
            result=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True,output_hidden_states=True)
            checks=[]
            for chunk,prefix in enumerate(row['prefill']):
                start=prefix['start'];count=prefix['valid_tokens']
                with np.load(references/f'case-{case}-original_fp64-prefill-{chunk}.npz',allow_pickle=False) as z:
                    checks.append(dict(stage='prefill-'+str(chunk)+'.logits',**comparison(result.logits[0,start:start+count].numpy(),z['logits'][:count])))
                    # Transformers hidden_states includes the embedding and each
                    # layer output; its final entry is additionally normalized.
                    for layer in range(23):
                        checks.append(dict(stage=f'prefill-{chunk}.layer-{layer}',**comparison(
                            result.hidden_states[layer+1][0,start:start+count].numpy(),z[f'layer.{layer}.ffn_residual'][:count])))
                    checks.append(dict(stage='prefill-'+str(chunk)+'.final_norm',**comparison(result.hidden_states[-1][0,start:start+count].numpy(),z['final_norm'][:count])))
            past=result.past_key_values
            lengths=[int(past.get_seq_length())]
            for step,decode in enumerate(row['decode']):
                token=torch.tensor([[decode['input_token']]],dtype=torch.long)
                result=model(input_ids=token,attention_mask=torch.ones((1,decode['position']+1),dtype=torch.long),past_key_values=past,use_cache=True)
                past=result.past_key_values;lengths.append(int(past.get_seq_length()))
                with np.load(references/f'case-{case}-original_fp64-decode-{step}.npz',allow_pickle=False) as z:
                    checks.append(dict(stage='decode-'+str(step)+'.logits',**comparison(result.logits[0].numpy(),z['logits'])))
            if lengths!=list(range(row['tokens'],row['tokens']+len(row['decode'])+1)):
                raise ValueError('framework cache length differs from reference')
            rows.append(dict(case=case,checks=checks,cache_lengths=lengths))
    report=dict(format='g17-qwen-independent-framework-control-v1',passed=all(c['passed'] for row in rows for c in row['checks']),
                gpu_dispatched=False,checkpoint=identity,torch_version=torch.__version__,transformers_version=transformers.__version__,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                reference_report_sha256=hashlib.sha256((references/'report.json').read_bytes()).hexdigest(),
                precision=precision,cases=rows,
                scope='eager CPU Transformers against independently implemented FP64 graph, prefill and two cached steps; library RMS and rotary internals retain their explicit FP32 operations')
    Path(receipt).write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('metadata','checkpoint','references','receipt'):parser.add_argument(name,type=Path)
    parser.add_argument('--precision',choices=('float32','float64'),default='float64')
    args=parser.parse_args();report=run(args.metadata,args.checkpoint,args.references,args.receipt,precision=args.precision)
    print(json.dumps(dict(passed=report['passed'],gpu_dispatched=False,cases=len(report['cases']))))
    raise SystemExit(0 if report['passed'] else 1)
