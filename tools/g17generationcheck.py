"""Offline independent Qwen framework check of captured native generation logits."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from g17decoderframework import load_eager_model

ROOT=Path(__file__).resolve().parents[1]


def run(generation,checkpoint,receipt):
    generation=Path(generation);native=json.loads(generation.read_text())
    if native['format']!='g17-native-qwen-generation-v1' or not native['passed']:
        raise ValueError('successful native generation receipt required')
    model,identity=load_eager_model(ROOT/'evidence/g17-inference-models-v1/qwen',checkpoint,precision='float64')
    requests=[]
    with torch.inference_mode():
        for request in native['requests']:
            ids=torch.tensor([request['prompt_token_ids']],dtype=torch.long)
            result=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True)
            full_logits=result.logits[0].numpy();past=result.past_key_values;checks=[]
            for frame in request['logit_frames']:
                path=Path(frame['path'])
                if path.stat().st_size!=frame['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=frame['sha256']:
                    raise ValueError('captured generation logit identity')
                actual=np.load(path,allow_pickle=False).astype(np.float64)
                if frame['mode']=='prefill':
                    expected=full_logits[frame['start']+frame['valid_tokens']-1]
                elif frame['mode']=='decode':
                    position=frame['position']
                    result=model(input_ids=torch.tensor([[frame['input_token']]],dtype=torch.long),
                        attention_mask=torch.ones((1,position+1),dtype=torch.long),past_key_values=past,use_cache=True)
                    past=result.past_key_values;expected=result.logits[0,0].numpy()
                    if int(past.get_seq_length())!=frame['cache_valid']:raise ValueError('framework/native generation cache length')
                else:raise ValueError('generation frame mode')
                if actual.shape!=expected.shape:raise ValueError('generation logit shape')
                error=np.abs(actual-expected);budget=.05+.003*np.abs(expected)
                checks.append(dict(mode=frame['mode'],cache_valid=frame['cache_valid'],
                    passed=bool(np.isfinite(actual).all() and np.isfinite(expected).all() and np.all(error<=budget)),
                    max_absolute_error=float(error.max()),max_budget_fraction=float(np.max(error/budget)),
                    native_top1=int(actual.argmax()),framework_top1=int(expected.argmax()),elements=actual.size))
            if len(checks)!=len(request['prefill'])+len(request['decode']):raise ValueError('generation frame coverage')
            requests.append(dict(prompt=request['prompt'],checks=checks))
    repeat=None
    for i,a in enumerate(native['requests']):
        for b in native['requests'][i+1:]:
            if a['prompt']==b['prompt']:
                repeat=a['generated_token_ids']==b['generated_token_ids'] and [f['sha256'] for f in a['logit_frames']]==[f['sha256'] for f in b['logit_frames']]
                if not repeat:raise ValueError('native generation changed after reset')
    report=dict(format='g17-native-qwen-generation-independent-check-v1',passed=all(c['passed'] for r in requests for c in r['checks']) and repeat is not False,
        gpu_dispatched=False,checkpoint=identity,requests=requests,repeat_exact=repeat,
        generation_receipt_sha256=hashlib.sha256(generation.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        framework_loader_sha256=hashlib.sha256((ROOT/'tools/g17decoderframework.py').read_bytes()).hexdigest(),
        scope='every captured last-valid-row native logit versus eager original-checkpoint FP64 framework, native tokens teacher-forced; no second GPU run')
    Path(receipt).write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('generation','checkpoint','receipt'):p.add_argument(name,type=Path)
    a=p.parse_args();r=run(a.generation,a.checkpoint,a.receipt)
    print(json.dumps(dict(passed=r['passed'],repeat_exact=r['repeat_exact'],requests=len(r['requests']))));raise SystemExit(0 if r['passed'] else 1)
