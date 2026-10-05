"""Greedy Qwen text generation using the shared persistent below-Metal runtime.

CPU tokenization and token selection only. All model arithmetic, including
logits and KV updates, runs in repository-authored native GPU programs.
"""
import argparse
import json
import re
import resource
from pathlib import Path
import time
import numpy as np
from g17textinput import prepare,validate_tokenizer
from g17modelimport import load
from g17inferencesession import DecoderSession,ROOT,R


def model_request(session,ids,position,valid,*,reset=False):
    rows=len(ids)
    inputs=dict(token_ids=np.array(ids,np.int32),position_ids=np.arange(position,position+rows,dtype=np.int32),
                attention_mask=(np.arange(256)<valid).astype(np.int32),cache_valid_length=np.array(valid,np.int32))
    outputs=session.execute(inputs,reset=reset,readback=False)
    if any(output['data'] for output in outputs[:-1]):raise RuntimeError('generation read back model intermediates')
    logits=np.frombuffer(outputs[-1]['data'],'<f4').reshape(rows,151936)
    if not np.isfinite(logits).all():raise RuntimeError('nonfinite generated logits')
    timing=dict(submissions=len(outputs),intermediate_readback_bytes=0,logit_readback_bytes=len(outputs[-1]['data']),
                client_roundtrip_ns=session.roundtrip_ns,submit_call_ns=sum(o['submit_ns'] for o in outputs),
                submit_to_completion_ns=sum(o['completion_ns'] for o in outputs),
                output_sha256=R.sha(outputs[-1]['data']),cache_valid=session.valid)
    return logits,timing


def generate(session,tokenizer_directory,tokenizer,prompt,max_new_tokens,eos_ids,*,stream=False,capture=None):
    if type(max_new_tokens) is not int or not 1<=max_new_tokens<=128:raise ValueError('generation token bound')
    encoded=prepare(ROOT/'evidence/g17-inference-models-v1/qwen',tokenizer_directory,prompt,chat=True,limit=256-max_new_tokens)
    ids=encoded['token_ids'];started=time.perf_counter_ns();prefill=[];decode=[];generated=[];first_token_ns=None;frames=[]
    def save(vector,**metadata):
        if capture is None:return
        path=Path(capture)/('logits-'+str(len(frames))+'.npy');np.save(path,vector)
        frames.append(dict(metadata,path=str(path),sha256=R.sha(path.read_bytes()),bytes=path.stat().st_size))
    for start in range(0,len(ids),32):
        count=min(32,len(ids)-start)
        logits,timing=model_request(session,ids[start:start+count]+[0]*(32-count),start,start+count,reset=start==0)
        prefill.append(timing);next_logits=logits[count-1]
        save(next_logits,mode='prefill',start=start,valid_tokens=count,cache_valid=start+count)
    previous_text=''
    for step in range(max_new_tokens):
        token=int(next_logits.argmax());generated.append(token)
        if first_token_ns is None:first_token_ns=time.perf_counter_ns()-started
        text=tokenizer.decode(generated,skip_special_tokens=True)
        if stream and text.startswith(previous_text):print(text[len(previous_text):],end='',flush=True)
        previous_text=text
        if token in eos_ids:break
        if step+1<max_new_tokens:
            position=len(ids)+step
            logits,timing=model_request(session,[token],position,position+1)
            decode.append(timing);next_logits=logits[0]
            save(next_logits,mode='decode',input_token=token,position=position,cache_valid=position+1)
    if stream:print(flush=True)
    elapsed=time.perf_counter_ns()-started;decode_ns=sum(t['client_roundtrip_ns'] for t in decode)
    return dict(prompt=prompt,prompt_tokens=len(ids),prompt_token_ids=ids,generated_token_ids=generated,text=previous_text,logit_frames=frames,
        stop_reason='eos' if generated[-1] in eos_ids else 'max_new_tokens',tokenizer=encoded['tokenizer'],
        tokenizer_engine=encoded['tokenizer_engine'],prefill=prefill,decode=decode,
        first_token_ns=first_token_ns,total_ns=elapsed,
        decode_tokens_per_second=len(decode)*1e9/decode_ns if decode_ns else None,
        submissions_per_decode_token=[t['submissions'] for t in decode])


def run(bundle,prepared,tokenizer_directory,prompts,receipt,*,max_new_tokens=24,interactive=False,capture=None):
    if type(max_new_tokens) is not int or not 1<=max_new_tokens<=128:raise ValueError('generation token bound')
    manifest,config,_,_=load(ROOT/'evidence/g17-inference-models-v1/qwen')
    validate_tokenizer(tokenizer_directory,manifest)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_directory),local_files_only=True,trust_remote_code=False)
    eos=config['eos_token_id'];eos_ids={eos} if type(eos) is int else set(eos)
    started=time.perf_counter();session=DecoderSession(bundle,prepared);verification=time.perf_counter()-started
    if capture is not None:Path(capture).mkdir(parents=True,exist_ok=False)
    def capture_path():
        if capture is None:return None
        path=Path(capture)/('request-'+str(len(report['requests'])));path.mkdir();return path
    report=dict(format='g17-native-qwen-generation-v1',passed=False,error=None,requests=[],
        scope='bounded Qwen greedy inference; CPU tokenization/control, all model arithmetic below Metal',
        model_revision=manifest['revision'],bundle_manifest_sha256=R.sha((Path(bundle)/'manifest.json').read_bytes()),
        application_sha256=R.sha(Path(__file__).read_bytes()),bundle_verification_seconds=verification)
    try:
        with session:
            for prompt in prompts:
                request=generate(session,tokenizer_directory,tokenizer,prompt,max_new_tokens,eos_ids,stream=True,capture=capture_path())
                report['requests'].append(request)
            if interactive:
                while True:
                    try:prompt=input('Prompt: ')
                    except EOFError:break
                    if not prompt.strip():break
                    report['requests'].append(generate(session,tokenizer_directory,tokenizer,prompt,max_new_tokens,eos_ids,stream=True,capture=capture_path()))
    except (RuntimeError,TimeoutError,EOFError) as error:report['error']=type(error).__name__+': '+str(error)
    report.update(native_returncode=getattr(session,'native_returncode',None),before_recovery=getattr(session,'before_recovery',None),
        after_recovery=getattr(session,'after_recovery',None),new_events=getattr(session,'new_events',None),
        native_log=getattr(session,'native_log',''),native_binary_sha256=getattr(session,'binary_sha256',None),
        runtime_source_sha256=R.sha(R.SOURCE.read_bytes()),runtime_header_sha256=R.sha((ROOT/'spike/agxsub/g17workload_decoder.h').read_bytes()),
        runtime_guard_header_sha256=R.sha((ROOT/'spike/agxsub/g17workload_inference.h').read_bytes()),
        graph_initializations=1,weight_uploads=1,program_uploads=1,
        driver_prepare_seconds=getattr(session,'prepare_seconds',None),native_build_seconds=getattr(session,'build_seconds',None),
        resident_allocation_bytes=sum(r['bytes'] for r in session.manifest['resource']['rows'] if r['kind']!=1),
        numerical_validation='Separate preregistered decoder receipt; meaningful text is not a numerical oracle',
        timing_scope='host guarded elapsed time; includes stage headers and full readonly checks; no speedup claim')
    peak=re.findall(r'^DECODER PEAK RSS BYTES: (\d+)\.$',report['native_log'],re.MULTILINE)
    report.update(native_process_peak_rss_bytes=int(peak[-1]) if len(peak)==1 else None,
                  host_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  memory_scope='macOS per-process RSS high-water marks; native includes readonly checkpoint snapshot. Host/native RSS peaks are separate, must not be added as a simultaneous physical peak; GPU allocations are reported separately.')
    report['passed']=report['error'] is None and bool(report['requests']) and report['native_returncode']==0 and report['before_recovery']==report['after_recovery'] and report['new_events']==[]
    Path(receipt).write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','prepared','tokenizer'):p.add_argument(name,type=Path)
    p.add_argument('--prompt',action='append',default=[]);p.add_argument('--interactive',action='store_true')
    p.add_argument('--max-new-tokens',type=int,default=24);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--capture-logits',type=Path,help='Optional final-logit files for independent application validation')
    a=p.parse_args()
    if not a.prompt and not a.interactive:p.error('a prompt or --interactive is required')
    r=run(a.bundle,a.prepared,a.tokenizer,a.prompt,a.receipt,max_new_tokens=a.max_new_tokens,interactive=a.interactive,capture=a.capture_logits)
    print(json.dumps(dict(passed=r['passed'],error=r['error'],requests=len(r['requests']))));raise SystemExit(0 if r['passed'] else 1)
