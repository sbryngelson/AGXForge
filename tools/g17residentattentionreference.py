"""Independent references for resident half-projection MiniLM attention.

Input is the NORMALIZED finite FP32 [32,384] embedding, not raw embeddings.
Q/K/V and output projections transport source/weights through half and preserve
64-K partial-fold ordering. Original FP64 and quantized FP64 application outputs
are reported separately. Scalar unary models are numpy estimates, not hardware
instruction oracles. Local references use ACTUAL GPU predecessor arrays.
"""
from pathlib import Path
import math
import numpy as np
import g17residentffnreference as F

ROWS, WIDTH, HEADS, DIM = 32, 384, 12, 32
PREFIX = 'encoder.layer.0.attention.'
PARAMETERS = {name+'_weight': PREFIX+'self.'+name+'.weight' for name in ('query','key','value')}
PARAMETERS.update({name+'_bias': PREFIX+'self.'+name+'.bias' for name in ('query','key','value')})
PARAMETERS.update(output_weight=PREFIX+'output.dense.weight',output_bias=PREFIX+'output.dense.bias',
                  gamma=PREFIX+'output.LayerNorm.weight',beta=PREFIX+'output.LayerNorm.bias')
PARAM_SHAPES = {name: (WIDTH,WIDTH) if name.endswith('_weight') else (WIDTH,) for name in PARAMETERS}
STAGE_SHAPES = dict(pack=(ROWS,WIDTH),q_projection=(ROWS,WIDTH),q_bias=(ROWS,WIDTH),
                    k_projection=(ROWS,WIDTH),k_bias=(ROWS,WIDTH),v_projection=(ROWS,WIDTH),v_bias=(ROWS,WIDTH),
                    scores=(HEADS,ROWS,ROWS),softmax=(HEADS,ROWS,ROWS),context=(ROWS,WIDTH),
                    context_pack=(ROWS,WIDTH),out_projection=(ROWS,WIDTH),residual=(ROWS,WIDTH),layernorm=(ROWS,WIDTH))
HALF_STAGES = ('pack','context_pack')


def arrays():
    """Existing fixture, embedding normalized independently in FP64 then FP32."""
    path=Path(__file__).resolve().parents[1]/'results/g17-attention-delivery-v1/fixture.npz'
    with np.load(path,allow_pickle=False) as z:
        source=z['source'].astype(np.float64)
        centered=source-source.mean(axis=1,keepdims=True)
        normalized=(centered/np.sqrt(np.mean(centered*centered,axis=1,keepdims=True)+1e-12)*
                    z['embeddings.LayerNorm.weight'].astype(np.float64)+z['embeddings.LayerNorm.bias'].astype(np.float64)).astype(np.float32)
        return dict(source=normalized,**{k:z[v].astype(np.float32).copy() for k,v in PARAMETERS.items()})


def require_arrays(ar):
    if set(ar)!= {'source',*PARAM_SHAPES}:
        raise ValueError('refused: exactly normalized source and ten attention parameters required')
    for name,shape in dict(source=(ROWS,WIDTH),**PARAM_SHAPES).items():
        x=np.asarray(ar[name])
        if x.shape!=shape or x.dtype!=np.float32 or not np.isfinite(x).all():
            raise ValueError('refused: finite measured FP32 attention array: '+name)
        if (name=='source' or name.endswith('_weight')) and np.any(np.abs(x)>65504):
            raise ValueError('refused: attention half transport overflow: '+name)


def _heads(x):
    return x.reshape(ROWS,HEADS,DIM).transpose(1,0,2)


def _norm64(x,g,b):
    x=np.asarray(x,dtype=np.float64)
    centered=x-x.mean(axis=1,keepdims=True)
    return centered/np.sqrt(np.mean(centered*centered,axis=1,keepdims=True)+1e-12)*g.astype(np.float64)+b.astype(np.float64)


def application_reference(ar,*,quantized=False):
    """Independent FP64 attention from normalized input; no embedding LN stage.

    quantized=True explicitly rounds only projection operands to half. It is a
    distinct application, not a replacement for the original FP32 reference.
    """
    require_arrays(ar)
    source=ar['source'].astype(np.float64)
    inp=source.astype(np.float16).astype(np.float64) if quantized else source
    weight=lambda name: ar[name].astype(np.float16).astype(np.float64) if quantized else ar[name].astype(np.float64)
    p={name:inp@weight(name+'_weight').T+ar[name+'_bias'].astype(np.float64) for name in ('query','key','value')}
    q,k,v=(_heads(p[n]) for n in ('query','key','value'))
    scores=(q@k.transpose(0,2,1))/math.sqrt(DIM)
    e=np.exp(scores-scores.max(axis=-1,keepdims=True));prob=e/e.sum(axis=-1,keepdims=True)
    context=(prob@v).transpose(1,0,2).reshape(ROWS,WIDTH)
    ci=context.astype(np.float16).astype(np.float64) if quantized else context
    projected=ci@weight('output_weight').T+ar['output_bias'].astype(np.float64)
    residual=projected+source
    return dict(query=p['query'],key=p['key'],value=p['value'],scores=scores,softmax=prob,
                context=context,projected=projected,residual=residual,output=_norm64(residual,ar['gamma'],ar['beta']))


def _fma_acc(a,b,acc):
    # Normal finite FP32 operands in the fixture; FP64 computes the product and
    # addition before the sole FP32 rounding (no float32 multiply then add).
    return F._f32(F._f32(a).astype(np.float64)*F._f32(b).astype(np.float64)+F._f32(acc).astype(np.float64))


def scalar_scores(q,k):
    q,k=_heads(q),_heads(k)
    out=np.zeros((HEADS,ROWS,ROWS),np.float32)
    for d in range(DIM): out=_fma_acc(q[:,:,d,None],k[:,None,:,d],out)
    return F._mul(out,np.float32(1/math.sqrt(DIM)))


def scalar_softmax(scores):
    maximum=np.max(scores,axis=-1,keepdims=True)
    exponent=F._mul(F._add(scores,F._mul(maximum,-1)),np.float32(math.log2(math.e)))
    terms=F._f32(np.exp2(exponent))
    total=np.zeros(scores.shape[:-1]+(1,),np.float32)
    for k in range(ROWS): total=F._add(total,terms[:,:,k:k+1])
    inverse=F._f32(np.float32(1)/total)
    return F._mul(terms,inverse)


def scalar_context(prob,v):
    vh=_heads(v);out=np.zeros((HEADS,ROWS,DIM),np.float32)
    for k in range(ROWS):out=_fma_acc(prob[:,:,k,None],vh[:,None,k,:],out)
    return out.transpose(1,0,2).reshape(ROWS,WIDTH)


def original_final_compare(got, reference):
    """Existing attention final 2e-4 factor; no quantization-driven relaxation."""
    got, reference = np.asarray(got), np.asarray(reference, dtype=np.float64)
    if got.shape != reference.shape:
        raise ValueError('attention final output shape differs')
    error = np.abs(got.astype(np.float64)-reference)
    budget = 2e-4*(1+np.abs(reference))
    finite = np.isfinite(got)&np.isfinite(reference)
    return dict(passed=bool(np.all(finite&(error<=budget))),
                failures=int(np.count_nonzero(~finite|(error>budget))),
                max_abs_error=float(np.max(np.where(finite,error,np.inf))),
                max_error_over_budget=float(np.max(np.where(finite,error/budget,np.inf))),
                error_bound='Existing attention final: 2e-4 * (1 + abs(original FP64 reference))')


def predictions(ar):
    require_arrays(ar)
    p=dict(pack=ar['source'].astype(np.float16))
    for short,long in (('q','query'),('k','key'),('v','value')):
        p[short+'_projection']=F.legacy_block_mma(p['pack'],ar[long+'_weight'])
        p[short+'_bias']=F._add(p[short+'_projection'],ar[long+'_bias'])
    p['scores']=scalar_scores(p['q_bias'],p['k_bias'])
    p['softmax']=scalar_softmax(p['scores'])
    p['context']=scalar_context(p['softmax'],p['v_bias'])
    p['context_pack']=p['context'].astype(np.float16)
    if not np.isfinite(p['context_pack']).all():raise ValueError('refused: attention context half transport overflow')
    p['out_projection']=F.legacy_block_mma(p['context_pack'],ar['output_weight'])
    p['residual']=F._add(F._add(p['out_projection'],ar['output_bias']),ar['source'])
    p['layernorm']=F.layernorm_model(p['residual'],ar['gamma'],ar['beta'])
    original=application_reference(ar);quantized=application_reference(ar,quantized=True)
    return dict(stages=p,application_fp64=original,quantized_fp64=quantized,
                application_error=original_final_compare(p['layernorm'],original['output']),application_local_error=F.compare(p['layernorm'],original['output']),quantized_error=F.compare(p['layernorm'],quantized['output']),
                scope='Normalized input; half projection MMA model; unary estimates; errors report unchanged 2e-5 local factor, not automatic full-block admission')


def stagewise_references(ar,p):
    """Independent local stage math on captured predecessors; exact transports."""
    require_arrays(ar)
    if set(p)!=set(STAGE_SHAPES):raise ValueError('refused: fourteen attention readbacks required')
    for name,shape in STAGE_SHAPES.items():
        x=np.asarray(p[name]);dtype=np.float16 if name in HALF_STAGES else np.float32
        if x.shape!=shape or x.dtype!=dtype or not np.isfinite(x).all():raise ValueError('refused: attention readback shape/dtype/finiteness: '+name)
    r=dict(pack=ar['source'].astype(np.float16))
    for short,long in (('q','query'),('k','key'),('v','value')):
        r[short+'_projection']=p['pack'].astype(np.float64)@ar[long+'_weight'].astype(np.float16).astype(np.float64).T
        r[short+'_bias']=F._add(p[short+'_projection'],ar[long+'_bias'])
    q,k=_heads(p['q_bias']).astype(np.float64),_heads(p['k_bias']).astype(np.float64)
    r['scores']=(q@k.transpose(0,2,1))/math.sqrt(DIM)
    s=p['scores'].astype(np.float64);e=np.exp(s-s.max(axis=-1,keepdims=True));r['softmax']=e/e.sum(axis=-1,keepdims=True)
    r['context']=(p['softmax'].astype(np.float64)@_heads(p['v_bias']).astype(np.float64)).transpose(1,0,2).reshape(ROWS,WIDTH)
    r['context_pack']=p['context'].astype(np.float16)
    r['out_projection']=p['context_pack'].astype(np.float64)@ar['output_weight'].astype(np.float16).astype(np.float64).T
    r['residual']=F._add(F._add(p['out_projection'],ar['output_bias']),ar['source'])
    r['layernorm']=_norm64(p['residual'],ar['gamma'],ar['beta'])
    return r


def validation_contract():
    return dict(input='Normalized finite FP32 embeddings [32,384]; raw embedding LayerNorm is outside this workload',
                stages=list(STAGE_SHAPES),submissions=14,local_bound='2e-5*(1+abs(reference))',
                original_application_final_bound='Existing attention factor 2e-4*(1+abs(original FP64)); report separately, never replace with quantized reference',
                exact_stages=['pack','q_bias','k_bias','v_bias','context_pack','residual'],
                readonly=['source',*PARAM_SHAPES],guards='128-byte canaries around every payload; unchanged after execution',
                repeats='At least three different normalized inputs; immutable parameters/programs; reset writable regions')
