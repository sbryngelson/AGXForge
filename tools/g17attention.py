"""Actual MiniLM layer-0 attention programs and independent FP64 stage reference."""
import math
import struct
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROWS, WIDTH, HEADS, DIM = 32, 384, 12, 32
EPSILON = 1e-12
LOCAL_ERROR_FACTOR = 2e-5
FINAL_ERROR_FACTOR = 2e-4


def bits(value):
    return struct.unpack("<I",struct.pack("<f",value))[0]


def program(name,names):
    import g17ir as ir
    buffers = [ir.Buffer(n,i+1,elem=ir.F32) for i,n in enumerate(names)]
    fn = ir.Function(name,buffers)
    return fn,ir.Builder(fn,fn.block("entry")),buffers


def scores_ir():
    """Grid (32,384): x=key token, y=head*32+query token; output [12,32,32]."""
    import g17ir as ir
    fn,b,(q,k,out)=program("minilm_attention_scores",("query","key","scores"))
    key=b.builtin("thread_position_in_grid",axis="x")
    headrow=b.builtin("thread_position_in_grid",axis="y")
    head=b.shr(headrow,ir.Imm(5))
    row=b.sub(headrow,b.shl(head,ir.Imm(5)))
    headbase=b.shl(head,ir.Imm(5))
    qb=b.add(b.mul(row,b.const(WIDTH)),headbase)
    kb=b.add(b.mul(key,b.const(WIDTH)),headbase)
    total=b.const(bits(0.))
    for d in range(DIM):
        qi=b.add(qb,ir.Imm(d));ki=b.add(kb,ir.Imm(d))
        total=b.fma(b.load(q,qi,width="word"),b.load(k,ki,width="word"),total)
    value=b.fmul(total,b.const(bits(1/math.sqrt(DIM))))
    index=b.add(b.shl(headrow,ir.Imm(5)),key)
    b.store_at(out,index,value,width="word");b.ret()
    return fn


def softmax_ir():
    """Grid (384,1): one stable softmax for each head/query row of 32 keys."""
    import g17ir as ir
    fn,b,(scores,out)=program("minilm_attention_softmax",("scores","probabilities"))
    if not callable(getattr(b,"fmax",None)):
        raise NotImplementedError("stable attention softmax requires IR floating-point fmax; integer csel is not a substitute")
    row=b.builtin("thread_position_in_grid")
    base=b.shl(row,ir.Imm(5))
    indices=[b.add(base,ir.Imm(i)) for i in range(ROWS)]
    values=[b.load(scores,i,width="word") for i in indices]
    maximum=values[0]
    for x in values[1:]:maximum=b.fmax(maximum,x)
    negative=b.fmul(maximum,b.const(bits(-1.)))
    scale=b.const(bits(math.log2(math.e)))
    terms=[b.exp2(b.fmul(b.fadd(x,negative),scale)) for x in values]
    total=b.const(bits(0.))
    for x in terms:total=b.fadd(total,x)
    inverse=b.recip(total)
    for index,x in zip(indices,terms):b.store_at(out,index,b.fmul(x,inverse),width="word")
    b.ret();return fn


def context_ir():
    """Grid (384,32): x=head*32+channel, y=query token; output [32,384]."""
    import g17ir as ir
    fn,b,(p,v,out)=program("minilm_attention_context",("probabilities","value","context"))
    channel=b.builtin("thread_position_in_grid",axis="x")
    row=b.builtin("thread_position_in_grid",axis="y")
    head=b.shr(channel,ir.Imm(5))
    pb=b.add(b.shl(head,ir.Imm(10)),b.shl(row,ir.Imm(5)))
    total=b.const(bits(0.))
    for token in range(ROWS):
        pi=b.add(pb,ir.Imm(token))
        vi=b.add(channel,b.const(token*WIDTH))
        total=b.fma(b.load(p,pi,width="word"),b.load(v,vi,width="word"),total)
    index=b.add(b.mul(row,b.const(WIDTH)),channel)
    b.store_at(out,index,total,width="word");b.ret();return fn


def residual_ir():
    fn,b,(source,projected,out)=program("minilm_attention_residual",("source","projected","residual"))
    index=b.builtin("thread_position_in_grid")
    value=b.fadd(b.load(source,index,width="word"),b.load(projected,index,width="word"))
    b.store_at(out,index,value,width="word");b.ret();return fn


def residual_grid_ir():
    """Grid (384,32): one output per feature and token row."""
    fn,b,(source,projected,out)=program("minilm_attention_residual_grid",("source","projected","residual"))
    column=b.builtin("thread_position_in_grid",axis="x")
    row=b.builtin("thread_position_in_grid",axis="y")
    index=b.add(b.mul(row,b.const(WIDTH)),column)
    value=b.fadd(b.load(source,index,width="word"),b.load(projected,index,width="word"))
    b.store_at(out,index,value,width="word");b.ret();return fn


def reference(source,parameters):
    """FP64 whole-block reference, independent of compiler and stage execution."""
    x=np.asarray(source,dtype=np.float64)
    if x.shape!=(ROWS,WIDTH) or not np.isfinite(x).all():raise ValueError("expected finite 32x384 input")
    p={k:np.asarray(v,dtype=np.float64) for k,v in parameters.items()}
    def norm(x,prefix):
        centered=x-x.mean(-1,keepdims=True)
        return centered/np.sqrt((centered*centered).mean(-1,keepdims=True)+EPSILON)*p[prefix+".weight"]+p[prefix+".bias"]
    def linear(x,prefix):return x@p[prefix+".weight"].T+p[prefix+".bias"]
    stages={"embedding_norm":norm(x,"embeddings.LayerNorm")}
    prefix="encoder.layer.0.attention."
    for name in ("query","key","value"):
        stages[name]=linear(stages["embedding_norm"],prefix+"self."+name)
    heads=lambda x:x.reshape(ROWS,HEADS,DIM).transpose(1,0,2)
    q,k,v=(heads(stages[n]) for n in ("query","key","value"))
    stages["scores"]=(q@k.transpose(0,2,1))/math.sqrt(DIM)
    exponent=np.exp(stages["scores"]-stages["scores"].max(-1,keepdims=True))
    stages["probabilities"]=exponent/exponent.sum(-1,keepdims=True)
    stages["context"]=(stages["probabilities"]@v).transpose(1,0,2).reshape(ROWS,WIDTH)
    stages["projected"]=linear(stages["context"],prefix+"output.dense")
    stages["residual"]=stages["embedding_norm"]+stages["projected"]
    stages["output"]=norm(stages["residual"],prefix+"output.LayerNorm")
    return stages


def deliver(model,destination):
    """Freeze real model parameters, FP64 references and actual compiler contracts."""
    import g17cc
    import g17minilmquery
    import g17layernorm
    destination=Path(destination)
    if destination.exists():raise ValueError("refusing to overwrite attention delivery")
    shapes={}
    for prefix in ("embeddings.LayerNorm","encoder.layer.0.attention.output.LayerNorm"):
        shapes.update({prefix+".weight":(WIDTH,),prefix+".bias":(WIDTH,)})
    for suffix in ("self.query","self.key","self.value","output.dense"):
        prefix="encoder.layer.0.attention."+suffix
        shapes.update({prefix+".weight":(WIDTH,WIDTH),prefix+".bias":(WIDTH,)})
    parameters={}
    with Path(model).open("rb") as stream:
        length=struct.unpack("<Q",stream.read(8))[0]
        if length>16*1024*1024:raise ValueError("invalid model header")
        header=json.loads(stream.read(length));base=8+length
        for name,shape in shapes.items():
            spec=header[name]
            if spec["dtype"]!="F32" or tuple(spec["shape"])!=shape:raise ValueError(name)
            start,end=spec["data_offsets"]
            if start<0 or end-start!=math.prod(shape)*4:raise ValueError("invalid tensor extent")
            stream.seek(base+start);raw=stream.read(end-start)
            parameters[name]=np.frombuffer(raw,dtype="<f4").copy().reshape(shape)
            if not np.isfinite(parameters[name]).all():raise ValueError("nonfinite model parameter")
    with Path(model).open("rb") as stream:
        model_hash=hashlib.file_digest(stream,"sha256").hexdigest()
    if model_hash!="53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db":
        raise ValueError("checkpoint differs from validated MiniLM query fixture")
    root=Path(__file__).resolve().parents[1]
    with np.load(root/"results/g17-layernorm-fixtures-v1/minilm_embeddings.npz",allow_pickle=False) as data:
        source=data["source"].copy()
        np.testing.assert_array_equal(data["gamma"],parameters["embeddings.LayerNorm.weight"])
        np.testing.assert_array_equal(data["beta"],parameters["embeddings.LayerNorm.bias"])
    refs=reference(source,parameters)
    destination.mkdir(parents=True)
    np.savez(destination/"fixture.npz",source=source,**parameters)
    np.savez(destination/"fp64-reference.npz",**refs)
    factories=dict(projection=g17minilmquery.query_projection_ir,layernorm=g17layernorm.layernorm_ir,
                   scores=scores_ir,softmax=softmax_ir,context=context_ir,residual=residual_ir)
    report=dict(model_sha256=model_hash,local_error_factor=LOCAL_ERROR_FACTOR,
                final_error_factor=FINAL_ERROR_FACTOR,gpu_dispatched=False,stages={})
    for name,factory in factories.items():
        try:
            fn=factory();(destination/(name+".ir.txt")).write_text(str(fn)+"\n")
            p=g17cc.compile_function(fn)
            (destination/(name+".bin")).write_bytes(p.code)
            (destination/(name+".abi.json")).write_text(json.dumps(p.abi_plain(p.abi()),indent=2)+"\n")
            report["stages"][name]=dict(status="compiled_unvalidated",bytes=len(p.code),
                                       code_sha256=hashlib.sha256(p.code).hexdigest())
        except (ValueError,RuntimeError,NotImplementedError) as error:
            report["stages"][name]=dict(status="refused",error=f"{type(error).__name__}: {error}")
    (destination/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model",type=Path);parser.add_argument("destination",type=Path)
    args=parser.parse_args()
    print(json.dumps(deliver(args.model,args.destination),indent=2))
