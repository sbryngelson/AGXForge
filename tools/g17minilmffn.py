"""Actual layer-0 MiniLM feed-forward operations, before native admission.

Source: spike/serve/mix_mlx.py embed_fn. GELU retains the erf definition used by
mlx.nn.gelu; no tanh or sigmoid replacement is silently selected.
"""
import math
import struct
import hashlib
import json
from pathlib import Path
import numpy as np

ROWS,WIDTH,HIDDEN=32,384,1536


def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]


def dense_ir(input_width,output_width,rows=ROWS):
    import g17ir as ir
    if (input_width,output_width) not in ((WIDTH,HIDDEN),(HIDDEN,WIDTH)) or rows not in (1,ROWS):
        raise ValueError('expected the MiniLM expansion or contraction at 1 or 32 rows')
    bufs=[ir.Buffer(n,i,elem=ir.F32) for i,n in enumerate(('source','weight','bias','output'),1)]
    source,weight,bias,output=bufs
    f=ir.Function('minilm_ffn_%d_%d'%(input_width,output_width),bufs)
    pre,loop,end=[f.block(n) for n in ('entry','reduction','exit')]
    b=ir.Builder(f,pre)
    column=b.builtin('thread_position_in_grid',axis='x',name='column')
    row=b.builtin('thread_position_in_grid',axis='y',name='row')
    stride=b.const(input_width,name='input_stride')
    source_base=b.mul(row,stride,name='source_base')
    weight_base=b.mul(column,stride,name='weight_base')
    output_base=b.mul(row,b.const(output_width),name='output_base')
    zero=b.const(0);total0=b.const(bits(0.0));b.br(loop)
    b.at(loop);k=b.phi(zero,name='k');total=b.phi(total0,type=ir.F32,name='sum')
    x=b.load(source,b.add(source_base,k),width='word')
    w=b.load(weight,b.add(weight_base,k),width='word')
    next_total=b.fma(x,w,total);next_k=b.add(k,ir.Imm(1))
    b.phi_latch(k,next_k);b.phi_latch(total,next_total)
    b.br_cond(b.cmp(next_k,input_width,'lt'),loop,end)
    b.at(end)
    result=b.fadd(next_total,b.load(bias,column,width='word'))
    b.store_at(output,b.add(output_base,column),result,width='word');b.ret()
    return f


def gelu_ir():
    import g17ir as ir
    source,output=ir.Buffer('source',1,elem=ir.F32),ir.Buffer('output',2,elem=ir.F32)
    f=ir.Function('minilm_ffn_gelu',[source,output]);b=ir.Builder(f,f.block('entry'))
    index=b.builtin('thread_position_in_grid',name='index')
    x=b.load(source,index,width='word')
    scaled=b.fmul(x,b.const(bits(1/math.sqrt(2))))
    # Keep the real missing operation explicit. The compiler must refuse this
    # until an implementation with a stated numerical domain/bound exists.
    e=b._def('erf',[scaled],ir.F32,name='erf')
    result=b.fmul(b.fmul(x,b.const(bits(.5))),b.fadd(e,b.const(bits(1.0))))
    b.store_at(output,index,result,width='word');b.ret()
    return f


def dense_reference(source,weight,bias):
    x,w,b=map(np.asarray,(source,weight,bias))
    if x.ndim!=2 or w.ndim!=2 or x.shape[1]!=w.shape[1] or b.shape!=(w.shape[0],):
        raise ValueError('dense shapes disagree')
    if any(v.dtype!=np.float32 or not np.isfinite(v).all() for v in (x,w,b)):
        raise ValueError('dense reference requires finite FP32 arrays')
    return x.astype(np.float64)@w.astype(np.float64).T+b.astype(np.float64)


def gelu_reference(source):
    x=np.asarray(source,dtype=np.float64)
    if not np.isfinite(x).all():raise ValueError('GELU reference requires finite inputs')
    erf=np.fromiter((math.erf(float(v)/math.sqrt(2)) for v in x.flat),float,count=x.size).reshape(x.shape)
    return .5*x*(1+erf)


def deliver(destination):
    """Retain actual compiled projections and the precise activation refusal.

    This is a compiler handoff, not an admitted image or execution receipt.
    """
    import g17cc
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(status='incomplete', gpu_dispatched=False, pipeline_created=False,
                  rows=ROWS, stages={}, source_sha256={})
    for module in (__file__, g17cc.__file__):
        path = Path(module)
        report['source_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, widths in [('expand', (WIDTH, HIDDEN)), ('contract', (HIDDEN, WIDTH))]:
        program = g17cc.compile_function(dense_ir(*widths))
        code = bytes(program.code)
        (destination / (name + '.program.bin')).write_bytes(code)
        abi = program.abi_plain(program.abi())
        (destination / (name + '.abi.json')).write_text(json.dumps(abi, indent=2) + '\n')
        report['stages'][name] = dict(status='compiled_unvalidated',
            input_width=widths[0], output_width=widths[1], grid=[widths[1], ROWS, 1],
            code_bytes=len(code), code_sha256=hashlib.sha256(code).hexdigest())
    try:
        g17cc.compile_function(gelu_ir())
    except Exception as error:
        report['stages']['gelu'] = dict(status='refused', error_type=type(error).__name__,
                                       error=str(error))
    else:
        report['stages']['gelu'] = dict(status='compiled_requires_delivery_and_validation')
    report['scope'] = ('Projection compiler bytes and ABI only; source hashes are not a '
                       'complete build-input audit. GELU retains the erf definition. '
                       'No image, numerical execution, or complete FFN claim.')
    (destination / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(deliver(args.destination), indent=2))
