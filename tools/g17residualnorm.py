"""Concrete five-buffer frontier program: residual add followed by LayerNorm.

The FP32 addition is rounded before normalization, matching the two-stage
attention graph. This constructs application IR only; it does not patch native
bytes or substitute compiler/metadata defaults.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def residualnorm_ir(rows=32, columns=384):
    import g17ir as ir
    import g17layernorm
    fn=g17layernorm.layernorm_ir(rows,columns)
    fn.name="minilm_residual_layernorm"
    source,gamma,beta,output=fn.buffers
    residual=ir.Buffer("residual",2,elem=ir.F32)
    gamma.slot,beta.slot,output.slot=3,4,5
    fn.buffers=[source,residual,gamma,beta,output]
    replacements=0
    for block in fn.blocks:
        ops=[]
        for op in block.ops:
            if op.kind=="load" and op.args[0] is source:
                left,right=ir.Value(op.dest.type),ir.Value(op.dest.type)
                ops += [ir.Op("load",left,[source,op.args[1]],**op.attrs),
                        ir.Op("load",right,[residual,op.args[1]],**op.attrs),
                        ir.Op("fadd",op.dest,[left,right])]
                replacements+=1
            else:
                ops.append(op)
        block.ops=ops
    if replacements!=columns:
        raise ValueError("LayerNorm input shape changed; residual fusion needs review")
    return fn


def reference(source,residual,gamma,beta):
    import numpy as np
    import g17layernorm
    if not isinstance(source,np.ndarray) or not isinstance(residual,np.ndarray) or \
       source.dtype!=np.float32 or residual.dtype!=np.float32 or source.shape!=residual.shape or \
       source.ndim!=2 or not np.isfinite(source).all() or not np.isfinite(residual).all():
        raise ValueError("residual inputs must be equally shaped finite FP32 matrices")
    with np.errstate(over="raise",invalid="raise"):
        summed=np.add(source,residual,dtype=np.float32)
    return g17layernorm.reference(summed,gamma,beta)


def compile_program(rows,columns):
    import g17cc
    program=g17cc.compile_function(residualnorm_ir(rows,columns))
    abi=program.abi_plain(program.abi())
    got=[(b["index"],b["offset"],b["written"],b["element_type"]) for b in abi["bindings"]]
    expected=[(i,2*(i-1),i==5,"float") for i in range(1,6)]
    if got!=expected:raise ValueError(f"five-buffer compiler contract differs: {got}")
    return program,abi


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows",type=int,default=32);parser.add_argument("--columns",type=int,default=384)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--verify-source",action="store_true")
    args=parser.parse_args()
    if args.out.exists():parser.error("refusing to overwrite a delivery")
    source=None
    try:
        if args.verify_source:
            import g17buildaudit
            (program,abi),source=g17buildaudit.verified_build(ROOT,lambda:compile_program(args.rows,args.columns))
        else:
            program,abi=compile_program(args.rows,args.columns)
        args.out.mkdir(parents=True)
        (args.out/"program.bin").write_bytes(program.code)
        (args.out/"abi.json").write_text(json.dumps(abi,indent=2)+"\n")
        report=dict(status="compiled_unvalidated",rows=args.rows,columns=args.columns,
            code_sha256=hashlib.sha256(program.code).hexdigest(),code_bytes=len(program.code),
            instruction_count=len(program.layout),source=source,gpu_dispatched=False,
            image_authored=False,bindings=abi["bindings"])
        (args.out/"delivery.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({k:v for k,v in report.items() if k!="source"},indent=2))
    except Exception as error:
        parser.exit(2,f"residual LayerNorm refused: {type(error).__name__}: {error}\n")


if __name__=="__main__":main()
