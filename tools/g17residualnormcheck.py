"""Check the delivered five-buffer residual LayerNorm bytes, entirely on CPU.

Reuses the generic delivered-instruction machine and asynchronous-load admission
check. It neither patches instruction semantics nor treats a sequential model
as hardware execution evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/"results/g17-attention-resident-32x384-v2"
BUDGET=2e-5


def simulate(decoded,bindings,source,residual,gamma,beta, *, executed=None, silicon=None):
    import g17normcheck as N
    arrays=[source,residual,gamma,beta,np.full(source.shape,np.nan,np.float32)]
    buffers={i:np.asarray(a,np.float32).reshape(-1).tolist() for i,a in enumerate(arrays)}
    writes=np.zeros(source.size,np.uint32)
    executed=N.executed_opcodes() if executed is None else frozenset(executed)
    silicon=N.silicon_agreement_opcodes() if silicon is None else frozenset(silicon)
    confidence=set();notes=set();loads=0
    for row in range(source.shape[0]):
        m=N.Machine(bindings,buffers,row,executed,silicon).run(decoded)
        loads+=len(m.loads)
        for rank,address,value in m.stores:
            if rank!=4:raise ValueError(f"delivered program writes read-only rank {rank}")
            if not 0<=address<source.size:raise ValueError("delivered output address out of range")
            buffers[4][address]=value;writes[address]+=1
        c,n=m.confidence();confidence.add(c);notes.update(n)
    if not np.all(writes==1):raise ValueError("every output must be written exactly once")
    output=np.asarray(buffers[4],np.float32).reshape(source.shape)
    if not np.isfinite(output).all():raise ValueError("nonfinite delivered output")
    return output,dict(loads=loads,stores=int(writes.sum()),confidence=sorted(confidence),notes=sorted(notes))


def check(delivery):
    import g17residualnorm as R
    import g17packedcheck
    import g17layernormimagecheck
    import g17normcheck as N
    delivery=Path(delivery)
    report=json.loads((delivery/"delivery.json").read_text())
    abi=json.loads((delivery/"abi.json").read_text())
    rows,columns=report["rows"],report["columns"]
    code=(delivery/"program.bin").read_bytes()
    if hashlib.sha256(code).hexdigest()!=report["code_sha256"]:raise ValueError("delivered code hash moved")
    program,current=R.compile_program(rows,columns)
    # Compare the serialized ABI: JSON object keys are strings on disk.
    # THE TWO SIDES ARE REPORTED SEPARATELY, at identical strictness. Conflated, this raised
    # "current compiler differs" both for changed native bytes and for an ABI that merely gained an
    # additive key, and the message could not tell a reviewer which had happened - the one
    # distinction that decides whether a retained delivery is stale or wrong. Neither branch is
    # weaker than the single condition it replaces; the ABI wording is unchanged because the
    # tampered-binding control tests for it.
    if program.code!=code:raise ValueError("delivered native bytes differ from the current compiler's")
    if json.loads(json.dumps(current))!=abi:raise ValueError("current compiler differs from the committed-source delivery")
    decoded=g17packedcheck.decode(code)
    emitted=[(i["offset"],i["length"],i["opcode"]) for i in program.contract().to_dict()["instructions"]]
    if emitted!=[(o,n,op) for o,n,op,_ in decoded]:raise ValueError("emitted and decoded instruction boundaries/forms differ")
    reuse=g17layernormimagecheck.check_load_reuse(decoded)
    bindings=[(b["index"],b["offset"],b["written"]) for b in abi["bindings"]]
    sizes=[rows*columns,rows*columns,columns,columns,rows*columns]
    addresses=N.check_operands(decoded,bindings,sizes,rows=rows)
    if addresses["stores"]!=rows*columns:raise ValueError("wrong decoded store coverage")
    controls=[]
    for rank in range(5):
        short=sizes.copy();short[rank]-=1
        memory={i:[0.]*size for i,size in enumerate(short)}
        try:
            N.Machine(bindings,memory,rows-1,frozenset(),frozenset()).run(decoded)
        except ValueError as error:
            controls.append(dict(rank=rank,error=str(error)))
        else:
            raise ValueError(f"undersized buffer at rank {rank} was accepted")
    initial=json.loads((BUNDLE/"full-initial/execution.json").read_text())
    path=BUNDLE/"full-initial"/initial["queries"][0]["artifact"]
    with np.load(path,allow_pickle=False) as f:
        x=f["embedding_norm"][:rows,:columns].copy()
        residual=f["projected"][:rows,:columns].copy()
    with np.load(BUNDLE/"fixture.npz",allow_pickle=False) as f:
        prefix="encoder.layer.0.attention.output.LayerNorm."
        gamma,beta=(f[prefix+s][:columns].copy() for s in ("weight","bias"))
    cases=[("real",x,residual,gamma,beta),
           ("changed_residual",x,np.roll(residual,1,axis=1),gamma,beta),
           ("cancel",x,-x,gamma,beta),
           ("changed_parameters",x,residual,-gamma,beta+np.float32(.25)),
           ("offset_pair",x+np.float32(1000),residual-np.float32(1000),gamma,beta),
           ("repeat_real",x,residual,gamma,beta)]
    reports=[];first=None
    for name,a,b,g,bias in cases:
        got,model=simulate(decoded,bindings,a,b,g,bias)
        expected=R.reference(a,b,g,bias)
        error=np.abs(got.astype(np.float64)-expected);limits=BUDGET*(1+np.abs(expected))
        failed=int(np.count_nonzero(error>limits))
        item=dict(case=name,outputs=int(got.size),failed_outputs=failed,
                  max_abs_error=float(error.max()),max_budget_fraction=float((error/limits).max()),model=model)
        reports.append(item)
        if failed:raise ValueError(f"delivered arithmetic failed: {name}: {item}")
        if name=="real":first=got.tobytes()
        if name=="repeat_real" and got.tobytes()!=first:raise ValueError("sequential interpretation not repeatable")
    return dict(status="interpreted",code_sha256=report["code_sha256"],rows=rows,columns=columns,
        instructions=len(decoded),bindings=bindings,addresses=addresses,load_reuse=reuse,
        undersized_controls=controls,cases=reports,gpu_dispatched=False,image_authored=False,
        error_budget="2e-5*(1+abs(FP64 LayerNorm of rounded FP32 residual sum))",
        limitations="Numerical agreement of delivered bytes under a sequential model. New image class, actual asynchronous execution and full hardware correctness remain unvalidated.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("delivery",type=Path);p.add_argument("--report",type=Path)
    a=p.parse_args()
    if a.report and a.report.exists():p.error("refusing to overwrite an interpretation")
    try:
        result=check(a.delivery)
        if a.report:a.report.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k!="cases"},indent=2))
    except Exception as error:p.exit(2,f"residual LayerNorm interpretation refused: {type(error).__name__}: {error}\n")
