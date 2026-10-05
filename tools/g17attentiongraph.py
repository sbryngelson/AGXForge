"""Application-owned resident graph; compiler ABI and allocation validation stay separate."""
import json


def graph():
    allocations={}
    def allocation(name,shape,role,tensor=None):
        words=1
        for n in shape:words*=n
        allocations[name]=dict(shape=list(shape),element_type="float",payload_bytes=words*4,
            offset=128,allocation_bytes=words*4+256,role=role)
        if tensor is not None:allocations[name]["tensor"]=tensor
    for name in ("source","embedding_norm","query","key","value","context","projected","residual","output"):
        allocation(name,(32,384),"input" if name=="source" else "output" if name=="output" else "intermediate")
    for name in ("scores","probabilities"):allocation(name,(12,32,32),"intermediate")
    stages=[]
    def stage(name,program,grid,buffers):
        stages.append(dict(name=name,program=program,grid=list(grid),bindings=[
            dict(index=i+1,allocation=allocation_name,offset=128,
                 length=allocations[allocation_name]["payload_bytes"],written=i==len(buffers)-1)
            for i,allocation_name in enumerate(buffers)]))
    def norm(name,source,output,prefix):
        for suffix in ("weight","bias"):
            allocation(name+"."+suffix,(384,),"parameter",prefix+"."+suffix)
        stage(name,"layernorm",(32,1,1),(source,name+".weight",name+".bias",output))
    def projection(name,source,output,prefix):
        allocation(name+".weight",(384,384),"parameter",prefix+".weight")
        allocation(name+".bias",(384,),"parameter",prefix+".bias")
        stage(name,"projection",(384,32,1),(source,name+".weight",name+".bias",output))
    norm("embedding_norm","source","embedding_norm","embeddings.LayerNorm")
    prefix="encoder.layer.0.attention."
    for name in ("query","key","value"):
        projection(name,"embedding_norm",name,prefix+"self."+name)
    stage("scores","scores",(32,384,1),("query","key","scores"))
    stage("softmax","softmax",(384,1,1),("scores","probabilities"))
    stage("context","context",(384,32,1),("probabilities","value","context"))
    projection("projected","context","projected",prefix+"output.dense")
    stage("residual","residual",(12288,1,1),("embedding_norm","projected","residual"))
    norm("output_norm","residual","output",prefix+"output.LayerNorm")
    return dict(format="g17-attention-graph-v1",status="proposed_not_executed",allocations=allocations,stages=stages,
        lifetime="all allocations retained until the session closes; no aliasing or in-place reuse",
        execution="sequential compute encoders in one command buffer; resource hazards tracked",
        uploads="source once per request; parameters once per session; intermediates never uploaded")


if __name__=="__main__":print(json.dumps(graph(),indent=2))
