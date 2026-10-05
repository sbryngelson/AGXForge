#!/usr/bin/env python3
"""Build CPU staging payloads from the A capture and compiler-authored code.

The binary is eleven fixed-size allocations in G17_LAYOUT_REQUESTS order. It
contains exact authored machine code but is not itself a firmware registration
or submission. The capture retains nonzero 4 KiB pages and omits verified zero
pages, which this generator reconstructs as zero. The new arithmetic and
40-byte bitwise variants are compiled locally; their full payloads were never
loaded through Metal.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import g17layoutrequests as L
import g17structuredrequests
import g17structuredgraphpayload
import g17structuredbasepayload
import g17cc as cc
import g17ir as ir
import g17packedcheck

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "evidence/g17-belowmetal-preflight/authored-pair-metal/passive-submit-with-requests.json"
SUPPORTED_VARIANTS=(('add',3,1),('add',3,2),('add',3,3),('add',3,7),('add',5,7),
                    ('sub',3,7),('xor',3,7),('and',3,7),('or',3,7),
                    ('input-add7',0,7),('input-sum',0,0),('input-muladd',0,7))


def authored_code(bias,scale=3,op='add'):
    if (op,scale,bias) not in SUPPORTED_VARIANTS:raise ValueError('unsupported authored variant')
    fn=ir.Function('pt',[ir.Buffer('A',0),ir.Buffer('B',1),ir.Buffer('C',2)])
    b=ir.Builder(fn,fn.block('e'))
    t=b.builtin('thread_position_in_grid',name='t')
    if op=='input-muladd':
        lhs=b.load(fn.buffers[0],t,name='lhs')
        rhs=b.load(fn.buffers[1],t,name='rhs')
        result=b.add(b.mul(lhs,rhs,name='product'),ir.Imm(7),name='v')
    elif op=='input-sum':
        lhs=b.load(fn.buffers[0],t,name='lhs')
        rhs=b.load(fn.buffers[1],t,name='rhs')
        result=b.add(lhs,rhs,name='v')
    elif op=='input-add7':
        lhs=b.load(fn.buffers[0],t,name='lhs')
        result=b.add(lhs,ir.Imm(7),name='v')
    else:
        result=getattr(b,op)(b.mul(t,ir.Imm(scale),name='m'),ir.Imm(bias),name='v')
    b.store_at(fn.buffers[2],t,result)
    b.ret();ir.verify(fn)
    code=bytes(cc.compile_function(fn,regs=range(0,24)).code)
    decoded=g17packedcheck.decode(code)
    if op=='input-muladd':
        if len(code)!=70 or [(x[0],x[1],x[2]) for x in decoded]!=[
                (0,4,14059),(4,14,12682),(18,14,12682),(32,14,10825),
                (46,12,10279),(58,8,17229),(66,4,684)]:
            raise ValueError('input-muladd changed the expected decoded instruction forms')
        return code
    if op=='input-sum':
        if len(code)!=56 or [(x[0],x[1],x[2]) for x in decoded]!=[
                (0,4,14059),(4,14,12682),(18,14,12682),(32,12,10282),
                (44,8,17229),(52,4,684)]:
            raise ValueError('input-sum changed the expected decoded instruction forms')
        return code
    if op=='input-add7':
        if len(code)!=42 or [(x[0],x[1],x[2]) for x in decoded]!=[
                (0,4,14059),(4,14,12682),(18,12,10279),
                (30,8,17229),(38,4,684)]:
            raise ValueError('input-add7 changed the expected decoded instruction forms')
        return code
    expected={'add':(42,12,10279,30,38),'sub':(42,12,11666,30,38),
              'xor':(40,10,17770,28,36),'and':(40,10,423,28,36),
              'or':(40,10,13574,28,36)}[op]
    size,arithmetic_bytes,opcode,store_offset,end_offset=expected
    if len(code)!=size or [(x[0],x[1],x[2]) for x in decoded]!=[
            (0,4,14059),(4,14,10822),(18,arithmetic_bytes,opcode),
            (store_offset,8,17229),(end_offset,4,684)]:
        raise ValueError('authored variant changed the expected instruction form')
    return code


def event(run):
    found = [m["payload"] for m in run["messages"]
             if m.get("payload", {}).get("kind") == "submit_enter"]
    if not run["passed"] or len(found) != 1:
        raise ValueError("capture did not complete exactly one passing Submit")
    return found[0]


def allocation(e, aperture, size):
    found = [w for w in e["windows"] if int(w["aperture"]) == aperture and w["size"] == size]
    if len(found) != 1:
        raise ValueError(f"missing mapped allocation {aperture:#x}, size {size:#x}")
    b = bytearray(size)
    for page in found[0]["pages"]:
        data = bytes.fromhex(page["hex"])
        if page["offset"] < 0 or page["offset"] + len(data) > size:
            raise ValueError("page exceeds allocation")
        b[page["offset"]:page["offset"] + len(data)] = data
    return bytes(b)


def build(path,bias=1,scale=3,op='add',placement='captured',structured_graph=False,structured_base=False):
    if structured_base and not structured_graph:
        raise ValueError('structured base construction requires structured_graph')
    if structured_graph and (bias,scale,op,placement)!=(1,3,'add','captured'):
        raise ValueError('structured graph construction supports only the baseline payload')
    if placement not in ('captured','tail','tail-control','second-allocation',
                         'second-allocation-candidate','second-allocation-control',
                         'third-allocation-candidate','third-allocation-control',
                         'alloc24-candidate','alloc24-control',
                         'input-add7-alloc1','input-sum-alloc1','input-muladd-alloc1',
                         'input-muladd-then-add7'):
        raise ValueError('unsupported code placement')
    if placement=='tail' and (op,scale,bias)!=('add',3,2):
        raise ValueError('tail placement is bounded to the authored B control')
    if placement=='tail-control' and (op,scale,bias)!=('add',3,1):
        raise ValueError('tail-control is bounded to active A and staged B')
    if placement.startswith('second-allocation') and (op,scale,bias)!=('add',3,2):
        if placement!='second-allocation-control' or (op,scale,bias)!=('add',3,1):
            raise ValueError('second-allocation placements are bounded to authored A/B')
    if placement=='third-allocation-candidate' and (op,scale,bias)!=('add',3,2):
        raise ValueError('third-allocation candidate is bounded to authored B')
    if placement=='third-allocation-control' and (op,scale,bias)!=('add',3,1):
        raise ValueError('third-allocation control is bounded to active A and staged B')
    if placement=='alloc24-candidate' and (op,scale,bias)!=('add',3,2):
        raise ValueError('allocation-24 candidate is bounded to authored B')
    if placement=='alloc24-control' and (op,scale,bias)!=('add',3,1):
        raise ValueError('allocation-24 control is bounded to active A and staged B')
    if placement=='input-sum-alloc1' and (op,scale,bias)!=('input-sum',0,0):
        raise ValueError('input-sum allocation-1 placement requires the exact load/add/store program')
    if placement=='input-add7-alloc1' and (op,scale,bias)!=('input-add7',0,7):
        raise ValueError('input-add7 allocation-1 placement requires the exact load/add/store program')
    if placement=='input-muladd-alloc1' and (op,scale,bias)!=('input-muladd',0,7):
        raise ValueError('input-muladd allocation-1 placement requires the exact load/mul/add/store program')
    if placement=='input-muladd-then-add7' and (op,scale,bias)!=('input-muladd',0,7):
        raise ValueError('two-program placement requires exact input-muladd first program')
    if op=='input-sum' and placement!='input-sum-alloc1':
        raise ValueError('input-sum requires the verified allocation-1 code slot')
    if op=='input-add7' and placement!='input-add7-alloc1':
        raise ValueError('input-add7 requires the verified allocation-1 code slot')
    if op=='input-muladd' and placement not in ('input-muladd-alloc1','input-muladd-then-add7'):
        raise ValueError('input-muladd requires the verified allocation-1 code slot')
    code_bias=2 if placement in ('tail-control','second-allocation-control',
                                 'third-allocation-control','alloc24-control') else bias
    code=authored_code(code_bias,scale,op)
    second_code=authored_code(7,0,'input-add7') if placement=='input-muladd-then-add7' else None
    staged_code=code.ljust(42,b'\0')
    raw = CAPTURE.read_bytes()
    runs = json.loads(raw)["runs"]
    if [r["bias"] for r in runs] != [1, 2]:
        raise ValueError("capture is not A/B")
    a, b = [event(r) for r in runs]
    rows = g17structuredrequests.physical_requests() if structured_graph else L.requests()
    payload = bytearray()
    manifest = []
    code_differences=None
    for index, request, aperture, size in rows:
        if structured_base and index in (1,2):
            av=bv=g17structuredbasepayload.allocation(index)
        elif structured_graph and index in g17structuredgraphpayload.INDICES:
            av=bv=g17structuredgraphpayload.allocation(index)
        else:
            av = allocation(a, aperture, size)
            bv = allocation(b, aperture, size)
        diff = [i for i,(x,y) in enumerate(zip(av,bv)) if x!=y]
        if diff != ([0x6D3] if index == 0 else []):
            raise ValueError(f"physical allocation {index} has unexpected A/B differences: {diff[:16]}")
        if index == 0 and (av[0x6D3], bv[0x6D3]) != (1,2):
            raise ValueError("authored immediate missing")
        chosen=bytearray(bv if placement=='captured' and (op,scale,bias)==('add',3,2) else av)
        if index==0:
            if code[:]!=av[0x6c0:0x6c0+len(code)] and (op,scale,code_bias)==('add',3,1):
                raise ValueError('compiler A differs from captured A')
            if code[:]!=bv[0x6c0:0x6c0+len(code)] and (op,scale,code_bias)==('add',3,2):
                raise ValueError('compiler B differs from captured B')
            visible_code=second_code if second_code is not None else staged_code[:42]
            code_differences=[i for i,(x,y) in enumerate(zip(av[0x6c0:0x6c0+42],visible_code)) if x!=y]
            expected_differences={('add',3,1):[],('add',3,2):[19],('add',3,3):[19],
                                  ('add',3,7):[19,21],('add',5,7):[12,13,19,21],
                                  ('sub',3,7):[19,23,24,26,27,28],
                                  ('xor',3,7):[18,19,20,21,22,23,26,27,28,29,30,31,32,33,34,35,36,37,38],
                                  ('and',3,7):[18,19,20,21,22,23,26,27,28,29,30,31,32,33,34,35,36,37,38],
                                  ('or',3,7):[18,19,20,21,22,23,26,27,28,29,30,31,32,33,34,35,36,37,38]}
            if op not in ('input-sum','input-add7','input-muladd') and code_differences!=expected_differences[(op,scale,code_bias)]:
                raise ValueError('unexpected authored code difference')
            code_offset=0xe500 if placement in ('tail','tail-control') else 0x6c0
            if placement in ('tail','tail-control') and av[code_offset:code_offset+42]!=bytes(42):
                raise ValueError('tail code slot is not zero in captured A')
            if not (placement.startswith('second-allocation') or
                    placement.startswith('third-allocation') or
                    placement.startswith('alloc24-') or
                    placement in ('input-sum-alloc1','input-add7-alloc1','input-muladd-alloc1',
                                  'input-muladd-then-add7')):
                chosen[code_offset:code_offset+42]=staged_code
            if second_code is not None:
                chosen[0x6c0:0x6c0+42]=second_code
        if index==1 and placement.startswith('second-allocation'):
            if av[0x500:0x500+42]!=bytes(42):
                raise ValueError('second allocation code slot is not zero in captured A')
            chosen[0x500:0x500+42]=staged_code
        if index==1 and placement in ('input-sum-alloc1','input-add7-alloc1','input-muladd-alloc1',
                                       'input-muladd-then-add7'):
            if av[0x500:0x500+len(code)]!=bytes(len(code)):
                raise ValueError('input-sum code slot is not zero in captured A')
            chosen[0x500:0x500+len(code)]=code
        if index==20 and placement.startswith('third-allocation'):
            if av[0x500:0x500+42]!=bytes(42):
                raise ValueError('third allocation code slot is not zero in captured A')
            chosen[0x500:0x500+42]=staged_code
        if index==24 and placement.startswith('alloc24-'):
            if av[0x500:0x500+42]!=bytes(42):
                raise ValueError('allocation-24 code slot is not zero in captured A')
            chosen[0x500:0x500+42]=staged_code
        if index==23 and placement=='tail':
            if struct.unpack_from('<I',chosen,0x40)[0]!=0x0e4006c7 or \
               struct.unpack_from('<H',chosen,0x48)[0]!=0x0100:
                raise ValueError('captured packet address changed')
            struct.pack_into('<I',chosen,0x40,0x0e40e507)
        if index==23 and placement=='second-allocation':
            if struct.unpack_from('<I',chosen,0x40)[0]!=0x0e4006c7 or \
               struct.unpack_from('<H',chosen,0x48)[0]!=0x0100:
                raise ValueError('captured packet address changed')
            struct.pack_into('<I',chosen,0x40,0x0e418507)
        if index==23 and placement=='second-allocation-candidate':
            if struct.unpack_from('<I',chosen,0x40)[0]!=0x0e4006c7 or \
               struct.unpack_from('<H',chosen,0x46)[0]!=0 or \
               struct.unpack_from('<H',chosen,0x48)[0]!=0x0100:
                raise ValueError('captured packet address changed')
            struct.pack_into('<H',chosen,0x40,0x8507)
            struct.pack_into('<H',chosen,0x46,0x0001)
        if index==23 and placement in ('input-sum-alloc1','input-add7-alloc1','input-muladd-alloc1',
                                        'input-muladd-then-add7'):
            if struct.unpack_from('<I',chosen,0x40)[0]!=0x0e4006c7 or \
               struct.unpack_from('<H',chosen,0x46)[0]!=0 or \
               struct.unpack_from('<H',chosen,0x48)[0]!=0x0100:
                raise ValueError('captured packet address changed')
            struct.pack_into('<H',chosen,0x40,0x8507)
            struct.pack_into('<H',chosen,0x46,0x0001)
        if index==23 and placement=='third-allocation-candidate':
            if struct.unpack_from('<I',chosen,0x40)[0]!=0x0e4006c7 or \
               struct.unpack_from('<H',chosen,0x46)[0]!=0 or \
               struct.unpack_from('<H',chosen,0x48)[0]!=0x0100:
                raise ValueError('captured packet address changed')
            struct.pack_into('<H',chosen,0x40,0x8507)
            struct.pack_into('<H',chosen,0x46,0x0005)
        if index==23 and placement=='alloc24-candidate':
            if struct.unpack_from('<I',chosen,0x40)[0]!=0x0e4006c7 or \
               struct.unpack_from('<H',chosen,0x46)[0]!=0 or \
               struct.unpack_from('<H',chosen,0x48)[0]!=0x0100:
                raise ValueError('captured packet address changed')
            struct.pack_into('<H',chosen,0x40,0x0507)
            struct.pack_into('<H',chosen,0x46,0x000a)
        payload += chosen
        manifest.append(dict(original_index=index,aperture=f"0x{aperture:x}",size=size,
                             sha256=hashlib.sha256(chosen).hexdigest(),ab_differences=diff))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    result=dict(scope=f"CPU staging image from A/B captures with compiler-authored {scale}*i {op} {bias} at {placement} placement; not execution",
                op=op,scale=scale,bias=bias,code_bias=code_bias,placement=placement,
                code_sha256=hashlib.sha256(code).hexdigest(),
                code_hex=code.hex(),
                second_code_hex=second_code.hex() if second_code is not None else None,
                code_bytes=len(code),
                staged_code_hex=staged_code.hex(),
                code_differences_from_a=code_differences,
                reconstruction="Captured nonzero 4 KiB pages copied verbatim; scanned zero pages reconstructed as zero",
                capture_sha256=hashlib.sha256(raw).hexdigest(),
                payload_sha256=hashlib.sha256(payload).hexdigest(),
                payload_bytes=len(payload),allocations=manifest)
    if structured_graph:
        result['reconstruction']='Base allocations 0/1/2 from capture; allocations 20 and 22..28 constructed from explicit fields'
        result['structured_allocation_indices']=list(g17structuredgraphpayload.INDICES)
        result['capture_allocation_indices']=[0,1,2]
    if structured_base:
        result['reconstruction']='Allocation 0 from capture; all other physical allocations constructed from explicit fields and arithmetic tables'
        result['structured_allocation_indices']=[1,2]+list(g17structuredgraphpayload.INDICES)
        result['capture_allocation_indices']=[0]
    (path.parent / (path.name + ".json")).write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    result=build(args.output)
    print("bytes",result["payload_bytes"],"sha256",result["payload_sha256"])
