#!/usr/bin/env python3
"""Exact 31-trip and predicted 32-trip twelve-group tensor graphs.

31 trips fill the 0x20000 B allocation up to its guarded tail.
32 trips grow it to 0x28000 and shift C and seven later resources.
"""
import hashlib
from pathlib import Path
import struct

import g17authoredtensorsg4h12loop24 as parent
import g17structuredrequests as request_base
import g17tensoraccregs

# Make agxforge.g17 importable when this tool runs from outside the checkout.
import sys as _g17_sys
from pathlib import Path as _G17Path
_g17_root = str(_G17Path(__file__).resolve().parents[1])
if _g17_root not in _g17_sys.path:
    _g17_sys.path.insert(0, _g17_root)
from agxforge.g17 import asm, cc, tensorlife


FIELDS = {
    31: dict(code='a1bb17eabb26140bd0b1998ad8944581fc59e5818472e9a3504a055682e0a597',
             b='ef0149763fc8da5f2e10cbfbc73e51cef45b7bfcde4317e4572d951942383f9b',
             output='ddc8fa5446686455b375b8f93df8849641821d285553afb85b0e834beb7ac0cc',
             page=parent.PAGE_SHA256,request=parent.REQUEST_SHA256,
             payload='f855d5e85921f044f56f1efd1b11614de47b00e423327a9412f1f30d7be45f42'),
    32: dict(code='1d93375800aa9272f98ae5851b0ab7891da5dd1446827e45d3bef456a0fa7ae4',
             b='0185b42c919c7d05573447afa87b7c9ee9fa90537d01a7a212927ff903b63033',
             output='1a132b14e17c752d8fe4d500515cf4900c66ac113aac268dcb7aa6e5a11db1e4',
             page='2acad2f9e12ea0f1a981276cb16ae65b22f4e87d5b80f57e75606909d0bf1e12',
             request='d9b48bc488ec7cefe1396e5bde6e8ffa3ec8365e8d9089a5a5e4b1d755ceb652',
             payload='e5de7047ea2947df15fb1530baf2e0dcd08da775a41bb4c77910eb071ec53dee'),
}
INDICES = parent.INDICES


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fields(trips):
    try:
        return FIELDS[trips]
    except KeyError:
        raise ValueError('only 31 and 32 trips are bounded here') from None


def code(trips):
    q=fields(trips)
    old=parent.code24()
    new=bytes(cc.compile_function(g17tensoraccregs.loop_fn(sg=4,heads=2,trips=trips)).code)
    if len(new)!=len(old) or sha(new)!=q['code']:
        raise ValueError('counted-loop code changed')
    diffs=[(i,x,y) for i,(x,y) in enumerate(zip(old,new)) if x!=y]
    want=([(3975,0x98,0x9e),(3976,0x05,0x15)] if trips==31 else [(3975,0x98,0xa0)])
    assert diffs==want
    assert asm.decode_cmp_imm(new[0xf86:0xf8a])=={'imm':trips,'rel':'lt','keep':True}
    tensorlife.counted_loop_check(new,trips,carried=tuple(cc._TENSOR_INDEX_USED))
    return new


def bstream(source,trips):
    q=fields(trips)
    result=parent.b24(source)+source[:4096]*(trips-24)
    if len(result)!=trips*4096 or sha(result)!=q['b']:
        raise ValueError('long B stream changed')
    return result


def sizes(trips):
    fields(trips)
    return tuple((0x20000 if trips==31 else 0x28000) if index==20 else size
                 for index,size in zip(INDICES,parent.SIZES))


def pages(trips):
    q=fields(trips)
    raw=bytearray(parent.pages())
    if trips==32:
        for offset in (0x174,0x1cc,0x1d4,0x1e4,0x1ec,0x1f4,0x1fc,0x3ec):
            struct.pack_into('<Q',raw,offset,struct.unpack_from('<Q',raw,offset)[0]+0x8000)
    result=bytes(raw)
    if sha(result)!=q['page']:
        raise ValueError('long-loop pages changed')
    return result


def requests(trips):
    q=fields(trips)
    if trips==31:
        result=parent.requests()
    else:
        source=parent.requests()
        rows=[list(request_base.ROW.unpack_from(source,index*request_base.ROW.size))
              for index in range(29)]
        rows[20][6]=0x28000
        rows[20][7]=request_base.request(0x10001,0x470,0x28000,base_buffer=True)
        for index in range(21,29):
            rows[index][5]+=0x8000
        result=b''.join(request_base.ROW.pack(*row) for row in rows)
    if sha(result)!=q['request']:
        raise ValueError('long-loop requests changed')
    return result


def allocations(program,a,b,c,trips):
    q=fields(trips)
    if program!=code(trips) or len(b)!=trips*4096 or sha(b)!=q['b']:
        raise ValueError('exact long-loop code and B required')
    result=parent.allocations(parent.code24(),a,b[:98304],c)
    raw=bytearray(result[0])
    raw[0x6c0+3975:0x6c0+3977]=program[3975:3977]
    result[0]=bytes(raw)
    raw=bytearray(0x20000 if trips==31 else 0x28000)
    raw[:0x80]=b'\xa5'*0x80
    raw[0x80:0x80+len(b)]=b
    raw[0x80+len(b):0x100+len(b)]=b'\xa5'*0x80
    result[20]=bytes(raw)
    if trips==32:
        raw=bytearray(result[23]);raw[6]=0xcc;result[23]=bytes(raw)
        raw=bytearray(result[25]);raw[9]=0x52;result[25]=bytes(raw)
        raw=bytearray(result[28]);struct.pack_into('<Q',raw,0x1bb0,0x100000d0080);result[28]=bytes(raw)
    if result[0][0x6c0:0x6c0+len(program)]!=program:
        raise ValueError('long-loop code not staged')
    if any(len(result[index])!=size for index,size in zip(INDICES,sizes(trips))):
        raise ValueError('long-loop physical size changed')
    return result


def build(request_path,page_path,payload_path,program,a,b,c,trips):
    q=fields(trips)
    req,page,alloc=requests(trips),pages(trips),allocations(program,a,b,c,trips)
    payload=b''.join(alloc[index] for index in INDICES)
    if len(payload)!=(1359872 if trips==31 else 1392640) or sha(payload)!=q['payload']:
        raise ValueError('long-loop payload changed')
    for path,raw in ((request_path,req),(page_path,page),(payload_path,payload)):
        Path(path).write_bytes(raw)
    return {'scope':f'field-built twelve-group {trips}-trip tensor graph',
            'capture_read_at_build':False,'program':f'sg4h12loop{trips}',
            'requests_sha256':sha(req),'pages_sha256':sha(page),
            'payload_sha256':sha(payload),'payload_bytes':len(payload),
            'physical_allocations':[{'index':index,'size':size,'sha256':sha(alloc[index])}
                                     for index,size in zip(INDICES,sizes(trips))]}
