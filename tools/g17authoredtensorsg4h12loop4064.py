#!/usr/bin/env python3
"""Predicted 40- and 64-trip twelve-group graphs from the verified 32-trip graph.

The 40-trip arm crosses one more B allocation boundary. The 64-trip
arm crosses four 32 KiB boundaries at once. Resource-correlate bytes
are extrapolated; their firmware meaning remains unknown.
"""
import hashlib
from pathlib import Path
import struct

import g17authoredtensorsg4h12loop3132 as parent
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
    40: dict(code='cc9fb05ca4fe9a640167a34ca8dc0a681502fa70ba1937e4ea1f1d519c956d39',
             b='af372a4057b27f2c3a83a2584d9d6d51b2a5437ca358be97bd054f482a2927c4',
             page='10aae65d9831ef1d6898e7bd4725a1a24ffc9c174527d9d75694ce66c3eadb8f',
             request='84b7a17c020cccd6ffd80acea15882e345325c522709ec83c56df0b7d68ca8ea',
             payload='4fbea256824ac0d7725280efdd45116a8349c27280e532cb645de006b0f6089a',
             output='17d7db7cccd052229a0f8e9c15d02e246c75bbd465ca1631d94b08d7c77f2da7'),
    64: dict(code='e4ce161d6be7f02f4d1f90a2e82f1672d9b25643fe9de6d462b56571df3a2bb5',
             b='a79f22d87ee2268260fce29981724832cdaa14629ff78062e6951f556182bb93',
             page='ec6f432c43ae217018a0ad6a9087880f4204d8142233c8f90de436c37b3c4d99',
             request='c7992ecd66bdb9d3eef6d31c9eeaea54f81e6b80569be32635c1f1023c6ea14f',
             payload='e7e053327e95b1a66a3352d5c7272b8b5d00524e9d367a99aa18e4d60206e574',
             output='ab876bf7579c40a68a79530aa09d49aeb570f791453c910ac2d70ffa1d74b7f6'),
}
INDICES = parent.INDICES
PAGE_POINTERS = (0x174,0x1cc,0x1d4,0x1e4,0x1ec,0x1f4,0x1fc,0x3ec)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fields(trips):
    try:
        return FIELDS[trips]
    except KeyError:
        raise ValueError('only precommitted 40 and 64 trip graphs are bounded here') from None


def shift(trips):
    fields(trips)
    return (trips-32)//8*0x8000


def code(trips):
    q=fields(trips)
    old=parent.code(32)
    new=bytes(cc.compile_function(g17tensoraccregs.loop_fn(sg=4,heads=2,trips=trips)).code)
    if len(new)!=len(old) or len(new)!=6074 or sha(new)!=q['code']:
        raise ValueError('counted-loop code changed')
    assert [i for i,(x,y) in enumerate(zip(old,new)) if x!=y]==[3975]
    assert asm.decode_cmp_imm(new[0xf86:0xf8a])=={'imm':trips,'rel':'lt','keep':True}
    tensorlife.counted_loop_check(new,trips,carried=tuple(cc._TENSOR_INDEX_USED))
    return new


def bstream(source,trips):
    q=fields(trips)
    result=parent.bstream(source,32)+source[:4096]*(trips-32)
    if len(result)!=trips*4096 or sha(result)!=q['b']:
        raise ValueError('long B stream changed')
    return result


def sizes(trips):
    delta=shift(trips)
    return tuple(0x28000+delta if index==20 else size
                 for index,size in zip(INDICES,parent.sizes(32)))


def pages(trips):
    q=fields(trips)
    delta=shift(trips)
    raw=bytearray(parent.pages(32))
    for offset in PAGE_POINTERS:
        struct.pack_into('<Q',raw,offset,struct.unpack_from('<Q',raw,offset)[0]+delta)
    result=bytes(raw)
    if sha(result)!=q['page']:
        raise ValueError('predicted pages changed')
    return result


def requests(trips):
    q=fields(trips)
    delta=shift(trips)
    source=parent.requests(32)
    rows=[list(request_base.ROW.unpack_from(source,index*request_base.ROW.size))
          for index in range(29)]
    rows[20][6]=0x28000+delta
    rows[20][7]=request_base.request(0x10001,0x470,0x28000+delta,base_buffer=True)
    for index in range(21,29):
        rows[index][5]+=delta
    result=b''.join(request_base.ROW.pack(*row) for row in rows)
    if sha(result)!=q['request']:
        raise ValueError('predicted requests changed')
    return result


def allocations(program,a,b,c,trips):
    q=fields(trips)
    delta=shift(trips)
    if program!=code(trips) or len(b)!=trips*4096 or sha(b)!=q['b']:
        raise ValueError('exact long-loop code and B required')
    old=parent.code(32)
    result=parent.allocations(old,a,b[:131072],c,32)
    raw=bytearray(result[0])
    raw[0x6c0+3975:0x6c0+3977]=program[3975:3977]
    result[0]=bytes(raw)
    raw=bytearray(0x28000+delta)
    raw[:0x80]=b'\xa5'*0x80
    raw[0x80:0x80+len(b)]=b
    raw[0x80+len(b):0x100+len(b)]=b'\xa5'*0x80
    result[20]=bytes(raw)
    raw=bytearray(result[23]);raw[6]=0xcc+4*delta//0x8000;result[23]=bytes(raw)
    raw=bytearray(result[25]);raw[9]=0x52+2*delta//0x8000;result[25]=bytes(raw)
    raw=bytearray(result[28]);struct.pack_into('<Q',raw,0x1bb0,0x100000d0080+delta);result[28]=bytes(raw)
    if result[0][0x6c0:0x6c0+len(program)]!=program:
        raise ValueError('long-loop code not staged')
    if any(len(result[index])!=size for index,size in zip(INDICES,sizes(trips))):
        raise ValueError('long-loop physical size changed')
    return result


def build(request_path,page_path,payload_path,program,a,b,c,trips):
    q=fields(trips)
    req,page,alloc=requests(trips),pages(trips),allocations(program,a,b,c,trips)
    payload=b''.join(alloc[index] for index in INDICES)
    if len(payload)!=(1425408 if trips==40 else 1523712) or sha(payload)!=q['payload']:
        raise ValueError('predicted payload changed')
    for path,raw in ((request_path,req),(page_path,page),(payload_path,payload)):
        Path(path).write_bytes(raw)
    return {'scope':f'field-built twelve-group {trips}-trip tensor graph',
            'capture_read_at_build':False,'program':f'sg4h12loop{trips}',
            'requests_sha256':sha(req),'pages_sha256':sha(page),
            'payload_sha256':sha(payload),'payload_bytes':len(payload),
            'physical_allocations':[{'index':index,'size':size,'sha256':sha(alloc[index])}
                                     for index,size in zip(INDICES,sizes(trips))]}
