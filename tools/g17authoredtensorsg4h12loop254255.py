#!/usr/bin/env python3
"""Predicted 254/255-trip twelve-group tensor graphs at the 8-bit bound.

Both variants use the same 1 MiB physical B allocation, command pages,
and selector requests. Distinctive final FP16 B blocks make the far
block reads observable despite the accumulator's fixed point.
"""
import hashlib
from pathlib import Path
import struct

import g17authoredtensorsg4h12loop4064 as parent
import g17structuredrequests as request_base
import g17tensoraccregs

# Make agxforge.g17 importable when this tool runs from outside the checkout.
import sys as _g17_sys
from pathlib import Path as _G17Path
_g17_root = str(_G17Path(__file__).resolve().parents[1])
if _g17_root not in _g17_sys.path:
    _g17_sys.path.insert(0, _g17_root)
from agxforge.g17 import asm, cc, tensorlife


FIELDS={
    254:dict(code='ee497e17d0f0374b1fbf79affb6eb94f11ecee2f6ab1fe16b7adda693a822cc6',
             b='e141a5ec176bc736511b3260b3ea033f7d30710f069a2539677a0e472572e34f',
             payload='7ccfbda98243f746436ac6a40c12d991a8701bf071a7291449bb4b8f6cf0ae96',
             output='65ee28aae3b8f72d0e4d00256e7a6a9d5c01b441ba1c7f519605c52089aeb914'),
    255:dict(code='29487df34d2092d0e7be54accd15baf333a5218cb03e6e71c953024621e08a50',
             b='c60c06956bff924ec605e2c43578a6945757955c6799018dcb1f49eaeba4a021',
             payload='c407b4b06c434ff2ffc04eb0d54e469cd4ae29dd015009370bfeef815bc62bf0',
             output='e30271c3b1cd7b2199e51e6b8167dc70cfd59ff8d52f100f9d65ec652793e517'),
}
PAGE_SHA256='7f333e3d3774d721c51d31005fedcf238a1f8e65d191301f5934eb49eb3cc8fb'
REQUEST_SHA256='fc307775a51b8e10e428451202fd2432df47284158ed5c75c5554b3d4cc565dc'
INDICES=parent.INDICES
PAGE_POINTERS=parent.PAGE_POINTERS
DELTA=0x100000-0x48000
B_SIZE=0x100000
C_BINDING=0x100001a8080


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fields(trips):
    try:
        return FIELDS[trips]
    except KeyError:
        raise ValueError('only precommitted 254 and 255 trip graphs are bounded here') from None


def code(trips):
    q=fields(trips)
    old=parent.code(64)
    new=bytes(cc.compile_function(g17tensoraccregs.loop_fn(sg=4,heads=2,trips=trips)).code)
    if len(new)!=len(old) or len(new)!=6074 or sha(new)!=q['code']:
        raise ValueError('counted-loop code changed')
    diffs=[i for i,(x,y) in enumerate(zip(old,new)) if x!=y]
    assert diffs==([3974,3975] if trips==254 else [3974,3975,3976])
    assert asm.decode_cmp_imm(new[0xf86:0xf8a])=={'imm':trips,'rel':'lt','keep':True}
    tensorlife.counted_loop_check(new,trips,carried=tuple(cc._TENSOR_INDEX_USED))
    return new


def marker(source,trips):
    fields(trips)
    factor=2 if trips==254 else -2
    return b''.join(struct.pack('<e',factor*struct.unpack_from('<e',source,i)[0])
                    for i in range(0,4096,2))


def bstream(source,trips):
    q=fields(trips)
    result=parent.bstream(source,64)+source[:4096]*(trips-65)+marker(source,trips)
    if len(result)!=trips*4096 or sha(result)!=q['b']:
        raise ValueError('marked B stream changed')
    return result


def sizes(trips):
    fields(trips)
    return tuple(B_SIZE if index==20 else size
                 for index,size in zip(INDICES,parent.sizes(64)))


def pages(trips):
    fields(trips)
    raw=bytearray(parent.pages(64))
    for offset in PAGE_POINTERS:
        struct.pack_into('<Q',raw,offset,struct.unpack_from('<Q',raw,offset)[0]+DELTA)
    result=bytes(raw)
    if sha(result)!=PAGE_SHA256:
        raise ValueError('predicted pages changed')
    return result


def requests(trips):
    fields(trips)
    source=parent.requests(64)
    rows=[list(request_base.ROW.unpack_from(source,index*request_base.ROW.size))
          for index in range(29)]
    rows[20][6]=B_SIZE
    rows[20][7]=request_base.request(0x10001,0x470,B_SIZE,base_buffer=True)
    for index in range(21,29):
        rows[index][5]+=DELTA
    result=b''.join(request_base.ROW.pack(*row) for row in rows)
    if sha(result)!=REQUEST_SHA256:
        raise ValueError('predicted requests changed')
    return result


def allocations(program,a,b,c,trips):
    q=fields(trips)
    if program!=code(trips) or len(b)!=trips*4096 or sha(b)!=q['b']:
        raise ValueError('exact marked-loop code and B required')
    result=parent.allocations(parent.code(64),a,b[:262144],c,64)
    raw=bytearray(result[0])
    raw[0x6c0+3974:0x6c0+3977]=program[3974:3977]
    result[0]=bytes(raw)
    raw=bytearray(B_SIZE)
    raw[:0x80]=b'\xa5'*0x80
    raw[0x80:0x80+len(b)]=b
    raw[0x80+len(b):0x100+len(b)]=b'\xa5'*0x80
    result[20]=bytes(raw)
    raw=bytearray(result[23]);struct.pack_into('<H',raw,6,0x138);result[23]=bytes(raw)
    raw=bytearray(result[25]);raw[9]=0x88;result[25]=bytes(raw)
    raw=bytearray(result[28]);struct.pack_into('<Q',raw,0x1bb0,C_BINDING);result[28]=bytes(raw)
    if result[0][0x6c0:0x6c0+len(program)]!=program:
        raise ValueError('marked-loop code not staged')
    if any(len(result[index])!=size for index,size in zip(INDICES,sizes(trips))):
        raise ValueError('marked-loop physical size changed')
    return result


def build(request_path,page_path,payload_path,program,a,b,c,trips):
    q=fields(trips)
    req,page,alloc=requests(trips),pages(trips),allocations(program,a,b,c,trips)
    payload=b''.join(alloc[index] for index in INDICES)
    if len(payload)!=2277376 or sha(payload)!=q['payload']:
        raise ValueError('predicted payload changed')
    for path,raw in ((request_path,req),(page_path,page),(payload_path,payload)):
        Path(path).write_bytes(raw)
    return {'scope':f'field-built twelve-group {trips}-trip marked tensor graph',
            'capture_read_at_build':False,'program':f'sg4h12loop{trips}',
            'requests_sha256':sha(req),'pages_sha256':sha(page),
            'payload_sha256':sha(payload),'payload_bytes':len(payload),
            'physical_allocations':[{'index':index,'size':size,'sha256':sha(alloc[index])}
                                     for index,size in zip(INDICES,sizes(trips))]}
