#!/usr/bin/env python3
"""One 255-trip counted loop with sixteen authored tensor bodies per trip.

The tensor index advances once for each body, addressing 4080 B blocks.
The final block is marked so its execution remains visible after the
repeated-positive stream reaches its FP32 fixed point.
"""
import hashlib
from pathlib import Path
import struct

import g17authoredtensorsg4h12loop2040oct as parent
import g17structuredrequests as request_base
import g17tensoraccregs as tensor

# Make agxforge.g17 importable when this tool runs from outside the checkout.
import sys as _g17_sys
from pathlib import Path as _G17Path
_g17_root = str(_G17Path(__file__).resolve().parents[1])
if _g17_root not in _g17_sys.path:
    _g17_sys.path.insert(0, _g17_root)
from agxforge.g17 import cc, ir, tensorlife


CODE_SHA256='477ed1350f6644783f9ff158118bd35821fdcf36ea5eb2fe71bb61cef132cc00'
B_SHA256='f17f9ebed3b9e3621aa1aa1668ddaba4150f78b8d40ca03f7303f6f1205b6ee1'
PAGE_SHA256='b06206c0af303dcc6602cd2b8a5dcdc9b44bde97b53e4acedf6ec524fd4ed9d8'
REQUEST_SHA256='31396d83394e3d09bc7305d819c8ce3538442b5b9cc4784893bc65a87b0c0b36'
PAYLOAD_SHA256='82a1f0c6cf0d4b2fac5f3bca80aea9911ac77853bbcd5657ab09f568b528821e'
OUTPUT_SHA256='eba184eac2c6ce9fcd69991612e5528160aca963a04f1dc52aa4a1510a8c66db'
INDICES=parent.INDICES
PAGE_POINTERS=parent.PAGE_POINTERS
B_SIZE=0x1100000
SHIFT=B_SIZE-parent.B_SIZE
C_BINDING=parent.C_BINDING+SHIFT


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def function():
    a=ir.Buffer('A',1,elem=ir.F16)
    b=ir.Buffer('B',2,elem=ir.F16)
    c=ir.Buffer('C',3,elem=ir.F32)
    fn=ir.Function(tensor.T.GENERIC_NAME,[a,b,c])
    bd=ir.Builder(fn,fn.block('entry'))
    zero=tensor._f32_const(bd,0.0,'zero')
    tiles=tensor.LOOP_N//16
    for tile in range(tiles):
        for slot in range(8):
            bd.tensor_acc_write('O',slot,zero,tile=(0,tile))
    bd.tensor_index_init('k',0)
    i0=bd.const(0,name='i_init')
    hdr,ex=fn.block('trips'),fn.block('done')
    bd.br(hdr)
    bd.at(hdr)
    i=bd.phi(i0,name='i')
    for body in range(16):
        half=tensor._f32_const(bd,0.5,f'scale{body}')
        for tile in range(tiles):
            for slot in range(8):
                bd.tensor_acc_write('O',slot,
                    bd.fmul(bd.tensor_acc_read('O',slot,tile=(0,tile),name=f'o{body}_{tile}_{slot}'),
                            half,name=f'h{body}_{tile}_{slot}'),tile=(0,tile))
        bd.tensor_matmul(a,b,c,M=64,N=tensor.LOOP_N,K=16,accumulate=True,acc='O',
                         offsetB_register='k',offsetB_step=4096,simdgroups=4,
                         head_stride=(2048,0,0))
    nxt=bd.add(i,ir.Imm(1),name='n')
    ir.Builder.phi_latch(i,nxt)
    bd.br_cond(bd.cmp(nxt,255,'lt',name='more'),hdr,ex)
    bd.at(ex)
    tensor._store_lane_registers(bd,c,bd.builtin('thread_index_in_simdgroup',name='lane'),
                                 'O',tiles=tiles,sg=4)
    bd.ret()
    ir.verify(fn)
    return fn


def code():
    result=bytes(cc.compile_function(function()).code)
    if len(result)!=56174 or sha(result)!=CODE_SHA256:
        raise ValueError('4080-body code changed')
    latch=tensorlife.counted_loop_check(result,255,carried=tuple(cc._TENSOR_INDEX_USED))
    if latch['back_edges']!=1 or latch['advances']!={'R125':16} or latch['hazards']:
        raise ValueError('sixteen-body counted loop changed')
    return result


def marker(source):
    return b''.join(struct.pack('<e',18*struct.unpack_from('<e',source,i)[0])
                    for i in range(0,4096,2))


def bstream(source):
    result=source[:16384]+source[:4096]*4075+marker(source)
    if len(result)!=4080*4096 or sha(result)!=B_SHA256:
        raise ValueError('4080-body B stream changed')
    return result


def sizes():
    return tuple(B_SIZE if index==20 else size
                 for index,size in zip(INDICES,parent.sizes()))


def pages():
    raw=bytearray(parent.pages())
    for offset in PAGE_POINTERS:
        struct.pack_into('<Q',raw,offset,struct.unpack_from('<Q',raw,offset)[0]+SHIFT)
    result=bytes(raw)
    if sha(result)!=PAGE_SHA256:
        raise ValueError('4080-body pages changed')
    return result


def requests():
    source=parent.requests()
    rows=[list(request_base.ROW.unpack_from(source,index*request_base.ROW.size))
          for index in range(29)]
    rows[20][6]=B_SIZE
    rows[20][7]=request_base.request(0x10001,0x470,B_SIZE,base_buffer=True)
    for index in range(21,29):
        rows[index][5]+=SHIFT
    result=b''.join(request_base.ROW.pack(*row) for row in rows)
    if sha(result)!=REQUEST_SHA256:
        raise ValueError('4080-body requests changed')
    return result


def allocations(program,a,b,c):
    if program!=code() or len(b)!=4080*4096 or sha(b)!=B_SHA256:
        raise ValueError('exact 4080-body code and B required')
    old_b=parent.bstream(b[:16384])
    result=parent.allocations(parent.code(),a,old_b,c)
    raw=bytearray(result[0])
    raw[0x6c0:0x6c0+len(program)]=program
    result[0]=bytes(raw)
    raw=bytearray(B_SIZE)
    raw[:0x80]=b'\xa5'*0x80
    raw[0x80:0x80+len(b)]=b
    raw[0x80+len(b):0x100+len(b)]=b'\xa5'*0x80
    result[20]=bytes(raw)
    raw=bytearray(result[23]);struct.pack_into('<H',raw,6,0x938);result[23]=bytes(raw)
    raw=bytearray(result[25]);struct.pack_into('<H',raw,9,0x0488);result[25]=bytes(raw)
    raw=bytearray(result[28]);struct.pack_into('<Q',raw,0x1bb0,C_BINDING);result[28]=bytes(raw)
    if result[0][0x6c0:0x6c0+len(program)]!=program:
        raise ValueError('4080-body code not staged')
    if any(len(result[index])!=size for index,size in zip(INDICES,sizes())):
        raise ValueError('4080-body physical size changed')
    return result


def build(request_path,page_path,payload_path,program,a,b,c):
    req,page,alloc=requests(),pages(),allocations(program,a,b,c)
    payload=b''.join(alloc[index] for index in INDICES)
    if len(payload)!=19054592 or sha(payload)!=PAYLOAD_SHA256:
        raise ValueError('4080-body payload changed')
    for path,raw in ((request_path,req),(page_path,page),(payload_path,payload)):
        Path(path).write_bytes(raw)
    return {'scope':'field-built twelve-group 4080-body tensor graph',
            'capture_read_at_build':False,'program':'sg4h12loop4080sixteen',
            'requests_sha256':sha(req),'pages_sha256':sha(page),
            'payload_sha256':sha(payload),'payload_bytes':len(payload),
            'physical_allocations':[{'index':index,'size':size,'sha256':sha(alloc[index])}
                                     for index,size in zip(INDICES,sizes())]}
