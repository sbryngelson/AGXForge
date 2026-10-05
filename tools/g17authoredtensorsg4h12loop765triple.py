#!/usr/bin/env python3
"""One 255-trip counted loop with three authored tensor bodies per trip.

The tensor index advances once for each body, addressing 765 B blocks.
The final block is marked so its execution remains visible after the
repeated-positive stream reaches its FP32 fixed point.
"""
import hashlib
from pathlib import Path
import struct

import g17authoredtensorsg4h12loop510dual as parent
import g17structuredrequests as request_base
import g17tensoraccregs as tensor

# Make agxforge.g17 importable when this tool runs from outside the checkout.
import sys as _g17_sys
from pathlib import Path as _G17Path
_g17_root = str(_G17Path(__file__).resolve().parents[1])
if _g17_root not in _g17_sys.path:
    _g17_sys.path.insert(0, _g17_root)
from agxforge.g17 import cc, ir, tensorlife


CODE_SHA256='2d29301d00bd5c372c6cd6ca228e6f004542f982bc8ec08284ecccf4805bb830'
B_SHA256='4ded9afb50a84e25ae6c6626658e3b9bb335b41e58f278aaed90a60fd7a94823'
PAGE_SHA256='2128acd83e74205561cd12de9ce2a244e068df458eda6167c4c0fa7989542968'
REQUEST_SHA256='44a1a7222f040daf2e4f9be5c1f44381fda153af59cfc94a1751edf7e6ad0003'
PAYLOAD_SHA256='c646e3c950fda66586decec50ff5ee4610e5eace36ac62dd1139a0a6ddae8e5b'
OUTPUT_SHA256='3b4feb66e052cf9420a9a9318a7a8d0d0be3d3cf2b5eb8695c5b37ece75c13ce'
INDICES=parent.INDICES
PAGE_POINTERS=parent.PAGE_POINTERS
B_SIZE=0x300000
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
    for body in range(3):
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
    if len(result)!=12754 or sha(result)!=CODE_SHA256:
        raise ValueError('765-body code changed')
    latch=tensorlife.counted_loop_check(result,255,carried=tuple(cc._TENSOR_INDEX_USED))
    if latch['back_edges']!=1 or latch['advances']!={'R125':3} or latch['hazards']:
        raise ValueError('dual-body counted loop changed')
    return result


def marker(source):
    return b''.join(struct.pack('<e',5*struct.unpack_from('<e',source,i)[0])
                    for i in range(0,4096,2))


def bstream(source):
    result=source[:16384]+source[:4096]*760+marker(source)
    if len(result)!=765*4096 or sha(result)!=B_SHA256:
        raise ValueError('765-body B stream changed')
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
        raise ValueError('765-body pages changed')
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
        raise ValueError('765-body requests changed')
    return result


def allocations(program,a,b,c):
    if program!=code() or len(b)!=765*4096 or sha(b)!=B_SHA256:
        raise ValueError('exact 765-body code and B required')
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
    raw=bytearray(result[23]);struct.pack_into('<H',raw,6,0x238);result[23]=bytes(raw)
    raw=bytearray(result[25]);struct.pack_into('<H',raw,9,0x0108);result[25]=bytes(raw)
    raw=bytearray(result[28]);struct.pack_into('<Q',raw,0x1bb0,C_BINDING);result[28]=bytes(raw)
    if result[0][0x6c0:0x6c0+len(program)]!=program:
        raise ValueError('765-body code not staged')
    if any(len(result[index])!=size for index,size in zip(INDICES,sizes())):
        raise ValueError('765-body physical size changed')
    return result


def build(request_path,page_path,payload_path,program,a,b,c):
    req,page,alloc=requests(),pages(),allocations(program,a,b,c)
    payload=b''.join(alloc[index] for index in INDICES)
    if len(payload)!=4374528 or sha(payload)!=PAYLOAD_SHA256:
        raise ValueError('765-body payload changed')
    for path,raw in ((request_path,req),(page_path,page),(payload_path,payload)):
        Path(path).write_bytes(raw)
    return {'scope':'field-built twelve-group 765-body tensor graph',
            'capture_read_at_build':False,'program':'sg4h12loop765triple',
            'requests_sha256':sha(req),'pages_sha256':sha(page),
            'payload_sha256':sha(payload),'payload_bytes':len(payload),
            'physical_allocations':[{'index':index,'size':size,'sha256':sha(alloc[index])}
                                     for index,size in zip(INDICES,sizes())]}
