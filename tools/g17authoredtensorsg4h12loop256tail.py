#!/usr/bin/env python3
"""One 255-trip tensor loop plus one authored tail operation below Metal.

The tail scales the carried accumulator and reads B block 256 through
the same tensor index. It tests execution beyond the compiler's
single-comparison eight-bit bound without changing that bound.
"""
import hashlib
from pathlib import Path
import struct

import g17authoredtensorsg4h12loop254255 as parent
import g17structuredrequests as request_base
import g17tensoraccregs as tensor

# Make agxforge.g17 importable when this tool runs from outside the checkout.
import sys as _g17_sys
from pathlib import Path as _G17Path
_g17_root = str(_G17Path(__file__).resolve().parents[1])
if _g17_root not in _g17_sys.path:
    _g17_sys.path.insert(0, _g17_root)
from agxforge.g17 import cc, ir, tensorlife


CODE_SHA256='8b0b121f188c06323df1ba6acd828381e4c7690b9059f3e45230ff57d811413d'
B_SHA256='c9fe846789662d141e0d24ea0bdb20477298382f27232ca437d5681d5ba470c5'
PAGE_SHA256='f0b60f46cb56fa6967ae8a3f1d5a54d49fa59e3d012988fee6e1432d57fd6e19'
REQUEST_SHA256='5dc2a22d20f191bf9f8943791b8c2dbfdcac4f54f9694b50e42e0efa3dfc1583'
PAYLOAD_SHA256='0753bd81a26177958700bbad1c58724ead8c61a2312879c5b46a2578eb44db16'
OUTPUT_SHA256='084cfa0a25ee137bb20ce97376387810500a4c04c3af2ab2b1f823365bcc1f63'
INDICES=parent.INDICES
PAGE_POINTERS=parent.PAGE_POINTERS
B_SIZE=0x108000
C_BINDING=0x100001b0080


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def function():
    """Preserve loop_fn's 255-body path and add the 256th body in done."""
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
    i0=bd.const(0,name='i0')
    hdr,ex=fn.block('trips'),fn.block('done')
    bd.br(hdr)
    bd.at(hdr)
    i=bd.phi(i0,name='i')
    half=tensor._f32_const(bd,0.5,'scale')
    for tile in range(tiles):
        for slot in range(8):
            bd.tensor_acc_write('O',slot,
                bd.fmul(bd.tensor_acc_read('O',slot,tile=(0,tile),name=f'o{tile}_{slot}'),
                        half,name=f'h{tile}_{slot}'),tile=(0,tile))
    bd.tensor_matmul(a,b,c,M=64,N=tensor.LOOP_N,K=16,accumulate=True,acc='O',
                     offsetB_register='k',offsetB_step=4096,simdgroups=4,
                     head_stride=(2048,0,0))
    nxt=bd.add(i,ir.Imm(1),name='n')
    ir.Builder.phi_latch(i,nxt)
    bd.br_cond(bd.cmp(nxt,255,'lt',name='more'),hdr,ex)
    bd.at(ex)
    half2=tensor._f32_const(bd,0.5,'scale_tail')
    for tile in range(tiles):
        for slot in range(8):
            bd.tensor_acc_write('O',slot,
                bd.fmul(bd.tensor_acc_read('O',slot,tile=(0,tile),name=f'tail_o{tile}_{slot}'),
                        half2,name=f'tail_h{tile}_{slot}'),tile=(0,tile))
    bd.tensor_matmul(a,b,c,M=64,N=tensor.LOOP_N,K=16,accumulate=True,acc='O',
                     offsetB_register='k',offsetB_step=4096,simdgroups=4,
                     head_stride=(2048,0,0))
    tensor._store_lane_registers(bd,c,bd.builtin('thread_index_in_simdgroup',name='lane'),
                                 'O',tiles=tiles,sg=4)
    bd.ret()
    ir.verify(fn)
    return fn


def compiled_code():
    return bytes(cc.compile_function(function()).code)


def code():
    result=compiled_code()
    if len(result)!=9414 or sha(result)!=CODE_SHA256:
        raise ValueError('256-body tensor code changed')
    tensorlife.counted_loop_check(result,255,carried=tuple(cc._TENSOR_INDEX_USED))
    return result


def marker(source):
    return b''.join(struct.pack('<e',3*struct.unpack_from('<e',source,i)[0])
                    for i in range(0,4096,2))


def bstream(source):
    result=parent.bstream(source,255)[:-4096]+source[:4096]+marker(source)
    if len(result)!=256*4096 or sha(result)!=B_SHA256:
        raise ValueError('256-body marked B stream changed')
    return result


def sizes():
    return tuple(B_SIZE if index==20 else size
                 for index,size in zip(INDICES,parent.sizes(255)))


def pages():
    raw=bytearray(parent.pages(255))
    for offset in PAGE_POINTERS:
        struct.pack_into('<Q',raw,offset,struct.unpack_from('<Q',raw,offset)[0]+0x8000)
    result=bytes(raw)
    if sha(result)!=PAGE_SHA256:
        raise ValueError('256-body pages changed')
    return result


def requests():
    source=parent.requests(255)
    rows=[list(request_base.ROW.unpack_from(source,index*request_base.ROW.size))
          for index in range(29)]
    rows[20][6]=B_SIZE
    rows[20][7]=request_base.request(0x10001,0x470,B_SIZE,base_buffer=True)
    for index in range(21,29):
        rows[index][5]+=0x8000
    result=b''.join(request_base.ROW.pack(*row) for row in rows)
    if sha(result)!=REQUEST_SHA256:
        raise ValueError('256-body requests changed')
    return result


def allocations(program,a,b,c):
    if program!=code() or len(b)!=256*4096 or sha(b)!=B_SHA256:
        raise ValueError('exact 256-body code and B required')
    old_b=parent.bstream(b[:16384],255)
    result=parent.allocations(parent.code(255),a,old_b,c,255)
    raw=bytearray(result[0])
    raw[0x6c0:0x6c0+len(program)]=program
    result[0]=bytes(raw)
    raw=bytearray(B_SIZE)
    raw[:0x80]=b'\xa5'*0x80
    raw[0x80:0x80+len(b)]=b
    raw[0x80+len(b):0x100+len(b)]=b'\xa5'*0x80
    result[20]=bytes(raw)
    raw=bytearray(result[23]);struct.pack_into('<H',raw,6,0x13c);result[23]=bytes(raw)
    raw=bytearray(result[25]);raw[9]=0x8a;result[25]=bytes(raw)
    raw=bytearray(result[28]);struct.pack_into('<Q',raw,0x1bb0,C_BINDING);result[28]=bytes(raw)
    if result[0][0x6c0:0x6c0+len(program)]!=program:
        raise ValueError('256-body code not staged')
    if any(len(result[index])!=size for index,size in zip(INDICES,sizes())):
        raise ValueError('256-body physical size changed')
    return result


def build(request_path,page_path,payload_path,program,a,b,c):
    req,page,alloc=requests(),pages(),allocations(program,a,b,c)
    payload=b''.join(alloc[index] for index in INDICES)
    if len(payload)!=2310144 or sha(payload)!=PAYLOAD_SHA256:
        raise ValueError('256-body payload changed')
    for path,raw in ((request_path,req),(page_path,page),(payload_path,payload)):
        Path(path).write_bytes(raw)
    return {'scope':'field-built twelve-group 256-body tensor graph',
            'capture_read_at_build':False,'program':'sg4h12loop256tail',
            'requests_sha256':sha(req),'pages_sha256':sha(page),
            'payload_sha256':sha(payload),'payload_bytes':len(payload),
            'physical_allocations':[{'index':index,'size':size,'sha256':sha(alloc[index])}
                                     for index,size in zip(INDICES,sizes())]}
