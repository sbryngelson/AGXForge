#!/usr/bin/env python3
"""One capped 510-trip runtime loop with sixteen authored tensor bodies per trip.

The tensor index advances once for each body, addressing 8160 B blocks.
The final block is marked so its execution remains visible after the
repeated-positive stream reaches its FP32 fixed point.
"""
import hashlib
from pathlib import Path
import struct

import g17authoredtensorsg4h12loop4080sixteen as parent
import g17structuredrequests as request_base
import g17tensoraccregs as tensor

# Make agxforge.g17 importable when this tool runs from outside the checkout.
import sys as _g17_sys
from pathlib import Path as _G17Path
_g17_root = str(_G17Path(__file__).resolve().parents[1])
if _g17_root not in _g17_sys.path:
    _g17_sys.path.insert(0, _g17_root)
from agxforge.g17 import cc, ir, tensorlife


CODE_SHA256='bfb97e255ee15e7220a17ad97efd864afff003c55eb087d3379b1a41a92c2ec2'
A_SHA256='fc03081c747e80652ac511a36a01d99bb9c2a4456231341c5176be9569f7c94b'
B_SHA256='2c9e5ec12c8c1a3721dff1298f41eba3667ae5d8f8e21826fa945aa08fe68b35'
PAGE_SHA256='16184534faa26c37d3599a507dc4e7ca2e20245820f3f33801e9cb3aad2a627e'
REQUEST_SHA256='7c195e3194ba967f72714a0f337d944bae2d486860ba0039981f7c0a0c7d43fc'
PAYLOAD_SHA256='7588e3c2647f32229c8c4ece095a57dbddedefde86ad98f907393d51da2c6b41'
OUTPUT_SHA256='8b0d5e72fa8a071275db73bbfe1073aa80b4f5f224b5c0c5c815395c271524e2'
BOUND_HASHES={
    1: ('04c40102f3b28c59b17989f0f59ffbb91f20bf975034ba60688bf876fd0eccd5',
        '58fb59abe3b6812befe65566098b3049cae6ce8afe95ff5ffb6357a391f77aa2',
        'b6ff64cbebc1ba3bda8899182074d824ebd4dbf61cfac45cacd3975a812bbe28'),
    255: ('a2cb34ca0e7e8e8e58117d7a617ad142a80d3c3de46ad44b9c4b48f37908b534',
          '839c518591492d996665b0200f118dfad460e3b0d15ae0a4324e35c43795514f',
          '6caffb49e8b1f7186fd1d23d0d5e768b7603b2ac90a6e0b67ad516bc39699a94'),
    510: (A_SHA256,PAYLOAD_SHA256,OUTPUT_SHA256),
}
CONTROL_PAYLOAD_SHA256='0bae7a351557f076342a28d723b640c9d6638ee675473e5a0e7c22c9b90ac8f7'
CONTROL_OUTPUT_SHA256=BOUND_HASHES[255][2]
INDICES=parent.INDICES
PAGE_POINTERS=parent.PAGE_POINTERS
B_SIZE=0x2200000
SHIFT=B_SIZE-parent.B_SIZE
C_BINDING=parent.C_BINDING+SHIFT


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def function(bodies=16, reuse_last=False, reuse_from=None, cap=510, fused=False,
             hoist_bodies=0, scale=0.5, count_witness=False):
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
    n=bd.load(a,bd.const(0x17ff0//4,name='bound_word'),type=ir.I32,name='runtime_bound')
    hdr,ex=fn.block('trips'),fn.block('done')
    bd.br(hdr)
    bd.at(hdr)
    i=bd.phi(i0,name='i')
    for body in range(bodies):
        half=tensor._f32_const(bd,scale,f'scale{body}')
        for tile in range(tiles):
            for slot in range(8):
                if fused:
                    bd.tensor_acc_scale('O',slot,half,tile=(0,tile))
                else:
                    bd.tensor_acc_write('O',slot,
                        bd.fmul(bd.tensor_acc_read('O',slot,tile=(0,tile),name=f'o{body}_{tile}_{slot}'),
                                half,name=f'h{body}_{tile}_{slot}'),tile=(0,tile))
        bd.tensor_matmul(a,b,c,M=64,N=tensor.LOOP_N,K=16,accumulate=True,acc='O',
                         offsetB_register='k',
                         offsetB_step=0 if ((reuse_last and body==bodies-1) or
                                           (reuse_from is not None and body>=reuse_from))
                         else 4096,
                         simdgroups=4,
                         hoist_prologue=body < hoist_bodies,
                         head_stride=(2048,0,0))
    nxt=bd.add(i,ir.Imm(1),name='n')
    ir.Builder.phi_latch(i,nxt)
    bd.br_cond(bd.cmp(nxt,n,'lt',cap=cap,name='more'),hdr,ex)
    bd.at(ex)
    tensor._store_lane_registers(bd,c,bd.builtin('thread_index_in_simdgroup',name='lane'),
                                 'O',tiles=tiles,sg=4,
                                 witness=nxt if count_witness else None)
    bd.ret()
    ir.verify(fn)
    return fn

def code():
    result=bytes(cc.compile_function(function()).code)
    if len(result)!=56240 or sha(result)!=CODE_SHA256:
        raise ValueError('8160-body code changed')
    latch=tensorlife.counted_loop_check(result,510,carried=tuple(cc._TENSOR_INDEX_USED),runtime=True)
    if latch['back_edges']!=1 or latch['trips']!=510 or not latch['runtime'] or latch['advances']!={'R125':16} or latch['hazards']:
        raise ValueError('sixteen-body counted loop changed')
    return result


def marker(source):
    return b''.join(struct.pack('<e',34*struct.unpack_from('<e',source,i)[0])
                    for i in range(0,4096,2))


def bstream(source):
    result=source[:16384]+source[:4096]*8155+marker(source)
    if len(result)!=8160*4096 or sha(result)!=B_SHA256:
        raise ValueError('8160-body B stream changed')
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
        raise ValueError('8160-body pages changed')
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
        raise ValueError('8160-body requests changed')
    return result


def allocations(program,a,b,c):
    bound=struct.unpack_from('<I',a,0x17ff0)[0]
    if (program not in (code(),parent.code()) or len(b)!=8160*4096 or sha(b)!=B_SHA256 or
            len(a)!=98304 or bound not in BOUND_HASHES or
            sha(a)!=BOUND_HASHES[bound][0]):
        raise ValueError('exact 8160-body code, runtime bound, and B required')
    old_b=parent.bstream(b[:16384])
    old_a=bytearray(a)
    old_a[0x17ff0:0x17ff4]=bytes(4)
    result=parent.allocations(parent.code(),bytes(old_a),old_b,c)
    raw=bytearray(result[19])
    struct.pack_into('<I',raw,0x80+0x17ff0,bound)
    result[19]=bytes(raw)
    raw=bytearray(result[0])
    raw[0x6c0:0x6c0+len(code())]=program+bytes(len(code())-len(program))
    result[0]=bytes(raw)
    raw=bytearray(B_SIZE)
    raw[:0x80]=b'\xa5'*0x80
    raw[0x80:0x80+len(b)]=b
    raw[0x80+len(b):0x100+len(b)]=b'\xa5'*0x80
    result[20]=bytes(raw)
    raw=bytearray(result[23]);struct.pack_into('<H',raw,6,0x11b8);result[23]=bytes(raw)
    raw=bytearray(result[25]);struct.pack_into('<H',raw,9,0x08c8);result[25]=bytes(raw)
    raw=bytearray(result[28]);struct.pack_into('<Q',raw,0x1bb0,C_BINDING);result[28]=bytes(raw)
    if result[0][0x6c0:0x6c0+len(program)]!=program:
        raise ValueError('8160-body code not staged')
    if any(len(result[index])!=size for index,size in zip(INDICES,sizes())):
        raise ValueError('8160-body physical size changed')
    return result


def build(request_path,page_path,payload_path,program,a,b,c):
    req,page,alloc=requests(),pages(),allocations(program,a,b,c)
    payload=b''.join(alloc[index] for index in INDICES)
    bound=struct.unpack_from('<I',a,0x17ff0)[0]
    expected=CONTROL_PAYLOAD_SHA256 if program==parent.code() and bound==510 else BOUND_HASHES[bound][1]
    if len(payload)!=36880384 or sha(payload)!=expected:
        raise ValueError('8160-body payload changed')
    for path,raw in ((request_path,req),(page_path,page),(payload_path,payload)):
        Path(path).write_bytes(raw)
    return {'scope':'field-built twelve-group 8160-body tensor graph',
            'capture_read_at_build':False,'program':'sg4h12loop8160runtime',
            'requests_sha256':sha(req),'pages_sha256':sha(page),
            'payload_sha256':sha(payload),'payload_bytes':len(payload),
            'physical_allocations':[{'index':index,'size':size,'sha256':sha(alloc[index])}
                                     for index,size in zip(INDICES,sizes())]}
