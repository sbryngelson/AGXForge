#!/usr/bin/env python3
"""Predict a ten-head tensor graph across the second A-allocation boundary."""
import hashlib
from pathlib import Path
import struct

import g17authoredtensorsg4h4 as h4
import g17authoredtensorsg4h8 as h8
import g17structuredrequests as request_base


INDICES=h8.INDICES
A_BYTES=81920
C_BYTES=327680
A_SIZE=0x18000
C_SIZE=0x54000
PAYLOAD_BYTES=1179648


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pages():
    raw=bytearray(h8.pages())
    for offset in (0x174,0x1cc,0x1d4,0x1e4,0x1ec,0x1f4,0x1fc):
        struct.pack_into('<Q',raw,offset,struct.unpack_from('<Q',raw,offset)[0]+0x18000)
    struct.pack_into('<Q',raw,0x3ec,0x100000b0000)
    struct.pack_into('<I',raw,0x3f8,0xa02)
    struct.pack_into('<I',raw,0x4000+0xe4,A_SIZE//0x400)
    struct.pack_into('<I',raw,0x4000+0xec,C_SIZE//0x400)
    return bytes(raw)


def requests():
    source=h8.requests()
    rows=[list(request_base.ROW.unpack_from(source,i*request_base.ROW.size))
          for i in range(29)]
    rows[19][6]=A_SIZE
    rows[19][7]=request_base.request(0x10001,0x470,A_SIZE,base_buffer=True)
    rows[20][5]+=0x8000
    rows[21][5]+=0x8000
    rows[21][6]=C_SIZE
    rows[21][7]=request_base.request(0x10001,0x470,C_SIZE,base_buffer=True)
    for index in range(22,29):
        rows[index][5]+=0x18000
    return b''.join(request_base.ROW.pack(*row) for row in rows)


def allocations(code,a,b,c):
    for head,value in ((8,1.0),(9,2.0)):
        identity=b''.join(struct.pack('<e',value if i==j else 0.0)
                          for sg in range(4) for i in range(16) for j in range(16))
        if a[head*2048:(head+1)*2048]!=identity:
            raise ValueError(f'head {head} A marker changed')
    if (len(a)!=A_BYTES or len(b)!=16384 or len(c)!=C_BYTES or
        a[20480:]!=bytes(A_BYTES-20480) or c!=b'\xff'*C_BYTES):
        raise ValueError('ten-head input pattern or size changed')
    baseline=a[:16384]+bytes(49152)
    result=dict(h8.allocations(code,baseline,b,b'\xff'*262144))
    result[19]=h4.guarded(a,A_SIZE)
    result[21]=h4.guarded(c,C_SIZE)
    raw=bytearray(result[22]);struct.pack_into('<I',raw,0xa8,1280);result[22]=bytes(raw)
    raw=bytearray(result[23]);raw[6]=0xb4;result[23]=bytes(raw)
    raw=bytearray(result[25]);raw[9]=0x46
    struct.pack_into('<I',raw,0x10,1280)
    result[25]=bytes(raw)
    raw=bytearray(result[28])
    struct.pack_into('<Q',raw,0x1ba8,0x100000a0080)
    struct.pack_into('<Q',raw,0x1bb0,0x100000b0080)
    result[28]=bytes(raw)
    sizes={**dict(zip(h8.INDICES,h8.SIZES)),19:A_SIZE,21:C_SIZE}
    for index in INDICES:
        if len(result[index])!=sizes[index]:
            raise ValueError(f'ten-head physical size {index} changed')
    return result


def build(request_path,page_path,payload_path,code,a,b,c):
    req,page,alloc=requests(),pages(),allocations(code,a,b,c)
    payload=b''.join(alloc[i] for i in INDICES)
    if len(payload)!=PAYLOAD_BYTES:
        raise ValueError('ten-head payload size changed')
    for path,raw in ((request_path,req),(page_path,page),(payload_path,payload)):
        Path(path).write_bytes(raw)
    return {'scope':'predicted ten-head graph across second A boundary',
            'capture_read_at_build':False,'program':'sg4h10',
            'requests_sha256':sha(req),'pages_sha256':sha(page),
            'payload_sha256':sha(payload),'payload_bytes':len(payload),
            'physical_allocations':[{'index':i,'size':len(alloc[i]),'sha256':sha(alloc[i])}
                                    for i in INDICES]}
