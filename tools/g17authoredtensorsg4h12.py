#!/usr/bin/env python3
"""Predict twelve tensor groups and test an A-size increase without an aperture move."""
import hashlib
from pathlib import Path
import struct

import g17authoredtensorsg4h4 as h4
import g17authoredtensorsg4h10 as h10
import g17structuredrequests as request_base


INDICES=h10.INDICES
A_BYTES=98304
C_BYTES=393216
A_SIZE=0x1c000
C_SIZE=0x64000
PAYLOAD_BYTES=1261568


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pages():
    raw=bytearray(h10.pages())
    for offset in (0x174,0x1cc,0x1d4,0x1e4,0x1ec,0x1f4,0x1fc):
        struct.pack_into('<Q',raw,offset,struct.unpack_from('<Q',raw,offset)[0]+0x10000)
    struct.pack_into('<I',raw,0x3f8,0xc02)
    struct.pack_into('<I',raw,0x4000+0xe4,A_SIZE//0x400)
    struct.pack_into('<I',raw,0x4000+0xec,C_SIZE//0x400)
    return bytes(raw)


def requests():
    source=h10.requests()
    rows=[list(request_base.ROW.unpack_from(source,i*request_base.ROW.size))
          for i in range(29)]
    rows[19][6]=A_SIZE
    rows[19][7]=request_base.request(0x10001,0x470,A_SIZE,base_buffer=True)
    rows[21][6]=C_SIZE
    rows[21][7]=request_base.request(0x10001,0x470,C_SIZE,base_buffer=True)
    for index in range(22,29):
        rows[index][5]+=0x10000
    return b''.join(request_base.ROW.pack(*row) for row in rows)


def allocations(code,a,b,c):
    for head in range(8,12):
        value=float(head-7)
        identity=b''.join(struct.pack('<e',value if i==j else 0.0)
                          for sg in range(4) for i in range(16) for j in range(16))
        if a[head*2048:(head+1)*2048]!=identity:
            raise ValueError(f'head {head} A marker changed')
    if (len(a)!=A_BYTES or len(b)!=16384 or len(c)!=C_BYTES or
        a[24576:]!=bytes(A_BYTES-24576) or c!=b'\xff'*C_BYTES):
        raise ValueError('twelve-head input pattern or size changed')
    baseline=a[:20480]+bytes(81920-20480)
    result=dict(h10.allocations(code,baseline,b,b'\xff'*327680))
    result[19]=h4.guarded(a,A_SIZE)
    result[21]=h4.guarded(c,C_SIZE)
    raw=bytearray(result[22]);struct.pack_into('<I',raw,0xa8,1536);result[22]=bytes(raw)
    raw=bytearray(result[23]);raw[6]=0xbc;result[23]=bytes(raw)
    raw=bytearray(result[25]);raw[9]=0x4a
    struct.pack_into('<I',raw,0x10,1536)
    result[25]=bytes(raw)
    sizes={**dict(zip(h4.INDICES,h4.SIZES)),19:A_SIZE,21:C_SIZE}
    for index in INDICES:
        if len(result[index])!=sizes[index]:
            raise ValueError(f'twelve-head physical size {index} changed')
    return result


def build(request_path,page_path,payload_path,code,a,b,c):
    req,page,alloc=requests(),pages(),allocations(code,a,b,c)
    payload=b''.join(alloc[i] for i in INDICES)
    if len(payload)!=PAYLOAD_BYTES:
        raise ValueError('twelve-head payload size changed')
    for path,raw in ((request_path,req),(page_path,page),(payload_path,payload)):
        Path(path).write_bytes(raw)
    return {'scope':'predicted twelve-head graph with stationary B/C apertures',
            'capture_read_at_build':False,'program':'sg4h12',
            'requests_sha256':sha(req),'pages_sha256':sha(page),
            'payload_sha256':sha(payload),'payload_bytes':len(payload),
            'physical_allocations':[{'index':i,'size':len(alloc[i]),'sha256':sha(alloc[i])}
                                    for i in INDICES]}
