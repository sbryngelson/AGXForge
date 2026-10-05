#!/usr/bin/env python3
"""Build the measured eight-head tensor command graph from fields."""
import hashlib
from pathlib import Path
import struct

import g17authoredtensorsg4h4 as h4
import g17structuredrequests as request_base


INDICES=h4.INDICES
SIZES=tuple(0x14000 if index==19 else 0x44000 if index==21 else size
            for index,size in zip(h4.INDICES,h4.SIZES))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pages():
    raw=bytearray(h4.pages())
    for offset in (0x174,0x1cc,0x1d4,0x1e4,0x1ec,0x1f4,0x1fc):
        struct.pack_into('<Q',raw,offset,struct.unpack_from('<Q',raw,offset)[0]+0x28000)
    struct.pack_into('<Q',raw,0x3ec,0x100000a8000)
    struct.pack_into('<I',raw,0x3f8,0x802)
    struct.pack_into('<I',raw,0x4000+0xe4,0x50)
    struct.pack_into('<I',raw,0x4000+0xec,0x110)
    return bytes(raw)


def requests():
    source=h4.requests()
    rows=[list(request_base.ROW.unpack_from(source,i*request_base.ROW.size))
          for i in range(29)]
    rows[19][6]=0x14000
    rows[19][7]=request_base.request(0x10001,0x470,0x14000,base_buffer=True)
    rows[20][5]+=0x8000
    rows[21][5]+=0x8000
    rows[21][6]=0x44000
    rows[21][7]=request_base.request(0x10001,0x470,0x44000,base_buffer=True)
    for index in range(22,29):
        rows[index][5]+=0x28000
    return b''.join(request_base.ROW.pack(*row) for row in rows)


def allocations(code,a,b,c):
    if len(a)!=65536 or len(b)!=16384 or len(c)!=262144 or a[16384:]!=bytes(49152):
        raise ValueError('eight-head input sizes or A extension changed')
    result=dict(h4.allocations(code,a[:32768],b,b'\xff'*131072))
    result[19]=h4.guarded(a,0x14000)
    result[21]=h4.guarded(c,0x44000)
    raw=bytearray(result[22]);struct.pack_into('<I',raw,0xa8,1024);result[22]=bytes(raw)
    raw=bytearray(result[23]);raw[6]=0xa8;result[23]=bytes(raw)
    raw=bytearray(result[25]);raw[9]=0x40
    struct.pack_into('<I',raw,0x10,1024)
    result[25]=bytes(raw)
    raw=bytearray(result[28])
    struct.pack_into('<Q',raw,0x1ba8,0x10000098080)
    struct.pack_into('<Q',raw,0x1bb0,0x100000a8080)
    result[28]=bytes(raw)
    for index,size in zip(INDICES,SIZES):
        if len(result[index])!=size:
            raise ValueError(f'eight-head physical size {index} changed')
    return result


def build(request_path,page_path,payload_path,code,a,b,c):
    req,page,alloc=requests(),pages(),allocations(code,a,b,c)
    payload=b''.join(alloc[i] for i in INDICES)
    if len(payload)!=1097728:
        raise ValueError('eight-head payload size changed')
    for path,raw in ((request_path,req),(page_path,page),(payload_path,payload)):
        Path(path).write_bytes(raw)
    return {'scope':'field-built measured eight-head four-SIMDgroup tensor graph',
            'capture_read_at_build':False,'program':'sg4h8',
            'requests_sha256':sha(req),'pages_sha256':sha(page),
            'payload_sha256':sha(payload),'payload_bytes':len(payload),
            'physical_allocations':[{'index':i,'size':s,'sha256':sha(alloc[i])}
                                    for i,s in zip(INDICES,SIZES)]}
