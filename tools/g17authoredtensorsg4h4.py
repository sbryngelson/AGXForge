#!/usr/bin/env python3
"""Build the measured four-head, four-SIMDgroup tensor graph from fields."""
import hashlib
from pathlib import Path
import struct

import g17authoredtensorsg4h as h2
import g17structuredrequests as request_base


INDICES=h2.INDICES
SIZES=tuple(0xc000 if index==19 else 0x24000 if index==21 else size
            for index,size in zip(h2.INDICES,h2.SIZES))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pages():
    raw=bytearray(h2.pages())
    for offset in (0x174,0x1cc,0x1d4,0x1e4,0x1ec,0x1f4,0x1fc):
        struct.pack_into('<Q',raw,offset,struct.unpack_from('<Q',raw,offset)[0]+0x10000)
    struct.pack_into('<I',raw,0x3f8,0x402)
    struct.pack_into('<I',raw,0x4000+0xe4,0x30)
    struct.pack_into('<I',raw,0x4000+0xec,0x90)
    return bytes(raw)


def requests():
    source=h2.requests()
    rows=[list(request_base.ROW.unpack_from(source,i*request_base.ROW.size))
          for i in range(29)]
    for index,size in ((19,0xc000),(21,0x24000)):
        rows[index][6]=size
        rows[index][7]=request_base.request(0x10001,0x470,size,base_buffer=True)
    for index in range(22,29):
        rows[index][5]+=0x10000
    return b''.join(request_base.ROW.pack(*row) for row in rows)


def guarded(raw,alloc_size):
    out=bytearray(alloc_size)
    out[:0x80]=b'\xa5'*0x80
    out[0x80:0x80+len(raw)]=raw
    out[0x80+len(raw):0x100+len(raw)]=b'\xa5'*0x80
    return bytes(out)


def allocations(code,a,b,c):
    if len(a)!=32768 or len(b)!=16384 or len(c)!=131072 or a[16384:]!=bytes(16384):
        raise ValueError('four-head input sizes or A extension changed')
    result=dict(h2.allocations(code,a[:16384],b,b'\xff'*65536))
    result[19]=guarded(a,0xc000)
    result[21]=guarded(c,0x24000)
    raw=bytearray(result[22]);struct.pack_into('<I',raw,0xa8,512);result[22]=bytes(raw)
    raw=bytearray(result[23]);raw[6]=0x94;result[23]=bytes(raw)
    raw=bytearray(result[25]);raw[9]=0x36
    struct.pack_into('<I',raw,0x10,512)
    result[25]=bytes(raw)
    for index,size in zip(INDICES,SIZES):
        if len(result[index])!=size:
            raise ValueError(f'four-head physical size {index} changed')
    return result


def build(request_path,page_path,payload_path,code,a,b,c):
    req,page,alloc=requests(),pages(),allocations(code,a,b,c)
    payload=b''.join(alloc[i] for i in INDICES)
    if len(payload)!=933888:
        raise ValueError('four-head payload size changed')
    for path,raw in ((request_path,req),(page_path,page),(payload_path,payload)):
        Path(path).write_bytes(raw)
    return {'scope':'field-built measured four-head four-SIMDgroup tensor graph',
            'capture_read_at_build':False,'program':'sg4h4',
            'requests_sha256':sha(req),'pages_sha256':sha(page),
            'payload_sha256':sha(payload),'payload_bytes':len(payload),
            'physical_allocations':[{'index':i,'size':s,'sha256':sha(alloc[i])}
                                    for i,s in zip(INDICES,SIZES)]}
