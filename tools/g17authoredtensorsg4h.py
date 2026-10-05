#!/usr/bin/env python3
"""Build the measured two-threadgroup, four-SIMDgroup tensor graph from fields.

A, B, and C occupy physical allocations 19, 20, and 21. The command
resources follow those allocations; no Metal capture is read at build time.
"""
import hashlib
from pathlib import Path
import struct

import g17authoredblock640payload as block
import g17authoredtensorphysical as physical
import g17authoredtensorsg4 as sg4
import g17structuredbasepayload as base
import g17structuredrequests as request_base


CODE_SHA256 = '402816df52cbbe3912b30a9064122319268735f50e685ef75fd09d9e88267386'
PAGE_SHA256 = 'a9b546ce2c4003e8a503a1662e2242e671448488747cc9392d4960cd9ecc40a1'
REQUEST_SHA256 = '7bfda5cea60ec41e8970008aa8d04bd88b12e9c0417bca77926908ea608cc57f'
ALLOCATION2_SHA256 = '1fb51e0645f6b2a07cf00bdc07e4ca4695071354f99c4670fb54e8af6e7fbffe'
ALLOCATION19_SHA256 = '15bf78444b82aa2367748ece3b8904440a029d3d015d795f4a55688b56013594'
ALLOCATION20_SHA256 = '830a6ca857720c821e42bcd7cc700faafaaf801463513f055db6c785c51dc32c'
ALLOCATION21_SHA256 = '761c4e4b886dbdd0eee124ba8dc4255a10e8d3b564be661748cdb09b48f34c80'
INDICES = (0,1,2,15,17,19,20,21,22,23,24,25,26,27,28)
SIZES = (0x10000,0x10000,0x20000,0x10000,0x20000,
         0x8000,0x8000,0x14000,0x8000,0x8000,0x8000,
         0x8000,0x8000,0x8000,0xc000)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pages():
    raw=bytearray(sg4.pages())
    for offset in (0x174,0x1cc,0x1d4,0x1e4,0x1ec,0x1f4,0x1fc):
        struct.pack_into('<Q',raw,offset,struct.unpack_from('<Q',raw,offset)[0]+0x18000)
    struct.pack_into('<Q',raw,0x3ec,0x100000a0000)
    struct.pack_into('<I',raw,0x3f8,0x202)
    struct.pack_into('<I',raw,0x4000+0xe4,0x20)
    struct.pack_into('<I',raw,0x4000+0xec,0x50)
    result=bytes(raw)
    if sha(result)!=PAGE_SHA256:
        raise ValueError('two-head pages changed')
    return result


def requests():
    source=sg4.requests()
    rows=[list(request_base.ROW.unpack_from(source,i*request_base.ROW.size))
          for i in range(29)]
    rows[19]=[19,0,0,0,0,request_base.BASE+0x80000,0x8000,
              request_base.request(0x10001,0x470,0x8000,base_buffer=True)]
    rows[20][5]=request_base.BASE+0x90000
    rows[21]=[21,0,0,0,0,request_base.BASE+0xa0000,0x14000,
              request_base.request(0x10001,0x470,0x14000,base_buffer=True)]
    for index in range(22,29):
        rows[index][5]+=0x18000
    result=b''.join(request_base.ROW.pack(*row) for row in rows)
    if sha(result)!=REQUEST_SHA256:
        raise ValueError('two-head requests changed')
    return result


def allocation2():
    raw=bytearray(0x20000)
    raw[:0x4d00]=base.allocation(2)[0xc00:0x5900]
    result=bytes(raw)
    if sha(result)!=ALLOCATION2_SHA256:
        raise ValueError('two-head control allocation changed')
    return result


def guarded_input(raw,size,expected,name):
    if len(raw)!=size:
        raise ValueError(f'{name} size changed')
    result=bytearray(0x8000)
    result[:0x80]=b'\xa5'*0x80
    result[0x80:0x80+size]=raw
    result[0x80+size:0x100+size]=b'\xa5'*0x80
    if sha(result)!=expected:
        raise ValueError(f'{name} allocation changed')
    return bytes(result)


def allocation21(c):
    if len(c)!=65536:
        raise ValueError('two-head C size changed')
    raw=bytearray(0x14000)
    raw[:0x80]=b'\xa5'*0x80
    raw[0x80:0x10080]=c
    raw[0x10080:0x10100]=b'\xa5'*0x80
    result=bytes(raw)
    if sha(result)!=ALLOCATION21_SHA256:
        raise ValueError('two-head C allocation changed')
    return result


def allocations(code,a,b,c):
    if len(code)!=6074 or sha(code)!=CODE_SHA256:
        raise ValueError('exact two-head program required')
    result={}
    raw=bytearray(block.allocation_zero(prefix_bytes=0x400))
    raw[0x6c0:0x6c0+len(code)]=code
    result[0]=bytes(raw)
    result[1]=base.allocation(1)
    result[2]=allocation2()
    result[15]=physical.allocation(15)
    result[17]=physical.allocation(17)
    result[19]=guarded_input(a,16384,ALLOCATION19_SHA256,'A')
    result[20]=guarded_input(b,16384,ALLOCATION20_SHA256,'B')
    result[21]=allocation21(c)
    for index in range(22,29):
        raw=bytearray(physical.allocation(index))
        if index==22:
            struct.pack_into('<I',raw,0xa8,256)
        elif index==23:
            raw[6]=0x8c
        elif index==25:
            raw[9]=0x32
            struct.pack_into('<I',raw,0x10,256)
            struct.pack_into('<I',raw,0x1c,128)
        elif index==28:
            for offset,address in ((0x1ba0,0x10000080080),
                                   (0x1ba8,0x10000090080),
                                   (0x1bb0,0x100000a0080)):
                struct.pack_into('<Q',raw,offset,address)
        result[index]=bytes(raw)
    for index,size in zip(INDICES,SIZES):
        if len(result[index])!=size:
            raise ValueError(f'two-head physical size {index} changed')
    return result


def build(request_path,page_path,payload_path,code,a,b,c):
    req,page,alloc=requests(),pages(),allocations(code,a,b,c)
    payload=b''.join(alloc[i] for i in INDICES)
    if len(payload)!=851968:
        raise ValueError('two-head payload length changed')
    for path,raw in ((request_path,req),(page_path,page),(payload_path,payload)):
        Path(path).write_bytes(raw)
    return {'scope':'field-built two-head four-SIMDgroup tensor graph',
            'capture_read_at_build':False,'program':'sg4h',
            'requests_sha256':sha(req),'pages_sha256':sha(page),
            'payload_sha256':sha(payload),'payload_bytes':len(payload),
            'physical_allocations':[{'index':i,'size':s,'sha256':sha(alloc[i])}
                                    for i,s in zip(INDICES,SIZES)]}
