#!/usr/bin/env python3
"""Build the measured four-SIMDgroup tensor-loop graph from fields.

This exact 128-thread case has A in allocation 2, B in allocation 20, and
C in a new physical allocation 21. No capture is read during construction.
"""
import hashlib
from pathlib import Path
import struct

import g17authoredtensorloop as loop
import g17authoredtensorphysical as physical
import g17authoredblock640payload as block
import g17structuredrequests as requests_base
import g17structuredbasepayload as base


CODE_SHA256 = '7e55d327e8f21cf74379d740ceed9b6b39f28a7a1c9ae136681f624c1c7a65f2'
LOOP16_CODE_SHA256 = 'f8525455f8d8ae8854f98a160d72b165779543362a9ab6eec6c43fcb73881c02'
LOOP16_B_SHA256 = '2c5ba1838948bb463945c1a62278c404a7b4e0d403123d65413676c1f12d14a1'
LOOP16_ALLOCATION20_SHA256 = 'ad5593daba89154478b72578966a3510c80649d9667be62ded5710a97469d854'
PAGE_SHA256 = 'ea267786d8b6a45fb9fb8ff2168abf031130048932b15e277abd3f7205542f6e'
LOOP16_PAGE_SHA256 = '57292ed7f00f8134a97ae64a01a17873ad4c73696826ca5c997a291b18e4c317'
REQUEST_SHA256 = '3ed570c576e6f23f01d66c3c1d68be315bca644650da70aa096591c25197d644'
LOOP16_REQUEST_SHA256 = '05c67af90e70e6800a7d5e41a65c185c6c26233acc156163f7fe7c4c9fd8ac77'
ALLOCATION2_SHA256 = 'ea5a3260d5be6b12a397d86733a952e58741d1bb2b902a4c12776e1d1b1c0f69'
ALLOCATION20_SHA256 = '49a04a349f590734187d7a23c4418d7a7b95851bb6b48863dd7c923dc85a3f99'
ALLOCATION21_SHA256 = '66f7710544799fc4be632eb1d9e130b0d6feede2a1e2f5f80d1f6e3f41bd6a22'
INDICES = (0,1,2,15,17,20,21,22,23,24,25,26,27,28)
SIZES = (0x10000,0x10000,0x20000,0x10000,0x20000,0x8000,0xc000,
         0x8000,0x8000,0x8000,0x8000,0x8000,0x8000,0xc000)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pages(program='sg4'):
    raw=bytearray(loop.pages())
    for offset in (0x174,0x1cc,0x1d4,0x1e4,0x1ec,0x1f4,0x1fc):
        struct.pack_into('<Q',raw,offset,struct.unpack_from('<Q',raw,offset)[0]+0x10000)
    struct.pack_into('<Q',raw,0x3ec,0x10000090000)
    struct.pack_into('<I',raw,0x3f8,0x102)
    struct.pack_into('<I',raw,0x4000+0xec,0x30)
    if program=='sg4loop16':
        for offset in (0x174,0x1cc,0x1d4,0x1e4,0x1ec,0x1f4,0x1fc,0x3ec):
            struct.pack_into('<Q',raw,offset,struct.unpack_from('<Q',raw,offset)[0]+0x10000)
    result=bytes(raw)
    if sha(result)!=(LOOP16_PAGE_SHA256 if program=='sg4loop16' else PAGE_SHA256):
        raise ValueError('four-SIMDgroup pages changed')
    return result


def requests(program='sg4'):
    source=loop.requests()
    rows=[list(requests_base.ROW.unpack_from(source,i*requests_base.ROW.size))
          for i in range(29)]
    if program=='sg4loop16':
        # Enlarging physical B shifts C and the seven later resources.
        rows[20][6]=0x18000
        rows[20][7]=requests_base.request(0x10001,0x470,0x18000,base_buffer=True)
    rows[21]=[21,0,0,0,0,requests_base.BASE+0x90000,0xc000,
              requests_base.request(0x10001,0x470,0xc000,base_buffer=True)]
    if program=='sg4loop16':
        rows[21][5]+=0x10000
    for index in range(22,29):
        rows[index][5]+=0x20000 if program=='sg4loop16' else 0x10000
    result=b''.join(requests_base.ROW.pack(*row) for row in rows)
    if sha(result)!=(LOOP16_REQUEST_SHA256 if program=='sg4loop16' else REQUEST_SHA256):
        raise ValueError('four-SIMDgroup requests changed')
    return result


def allocation2(a):
    if len(a)!=8192:
        raise ValueError('four-SIMDgroup A shape changed')
    raw=bytearray(0x20000)
    raw[:0x4d00]=base.allocation(2)[0xc00:0x5900]
    raw[0x4d00:0x4d80]=b'\xa5'*0x80
    raw[0x4d80:0x6d80]=a
    raw[0x6d80:0x6e00]=b'\xa5'*0x80
    result=bytes(raw)
    if sha(result)!=ALLOCATION2_SHA256:
        raise ValueError('four-SIMDgroup allocation 2 changed')
    return result


def long_b_blocks(b):
    if len(b)!=16384:
        raise ValueError('four-SIMDgroup B source changed')
    neg=bytes(value^(0x80 if index&1 else 0) for index,value in enumerate(b))
    result=b+neg*3
    if sha(result)!=LOOP16_B_SHA256:
        raise ValueError('four-SIMDgroup long B input changed')
    return result


def allocation20(b,program='sg4'):
    if len(b)!=(65536 if program=='sg4loop16' else 16384):
        raise ValueError('four-SIMDgroup B shape changed')
    raw=bytearray(0x18000 if program=='sg4loop16' else 0x8000)
    raw[:0x80]=b'\xa5'*0x80
    raw[0x80:0x80+len(b)]=b
    raw[0x80+len(b):0x100+len(b)]=b'\xa5'*0x80
    result=bytes(raw)
    if sha(result)!=(LOOP16_ALLOCATION20_SHA256 if program=='sg4loop16' else ALLOCATION20_SHA256):
        raise ValueError('four-SIMDgroup allocation 20 changed')
    return result


def allocation21(c):
    if len(c)!=32768:
        raise ValueError('four-SIMDgroup C shape changed')
    raw=bytearray(0xc000)
    raw[:0x80]=b'\xa5'*0x80
    raw[0x80:0x8080]=c
    raw[0x8080:0x8100]=b'\xa5'*0x80
    result=bytes(raw)
    if sha(result)!=ALLOCATION21_SHA256:
        raise ValueError('four-SIMDgroup allocation 21 changed')
    return result


def allocations(code,a,b,c,program='sg4'):
    if len(code)!=6034 or sha(code)!=(LOOP16_CODE_SHA256 if program=='sg4loop16' else CODE_SHA256):
        raise ValueError('exact four-SIMDgroup program required')
    result={}
    raw=bytearray(block.allocation_zero(prefix_bytes=0x400))
    raw[0x6c0:0x6c0+len(code)]=code
    result[0]=bytes(raw)
    result[1]=base.allocation(1)
    result[2]=allocation2(a)
    result[15]=physical.allocation(15)
    result[17]=physical.allocation(17)
    result[20]=allocation20(b,program=program)
    result[21]=allocation21(c)
    for index in range(22,29):
        raw=bytearray(physical.allocation(index))
        if index==22:
            struct.pack_into('<I',raw,0xa8,128)
        elif index==23:
            # Empirical resource-base correlate: +8 for this +0x10000 shift.
            raw[6]=0x88 if program=='sg4loop16' else 0x80
        elif index==25:
            # Paired correlate: +4 for the same shift; semantics still opaque.
            raw[9]=0x30 if program=='sg4loop16' else 0x2c
            struct.pack_into('<I',raw,0x10,128)
            struct.pack_into('<I',raw,0x1c,128)
        elif index==28:
            struct.pack_into('<Q',raw,0x1ba8,0x10000080080)
            struct.pack_into('<Q',raw,0x1bb0,0x100000a0080 if program=='sg4loop16' else 0x10000090080)
        result[index]=bytes(raw)
    for index,size in zip(INDICES,sizes(program)):
        if len(result[index])!=size:
            raise ValueError(f'four-SIMDgroup physical size {index} changed')
    return result


def sizes(program='sg4'):
    if program=='sg4loop16':
        return tuple(0x18000 if index==20 else size for index,size in zip(INDICES,SIZES))
    return SIZES


def build(request_path,page_path,payload_path,code,a,b,c,program='sg4'):
    req,page,alloc=requests(program),pages(program),allocations(code,a,b,c,program=program)
    payload=b''.join(alloc[i] for i in INDICES)
    if len(payload)!=(851968 if program=='sg4loop16' else 786432):
        raise ValueError('four-SIMDgroup payload length changed')
    for path,raw in ((request_path,req),(page_path,page),(payload_path,payload)):
        Path(path).write_bytes(raw)
    return {'scope':'field-built four-SIMDgroup tensor loop graph',
            'capture_read_at_build':False,'program':program,
            'requests_sha256':sha(req),'pages_sha256':sha(page),
            'payload_sha256':sha(payload),'payload_bytes':len(payload),
            'physical_allocations':[{'index':i,'size':s,'sha256':sha(alloc[i])}
                                    for i,s in zip(INDICES,sizes(program))]}
