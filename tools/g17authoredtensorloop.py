#!/usr/bin/env python3
"""Build the measured four-trip tensor accumulator loop graph from fields.

The B stream lives in a new physical allocation 20; the seven command
resources move one 64 KiB aperture. No Metal capture is read at build time.
"""
import hashlib
from pathlib import Path
import struct

import g17authoredblock640payload as block
import g17authoredtensorpages as tensor_pages
import g17authoredtensorphysical as physical
import g17authoredtensorrequests as tensor_requests
import g17structuredbasepayload as base
import g17structuredrequests as scalar_requests


PAGE_SHA256 = '813a0034cf988cbc8269564bd1165ed448faa502ad9f35a3ee8785a80b83fc70'
REQUEST_SHA256 = 'fb3e39ade5d3a7605b4823e948456bfa248e204f419db879b99329b91b7ec876'
ALLOCATION2_SHA256 = 'd7298772350311d47fd712fbbbca5e6ea7eb0d1221b91e3dcfe6a5d8f8d060b3'
ALLOCATION20_SHA256 = '0beb9a2afb0f5704893a42a975ead7fdd7c9f78057689046ae41b0140f4e7fbf'
CODE_SHA256 = {
    5940: 'b0a97b4c2da0f906796690153bc1bbdb590abd77d9fb48f1a53219818a025502',
    5092: '08d7eb21dbf91e1d38555ef1b28a2117142b09ce8fa1f49b3a5f6bfd9fb158c4',
}
LOOP3_CODE_SHA256 = '0a83c98f5aeb9e22bba58bae711d456698413ade39abb679d5852912e01137f0'
LOOP5_CODE_SHA256 = '332b88efc0e06bbc26855be11bf2d6699e69909f5cdccebbb1904e16c9490777'
LOOP5_B_SHA256 = '28daba513676a31e94acf8a244bf4a0be69ea8373ace9c8e1dd3b5a214a0182c'
LOOP5_ALLOCATION20_SHA256 = 'ab48f4b75b7733524a3c35ee33352fdcdbac3b5859cdaf52d23266eee7da62e8'
LOOP7_CODE_SHA256 = 'dffbd0750a0229da58a40bcb68a9ff4fe12220bf41dfdc2537d9bb2e4d1111f3'
LOOP7_B_SHA256 = '562bdf4b17cad062bea0ad5064df850011a16f34313a630fc3d5c386d12ac80b'
LOOP7_ALLOCATION20_SHA256 = '5810fb9f45ce146f9f08084448782adaefb28493decd35e4790dfd456d76d906'
LOOP8_CODE_SHA256 = '5a7368d2e79825c913a991cce8f30fe707e9fd867638e69c972e6e8f0a07e2c1'
LOOP8_B_SHA256 = '516bf7c14cc056a004f14449a451b4925f703ce56879a7589eece27b1a433cbe'
LOOP8_ALLOCATION20_SHA256 = 'b5688ea4d1170259633dd36121cceaedfdbf74511cb0dcce0ec40bc1cea86b32'
LOOP8_PAGE_SHA256 = 'f9df5fd6bc2c10a0612bc8fe6832099fbd5131fd289323798c74e1fdc9a0be91'
LOOP8_REQUEST_SHA256 = '143d5c88e9327694df5b4b195283e246d257a97d1ce668159388d8c9bc3e0a83'
LOOP15_CODE_SHA256 = '504b0e9862fca82baba6810ea8e6449dbc343187e6f1a91b504ddb832804bde6'
LOOP15_B_SHA256 = '21f9511e184348580a738c987407ce5ef1a4fc2fc998d58194bd61234a5c7014'
LOOP15_ALLOCATION20_SHA256 = '93551d31c3fb8c3f536093a99a6eab2378e9120f104f180a5f622c65a51208fb'
LOOP16_CODE_SHA256 = '451ee18d188be978cbc37ab6f7fe761cb2e0540a72dddd9c260911f38d8e83cc'
LOOP16_B_SHA256 = 'bd04fc5a16fa325bc0b4b477cf8babef42d006938892f27f835e3b88db3e1533'
LOOP16_ALLOCATION20_SHA256 = '6e3557683044cc20d2834d382fa66448bc80f80f305edcf1227f545a9edc3c2a'
LOOP16_PAGE_SHA256 = 'e2fe522b4cdb0657c730b92d4495342d700c518bd897fb7d25f8e606e51417f9'
LOOP16_REQUEST_SHA256 = 'a341e25fac55e02d19b85b6a3949e2d116546599f443b97ccc838850160b8327'
SCALE_CODE_SHA256 = {
    'loopquarter': '3f02e027299aac3e6c812b8eb06fe1e3b3dd28e7901cc972be1366ff7b2a640d',
    'loopthreequarter': '3c072fb500e808c30d3ba9e2d212fd9f44312524e1910f2a5e4a9c28cb36c319',
}
INDICES = (0,1,2,15,17,20,22,23,24,25,26,27,28)
SIZES = (0x10000,0x10000,0x20000,0x10000,0x20000,0x8000,
         0x8000,0x8000,0x8000,0x8000,0x8000,0x8000,0xc000)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def resource_shift(program):
    return 0x10000 if program=='loop16' else 0x8000 if program in ('loop8','loop15') else 0


def pages(program='loop'):
    raw=bytearray(tensor_pages.serialize('probe'))
    for offset in (0x174,0x1cc,0x1d4,0x1e4,0x1ec,0x1f4,0x1fc):
        old=struct.unpack_from('<Q',raw,offset)[0]
        struct.pack_into('<Q',raw,offset,old+0x10000)
    struct.pack_into('<Q',raw,0x3ec,0x10000035600)
    struct.pack_into('<I',raw,0x3f8,0x42)
    struct.pack_into('<I',raw,0x4000+0xe8,0x20)
    shift=resource_shift(program)
    if shift:
        # Seven command-page resource pointers follow the driver-assigned
        # shift caused by enlarging physical allocation 20.
        for offset in (0x174,0x1cc,0x1d4,0x1e4,0x1ec,0x1f4,0x1fc):
            struct.pack_into('<Q',raw,offset,struct.unpack_from('<Q',raw,offset)[0]+shift)
    result=bytes(raw)
    if sha(result)!=(LOOP16_PAGE_SHA256 if program=='loop16' else
                    LOOP8_PAGE_SHA256 if shift else PAGE_SHA256):
        raise ValueError('tensor loop pages changed')
    return result


def requests(program='loop'):
    source=tensor_requests.serialize('probe')
    rows=[list(scalar_requests.ROW.unpack_from(source,i*scalar_requests.ROW.size))
          for i in range(29)]
    b_size=0x18000 if program=='loop16' else 0x10000 if resource_shift(program) else 0x8000
    rows[20]=[20,0,0,0,0,scalar_requests.BASE+0x80000,b_size,
              scalar_requests.request(0x10001,0x470,b_size,base_buffer=True)]
    rows[21][4]=0x5600
    rows[21][5]=scalar_requests.BASE+0x35600
    for index in range(22,29):
        rows[index][5]+=0x10000+resource_shift(program)
    result=b''.join(scalar_requests.ROW.pack(*row) for row in rows)
    if sha(result)!=(LOOP16_REQUEST_SHA256 if program=='loop16' else
                    LOOP8_REQUEST_SHA256 if resource_shift(program) else REQUEST_SHA256):
        raise ValueError('tensor loop requests changed')
    return result


def allocation2(a,c):
    if len(a)!=2048 or len(c)!=8192:
        raise ValueError('four-trip A/C buffer shape changed')
    raw=bytearray(0x20000)
    scalar=base.allocation(2)
    raw[:0x4d00]=scalar[0xc00:0x5900]
    for offset,size in ((0x4d00,0x80),(0x5580,0x100),(0x7680,0x80)):
        raw[offset:offset+size]=b'\xa5'*size
    raw[0x4d80:0x5580]=a
    raw[0x5680:0x7680]=c
    result=bytes(raw)
    if sha(result)!=ALLOCATION2_SHA256:
        raise ValueError('tensor loop allocation 2 changed')
    return result


def fifth_b_block(b):
    """Append a sign-negated copy of B block 0 in the reserved B aperture."""
    if len(b)!=16384:
        raise ValueError('four-trip B source changed')
    result=b+bytes(value^(0x80 if index&1 else 0)
                   for index,value in enumerate(b[:4096]))
    if sha(result)!=LOOP5_B_SHA256:
        raise ValueError('five-trip B input changed')
    return result


def seventh_b_block(b):
    """Append sign-negated copies of B blocks 0..2 without enlarging B's allocation."""
    if len(b)!=16384:
        raise ValueError('four-trip B source changed')
    result=b+bytes(value^(0x80 if index&1 else 0)
                   for index,value in enumerate(b[:12288]))
    if sha(result)!=LOOP7_B_SHA256:
        raise ValueError('seven-trip B input changed')
    return result


def eighth_b_block(b):
    """Append sign-negated B blocks 0..3 in an enlarged physical allocation."""
    if len(b)!=16384:
        raise ValueError('four-trip B source changed')
    result=b+bytes(value^(0x80 if index&1 else 0)
                   for index,value in enumerate(b))
    if sha(result)!=LOOP8_B_SHA256:
        raise ValueError('eight-trip B input changed')
    return result


def long_b_blocks(b,trips):
    """Repeat sign-negated blocks 0..3 through trip 15 or 16."""
    expected={15:LOOP15_B_SHA256,16:LOOP16_B_SHA256}
    if trips not in expected or len(b)!=16384:
        raise ValueError('long-loop B source or trip count changed')
    neg=bytes(value^(0x80 if index&1 else 0)
              for index,value in enumerate(b))
    result=b+(neg*3)[:(trips-4)*4096]
    if sha(result)!=expected[trips]:
        raise ValueError('long-loop B input changed')
    return result


def allocation20(b,program='loop'):
    if len(b)!=(65536 if program=='loop16' else 61440 if program=='loop15' else
                32768 if program=='loop8' else 28672 if program=='loop7' else 20480 if program=='loop5' else 16384):
        raise ValueError('tensor-loop B buffer shape changed')
    raw=bytearray(0x18000 if program=='loop16' else 0x10000 if resource_shift(program) else 0x8000)
    raw[:0x80]=b'\xa5'*0x80
    raw[0x80:0x80+len(b)]=b
    raw[0x80+len(b):0x100+len(b)]=b'\xa5'*0x80
    result=bytes(raw)
    if sha(result)!=(LOOP16_ALLOCATION20_SHA256 if program=='loop16' else
                    LOOP15_ALLOCATION20_SHA256 if program=='loop15' else
                    LOOP8_ALLOCATION20_SHA256 if program=='loop8' else
                    LOOP7_ALLOCATION20_SHA256 if program=='loop7' else
                    LOOP5_ALLOCATION20_SHA256 if program=='loop5' else ALLOCATION20_SHA256):
        raise ValueError('tensor loop allocation 20 changed')
    return result


def allocations(code,a,b,c,program='loop'):
    expected = (LOOP3_CODE_SHA256 if program=='loop3' else
                LOOP5_CODE_SHA256 if program=='loop5' else
                LOOP7_CODE_SHA256 if program=='loop7' else
                LOOP8_CODE_SHA256 if program=='loop8' else
                LOOP15_CODE_SHA256 if program=='loop15' else
                LOOP16_CODE_SHA256 if program=='loop16' else
                SCALE_CODE_SHA256[program] if program in SCALE_CODE_SHA256 else
                CODE_SHA256.get(len(code)))
    if sha(code)!=expected or (program in ('loop3','loop5','loop7','loop8','loop15','loop16',*SCALE_CODE_SHA256) and len(code)!=5940):
        raise ValueError('exact loop variant or fused-scale program required')
    result={}
    raw=bytearray(block.allocation_zero(prefix_bytes=0x400))
    raw[0x6c0:0x6c0+len(code)]=code
    result[0]=bytes(raw)
    result[1]=base.allocation(1)
    result[2]=allocation2(a,c)
    result[15]=physical.allocation(15)
    result[17]=physical.allocation(17)
    result[20]=allocation20(b,program=program)
    for index in range(22,29):
        raw=bytearray(physical.allocation(index))
        if index==23:
            # Empirical resource-base correlate, +4 per +0x8000 shift;
            # its exact firmware field meaning remains undecoded.
            raw[6]=0x78+resource_shift(program)//0x2000
        elif index==25:
            # Paired correlate, +2 per +0x8000 shift. Omitting this pair
            # caused a no-output GPU recovery in the eight-trip trial.
            raw[9]=0x28+resource_shift(program)//0x4000
        elif index==28:
            struct.pack_into('<Q',raw,0x1ba8,0x10000080080)
            struct.pack_into('<Q',raw,0x1bb0,0x10000035680)
        result[index]=bytes(raw)
    for index,size in zip(INDICES,sizes(program)):
        if len(result[index])!=size:
            raise ValueError(f'tensor loop physical size {index} changed')
    return result


def sizes(program='loop'):
    if resource_shift(program):
        b_size=0x18000 if program=='loop16' else 0x10000
        return tuple(b_size if index==20 else size for index,size in zip(INDICES,SIZES))
    return SIZES


def build(request_path,page_path,payload_path,code,a,b,c,program='loop'):
    req,page,alloc=requests(program),pages(program),allocations(code,a,b,c,program=program)
    payload=b''.join(alloc[i] for i in INDICES)
    if len(payload)!=(802816 if program=='loop16' else 770048 if resource_shift(program) else 737280):
        raise ValueError('tensor loop payload length changed')
    for path,raw in ((request_path,req),(page_path,page),(payload_path,payload)):
        Path(path).write_bytes(raw)
    return {'scope':'field-built tensor loop graph',
            'capture_read_at_build':False,'program':program if program in ('loop3','loop5','loop7','loop8','loop15','loop16',*SCALE_CODE_SHA256) else 'loopscale' if len(code)==5092 else 'loop',
            'requests_sha256':sha(req),'pages_sha256':sha(page),
            'payload_sha256':sha(payload),'payload_bytes':len(payload),
            'physical_allocations':[{'index':i,'size':s,'sha256':sha(alloc[i])}
                                    for i,s in zip(INDICES,sizes(program))]}
