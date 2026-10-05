#!/usr/bin/env python3
"""Construct the measured 29-request allocation layout without capture reads.

Call order, expected apertures, request flags and opaque words are retained
observations. This is not an allocator for arbitrary sizes or memory types.
View CPU pointers and class pointers remain zero for native reconstruction.
"""
import hashlib
import json
from pathlib import Path
import struct


ROW=struct.Struct('<IIIIQQQ104s')
BASE=0x10000000000
VIEW_OFFSETS=(0,0x400,0x800,0xc00,0xe00,0x1000,0x1100,0x1300,
              0x1400,0x1500,0x2500,0x2600,0x2700,0x2800,0x3800)


def request(word08,word14,length,*,word58=0,word5c=0,parent=None,base_buffer=False):
    data=bytearray(104)
    for offset,value in ((8,word08),(12,1),(16,0x1000101),(20,word14),
                          (0x58,word58),(0x5c,word5c)):
        struct.pack_into('<I',data,offset,value)
    struct.pack_into('<Q',data,0x48,length)
    if base_buffer:
        struct.pack_into('<I',data,0x30,1)
    if parent is not None:
        struct.pack_into('<I',data,0,0x80)
        struct.pack_into('<I',data,0x50,parent+1)
    return bytes(data)


def rows():
    result=[]
    def add(index,kind,parent,offset,aperture,size,data):
        assert index==len(result)
        result.append((index,kind,parent,0,offset,aperture,size,data))
    for index,offset in ((0,0),(1,0x18000)):
        add(index,0,0,0,BASE+offset,0x10000,
            request(0x10001,0x430,0x10000,word58=0x8000000,word5c=0x18))
    add(2,0,0,0,BASE+0x30000,0x20000,request(0x10001,0x470,0x20000,base_buffer=True))
    for index,offset in enumerate(VIEW_OFFSETS,3):
        add(index,1,2,offset,BASE+0x30000+offset,0x20000,
            request(0x10001,0xc30,0x20000,parent=2,word5c=0 if index<6 else 0x18))
    add(18,2,0,0,0,0x10000,request(0x10001,0x8430,0x10000,word58=0x38000000,word5c=0x18))
    add(19,1,2,0x3900,BASE+0x33900,0x20000,
        request(0x10001,0xc30,0x20000,parent=2,word5c=0x18))
    add(20,0,0,0,BASE+0x58000,0x20000,request(0x10001,0x470,0x20000,base_buffer=True))
    add(21,1,20,0,BASE+0x58000,0x20000,
        request(0x10001,0xc30,0x20000,parent=20,word5c=0x18))
    for index in range(22,29):
        length=0x9b80 if index==28 else 0x8000
        add(index,0,0,0,BASE+0x80000+(index-22)*0x10000,
            0xc000 if index==28 else 0x8000,
            request(0x10000+length,0x4430,length,
                    word58=0x18000000 if index in (23,28) else 0x8000000,
                    word5c={22:0x48,25:0x40}.get(index,0)))
    assert len(result)==29
    return result


def serialize():
    return b''.join(ROW.pack(*row) for row in rows())


def physical_requests():
    return [(i,data,aperture,size) for i,kind,_,_,_,aperture,size,data in rows() if kind==0]


def build(path):
    raw=serialize();path=Path(path)
    manifest_rows=[{'index':i,'kind':k,'parent':parent,'offset':offset,
                    'aperture':hex(aperture),'size':size}
                   for i,k,parent,_,offset,aperture,size,_ in rows()]
    result={'scope':'structured measured selector-9 request sequence; no Submit',
            'builder':'g17structuredrequests','capture_read_at_build':False,
            'undecoded_request_words_retained':True,
            'template_sha256':hashlib.sha256(raw).hexdigest(),
            'template_bytes':len(raw),'row_bytes':ROW.size,'rows':manifest_rows}
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    path.with_name(path.name+'.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(parser.parse_args().output),indent=2))
