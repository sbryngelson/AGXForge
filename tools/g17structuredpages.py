#!/usr/bin/env python3
"""Serialize the measured first/second graph command pages without capture I/O.

Resource pointers and handle slots are measured fields. The other nonzero
words below are retained constants whose full semantics are NOT decoded.
Trace IDs remain zero and ready remains clear; the native process fills them.
This is a bounded serializer for the measured layout, not a general encoder.
"""
import hashlib
import json
from pathlib import Path
import struct


KERNEL_CONSTANTS = {
    0x000:0x0f, 0x004:0xac, 0x0ac:0x10000, 0x0b0:0x358,
    0x134:4, 0x144:0x18, 0x150:0x268, 0x154:3, 0x168:1,
    0x204:1, 0x214:2, 0x218:0x1c, 0x240:0xffffffff,
    0x2c4:0x010b0000, 0x2cc:0x02820000, 0x2d4:0x1ea,
    0x2ec:0x54b, 0x304:0xf2, 0x3a8:8,
    0x3c8:0xffffffff, 0x3cc:0xffffffff, 0x3d0:0xffffffff,
    0x3d4:2, 0x3e0:1, 0x3e4:1, 0x3f4:6, 0x3f8:8,
    0x3fc:3, 0x400:0x01000064,
}
SEGMENT_CONSTANTS = {
    0x008:1, 0x00c:0x40000001, 0x014:0xac, 0x020:1,
    0x024:0xf0, 0x030:0xac, 0x034:0x404, 0x040:0x10,
    0x044:3, 0x078:0x10001, 0x07c:0x10001, 0x080:0x10001,
    0x084:0x60000, 0x088:1, 0x090:0x13, 0x094:0x16,
    0x098:0x0f, 0x09c:0x0e, 0x0a0:0x40, 0x0a4:0x30,
    0x0a8:0x40, 0x0ac:0x80, 0x0b0:0x80, 0x0b4:0x80,
    0x0b8:0x10080, 0x0bc:0x800080, 0x0c0:0x800080,
    0x0c4:0x60000, 0x0c8:2, 0x0cc:4, 0x0d0:5, 0x0d4:6,
    0x0e0:0x40, 0x0e4:0x80, 0x0e8:0x80, 0x0ec:0x80,
    0x0f8:0x10080, 0x0fc:0x30001, 0x104:0x40000,
}


def serialize(graph=0):
    if graph not in (0,1):
        raise ValueError('only the measured first and second graph layouts are supported')
    raw=bytearray(0x8000)
    for base,constants in ((0,KERNEL_CONSTANTS),(0x4000,SEGMENT_CONSTANTS)):
        for offset,value in constants.items():
            struct.pack_into('<I',raw,base+offset,value)
    # Seven resources per graph; the measured aperture stride is 64 KiB.
    first_index=22+7*graph
    first_aperture=0x10000080000+7*0x10000*graph
    command_stream=first_aperture+3*0x10000
    bindings=first_aperture+6*0x10000
    pointers={0x174:command_stream,0x1cc:bindings,0x1d4:command_stream+0x2c,
              0x20c:0x10000031100,0x3ec:0x10000030800}
    for j in range(4):
        pointers[0x1e4+8*j]=bindings+0x1b80+8*j
    for offset,value in pointers.items():
        struct.pack_into('<Q',raw,offset,value)
    # Preserve slot order and per-slot words; no resource reordering is inferred.
    for j in range(6):
        struct.pack_into('<I',raw,0x4048+4*j,first_index+1+j)
        struct.pack_into('<I',raw,0x4060+4*j,0x20)
    struct.pack_into('<I',raw,0x408c,first_index+7)
    return bytes(raw)


def build(path,graph=0):
    raw=serialize(graph)
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(raw)
    manifest={'scope':'structured serialization of measured command-page layout; no Submit',
              'builder':'g17structuredpages','graph':graph,
              'template_bytes':len(raw),'template_sha256':hashlib.sha256(raw).hexdigest(),
              'capture_read_at_build':False,'undecoded_constants_retained':True,
              'kernel_trace_low32_offset':'0x234',
              'segment_trace_qword_offsets':['0x0','0x18','0x28'],
              'segment_ready_word_offset':'0x24','ready_word_staged':'0x000000f0'}
    path.with_name(path.name+'.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--graph',type=int,choices=(0,1),default=0)
    args=parser.parse_args()
    print(json.dumps(build(args.output,args.graph),indent=2))
