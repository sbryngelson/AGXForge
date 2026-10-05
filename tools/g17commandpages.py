#!/usr/bin/env python3
"""Build a no-Submit selector-14 page template from authored A/B captures.

Output is kernel then segment, 16 KiB each. Process-specific trace IDs are
zeroed, and the segment ready bit is cleared. A pure-process preflight must
insert its own trace IDs before CPU staging. This is not a valid submission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[1]
CAPTURE=ROOT/'evidence/g17-belowmetal-preflight/authored-pair-metal/passive-submit-with-requests.json'


def event(run):
    found=[m['payload'] for m in run['messages']
           if m.get('payload',{}).get('kind')=='submit_enter']
    if not run['passed'] or len(found)!=1:raise ValueError('capture must have one passing Submit')
    return found[0]


def template(e):
    sh=e['snapshot']['shmems']
    k=bytearray.fromhex(sh['kernel']['hex']);s=bytearray.fromhex(sh['segment']['hex'])
    if len(k)!=0x4000 or len(s)!=0x4000 or (sh['kernel']['id'],sh['segment']['id'])!=(2,1):
        raise ValueError('unexpected selector-14 page size or IDs')
    first=struct.unpack_from('<Q',s,0)[0];second=first+1
    if (struct.unpack_from('<Q',s,0x18)[0]!=first or
        struct.unpack_from('<Q',s,0x28)[0]!=second or
        struct.unpack_from('<I',k,0x234)[0]!=(second&0xffffffff) or
        struct.unpack_from('<I',k,0x230)[0]!=0 or
        struct.unpack_from('<I',s,0x24)[0]!=0x800000f0):
        raise ValueError('trace-ID or ready-word layout changed')
    for offset in (0,0x18,0x28):struct.pack_into('<Q',s,offset,0)
    struct.pack_into('<I',k,0x234,0)
    struct.pack_into('<I',s,0x24,0xf0)
    return bytes(k+s),first


def build(path):
    raw=CAPTURE.read_bytes();runs=json.loads(raw)['runs']
    if [r['bias'] for r in runs]!=[1,2]:raise ValueError('capture is not authored A/B')
    a,aid=template(event(runs[0]));b,bid=template(event(runs[1]))
    if a!=b:raise ValueError('A/B selector-14 pages differ beyond trace IDs')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(a)
    manifest=dict(scope='CPU selector-14 staging template; no Submit or GPU execution',
                  capture_sha256=hashlib.sha256(raw).hexdigest(),
                  template_sha256=hashlib.sha256(a).hexdigest(),template_bytes=len(a),
                  captured_trace_ids=[hex(aid),hex(bid)],
                  kernel_trace_low32_offset='0x234',
                  segment_trace_qword_offsets=['0x0','0x18','0x28'],
                  segment_ready_word_offset='0x24',ready_word_staged='0x000000f0')
    path.with_name(path.name+'.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(ap.parse_args().output),indent=2))
