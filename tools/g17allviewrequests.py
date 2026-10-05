#!/usr/bin/env python3
"""Normalize the 17 captured selector-9 views with nonzero GPU apertures."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[1]
CAPTURE=ROOT/'evidence/g17-belowmetal-preflight/authored-pair-metal/passive-submit-with-requests.json'
INDICES=tuple(range(3,18))+(19,21)
ROW=struct.Struct('<IIQQQ104s')

def build(path):
    raw=CAPTURE.read_bytes();runs=json.loads(raw)['runs']
    if [r['bias'] for r in runs]!=[1,2]:raise ValueError('capture is not A/B')
    events=[]
    for run in runs:
        found=[m['payload'] for m in run['messages']
               if m.get('payload',{}).get('kind')=='submit_enter']
        if not run['passed'] or len(found)!=1:raise ValueError('capture did not complete')
        rows=found[0]['allocationRequests']
        if len(rows)!=29:raise ValueError('allocation count')
        events.append(rows)
    data=bytearray();manifest=[]
    for index in INDICES:
        normalized=[];offsets=[]
        for rows in events:
            record=bytearray.fromhex(rows[index]['input_hex'])
            if len(record)!=104 or struct.unpack_from('<Q',record,0)[0]!=0x80:
                raise ValueError(f'view {index} shape')
            q7,q8=struct.unpack_from('<QQ',record,56)
            if not q8 or q7<q8:raise ValueError(f'view {index} CPU base')
            offsets.append(q7-q8)
            struct.pack_into('<QQ',record,56,0,0)
            struct.pack_into('<Q',record,96,0)
            normalized.append(bytes(record))
        if offsets[0]!=offsets[1] or normalized[0]!=normalized[1]:
            raise ValueError(f'view {index} not stable after pointer normalization')
        aperture=[int(rows[index]['output_words'][0]) for rows in events]
        size=[int(rows[index]['output_words'][5]) for rows in events]
        if aperture[0]!=aperture[1] or size!=[0x20000]*2 or aperture[0]==0:
            raise ValueError(f'view {index} output changed')
        base_slot=3 if index==21 else 2
        data+=ROW.pack(index,base_slot,offsets[0],aperture[0],size[0],normalized[0])
        manifest.append({'index':index,'base_slot':base_slot,
                         'offset':offsets[0],'aperture':f'{aperture[0]:#x}'})
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    result={'scope':'CPU-only 17-view selector-9 templates, no Submit',
            'capture_sha256':hashlib.sha256(raw).hexdigest(),
            'template_sha256':hashlib.sha256(data).hexdigest(),
            'template_bytes':len(data),'row_bytes':ROW.size,'rows':manifest}
    path.with_name(path.name+'.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(ap.parse_args().output),indent=2))
