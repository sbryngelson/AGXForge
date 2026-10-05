#!/usr/bin/env python3
"""Build selected selector-9 views with process-local pointers removed."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[1]
CAPTURE=ROOT/'evidence/g17-belowmetal-preflight/authored-pair-metal/passive-submit-with-requests.json'


SPECS={5:(0x800,0x10000030800),9:(0x1100,0x10000031100)}

def build(path,index=5):
    if index not in SPECS:raise ValueError('unsupported view index')
    raw=CAPTURE.read_bytes();report=json.loads(raw)
    runs=report['runs']
    if [r['bias'] for r in runs]!=[1,2]:raise ValueError('capture is not A/B')
    rows=[]
    for run in runs:
        events=[m['payload'] for m in run['messages']
                if m.get('payload',{}).get('kind')=='submit_enter']
        if not run['passed'] or len(events)!=1:raise ValueError('capture did not complete')
        requests=events[0]['allocationRequests']
        if len(requests)!=29:raise ValueError('allocation count')
        row=requests[index]
        data=bytearray.fromhex(row['input_hex'])
        if len(data)!=104 or struct.unpack_from('<Q',data,0)[0]!=0x80 or \
           struct.unpack_from('<Q',data,80)[0]!=3 or \
           (index==5 and struct.unpack_from('<Q',data,96)[0]!=0) or \
           (index==9 and struct.unpack_from('<Q',data,96)[0]==0):
            raise ValueError('view #5 shape changed')
        base=struct.unpack_from('<Q',data,64)[0]
        offset=struct.unpack_from('<Q',data,56)[0]-base
        if offset!=SPECS[index][0]:raise ValueError('view offset changed')
        aperture=int(row['output_words'][0]);size=int(row['output_words'][5])
        if (aperture,size)!=(SPECS[index][1],0x20000):raise ValueError('view output changed')
        struct.pack_into('<QQ',data,56,0,0)
        if index==9:struct.pack_into('<Q',data,96,0)
        rows.append(bytes(data))
    if rows[0]!=rows[1]:raise ValueError('normalized views differ')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(rows[0])
    result={'scope':f'CPU-only selector-9 view #{index} template, no Submit',
            'view_index':index,
            'capture_sha256':hashlib.sha256(raw).hexdigest(),
            'template_sha256':hashlib.sha256(rows[0]).hexdigest(),
            'template_bytes':104,'view_offset':SPECS[index][0],
            'expected_aperture':f'{SPECS[index][1]:#x}','expected_size':0x20000}
    path.with_name(path.name+'.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--index',type=int,choices=tuple(SPECS),default=5)
    args=ap.parse_args()
    print(json.dumps(build(args.output,args.index),indent=2))
