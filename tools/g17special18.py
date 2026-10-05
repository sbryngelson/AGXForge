#!/usr/bin/env python3
"""Extract the authored A/B selector-9 request #18 with no GPU aperture."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CAPTURE=ROOT/'evidence/g17-belowmetal-preflight/authored-pair-metal/passive-submit-with-requests.json'

def build(path):
    raw=CAPTURE.read_bytes();runs=json.loads(raw)['runs']
    if [r['bias'] for r in runs]!=[1,2]:raise ValueError('capture is not A/B')
    rows=[]
    for run in runs:
        found=[m['payload'] for m in run['messages']
               if m.get('payload',{}).get('kind')=='submit_enter']
        if not run['passed'] or len(found)!=1:raise ValueError('capture did not complete')
        row=found[0]['allocationRequests'][18]
        data=bytes.fromhex(row['input_hex'])
        if len(data)!=104 or int(row['output_words'][0])!=0 or int(row['output_words'][5])!=0x10000:
            raise ValueError('special request shape changed')
        rows.append(data)
    if rows[0]!=rows[1]:raise ValueError('special request differs across A/B')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(rows[0])
    result={'scope':'CPU-only selector-9 request #18, no GPU aperture or Submit',
            'capture_sha256':hashlib.sha256(raw).hexdigest(),
            'template_sha256':hashlib.sha256(rows[0]).hexdigest(),
            'template_bytes':len(rows[0])}
    path.with_name(path.name+'.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(ap.parse_args().output),indent=2))
