"""Recover exact previously measured large graph surfaces; no GPU admission."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import g17tensor16320unique as prior
import g17structuredrequests as rowspec


def recover(mib):
    if mib not in (128,1024):raise ValueError('refused: inference resource graph requires measured 128 MiB or 1 GiB class')
    trips=2040 if mib==128 else 16320
    pin=prior.VARIANTS[trips]
    shift=pin['b_size']-prior.parent.B_SIZE
    raw=prior.parent.requests()
    rows=[list(rowspec.ROW.unpack_from(raw,i*rowspec.ROW.size)) for i in range(29)]
    rows[20][6]=pin['b_size']
    rows[20][7]=rowspec.request(0x10001,0x470,pin['b_size'],base_buffer=True)
    for i in range(21,29):rows[i][5]+=shift
    requests=b''.join(rowspec.ROW.pack(*r) for r in rows)
    pages=bytearray(prior.parent.pages())
    for offset in prior.parent.PAGE_POINTERS:
        struct.pack_into('<Q',pages,offset,struct.unpack_from('<Q',pages,offset)[0]+shift)
    pages=bytes(pages)
    for data,key in ((requests,'request'),(pages,'page')):
        if hashlib.sha256(data).hexdigest()!=pin[key]:raise ValueError('previously measured '+key+' identity changed')
    # These are the two specific previously fired graph classes, not a formula
    # extending their address fields to an arbitrary allocation or location.
    report=dict(format='g17-inference-resource-surfaces-v1',status='prior_graph_recovered_model_launch_not_admitted',
                gpu_admitted=False,gpu_dispatched=False,mib=mib,
                requests_sha256=pin['request'],pages_sha256=pin['page'],
                rows=[dict(index=r[0],kind=r[1],parent=r[2],view_offset=r[4],gpu_address=r[5],bytes=r[6]) for r in rows],
                binding_metadata=dict(source_allocation=28,destination_allocation=17,destination_offset=0,bytes=0xc000,
                                      shader_allocation=23,coordinate_offset=6,coordinate=0x002c),
                shader_record=dict(source_allocation=23,destination_allocation=17 if mib==1024 else 23,
                                   destination_offset=0x10000 if mib==1024 else 0,bytes=0x8000,
                                   selector_allocation=25,coordinate_offset=9,coordinate=0x001a if mib==1024 else 0x2048),
                remaining=['model program/grid/binding admission','writable activation placement','persistent complete-model schedule'])
    return requests,pages,report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mib',type=int);p.add_argument('destination',type=Path)
    a=p.parse_args();request,page,report=recover(a.mib)
    a.destination.mkdir(parents=True,exist_ok=False)
    (a.destination/'ordered.bin').write_bytes(request);(a.destination/'pages.bin').write_bytes(page)
    (a.destination/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','mib','requests_sha256','pages_sha256','gpu_admitted')}))
