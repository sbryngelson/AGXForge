#!/usr/bin/env python3
"""Construct the measured physical payload without capture or prefix fixtures.

Allocation 0 is an explicit 686-byte prefix followed by zeroes. Its retained
words are observations of one passing pure-process configuration, not a
general encoding of AGX allocation metadata.
"""
import hashlib
import json
from pathlib import Path
import struct

import g17structuredbasepayload as base
import g17structuredgraphpayload as graph
import g17structuredrequests as requests


ALLOCATION0_BYTES = 0x10000
PREFIX_BYTES = 0x2ae
PAYLOAD_BYTES = 638976
PAYLOAD_SHA256 = 'aba0556d5032bdada17b9130382f295d923702630af396b6f4d60394e9cca668'
ALLOCATION0_SHA256 = '93b903e0930fce3005e2774da0bfc3586a868047a29d76fc74e21dd424593fd2'
PHYSICAL_INDICES = (0, 1, 2, 20, 22, 23, 24, 25, 26, 27, 28)
DISPLACEMENT_LOW_BYTES = (0x33, 0x31, 0x31, 0x1b, 0x1b,
                          0x19, 0x19, 0x13, 0x13, 0x11)
TRAILER = bytes.fromhex('0700100f0000270004002000a50a')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def allocation_zero():
    """Return the observed allocation-0 prefix and zero tail, generated anew."""
    raw = bytearray(ALLOCATION0_BYTES)
    struct.pack_into('<I', raw, 0, 0x640)
    for offset in range(0x40, 0x200, 2):
        struct.pack_into('<H', raw, offset, 6)
    for index, low in enumerate(DISPLACEMENT_LOW_BYTES):
        offset = 0x200 + 16 * index
        # Only the low byte of this 0x0eXX word varies in these ten records.
        struct.pack_into('<HH', raw, offset,
                         0x005e | (0x4000 if index & 1 else 0), 0x0e00 | low)
        for field in (10, 12, 14):
            struct.pack_into('<H', raw, offset + field, 6)
    raw[0x2a0:PREFIX_BYTES] = TRAILER
    if sha(raw) != ALLOCATION0_SHA256:
        raise ValueError('authored allocation-0 bytes changed')
    return bytes(raw)


def build(path):
    """Write the full measured physical payload and provenance manifest."""
    path = Path(path)
    rows = requests.physical_requests()
    if tuple(row[0] for row in rows) != PHYSICAL_INDICES:
        raise ValueError('measured physical allocation order changed')
    raw = bytearray()
    manifest = []
    for index, _, aperture, size in rows:
        if index == 0:
            allocation = allocation_zero()
            source = 'g17authoredalloc0payload'
        elif index in (1, 2):
            allocation = base.allocation(index)
            source = 'g17structuredbasepayload'
        else:
            allocation = graph.allocation(index)
            source = 'g17structuredgraphpayload'
        if len(allocation) != size:
            raise ValueError(f'allocation {index} has unexpected size')
        raw.extend(allocation)
        manifest.append({'original_index': index, 'aperture': hex(aperture),
                         'size': size, 'sha256': sha(allocation), 'source': source})
    if len(raw) != PAYLOAD_BYTES or sha(raw) != PAYLOAD_SHA256:
        raise ValueError('authored physical payload bytes changed')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    result = {
        'scope': 'measured pure-process physical payload with constructed allocation 0; no Submit',
        'capture_read_at_build': False,
        'fixture_read_at_build': False,
        'capture_allocation_indices': [],
        'fixture_bytes': 0,
        'allocation0_authored_prefix_bytes': PREFIX_BYTES,
        'allocation0_zero_tail_bytes': ALLOCATION0_BYTES - PREFIX_BYTES,
        'structured_allocation_indices': list(PHYSICAL_INDICES),
        'source_sha256': {
            'g17authoredalloc0payload': sha(Path(__file__).read_bytes()),
            'g17structuredbasepayload': sha(Path(base.__file__).read_bytes()),
            'g17structuredgraphpayload': sha(Path(graph.__file__).read_bytes()),
            'g17structuredrequests': sha(Path(requests.__file__).read_bytes()),
        },
        'payload_sha256': sha(raw),
        'payload_bytes': len(raw),
        'allocations': manifest,
    }
    path.with_name(path.name + '.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output), indent=2))
