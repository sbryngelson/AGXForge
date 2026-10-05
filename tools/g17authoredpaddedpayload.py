#!/usr/bin/env python3
"""Build the measured physical payload with a generated 0x300-byte prefix.

The first 0x2ae bytes come from the explicit allocation-0 constructor. The
following bytes reproduce the observed NOP, duplicate 14-byte trailer, and
NOP padding through 0x300. The rest of allocation 0 is zero. This is one
measured configuration, not an encoder for arbitrary AGX metadata.
"""
import hashlib
import json
from pathlib import Path
import struct

import g17authoredalloc0payload as authored
import g17structuredbasepayload as base
import g17structuredgraphpayload as graph
import g17structuredrequests as requests


PREFIX_BYTES = 0x300
ALLOCATION0_BYTES = 0x10000
ALLOCATION0_SHA256 = 'fc6f6cfe8643aadbad68237c31a9b17196e1a7068bf7bb53998e1647f0ce12bc'
PAYLOAD_BYTES = 638976
PAYLOAD_SHA256 = 'e36e52a50c825b215636c10efe314cc337f58ad71b11767bdf8330b294d291b8'
PHYSICAL_INDICES = (0, 1, 2, 20, 22, 23, 24, 25, 26, 27, 28)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def allocation_zero():
    raw = bytearray(authored.allocation_zero())
    if len(raw) != ALLOCATION0_BYTES or any(raw[authored.PREFIX_BYTES:]):
        raise ValueError('base authored allocation-0 tail changed')
    struct.pack_into('<H', raw, 0x2ae, 6)
    raw[0x2b0:0x2be] = authored.TRAILER
    for offset in range(0x2be, PREFIX_BYTES, 2):
        struct.pack_into('<H', raw, offset, 6)
    if sha(raw) != ALLOCATION0_SHA256:
        raise ValueError('padded allocation-0 bytes changed')
    return bytes(raw)


def build(path):
    """Write the full fixture-free physical payload and provenance manifest."""
    path = Path(path)
    rows = requests.physical_requests()
    if tuple(row[0] for row in rows) != PHYSICAL_INDICES:
        raise ValueError('measured physical allocation order changed')
    raw = bytearray()
    manifest = []
    for index, _, aperture, size in rows:
        if index == 0:
            allocation = allocation_zero()
            source = 'g17authoredpaddedpayload'
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
        raise ValueError('padded physical payload bytes changed')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    result = {
        'scope': 'measured pure-process physical payload with generated padded allocation 0; no Submit',
        'capture_read_at_build': False,
        'fixture_read_at_build': False,
        'capture_allocation_indices': [],
        'fixture_bytes': 0,
        'allocation0_authored_prefix_bytes': PREFIX_BYTES,
        'allocation0_zero_tail_bytes': ALLOCATION0_BYTES - PREFIX_BYTES,
        'structured_allocation_indices': list(PHYSICAL_INDICES),
        'source_sha256': {
            'g17authoredpaddedpayload': sha(Path(__file__).read_bytes()),
            'g17authoredalloc0payload': sha(Path(authored.__file__).read_bytes()),
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
