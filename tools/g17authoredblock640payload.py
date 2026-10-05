#!/usr/bin/env python3
"""Construct the observed allocation-0 first 0x640 bytes without fixture reads.

The repeated 0x300..0x5ff records are measured instruction-shaped words;
their semantic role beyond the bounded launch remains to be determined.
"""
import hashlib
from pathlib import Path
import struct

import g17authoredalloc0payload as authored
import g17authoredpaddedpayload as padded
import g17structuredbasepayload as base
import g17structuredgraphpayload as graph
import g17structuredrequests as requests


PREFIX_BYTES = 0x640
ALLOCATION0_SHA256 = '95641bc00801d79c6b2488c635fd2885611730e726b8ad63f561565c71e81053'
PAYLOAD_SHA256 = 'c07d32b850ce0dfa51bbbc15a32755a154c227e77d68c307c2caa784a638590a'
SHORT_PREFIX_BYTES = 0x3a0
SHORT_ALLOCATION0_SHA256 = '942237f01e32a620968ab1c71c2b1c1c16c647a52db5b289d5810b23ad1e8a28'
SHORT_PAYLOAD_SHA256 = 'a45428daa1135c44dcd018e62a00c30069f54f6d7e42f8a00689c40d4cd384d9'
BLOCK_PREFIX_BYTES = 0x400
BLOCK_ALLOCATION0_SHA256 = 'c235c24a740b6d3290aa5678b251a1efd91f50e32c35d2e628027d995c86ce38'
BLOCK_PAYLOAD_SHA256 = '1f5f8c3a3c1c2c880cc7d7aae94e93a97559de02814572c6f032c6da0e0955c4'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def allocation_zero(prefix_bytes=PREFIX_BYTES):
    if prefix_bytes not in (PREFIX_BYTES, BLOCK_PREFIX_BYTES, SHORT_PREFIX_BYTES):
        raise ValueError('unsupported first-block prefix')
    raw = bytearray(padded.allocation_zero())
    for block in (0x300, 0x400, 0x500):
        for index, low in enumerate(authored.DISPLACEMENT_LOW_BYTES):
            offset = block + 16 * index
            struct.pack_into('<HH', raw, offset,
                             0x005e | (0x4000 if index & 1 else 0), 0x0e00 | low)
            for field in (10, 12, 14):
                struct.pack_into('<H', raw, offset + field, 6)
        for offset in (block + 0xa0, block + 0xb0):
            raw[offset:offset+14] = authored.TRAILER
            struct.pack_into('<H', raw, offset+14, 6)
        for offset in range(block + 0xc0, block + 0x100, 2):
            struct.pack_into('<H', raw, offset, 6)
    for offset in range(0x600, PREFIX_BYTES, 2):
        struct.pack_into('<H', raw, offset, 6)
    if prefix_bytes != PREFIX_BYTES:
        raw[prefix_bytes:] = bytes(len(raw) - prefix_bytes)
    expected = {SHORT_PREFIX_BYTES: SHORT_ALLOCATION0_SHA256,
                BLOCK_PREFIX_BYTES: BLOCK_ALLOCATION0_SHA256,
                PREFIX_BYTES: ALLOCATION0_SHA256}[prefix_bytes]
    if sha(raw) != expected:
        raise ValueError('constructed allocation-0 bytes changed')
    return bytes(raw)


def build(path, prefix_bytes=PREFIX_BYTES):
    """Write all physical allocations using field constructors only."""
    path = Path(path)
    raw = bytearray()
    manifest = []
    for index, _, aperture, size in requests.physical_requests():
        allocation = (allocation_zero(prefix_bytes) if index == 0 else
                      base.allocation(index) if index in (1, 2) else
                      graph.allocation(index))
        if len(allocation) != size:
            raise ValueError(f'allocation {index} has unexpected size')
        raw.extend(allocation)
        manifest.append({'original_index': index, 'aperture': hex(aperture),
                         'size': size, 'sha256': sha(allocation)})
    expected = {SHORT_PREFIX_BYTES: SHORT_PAYLOAD_SHA256,
                BLOCK_PREFIX_BYTES: BLOCK_PAYLOAD_SHA256,
                PREFIX_BYTES: PAYLOAD_SHA256}[prefix_bytes]
    if len(raw) != 638976 or sha(raw) != expected:
        raise ValueError('constructed physical payload bytes changed')
    path.write_bytes(raw)
    return {'scope': 'constructed first-block allocation-0 payload; no Submit',
            'capture_read_at_build': False, 'fixture_read_at_build': False,
            'capture_allocation_indices': [], 'fixture_bytes': 0,
            'allocation0_authored_prefix_bytes': prefix_bytes,
            'allocation0_zero_tail_bytes': 0x10000 - prefix_bytes,
            'payload_sha256': sha(raw), 'payload_bytes': len(raw),
            'allocations': manifest,
            'source_sha256': {
                'g17authoredblock640payload': sha(Path(__file__).read_bytes()),
                'g17authoredpaddedpayload': sha(Path(padded.__file__).read_bytes()),
                'g17authoredalloc0payload': sha(Path(authored.__file__).read_bytes()),
                'g17structuredbasepayload': sha(Path(base.__file__).read_bytes()),
                'g17structuredgraphpayload': sha(Path(graph.__file__).read_bytes()),
                'g17structuredrequests': sha(Path(requests.__file__).read_bytes()),
            }}
