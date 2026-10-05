#!/usr/bin/env python3
"""Build measured tensor physical allocations except input/control allocation 2.

Allocation 2 is supplied by the caller. This builder reads no Metal capture;
it reuses bounded scalar allocation constructors and explicit tensor deltas.
"""
import hashlib
import struct

import g17authoredblock640payload as block
import g17structuredbasepayload as base
import g17structuredgraphpayload as graph


INDICES = (0, 1, 2, 15, 17, 22, 23, 24, 25, 26, 27, 28)
SIZES = (0x10000, 0x10000, 0x20000, 0x10000, 0x20000,
         0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0xc000)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def allocation(index, code=None, allocation2=None, *, program='probe'):
    if program not in ('probe','common17x19'):
        raise ValueError('unsupported tensor physical program')
    if index == 0:
        expected_code_bytes = 1232 if program == 'common17x19' else 648
        if code is None or len(code) != expected_code_bytes:
            raise ValueError(f'exact {expected_code_bytes}-byte tensor program required')
        raw = bytearray(block.allocation_zero(prefix_bytes=0x400))
        raw[0x6c0:0x6c0 + len(code)] = code
    elif index == 1:
        raw = bytearray(base.allocation(1))
    elif index == 2:
        if allocation2 is None or len(allocation2) != 0x20000:
            raise ValueError('exact allocation-2 window required')
        raw = bytearray(allocation2)
    elif index == 15:
        raw = bytearray(0x10000)
        for offset in (0, 0x80, 0x100):
            raw[offset] = 0x80
            raw[offset + 0x42:offset + 0x44] = b'\x10\x10'
        raw[0x180:0x182] = b'\x40\xfe'
    elif index == 17:
        raw = bytearray(0x20000)
    elif 22 <= index <= 28:
        raw = bytearray(graph.allocation(index))
        if index == 22:
            struct.pack_into('<QQ', raw, 0x50, 0x10000031a00, 0x10000031900)
            struct.pack_into('<I', raw, 0xa8, 32)
            struct.pack_into('<I', raw, 0xb4, 1)
        elif index == 23:
            struct.pack_into('<H', raw, 0x42, 0x0e58)
        elif index == 25:
            struct.pack_into('<I', raw, 0, 0x00880000)
            struct.pack_into('<I', raw, 0x10, 32)
            struct.pack_into('<I', raw, 0x1c, 32)
        elif index == 28:
            pointers = (0x10000034d80,
                        0x10000035180 if program == 'common17x19' else 0x10000035080,
                        0x10000035580 if program == 'common17x19' else 0x10000035380)
            for j, value in enumerate(pointers):
                struct.pack_into('<Q', raw, 0x1ba0 + 8*j, value)
    else:
        raise ValueError(f'unsupported tensor allocation {index}')
    expected_size = SIZES[INDICES.index(index)]
    if len(raw) != expected_size:
        raise ValueError(f'allocation {index} size {len(raw)} != {expected_size}')
    return bytes(raw)


def serialize(code, allocation2, *, program='probe'):
    parts = [allocation(index, code, allocation2, program=program) for index in INDICES]
    result = b''.join(parts)
    if len(result) != 704512:
        raise ValueError('tensor physical graph size changed')
    return result


def describe(code, allocation2, *, allocation2_authored=False, program='probe'):
    return {'scope':('field-built complete tensor physical graph' if allocation2_authored else
                     'field-built physical allocations except captured allocation 2'),
            'capture_read_at_build':False,
            'program':program,
            'captured_allocation_indices':[] if allocation2_authored else [2],
            'authored_allocation_indices':[i for i in INDICES if allocation2_authored or i!=2],
            'payload_bytes':704512,
            'payload_sha256':sha(serialize(code,allocation2,program=program))}
