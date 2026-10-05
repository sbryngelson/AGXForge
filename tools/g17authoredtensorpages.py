#!/usr/bin/env python3
"""Build the measured single-MMA command pages from explicit fields.

This bounded serializer does not read a Metal capture. Unknown scalar-layout
constants are inherited from g17structuredpages; the tensor-specific words
are explicit below. A pinned hash detects accidental drift in the result.
"""
import hashlib
from pathlib import Path
import struct

import g17structuredpages


EXPECTED_SHA256 = 'be051b317d3c5ebfdd36b87a61f97c6777a3fa3a2f9af3aee719ecdbd8608093'
COMMON_SHA256 = 'd8fce6179acd87cae491a0d204d70b8be18d548980dfce8aee4329b37374c213'
KERNEL_U32 = {
    0x3c0: 0x01000000,  # causal for MMA contribution in this measured layout
    0x3f4: 0x16,
    0x3f8: 0x0a,
}
KERNEL_U64 = {
    0x20c: 0x10000030500,
    0x3ec: 0x10000035300,
}
SEGMENT_U32 = {
    0x090: 0x10,
    0x094: 0x13,
    0x098: 0x0c,
    0x09c: 0x0b,
    0x0cc: 0x14,
    0x0d0: 0x15,
    0x0d4: 0x16,
}


def serialize(program='probe'):
    if program not in ('probe','common17x19'):
        raise ValueError('unsupported tensor command-page program')
    raw = bytearray(g17structuredpages.serialize(0))
    for offset, value in KERNEL_U32.items():
        struct.pack_into('<I', raw, offset, value)
    for offset, value in KERNEL_U64.items():
        struct.pack_into('<Q', raw, offset, value)
    for offset, value in SEGMENT_U32.items():
        struct.pack_into('<I', raw, 0x4000 + offset, value)
    if program == 'common17x19':
        struct.pack_into('<Q', raw, 0x3ec, 0x10000035500)
        struct.pack_into('<I', raw, 0x3f8, 0x0c)
    result = bytes(raw)
    digest = hashlib.sha256(result).hexdigest()
    expected = COMMON_SHA256 if program == 'common17x19' else EXPECTED_SHA256
    if digest != expected:
        raise ValueError(f'authored tensor pages changed: {digest}')
    return result


def build(path, program='probe'):
    raw = serialize(program)
    Path(path).write_bytes(raw)
    return {'scope': 'field-built pages for measured single-MMA tensor layout',
            'capture_read_at_build': False, 'tensor_specific_dwords': 12,
            'program': program, 'template_bytes': len(raw),
            'template_sha256': COMMON_SHA256 if program == 'common17x19' else EXPECTED_SHA256,
            'undecoded_scalar_constants_retained': True,
            'kernel_trace_low32_offset': '0x234',
            'segment_trace_qword_offsets': ['0x0', '0x18', '0x28'],
            'segment_ready_word_offset': '0x24',
            'ready_word_staged': '0x000000f0'}
