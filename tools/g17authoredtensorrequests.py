#!/usr/bin/env python3
"""Serialize the measured single-MMA selector-9 request graph from fields.

This bounded layout uses the known request constructor and reads no capture.
Physical allocation contents are a separate concern.
"""
import hashlib
from pathlib import Path

import g17structuredrequests as scalar


EXPECTED_SHA256 = '49822529a735a9c0d39579c19d5dca1b512a0627455023fe78e744de7fc51707'
COMMON_SHA256 = '2a4ae5243ae4a329a8418dc6c4351a6a1d3cc961a6a94d99af645b37a7c738b0'
VIEW_OFFSETS = {
    3: 0x0000, 4: 0x0200, 5: 0x0400, 6: 0x0500,
    7: 0x0700, 8: 0x0800, 9: 0x0900, 10: 0x1900,
    11: 0x1a00, 12: 0x1b00, 13: 0x1c00, 14: 0x2c00,
    16: 0x2d00, 19: 0x4d00, 20: 0x5000, 21: 0x5300,
}


def serialize(program='probe'):
    if program not in ('probe','common17x19'):
        raise ValueError('unsupported tensor request program')
    rows = list(scalar.rows())
    base = scalar.BASE + 0x30000
    for index, offset in VIEW_OFFSETS.items():
        if program == 'common17x19' and index in (20,21):
            offset += 0x100 if index == 20 else 0x200
        word5c = 0 if index in (19, 20, 21) else 0x18
        rows[index] = (index, 1, 2, 0, offset, base + offset, 0x20000,
                       scalar.request(0x10001, 0xc30, 0x20000,
                                      parent=2, word5c=word5c))
    rows[15] = (15, 2, 0, 0, 0, 0, 0x10000,
                scalar.request(0x10001, 0x8430, 0x10000,
                               word58=0x38000000, word5c=0x18))
    rows[17] = (17, 0, 0, 0, 0, scalar.BASE + 0x58000, 0x20000,
                scalar.request(0x10001, 0x470, 0x20000, base_buffer=True))
    rows[18] = (18, 1, 17, 0, 0, scalar.BASE + 0x58000, 0x20000,
                scalar.request(0x10001, 0xc30, 0x20000,
                               parent=17, word5c=0x18))
    raw = b''.join(scalar.ROW.pack(*row) for row in rows)
    digest = hashlib.sha256(raw).hexdigest()
    expected = COMMON_SHA256 if program == 'common17x19' else EXPECTED_SHA256
    if digest != expected:
        raise ValueError(f'authored tensor requests changed: {digest}')
    return raw


def build(path, program='probe'):
    raw = serialize(program)
    Path(path).write_bytes(raw)
    return {'scope': 'field-built selector-9 requests for measured single-MMA graph',
            'capture_read_at_build': False, 'program': program,
            'template_bytes': len(raw),
            'template_sha256': hashlib.sha256(raw).hexdigest(),
            'undecoded_request_words_retained': True}
