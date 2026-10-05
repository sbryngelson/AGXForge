#!/usr/bin/env python3
"""Field-build twelve tensor groups with the verified sixteen-trip loop.

The twelve-group geometry and the sixteen-trip binary each passed below
Metal independently. This combined graph has no Metal capture or dispatch.
"""
import hashlib
from pathlib import Path
import struct

import g17authoredtensorsg4h4 as h4
import g17authoredtensorsg4h as h2
import g17authoredtensorsg4h12 as h12
import g17authoredtensorsg4hloop16 as h2long
import g17structuredrequests as request_base


INDICES = h12.INDICES
SIZES = tuple(0x18000 if index == 20 else
              h12.A_SIZE if index == 19 else
              h12.C_SIZE if index == 21 else
              0x10000 if index in (0, 1, 15) else
              0x20000 if index in (2, 17) else
              0xc000 if index == 28 else 0x8000
              for index in INDICES)
CODE_SHA256 = h2long.CODE_SHA256
OUTPUT_SHA256 = '81934bad6a15f1d97b3a62b60e07d1b1ed853171daa5a518a8e27d46604446c4'
A_SHA256 = '2e24688faf98ac43e07bc4cbb7e4f2376067a21b61edef68803f0912a31f6e98'
PAGE_SHA256 = '41f0db8f7b6f08fbe223f03419fd7f299764577118479a8d2665c7db18c5fb11'
REQUEST_SHA256 = '787706eabc7b5aab69f5048e397185320df7c51ede13dddd70ba5df810c471ad'
PAYLOAD_SHA256 = '389c8a74be732ee79250663982be03f6ca677db42d8cfff8318cc6f9df6497d0'
LATE_HEAD_SHA256 = (
    '78c9e3fcf3385ffdad07e4880d31890ed7fee52d7bd3ce889a898b212475161f',
    'c40630d0d0e0bd54d8d8a95750ebefc0839864c6a7b59560e5b36a77f47558f9',
    '4289e6d522545c65d19f3665a3034e14a136c98b274748a3c52c24383c1a09e9',
    '760feaf46dd2bbf8585fcb3e4a1d6a9dd4777c22cadc792213d63dc264820abe')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pages():
    raw = bytearray(h12.pages())
    for offset in (0x174, 0x1cc, 0x1d4, 0x1e4, 0x1ec, 0x1f4, 0x1fc, 0x3ec):
        struct.pack_into('<Q', raw, offset, struct.unpack_from('<Q', raw, offset)[0] + 0x10000)
    result = bytes(raw)
    if sha(result) != PAGE_SHA256:
        raise ValueError('twelve-group sixteen-trip pages changed')
    return result


def requests():
    source = h12.requests()
    rows = [list(request_base.ROW.unpack_from(source, index * request_base.ROW.size))
            for index in range(29)]
    rows[20][6] = 0x18000
    rows[20][7] = request_base.request(0x10001, 0x470, 0x18000, base_buffer=True)
    rows[21][5] += 0x10000
    for index in range(22, 29):
        rows[index][5] += 0x10000
    result = b''.join(request_base.ROW.pack(*row) for row in rows)
    if sha(result) != REQUEST_SHA256:
        raise ValueError('twelve-group sixteen-trip requests changed')
    return result


def allocations(code, a, b, c):
    if (len(code) != 6074 or sha(code) != CODE_SHA256 or code[0xf87] != 0x90):
        raise ValueError('exact sixteen-trip two-head code required')
    if len(a) != h12.A_BYTES or sha(a) != A_SHA256:
        raise ValueError('twelve-group A changed')
    if len(b) != 65536 or sha(b) != h2long.B_SHA256:
        raise ValueError('sixteen-trip B changed')
    if len(c) != h12.C_BYTES or c != b'\xff' * h12.C_BYTES:
        raise ValueError('twelve-group C sentinels changed')
    original = code[:0xf87] + b'\x84' + code[0xf88:]
    if sha(original) != h2.CODE_SHA256:
        raise ValueError('four-trip code cannot be reconstructed')
    result = h12.allocations(original, a, b[:16384], c)
    raw = bytearray(result[0])
    raw[0x6c0 + 0xf87] = 0x90
    result[0] = bytes(raw)
    result[20] = h4.guarded(b, 0x18000)
    raw = bytearray(result[23])
    raw[6] = 0xc4  # Empirical resource-base correlate, +8 for B's +0x10000.
    result[23] = bytes(raw)
    raw = bytearray(result[25])
    raw[9] = 0x4e  # Paired correlate, +4; semantics still unknown.
    result[25] = bytes(raw)
    raw = bytearray(result[28])
    struct.pack_into('<Q', raw, 0x1bb0, 0x100000c0080)
    result[28] = bytes(raw)
    for index, size in zip(INDICES, SIZES):
        if len(result[index]) != size:
            raise ValueError(f'twelve-group sixteen-trip allocation {index} size changed')
    return result


def build(request_path, page_path, payload_path, code, a, b, c):
    req, page, alloc = requests(), pages(), allocations(code, a, b, c)
    payload = b''.join(alloc[index] for index in INDICES)
    if len(payload) != 1327104 or sha(payload) != PAYLOAD_SHA256:
        raise ValueError('twelve-group sixteen-trip payload length changed')
    for path, raw in ((request_path, req), (page_path, page), (payload_path, payload)):
        Path(path).write_bytes(raw)
    return {'scope': 'field-built twelve-group sixteen-trip tensor graph',
            'capture_read_at_build': False, 'program': 'sg4h12loop16',
            'requests_sha256': sha(req), 'pages_sha256': sha(page),
            'payload_sha256': sha(payload), 'payload_bytes': len(payload),
            'physical_allocations': [{'index': index, 'size': size, 'sha256': sha(alloc[index])}
                                     for index, size in zip(INDICES, SIZES)]}
