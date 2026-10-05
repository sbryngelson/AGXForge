#!/usr/bin/env python3
"""Field-build the two-threadgroup, sixteen-trip tensor graph.

This combines the measured two-head geometry with the independently
authored sixteen-trip latch and 64 KiB B stream. No capture is read.
"""
import hashlib
from pathlib import Path
import struct

import g17authoredblock640payload as block
import g17authoredtensorphysical as physical
import g17authoredtensorsg4h as h2
import g17structuredbasepayload as base
import g17structuredrequests as request_base


CODE_SHA256 = '6cd826583c8a0778034c8137d9bb3f03a11ddfbe938337b98fcbd8a316050bc6'
B_SHA256 = '26484578e8c7e8f8868afb2e105ed9318368777cb091dae52e9836dcf936b9ff'
A_SHA256 = '9f4b2c96e490b45d436c376f3e994b83577a2959cf8b4d4b99f905fc57183065'
C_SHA256 = '71189f7fb6aed638640078fba3a35fda6c39c8962e74dcc75935aac948da9063'
OUTPUT_SHA256 = '3dba58b966fce9378e0d5e43b6f94695893dd1376ded1d0fc21e259bb8c13fd6'
PAGE_SHA256 = 'b2b4a63b0a388c67c157e009d01f71fd54b6ad89492f6f96895bc8836af9fee8'
REQUEST_SHA256 = 'a95dfe40534e31aa51c3e852f002a4e7a04ff2eb63b98584fe8000b88e6a2a8c'
ALLOCATION20_SHA256 = 'c1980cce3059eed6c38b054bb21c50f4837ccbd196ec74c05c7ebc0886d7f2b1'
PAYLOAD_SHA256 = '92b1d5f6388d191c7443e008fa40f919fb3e1bafc67aeaf9542e872d4f2e8ae5'
INDICES = h2.INDICES
SIZES = tuple(0x18000 if i == 20 else size for i, size in zip(INDICES, h2.SIZES))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def long_b_blocks(b):
    if len(b) != 16384 or sha(b) != '04af63901ab2b99d0db8426de063ce9a87080a58ece0862bac3be9ce8e70bf32':
        raise ValueError('two-head B source changed')
    neg = bytes(value ^ (0x80 if index & 1 else 0) for index, value in enumerate(b))
    result = b + neg * 3
    if sha(result) != B_SHA256:
        raise ValueError('two-head long B changed')
    return result


def pages():
    raw = bytearray(h2.pages())
    for offset in (0x174, 0x1cc, 0x1d4, 0x1e4, 0x1ec, 0x1f4, 0x1fc, 0x3ec):
        struct.pack_into('<Q', raw, offset, struct.unpack_from('<Q', raw, offset)[0] + 0x10000)
    result = bytes(raw)
    if sha(result) != PAGE_SHA256:
        raise ValueError('two-head sixteen-trip pages changed')
    return result


def requests():
    source = h2.requests()
    rows = [list(request_base.ROW.unpack_from(source, i * request_base.ROW.size)) for i in range(29)]
    rows[20][6] = 0x18000
    rows[20][7] = request_base.request(0x10001, 0x470, 0x18000, base_buffer=True)
    rows[21][5] += 0x10000
    for index in range(22, 29):
        rows[index][5] += 0x10000
    result = b''.join(request_base.ROW.pack(*row) for row in rows)
    if sha(result) != REQUEST_SHA256:
        raise ValueError('two-head sixteen-trip requests changed')
    return result


def allocations(code, a, b, c):
    if len(code) != 6074 or sha(code) != CODE_SHA256:
        raise ValueError('exact two-head sixteen-trip program required')
    if len(a) != 16384 or sha(a) != A_SHA256:
        raise ValueError('two-head A changed')
    if len(b) != 65536 or sha(b) != B_SHA256:
        raise ValueError('two-head long B changed')
    if len(c) != 65536 or sha(c) != C_SHA256:
        raise ValueError('two-head C changed')
    result = {}
    raw = bytearray(block.allocation_zero(prefix_bytes=0x400))
    raw[0x6c0:0x6c0 + len(code)] = code
    result[0] = bytes(raw)
    result[1] = base.allocation(1)
    result[2] = h2.allocation2()
    result[15] = physical.allocation(15)
    result[17] = physical.allocation(17)
    result[19] = h2.guarded_input(a, 16384, h2.ALLOCATION19_SHA256, 'A')
    raw = bytearray(0x18000)
    raw[:0x80] = b'\xa5' * 0x80
    raw[0x80:0x10080] = b
    raw[0x10080:0x10100] = b'\xa5' * 0x80
    result[20] = bytes(raw)
    if sha(result[20]) != ALLOCATION20_SHA256:
        raise ValueError('two-head sixteen-trip B allocation changed')
    result[21] = h2.allocation21(c)
    for index in range(22, 29):
        raw = bytearray(physical.allocation(index))
        if index == 22:
            struct.pack_into('<I', raw, 0xa8, 256)
        elif index == 23:
            raw[6] = 0x94  # Empirical resource-base correlate, +8 for B's +0x10000.
        elif index == 25:
            raw[9] = 0x36  # Paired correlate, +4; semantic names remain unknown.
            struct.pack_into('<I', raw, 0x10, 256)
            struct.pack_into('<I', raw, 0x1c, 128)
        elif index == 28:
            for offset, address in ((0x1ba0, 0x10000080080),
                                    (0x1ba8, 0x10000090080),
                                    (0x1bb0, 0x100000b0080)):
                struct.pack_into('<Q', raw, offset, address)
        result[index] = bytes(raw)
    for index, size in zip(INDICES, SIZES):
        if len(result[index]) != size:
            raise ValueError(f'two-head sixteen-trip allocation {index} size changed')
    return result


def build(request_path, page_path, payload_path, code, a, b, c):
    req, page, alloc = requests(), pages(), allocations(code, a, b, c)
    payload = b''.join(alloc[index] for index in INDICES)
    if len(payload) != 917504 or sha(payload) != PAYLOAD_SHA256:
        raise ValueError('two-head sixteen-trip payload size changed')
    for path, raw in ((request_path, req), (page_path, page), (payload_path, payload)):
        Path(path).write_bytes(raw)
    return {'scope': 'field-built two-head sixteen-trip tensor graph',
            'capture_read_at_build': False, 'program': 'sg4hloop16',
            'requests_sha256': sha(req), 'pages_sha256': sha(page),
            'payload_sha256': sha(payload), 'payload_bytes': len(payload),
            'physical_allocations': [{'index': index, 'size': size, 'sha256': sha(alloc[index])}
                                     for index, size in zip(INDICES, SIZES)]}
