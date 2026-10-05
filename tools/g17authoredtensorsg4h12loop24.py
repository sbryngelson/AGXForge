#!/usr/bin/env python3
"""Predict the first twelve-group loop crossing the B-allocation boundary.

From the passing 23-trip graph, B grows from 0x18000 to 0x20000.
Seven later resources and their command-page pointers move +0x8000.
The resource-correlate pair is an extrapolation and remains unnamed.
"""
import hashlib
from pathlib import Path
import struct

import g17authoredtensorsg4h12loop23 as parent
import g17structuredrequests as request_base
import g17tensoraccregs

# Make agxforge.g17 importable when this tool runs from outside the checkout.
import sys as _g17_sys
from pathlib import Path as _G17Path
_g17_root = str(_G17Path(__file__).resolve().parents[1])
if _g17_root not in _g17_sys.path:
    _g17_sys.path.insert(0, _g17_root)
from agxforge.g17 import asm, cc, tensorlife


CODE_SHA256 = '74200b7ff33c9503b711cdcd74880aa5ce20cedaf5f25a75013fbd85d11c9b33'
B_SHA256 = 'b1d9364fa03b71904adb15a2cbf8c87d6d7ce3f29748e95925bd869b55f32ea7'
OUTPUT_SHA256 = '50f162facbf103efaf54b9b9b336c46f754fc247fe8039983659682ce413cc08'
PAGE_SHA256 = 'e79b15b7cb36f1134489c4a87c319188867b36616b2ce44cf688895e389a3e4b'
REQUEST_SHA256 = 'f9f247fec4db6a64d816c018c69df86b42577435e65bc10028d46ad44541324d'
PAYLOAD_SHA256 = '75bc8dc74b4611f918b1f81dbdac38d3646c55bba44516b50a7fe927386784f4'
INDICES = parent.INDICES
SIZES = tuple(0x20000 if index == 20 else size for index, size in zip(INDICES, parent.SIZES))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def code24():
    old = parent.code23()
    new = bytes(cc.compile_function(g17tensoraccregs.loop_fn(sg=4, heads=2, trips=24)).code)
    assert len(new) == len(old) == 6074 and sha(new) == CODE_SHA256
    assert [(i, x, y) for i, (x, y) in enumerate(zip(old, new)) if x != y] == [
        (3975, 0x96, 0x98), (3976, 0x15, 0x05)]
    assert asm.decode_cmp_imm(new[0xf86:0xf8a]) == {'imm': 24, 'rel': 'lt', 'keep': True}
    tensorlife.counted_loop_check(new, 24, carried=tuple(cc._TENSOR_INDEX_USED))
    return new


def b24(source):
    result = parent.b23(source) + source[:4096]
    if len(result) != 98304 or sha(result) != B_SHA256:
        raise ValueError('twenty-four-trip B stream changed')
    return result


def pages():
    raw = bytearray(parent.pages())
    for offset in (0x174, 0x1cc, 0x1d4, 0x1e4, 0x1ec, 0x1f4, 0x1fc, 0x3ec):
        struct.pack_into('<Q', raw, offset, struct.unpack_from('<Q', raw, offset)[0] + 0x8000)
    result = bytes(raw)
    if sha(result) != PAGE_SHA256:
        raise ValueError('twenty-four-trip page prediction changed')
    return result


def requests():
    source = parent.requests()
    rows = [list(request_base.ROW.unpack_from(source, index * request_base.ROW.size))
            for index in range(29)]
    rows[20][6] = 0x20000
    rows[20][7] = request_base.request(0x10001, 0x470, 0x20000, base_buffer=True)
    for index in range(21, 29):
        rows[index][5] += 0x8000
    result = b''.join(request_base.ROW.pack(*row) for row in rows)
    if sha(result) != REQUEST_SHA256:
        raise ValueError('twenty-four-trip request prediction changed')
    return result


def allocations(code, a, b, c):
    if code != code24() or len(b) != 98304 or sha(b) != B_SHA256:
        raise ValueError('exact twenty-four-trip code and B required')
    result = parent.allocations(parent.code23(), a, b[:94208], c)
    raw = bytearray(result[0])
    raw[0x6c0+3975:0x6c0+3977] = code[3975:3977]
    result[0] = bytes(raw)
    raw = bytearray(0x20000)
    raw[:0x80] = b'\xa5' * 0x80
    raw[0x80:0x80+len(b)] = b
    raw[0x80+len(b):0x100+len(b)] = b'\xa5' * 0x80
    result[20] = bytes(raw)
    raw = bytearray(result[23]); raw[6] = 0xc8; result[23] = bytes(raw)
    raw = bytearray(result[25]); raw[9] = 0x50; result[25] = bytes(raw)
    raw = bytearray(result[28])
    struct.pack_into('<Q', raw, 0x1bb0, 0x100000c8080)
    result[28] = bytes(raw)
    if result[0][0x6c0:0x6c0+len(code)] != code:
        raise ValueError('twenty-four-trip code not staged')
    if any(len(result[index]) != size for index, size in zip(INDICES, SIZES)):
        raise ValueError('twenty-four-trip physical size changed')
    return result


def build(request_path, page_path, payload_path, code, a, b, c):
    req, page, alloc = requests(), pages(), allocations(code, a, b, c)
    payload = b''.join(alloc[index] for index in INDICES)
    if len(payload) != 1359872 or sha(payload) != PAYLOAD_SHA256:
        raise ValueError('twenty-four-trip payload changed')
    for path, raw in ((request_path, req), (page_path, page), (payload_path, payload)):
        Path(path).write_bytes(raw)
    return {'scope': 'predicted twelve-group twenty-four-trip B-boundary graph',
            'capture_read_at_build': False, 'program': 'sg4h12loop24',
            'requests_sha256': sha(req), 'pages_sha256': sha(page),
            'payload_sha256': sha(payload), 'payload_bytes': len(payload),
            'physical_allocations': [{'index': index, 'size': size, 'sha256': sha(alloc[index])}
                                     for index, size in zip(INDICES, SIZES)]}
