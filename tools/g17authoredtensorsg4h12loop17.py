#!/usr/bin/env python3
"""Seventeen tensor trips on the verified twelve-group command graph.

The 16-trip parent provides every physical allocation and both command pages.
Only its one-byte loop bound and the seventeenth 4 KiB B block change.
"""
import hashlib
from pathlib import Path

import g17authoredtensorsg4h12loop16 as parent
import g17authoredtensorsg4hloop16 as two_head
import g17tensoraccregs

# Make agxforge.g17 importable when this tool runs from outside the checkout.
import sys as _g17_sys
from pathlib import Path as _G17Path
_g17_root = str(_G17Path(__file__).resolve().parents[1])
if _g17_root not in _g17_sys.path:
    _g17_sys.path.insert(0, _g17_root)
from agxforge.g17 import asm, cc, tensorlife


CODE_SHA256 = '4e4ace0ce823077f0539d832deb86859fd5b4da98088988dc8c68dfd3d09f161'
B_SHA256 = '3bc76757323247f92d8075fe24e4f13e3ba94cfc03ea697ca16fbf9757697a1a'
OUTPUT_SHA256 = 'd97adc248041a79b7bc6a20778db437ecf286539c3260c25f27a8ba6bf69d5cb'
PAYLOAD_SHA256 = '7457676408d7b75fecee2c9c37140e1736ff8d4a2ce11c9099b0809b5ae35d35'
INDICES = parent.INDICES
SIZES = parent.SIZES


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def code17():
    old = bytes(cc.compile_function(g17tensoraccregs.loop_fn(sg=4, heads=2, trips=16)).code)
    new = bytes(cc.compile_function(g17tensoraccregs.loop_fn(sg=4, heads=2, trips=17)).code)
    assert sha(old) == parent.CODE_SHA256
    assert len(new) == len(old) == 6074 and sha(new) == CODE_SHA256
    assert [(i, x, y) for i, (x, y) in enumerate(zip(old, new)) if x != y] == [(3976, 5, 21)]
    assert asm.decode_cmp_imm(new[0xf86:0xf8a]) == {'imm': 17, 'rel': 'lt', 'keep': True}
    tensorlife.counted_loop_check(new, 17, carried=tuple(cc._TENSOR_INDEX_USED))
    return new


def b17(source):
    b16 = two_head.long_b_blocks(source)
    result = b16 + source[:4096]
    if len(result) != 69632 or sha(result) != B_SHA256:
        raise ValueError('seventeenth B block changed')
    return result


def pages():
    return parent.pages()


def requests():
    return parent.requests()


def allocations(code, a, b, c):
    if code != code17() or sha(b) != B_SHA256 or len(b) != 69632:
        raise ValueError('exact seventeen-trip code and B required')
    old_code = bytes(cc.compile_function(g17tensoraccregs.loop_fn(sg=4, heads=2, trips=16)).code)
    result = parent.allocations(old_code, a, b[:65536], c)
    raw = bytearray(result[0])
    raw[0x6c0 + 3976] = 21
    result[0] = bytes(raw)
    raw = bytearray(result[20])
    raw[0x80:0x80 + len(b)] = b
    raw[0x80 + len(b):0x100 + len(b)] = b'\xa5' * 0x80
    result[20] = bytes(raw)
    if result[0][0x6c0:0x6c0 + len(code)] != code:
        raise ValueError('seventeen-trip code not staged')
    if any(len(result[index]) != size for index, size in zip(INDICES, SIZES)):
        raise ValueError('seventeen-trip allocation size changed')
    return result


def build(request_path, page_path, payload_path, code, a, b, c):
    req, page, alloc = requests(), pages(), allocations(code, a, b, c)
    payload = b''.join(alloc[index] for index in INDICES)
    if len(payload) != 1327104 or sha(payload) != PAYLOAD_SHA256:
        raise ValueError('seventeen-trip payload changed')
    for path, raw in ((request_path, req), (page_path, page), (payload_path, payload)):
        Path(path).write_bytes(raw)
    return {'scope': 'field-built twelve-group seventeen-trip tensor graph',
            'capture_read_at_build': False, 'program': 'sg4h12loop17',
            'requests_sha256': sha(req), 'pages_sha256': sha(page),
            'payload_sha256': sha(payload), 'payload_bytes': len(payload),
            'physical_allocations': [{'index': index, 'size': size, 'sha256': sha(alloc[index])}
                                     for index, size in zip(INDICES, SIZES)]}
