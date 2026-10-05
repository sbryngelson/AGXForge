#!/usr/bin/env python3
"""Twenty-three tensor trips at the guarded limit of the 0x18000 B allocation."""
import hashlib
from pathlib import Path

import g17authoredtensorsg4h12loop17 as parent
import g17tensoraccregs

# Make agxforge.g17 importable when this tool runs from outside the checkout.
import sys as _g17_sys
from pathlib import Path as _G17Path
_g17_root = str(_G17Path(__file__).resolve().parents[1])
if _g17_root not in _g17_sys.path:
    _g17_sys.path.insert(0, _g17_root)
from agxforge.g17 import asm, cc, tensorlife


CODE_SHA256 = '4167de4342556a7eff9cd75aa30b69c9b97838e318201f579caa8b5a4734bb97'
B_SHA256 = '8b15ca51ecfaf4f5a93fb13f00870175d7fb96f3edb3f6da94f2aa5996604088'
OUTPUT_SHA256 = 'b8a975eddfdc7261dd050def9fb97da5fde67e3fdc0bdbc13d896d09a870c593'
PAYLOAD_SHA256 = '0746cab6fc1e08b7fb5bfaf2b942a75c27e9a5b0012471fb424ad8b205735137'
INDICES = parent.INDICES
SIZES = parent.SIZES


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def code23():
    old = parent.code17()
    new = bytes(cc.compile_function(g17tensoraccregs.loop_fn(sg=4, heads=2, trips=23)).code)
    assert len(new) == len(old) == 6074 and sha(new) == CODE_SHA256
    assert [(i, x, y) for i, (x, y) in enumerate(zip(old, new)) if x != y] == [(3975, 0x90, 0x96)]
    assert asm.decode_cmp_imm(new[0xf86:0xf8a]) == {'imm': 23, 'rel': 'lt', 'keep': True}
    tensorlife.counted_loop_check(new, 23, carried=tuple(cc._TENSOR_INDEX_USED))
    return new


def b23(source):
    result = parent.b17(source) + source[:4096] * 6
    if len(result) != 94208 or sha(result) != B_SHA256:
        raise ValueError('twenty-three-trip B stream changed')
    return result


def pages():
    return parent.pages()


def requests():
    return parent.requests()


def allocations(code, a, b, c):
    if code != code23() or len(b) != 94208 or sha(b) != B_SHA256:
        raise ValueError('exact twenty-three-trip code and B required')
    result = parent.allocations(parent.code17(), a, b[:69632], c)
    raw = bytearray(result[0])
    raw[0x6c0 + 3975] = 0x96
    result[0] = bytes(raw)
    raw = bytearray(result[20])
    raw[0x80:0x80 + len(b)] = b
    raw[0x80 + len(b):0x100 + len(b)] = b'\xa5' * 0x80
    result[20] = bytes(raw)
    if result[0][0x6c0:0x6c0 + len(code)] != code:
        raise ValueError('twenty-three-trip code not staged')
    if any(len(result[index]) != size for index, size in zip(INDICES, SIZES)):
        raise ValueError('twenty-three-trip physical size changed')
    return result


def build(request_path, page_path, payload_path, code, a, b, c):
    req, page, alloc = requests(), pages(), allocations(code, a, b, c)
    payload = b''.join(alloc[index] for index in INDICES)
    if len(payload) != 1327104 or sha(payload) != PAYLOAD_SHA256:
        raise ValueError('twenty-three-trip payload changed')
    for path, raw in ((request_path, req), (page_path, page), (payload_path, payload)):
        Path(path).write_bytes(raw)
    return {'scope': 'field-built twelve-group twenty-three-trip tensor graph',
            'capture_read_at_build': False, 'program': 'sg4h12loop23',
            'requests_sha256': sha(req), 'pages_sha256': sha(page),
            'payload_sha256': sha(payload), 'payload_bytes': len(payload),
            'physical_allocations': [{'index': index, 'size': size, 'sha256': sha(alloc[index])}
                                     for index, size in zip(INDICES, SIZES)]}
