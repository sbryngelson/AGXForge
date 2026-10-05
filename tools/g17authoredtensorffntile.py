#!/usr/bin/env python3
"""Build the first FFN tensor tile's pure-IOGPU graph from repository fields.

The builder reads no Metal capture. The separate audit function compares its
objects with a passive measurement and labels the intentional differences.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import g17authoredblock640payload as block
import g17authoredtensorpages as tensor_pages
import g17authoredtensorphysical as physical
import g17authoredtensorrequests as tensor_requests
import g17ffnchain
import g17structuredbasepayload as base
import g17structuredrequests as requests_base
import g17tensorprojection

CODE_ADDRESS = 0x100000006c0
VIEW_OFFSETS = (0x4d00, 0x5e00, 0x6f00)
BINDINGS = tuple(requests_base.BASE + 0x30000 + offset + 0x80 for offset in VIEW_OFFSETS)
RELOCATION = 0x4000
SENTINEL = 0x7fc01234
EXPECTED_SHA256 = {
    'program.bin': '472570faa7d227ee02179ac3dc83e0f81155913b6b3e6f7d7210f5fc8df18283',
    'a.f16': 'd25820be2c155436d6aa7ff0563867b765d250c7d1b25ecb628fbf469493ba82',
    'b.f16': 'b06a6cd9536605243875b20ba2363c6a60b529d33537a7cdf76952b23b2ef031',
    'c.f32': 'a66cc7f90ba3924586f7bb3893e064cbd4e34b0fe3bb2427feced9e6080876d6',
    'pages.bin': 'ddb7a261ae97ad4e146162ed35b7179c30a3afaa0dcb01653825e4027a741888',
    'ordered.bin': '9bd00a3f24a44abbb35bc897d8f1ce79e165cd97ca817908fe35077ecac3eac5',
    'physical.bin': '6bf2b7fcfa86455eba0175a7783acf4de9e77566d882c70b8d0dcdc2c10348aa',
}
K1_EXPECTED_SHA256 = {
    **EXPECTED_SHA256,
    'a.f16': '7e2394ee302a0b44ad81de8d47907925df9ed93ab0e01bb790262beb5a15320f',
    'b.f16': '5253932fd17d66e2d26049d6f6e330a4c17c1cc0784f1f9256fbd3affe3e2343',
    'physical.bin': 'c4ad9df98a7cb1aca75d33f599cc492c9483fd28e2fe3a7192b3e4c3ca0358be',
}
RELOCATED_SHA256 = {
    **EXPECTED_SHA256,
    'pages.bin': '11f26c100bc8db963e47d5d992c041479cfbbe3b8ec8bfec3b86a202d31124c0',
    'ordered.bin': 'bb6701c0fc9d63086bcfa7ef61569e41acd257c93b964550c227da68d7573b3b',
    'physical.bin': '63ce1c65d99351fad7570767a21bc4670b5aea4e65fb3676a0d8679ad289961c',
}
C_OFFSET_BINDING_SHA256 = '4d09cc4d17c42d466e87ea4fd54716705301f54da04bb0a4b4ba187a081e9eaf'
C_BASE_BINDING_SHA256 = '6a3d4c6d6d3494aead92559b1c2e77ec116b2c9a28a8bf0e9852bf36e916e837'
B_PREFIX = struct.pack('<H', 0x3c00) * 64
A_PREFIX = B_PREFIX
B_OFFSET_BINDING_SHA256 = 'b085d4b1452dab81fb8c6c1c3faa5e7301e212ac0a99c2747bc44614bf7ecacf'
B_BASE_BINDING_SHA256 = '7d9300cfd64910f6416ca5f8ba08312d83c59f07c268e64f40f18fb8c2020acc'
A_OFFSET_BINDING_SHA256 = '1da12c35c6c2a3713b741cea2e9db9eb937e6aa1c2e4b9413dff1b2446fb982e'
A_BASE_BINDING_SHA256 = 'c45d145b0e76a3dbec80373c344014f3bf5ec07890b6de6b55fb67e45f70ae69'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def inputs(k_tile=0):
    if k_tile not in (0, 1):
        raise ValueError('only pinned FFN reduction tiles k=0 and k=1 are supported')
    arrays = g17ffnchain.arrays()
    packed_a, packed_b = g17tensorprojection.packed_inputs(arrays['source'], arrays['expand_weight'])
    a, b = packed_a[k_tile].tobytes(), packed_b[0, k_tile].tobytes()
    c = struct.pack('<I', SENTINEL) * 1024
    code = (ROOT / 'results/g17-tensor-ffn-block-runtime-v2/programs/matmul/program.bin').read_bytes()
    if len(code) != 1276 or sha(code) != '472570faa7d227ee02179ac3dc83e0f81155913b6b3e6f7d7210f5fc8df18283':
        raise ValueError('first FFN tensor image changed')
    if any(len(value) != 4096 for value in (a, b, c)):
        raise ValueError('first FFN tile buffer lengths differ')
    return code, a, b, c


def allocation2(a, b, c, shift=0, c_competition=False, b_competition=False,
                a_competition=False):
    if any(len(value) != 4096 for value in (a, b, c)):
        raise ValueError('first FFN tile requires three 4 KiB buffers')
    raw = bytearray(0x20000)
    raw[:0x4d00] = base.allocation(2)[0xc00:0x5900]
    for offset, fill in ((0x4d00, 0xa5), (0x5d80, 0xa5),
                         (0x5e00, 0xa4), (0x6e80, 0xa4),
                         (0x6f00, 0xa7), (0x7f80, 0xa7)):
        raw[offset+shift:offset+shift+0x80] = bytes([fill]) * 0x80
    for offset, value in ((0x4d80, a), (0x5e80, b), (0x6f80, c)):
        raw[offset+shift:offset+shift+4096] = value
    if c_competition:
        raw[0x6f00+shift:0x7f80+shift] = struct.pack('<I', SENTINEL) * 1056
    if b_competition:
        raw[0x5e00+shift:0x5e80+shift] = B_PREFIX
    if a_competition:
        raw[0x4d00+shift:0x4d80+shift] = A_PREFIX
    return bytes(raw)


def pages(shift=0, stale_c_page=False):
    if stale_c_page and shift != RELOCATION:
        raise ValueError('stale C page control requires relocated views')
    raw = bytearray(tensor_pages.serialize('probe'))
    struct.pack_into('<Q', raw, 0x3ec, BINDINGS[2] - 0x80 + (0 if stale_c_page else shift))
    struct.pack_into('<I', raw, 0x3f8, 0x22)
    return bytes(raw)


def requests(shift=0):
    source = tensor_requests.serialize('probe')
    rows = [list(requests_base.ROW.unpack_from(source, i*requests_base.ROW.size)) for i in range(29)]
    for index, offset in zip((19, 20, 21), VIEW_OFFSETS):
        rows[index][4] = offset + shift
        rows[index][5] = requests_base.BASE + 0x30000 + offset + shift
    return b''.join(requests_base.ROW.pack(*row) for row in rows)


def payload(code, a, b, c, shift=0, c_competition=False, c_binding_base=False,
            b_competition=False, b_binding_base=False,
            a_competition=False, a_binding_base=False):
    parts = []
    for index in physical.INDICES:
        if index == 0:
            raw = bytearray(block.allocation_zero(prefix_bytes=0x400))
            raw[0x6c0:0x6c0+len(code)] = code
            part = bytes(raw)
        elif index == 2:
            part = allocation2(a, b, c, shift, c_competition, b_competition,
                               a_competition)
        elif index == 28:
            raw = bytearray(physical.allocation(index))
            for j, address in enumerate(BINDINGS):
                struct.pack_into('<Q', raw, 0x1ba0+8*j,
                                 address + shift - (0x80 if (j == 2 and c_binding_base) or
                                                   (j == 1 and b_binding_base) or
                                                   (j == 0 and a_binding_base) else 0))
            part = bytes(raw)
        else:
            part = physical.allocation(index)
        if len(part) != physical.SIZES[physical.INDICES.index(index)]:
            raise ValueError(f'physical allocation {index} length changed')
        parts.append(part)
    return b''.join(parts)


def build(destination, shift=0, stale_c_page=False, c_competition=False, c_binding_base=False,
          b_competition=False, b_binding_base=False,
          a_competition=False, a_binding_base=False, k_tile=0):
    if k_tile not in (0, 1) or (k_tile == 1 and
                               (shift or stale_c_page or c_competition or c_binding_base or
                                b_competition or b_binding_base or a_competition or a_binding_base)):
        raise ValueError('k=1 uses the pinned original-placement FFN graph only')
    if shift not in (0, RELOCATION):
        raise ValueError('FFN view relocation must be 0 or 0x4000')
    if stale_c_page and shift != RELOCATION:
        raise ValueError('stale C page control requires relocated views')
    if (c_competition and shift != RELOCATION) or (c_binding_base and not c_competition):
        raise ValueError('C binding competition requires relocated views and shared C sentinel field')
    if (b_competition and shift != RELOCATION) or (b_binding_base and not b_competition) or (b_competition and c_competition):
        raise ValueError('B binding competition requires relocated views without C competition')
    if (a_competition and shift != RELOCATION) or (a_binding_base and not a_competition) or (a_competition and (b_competition or c_competition)):
        raise ValueError('A binding competition requires relocated views without other competitions')
    destination = Path(destination)
    if destination.exists():
        raise ValueError('refusing to overwrite authored FFN tile graph')
    destination.mkdir(parents=True)
    code, a, b, c = inputs(k_tile)
    blobs = {'program.bin': code, 'a.f16': a, 'b.f16': b, 'c.f32': c,
             'pages.bin': pages(shift, stale_c_page), 'ordered.bin': requests(shift),
             'physical.bin': payload(code, a, b, c, shift, c_competition, c_binding_base,
                                     b_competition, b_binding_base,
                                     a_competition, a_binding_base)}
    expected = (K1_EXPECTED_SHA256 if k_tile == 1 else EXPECTED_SHA256 if shift == 0 else
                {**RELOCATED_SHA256, 'pages.bin': EXPECTED_SHA256['pages.bin']} if stale_c_page else
                RELOCATED_SHA256)
    if c_competition:
        expected = {**expected, 'physical.bin': (C_BASE_BINDING_SHA256 if c_binding_base
                                                else C_OFFSET_BINDING_SHA256)}
    if b_competition:
        expected = {**expected, 'physical.bin': (B_BASE_BINDING_SHA256 if b_binding_base
                                                else B_OFFSET_BINDING_SHA256)}
    if a_competition:
        expected = {**expected, 'physical.bin': (A_BASE_BINDING_SHA256 if a_binding_base
                                                else A_OFFSET_BINDING_SHA256)}
    for name, raw in blobs.items():
        if sha(raw) != expected[name]:
            raise ValueError(f'authored FFN graph changed: {name}')
    for name, raw in blobs.items():
        (destination / name).write_bytes(raw)
    report = {'scope': 'capture-independent authored first FFN tile graph; no Submit',
              'code_address': hex(CODE_ADDRESS),
              'stage': f'expand_matmul_n0_k{k_tile}',
              'view_shift': hex(shift),
              'stale_c_page': stale_c_page,
              'c_competition': c_competition,
              'c_binding_base': c_binding_base,
              'b_competition': b_competition,
              'b_binding_base': b_binding_base,
              'a_competition': a_competition,
              'a_binding_base': a_binding_base,
              'binding_addresses': [hex(x + shift - (0x80 if (j == 2 and c_binding_base) or
                                                  (j == 1 and b_binding_base) or
                                                  (j == 0 and a_binding_base) else 0))
                                    for j, x in enumerate(BINDINGS)],
              'sentinel': hex(SENTINEL),
              'files': {name: {'bytes': len(raw), 'sha256': sha(raw)} for name, raw in blobs.items()}}
    (destination / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('destination', type=Path)
    ap.add_argument('--shift', type=lambda x: int(x, 0), default=0)
    ap.add_argument('--stale-c-page', action='store_true')
    ap.add_argument('--c-competition', action='store_true')
    ap.add_argument('--c-binding-base', action='store_true')
    ap.add_argument('--b-competition', action='store_true')
    ap.add_argument('--b-binding-base', action='store_true')
    ap.add_argument('--a-competition', action='store_true')
    ap.add_argument('--a-binding-base', action='store_true')
    ap.add_argument('--k-tile', type=int, choices=(0, 1), default=0)
    args = ap.parse_args()
    result = build(args.destination, args.shift, args.stale_c_page,
                   args.c_competition, args.c_binding_base,
                   args.b_competition, args.b_binding_base,
                   args.a_competition, args.a_binding_base, args.k_tile)
    print(json.dumps({'scope': result['scope'], 'files': result['files']}, indent=2))


if __name__ == '__main__':
    main()
