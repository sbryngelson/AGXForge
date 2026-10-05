#!/usr/bin/env python3
"""Small authored G17 program for competing valid lookup records.

The high record selects the 56,240-byte tensor image at allocation-0
offset 0x6c0. The low alias can instead select this 80-byte FP16-add
program at offset 0xf000. Both use the tensor graph's A/B/C binding
indices; the outputs are deliberately distinguishable.
"""
import hashlib
import struct


# Make agxforge.g17 importable when this tool runs from outside the checkout.
import sys as _g17_sys
from pathlib import Path as _G17Path
_g17_root = str(_G17Path(__file__).resolve().parents[1])
if _g17_root not in _g17_sys.path:
    _g17_sys.path.insert(0, _g17_root)
from agxforge.g17 import cc, ir


CODE_OFFSET = 0xf000
CODE_VA = 0x1000000f000
CODE_SHA256 = '0f58e7e9f488a721874c79b0010ed3a8cdae11127804c3b6f76d404df5edf2b4'
THREADS = 1536
OUTPUT_WORDS = 98304


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def code():
    a = ir.Buffer('A', 1, elem=ir.F16)
    b = ir.Buffer('B', 2, elem=ir.F16)
    c = ir.Buffer('C', 3, elem=ir.F32)
    fn = ir.Function('g17_half_witness', [a, b, c])
    q = ir.Builder(fn, fn.block('entry'))
    t = q.builtin('thread_position_in_grid', name='t')
    av = q.f16_to_f32(q.load(a, t, width='half', name='a'), name='af')
    bv = q.f16_to_f32(q.load(b, t, width='half', name='b'), name='bf')
    q.store_at(c, t, q.fadd(av, bv, name='sum'))
    q.ret()
    ir.verify(fn)
    program = cc.compile_function(fn)
    raw = bytes(program.code)
    if (len(raw) != 80 or sha(raw) != CODE_SHA256 or
            [(x['index'], x['offset']) for x in program.abi()['bindings']]
            != [(1, 0), (2, 2), (3, 4)]):
        raise ValueError('alternate authored witness changed')
    return raw


def expected_words(a, b):
    if len(a) < THREADS * 2 or len(b) < THREADS * 2:
        raise ValueError('short alternate witness input')
    result = [0xffffffff] * OUTPUT_WORDS
    for i in range(THREADS):
        av = struct.unpack_from('<e', a, 2 * i)[0]
        bv = struct.unpack_from('<e', b, 2 * i)[0]
        result[i] = struct.unpack('<I', struct.pack('<f', float(av + bv)))[0]
    return result
