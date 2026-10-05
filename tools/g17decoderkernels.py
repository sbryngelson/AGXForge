"""Bounded Qwen native stage IR; compilation is not hardware admission.

No Metal kernels or Apple native bodies are used. Position/trigonometric tables
are fixed model constants; rotations and KV writes execute on the GPU.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agxforge.g17 import ir, cc
from g17inferencelower import function, bits

WIDTH, HEADS, KV_HEADS, DIM, CAPACITY = 896, 14, 2, 64, 256


def domain(rows):
    if type(rows) is not int or rows not in (1, 32):
        raise ValueError('refused: Qwen stage supports prefill 32 or decode 1 only')


def rms_ir(rows):
    domain(rows)
    fn, b, (source, gamma, out) = function('qwen_rms', ['source', 'gamma', 'output'])
    lane = b.builtin('thread_position_in_grid', axis='x')
    row = b.builtin('thread_position_in_grid', axis='y')
    base = b.mul(row, b.const(WIDTH))
    total = b.const(bits(0))
    for j in range(WIDTH//32):
        index = b.add(base, b.add(lane, b.const(j*32)))
        x = b.load(source, index, type=ir.F32)
        total = b.fadd(total, b.fmul(x, x))
    for mask in (1, 2, 4, 8, 16):
        total = b.fadd(total, b.simd_shuffle_xor(total, mask))
    inv = b.rsqrt(b.fadd(b.fmul(total, b.const(bits(1/WIDTH))), b.const(bits(1e-6))))
    for j in range(WIDTH//32):
        col = b.add(lane, b.const(j*32)); index = b.add(base, col)
        value = b.fmul(b.fmul(b.load(source, index, type=ir.F32), inv), b.load(gamma, col, type=ir.F32))
        b.store_at(out, index, value)
    b.ret(); return fn


def rotary_ir(rows, heads):
    domain(rows)
    if heads not in (HEADS, KV_HEADS):
        raise ValueError('refused: Qwen rotary head count')
    # The preceding generic gather reads [position, cos32+sin32] from a fixed
    # FP64-generated model table rounded to FP32, with positions checked by host.
    fn, b, (source, phase, out) = function('qwen_rotary', ['source', 'phase', 'output'])
    col = b.builtin('thread_position_in_grid', axis='x')
    row = b.builtin('thread_position_in_grid', axis='y')
    d = b._def('and', [col, b.const(63)])
    half = b.shr(d, b.const(5)); pair = b._def('and', [d, b.const(31)])
    index = b.add(b.mul(row, b.const(heads*DIM)), col)
    other = b._def('xor', [col, b.const(32)])
    other_index = b.add(b.mul(row, b.const(heads*DIM)), other)
    phase_base = b.add(b.mul(row, b.const(DIM)), pair)
    cos = b.load(phase, phase_base, type=ir.F32)
    sin = b.load(phase, b.add(phase_base, b.const(32)), type=ir.F32)
    sign = b.fadd(b.fmul(b.u32_to_f32(half), b.const(bits(2))), b.const(bits(-1)))
    value = b.fadd(b.fmul(b.load(source, index, type=ir.F32), cos),
                   b.fmul(b.fmul(b.load(source, other_index, type=ir.F32), sin), sign))
    b.store_at(out, index, value); b.ret(); return fn


def cache_append_ir(rows):
    domain(rows)
    fn, b, (source, positions, cache) = function('qwen_cache_append', ['source', 'positions', 'cache'], [ir.F32, ir.I32, ir.F32])
    col = b.builtin('thread_position_in_grid', axis='x')
    row = b.builtin('thread_position_in_grid', axis='y')
    head = b.shr(col, b.const(6)); element = b._def('and', [col, b.const(63)])
    position = b.load(positions, row)
    index = b.add(b.add(b.mul(head, b.const(CAPACITY*DIM)), b.mul(position, b.const(DIM))), element)
    value = b.load(source, b.add(b.mul(row, b.const(KV_HEADS*DIM)), col), type=ir.F32)
    b.store_at(cache, index, value); b.ret(); return fn


def silu_ir(rows):
    domain(rows)
    fn, b, (source, out) = function('qwen_silu', ['source', 'output'])
    col = b.builtin('thread_position_in_grid', axis='x'); row = b.builtin('thread_position_in_grid', axis='y')
    index = b.add(b.mul(row, b.const(4864)), col)
    x = b.load(source, index, type=ir.F32)
    exponent = b.fmul(x, b.const(bits(-math.log2(math.e))))
    exponent = b.fmax(b.fmin(exponent, b.const(bits(125))), b.const(bits(-125)))
    value = b.fmul(x, b.recip(b.fadd(b.exp2(exponent), b.const(bits(1)))))
    b.store_at(out, index, value); b.ret(); return fn


def scores_ir(rows):
    domain(rows)
    fn, b, (q, cache, out) = function('qwen_gqa_scores', ['query', 'cache_k', 'scores'])
    key = b.builtin('thread_position_in_grid', axis='x')
    headrow = b.builtin('thread_position_in_grid', axis='y')
    head = b.shr(headrow, b.const(5 if rows == 32 else 0))
    row = b._def('and', [headrow, b.const(rows-1)])
    # 14 query heads map in contiguous groups of seven onto two KV heads.
    kvhead = b.icmp(head, b.const(7), rel='ge')
    qb = b.add(b.mul(row, b.const(WIDTH)), b.mul(head, b.const(DIM)))
    kb = b.add(b.mul(kvhead, b.const(CAPACITY*DIM)), b.mul(key, b.const(DIM)))
    total = b.const(bits(0))
    for d in range(DIM):
        total = b.fma(b.load(q, b.add(qb, b.const(d)), type=ir.F32),
                      b.load(cache, b.add(kb, b.const(d)), type=ir.F32), total)
    b.store_at(out, b.add(b.mul(headrow, b.const(CAPACITY)), key), b.fmul(total, b.const(bits(1/8))))
    b.ret(); return fn


def mask_ir(rows):
    domain(rows)
    fn, b, (scores, positions, mask) = function('qwen_causal_mask', ['scores', 'positions', 'mask'], [ir.F32, ir.I32, ir.I32])
    key = b.builtin('thread_position_in_grid', axis='x'); headrow = b.builtin('thread_position_in_grid', axis='y')
    row = b._def('and', [headrow, b.const(rows-1)])
    causal = b.icmp(key, b.load(positions, row), rel='le')
    valid = b._def('and', [causal, b.load(mask, key)])
    invalid = b.fadd(b.const(bits(1)), b.fmul(b.u32_to_f32(valid), b.const(bits(-1))))
    index = b.add(b.mul(headrow, b.const(CAPACITY)), key)
    value = b.fadd(b.load(scores, index, type=ir.F32), b.fmul(invalid, b.const(bits(-3.4028234663852886e38))))
    b.store_at(scores, index, value); b.ret(); return fn


def softmax_ir(rows):
    domain(rows)
    fn, b, (source, out) = function('qwen_softmax', ['scores', 'probabilities'])
    lane = b.builtin('thread_position_in_grid', axis='x'); row = b.builtin('thread_position_in_grid', axis='y')
    base = b.mul(row, b.const(CAPACITY))
    indices = [b.add(base, b.add(lane, b.const(j*32))) for j in range(8)]
    values = [b.load(source, i, type=ir.F32) for i in indices]
    maximum = b.const(bits(-3.4028234663852886e38))
    for value in values: maximum = b.fmax(maximum, value)
    for mask in (1, 2, 4, 8, 16): maximum = b.fmax(maximum, b.simd_shuffle_xor(maximum, mask))
    terms = [b.exp2(b.fmul(b.fadd(value, b.fmul(maximum, b.const(bits(-1)))), b.const(bits(math.log2(math.e))))) for value in values]
    total = b.const(bits(0))
    for value in terms: total = b.fadd(total, value)
    for mask in (1, 2, 4, 8, 16): total = b.fadd(total, b.simd_shuffle_xor(total, mask))
    inv = b.recip(total)
    for index, value in zip(indices, terms): b.store_at(out, index, b.fmul(value, inv))
    b.ret(); return fn


def context_ir(rows):
    domain(rows)
    fn, b, (prob, cache, out) = function('qwen_gqa_context', ['probabilities', 'cache_v', 'context'])
    col = b.builtin('thread_position_in_grid', axis='x'); row = b.builtin('thread_position_in_grid', axis='y')
    head = b.shr(col, b.const(6)); element = b._def('and', [col, b.const(63)])
    kvhead = b.icmp(head, b.const(7), rel='ge')
    pb = b.mul(b.add(b.mul(head, b.const(rows)), row), b.const(CAPACITY))
    vb = b.add(b.mul(kvhead, b.const(CAPACITY*DIM)), element)
    seed = b.const(bits(0)); zero = b.const(0); bound = b.const(CAPACITY)
    loop = fn.block('key_loop'); done = fn.block('done'); b.br(loop); b.at(loop)
    key = b.phi(zero); acc = b.phi(seed, type=ir.F32)
    next_acc = b.fma(b.load(prob, b.add(pb, key), type=ir.F32),
                     b.load(cache, b.add(vb, b.mul(key, b.const(DIM))), type=ir.F32), acc)
    nxt = b.add(key, b.const(1)); b.phi_latch(key, nxt); b.phi_latch(acc, next_acc)
    b.br_cond(b.cmp(nxt, bound, cap=CAPACITY), loop, done); b.at(done)
    b.store_at(out, b.add(b.mul(row, b.const(WIDTH)), col), next_acc); b.ret(); return fn


def logits_ir(rows):
    """Reuse tied BF16 embeddings directly, avoiding a second 272 MiB matrix.

    A bounded scalar FMA loop is the initial correctness path for logits; the
    decoder's attention/MLP projections still target the accelerator. Each loop
    trip widens one BF16 pair without floating arithmetic. No host dot product.
    """
    domain(rows)
    fn, b, (source, table, out) = function('qwen_tied_logits', ['source', 'embedding_table', 'logits'], [ir.F32, ir.I32, ir.F32])
    word = b.builtin('thread_position_in_grid', axis='x'); row = b.builtin('thread_position_in_grid', axis='y')
    sb = b.mul(row, b.const(WIDTH)); wb = b.mul(word, b.const(WIDTH//2))
    zero = b.const(0); seed = b.const(bits(0)); bound = b.const(WIDTH//2)
    loop = fn.block('width_loop'); done = fn.block('done'); b.br(loop); b.at(loop)
    pair = b.phi(zero); acc = b.phi(seed, type=ir.F32)
    packed = b.load(table, b.add(wb, pair))
    lo = b.shl(b._def('and', [packed, b.const(0xffff)]), b.const(16))
    hi = b._def('and', [packed, b.const(0xffff0000)])
    index = b.add(sb, b.mul(pair, b.const(2)))
    first = b.fma(b.load(source, index, type=ir.F32), lo, acc)
    next_acc = b.fma(b.load(source, b.add(index, b.const(1)), type=ir.F32), hi, first)
    nxt = b.add(pair, b.const(1)); b.phi_latch(pair, nxt); b.phi_latch(acc, next_acc)
    b.br_cond(b.cmp(nxt, bound, cap=WIDTH//2), loop, done); b.at(done)
    b.store_at(out, b.add(b.mul(row, b.const(151936)), word), next_acc); b.ret(); return fn


def compile_all():
    programs = {}; records = []
    for rows in (1, 32):
        factories = dict(rms=lambda: rms_ir(rows), rotary_q=lambda: rotary_ir(rows, HEADS),
                         rotary_k=lambda: rotary_ir(rows, KV_HEADS), cache_append=lambda: cache_append_ir(rows),
                         silu=lambda: silu_ir(rows), scores=lambda: scores_ir(rows),
                         mask=lambda: mask_ir(rows), softmax=lambda: softmax_ir(rows), context=lambda: context_ir(rows),
                         logits=lambda: logits_ir(rows))
        for name, factory in factories.items():
            p = cc.compile_function(factory()); digest = hashlib.sha256(p.code).hexdigest(); programs[digest] = p.code
            records.append(dict(operation=name, tokens=rows, code_sha256=digest, code_bytes=len(p.code), abi=p.abi_plain(p.abi())))
    return programs, dict(format='g17-native-qwen-stages-v1', status='compiled_not_executed', gpu_admitted=False,
                         rows=[1, 32], cache_capacity=CAPACITY, records=records,
                         preconditions=['finite FP32 activations', 'positions unique and increasing inside 0..255',
                                        '0/1 key mask, at least one valid causal key per query',
                                        'zero cache on reset; preserve live cache across decode requests',
                                        'exact stage grids; no unmeasured resource metadata admission'],
                         pending=['independent references and GPU validation', 'complete decoder lowering',
                                  'stateful resource planning and persistent decoder executor'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('destination', type=Path)
    args = parser.parse_args(); programs, report = compile_all()
    args.destination.mkdir(parents=True, exist_ok=False)
    for digest, code in programs.items(): (args.destination/(digest+'.bin')).write_bytes(code)
    (args.destination/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(stages=len(report['records']), unique_programs=len(programs), code_bytes=sum(map(len, programs.values())), gpu_admitted=False)))
