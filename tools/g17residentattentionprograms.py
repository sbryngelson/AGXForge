"""CPU artifacts for 14 resident MiniLM attention stages, never GPU admission.

Input is already normalized FP32 [32,384]. Four projections deliberately narrow
their input and checkpoint weights to half; this is a separate numerical domain
from g17attention.reference's original FP32 checkpoint application. Scores,
softmax and context reuse the existing ordinary scalar compiler IR unchanged.
Every stage uses at most three published buffers: fourth-buffer native
publication remains unresolved. No fused/flash attention is introduced here.
"""
import hashlib
import json
from pathlib import Path

ROWS, WIDTH, HEADS, DIM = 32, 384, 12, 32
FLOAT_BYTES, HALF_BYTES, WEIGHT_BYTES = 49152, 24576, 294912


def tensor_ir(stage):
    import g17ir as ir
    if stage not in ('q_projection', 'k_projection', 'v_projection', 'out_projection'):
        raise ValueError('refused: unknown resident attention tensor stage')
    buffers = [ir.Buffer(n, slot, elem=elem) for n, slot, elem in
               [('input', 1, ir.F16), ('weight', 2, ir.F16), ('output', 3, ir.F32)]]
    fn = ir.Function('resident_attention_' + stage, buffers)
    b = ir.Builder(fn, fn.block('entry'))
    b.tensor_matmul(*buffers, M=ROWS, N=WIDTH, K=WIDTH,
                    kloop=True, kloop_chunk=64, grid_n=12)
    b.ret()
    return fn


def pack_ir(stage='pack'):
    import g17ir as ir
    if stage not in ('pack', 'context_pack'):
        raise ValueError('refused: unknown resident attention pack stage')
    source, output = ir.Buffer('source', 1, elem=ir.F32), ir.Buffer('output', 2, elem=ir.F16)
    fn = ir.Function('resident_attention_' + stage, [source, output])
    b = ir.Builder(fn, fn.block('entry'))
    index = b.builtin('thread_position_in_grid')
    b.store_at(output, index, b.f32_to_f16_rte(b.load(source, index, width='word')), width='half')
    b.ret()
    return fn


def bias_ir(stage):
    import g17ir as ir
    if stage not in ('q_bias', 'k_bias', 'v_bias'):
        raise ValueError('refused: unknown resident attention bias stage')
    source, bias = [ir.Buffer(n, i, elem=ir.F32) for i, n in enumerate(('projection', 'bias'), 1)]
    fn = ir.Function('resident_attention_' + stage, [source, bias])
    b = ir.Builder(fn, fn.block('entry'))
    column = b.builtin('thread_position_in_grid', axis='x')
    row = b.builtin('thread_position_in_grid', axis='y')
    index = b.add(b.mul(row, b.const(WIDTH)), column)
    value = b.fadd(b.load(source, index, width='word'), b.load(bias, column, width='word'))
    b.store_at(source, index, value, width='word')
    b.ret()
    return fn


def stage(buffers, lengths, grid, *, output=None, shape=(ROWS, WIDTH), dtype='float32', tg_bytes=0):
    result = dict(buffers=buffers, binding_bytes=lengths, grid=grid,
                  threadgroup=[32, 1, 1], output_shape=list(shape), output_dtype=dtype)
    if output is not None:
        result['output_buffer'] = output
    if tg_bytes:
        result['threadgroup_bytes'] = tg_bytes
    return result


STAGES = {
    'pack': stage(['source', 'half_source'], [FLOAT_BYTES, HALF_BYTES], [12288, 1, 1], dtype='float16'),
    'q_projection': stage(['half_source', 'q_weight', 'query'], [HALF_BYTES, WEIGHT_BYTES, FLOAT_BYTES], [384, 1, 1]),
    'q_bias': stage(['query', 'q_bias'], [FLOAT_BYTES, 1536], [384, 32, 1], output='query'),
    'k_projection': stage(['half_source', 'k_weight', 'key'], [HALF_BYTES, WEIGHT_BYTES, FLOAT_BYTES], [384, 1, 1]),
    'k_bias': stage(['key', 'k_bias'], [FLOAT_BYTES, 1536], [384, 32, 1], output='key'),
    'v_projection': stage(['half_source', 'v_weight', 'value'], [HALF_BYTES, WEIGHT_BYTES, FLOAT_BYTES], [384, 1, 1]),
    'v_bias': stage(['value', 'v_bias'], [FLOAT_BYTES, 1536], [384, 32, 1], output='value'),
    'scores': stage(['query', 'key', 'scores'], [FLOAT_BYTES]*3, [32, 384, 1], shape=(12, 32, 32)),
    'softmax': stage(['scores', 'probabilities'], [FLOAT_BYTES]*2, [384, 1, 1], shape=(12, 32, 32)),
    'context': stage(['probabilities', 'value', 'context'], [FLOAT_BYTES]*3, [384, 32, 1]),
    'context_pack': stage(['context', 'half_context'], [FLOAT_BYTES, HALF_BYTES], [12288, 1, 1], dtype='float16'),
    'out_projection': stage(['half_context', 'out_weight', 'projected'], [HALF_BYTES, WEIGHT_BYTES, FLOAT_BYTES], [384, 1, 1]),
    'residual': stage(['projected', 'out_bias', 'source'], [FLOAT_BYTES, 1536, FLOAT_BYTES], [384, 32, 1], output='projected'),
    'layernorm': stage(['projected', 'norm_parameters', 'output'], [FLOAT_BYTES, 3072, FLOAT_BYTES], [32, 32, 1], tg_bytes=128),
}


def programs():
    import g17cc
    import g17attention
    import g17residentffnprograms as ffn
    factories = {
        'pack': pack_ir,
        'q_projection': lambda: tensor_ir('q_projection'), 'q_bias': lambda: bias_ir('q_bias'),
        'k_projection': lambda: tensor_ir('k_projection'), 'k_bias': lambda: bias_ir('k_bias'),
        'v_projection': lambda: tensor_ir('v_projection'), 'v_bias': lambda: bias_ir('v_bias'),
        'scores': g17attention.scores_ir, 'softmax': g17attention.softmax_ir,
        'context': g17attention.context_ir, 'context_pack': lambda: pack_ir('context_pack'),
        'out_projection': lambda: tensor_ir('out_projection'),
        'residual': ffn.residual_ir, 'layernorm': ffn.layernorm_ir,
    }
    return {name: g17cc.compile_function(factory()) for name, factory in factories.items()}


def requirements(compiled):
    if list(compiled) != list(STAGES):
        raise ValueError('refused: attention stage inventory/order differs')
    return {name: dict(STAGES[name], code_bytes=len(p.code),
                       code_sha256=hashlib.sha256(p.code).hexdigest(),
                       abi=p.abi_plain(p.abi()), status='compiled_not_executed')
            for name, p in compiled.items()}


def prepare(destination):
    destination = Path(destination)
    compiled = programs()
    req = requirements(compiled)
    destination.mkdir(parents=True, exist_ok=False)
    for name, p in compiled.items():
        (destination / (name + '.bin')).write_bytes(p.code)
    report = dict(status='compiled_not_executed', gpu_dispatched=False,
                  submission_count=14, stages=req,
                  numerical_domain='Normalized FP32 source; half source/context transport and K-by-N half projection weights; FP32 scalar attention and LayerNorm. Original unquantized FP32 application reference is a separate accuracy diagnostic.',
                  aliases={},
                  scope='CPU artifacts only: resource/launch admission, independent half-domain references and hardware validation remain required.')
    (destination / 'requirements.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    print(json.dumps(prepare(parser.parse_args().destination), indent=2))
