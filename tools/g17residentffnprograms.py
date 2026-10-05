"""Compiler-built, row-major stages for the resident 32-token MiniLM FFN.

This is a CPU artifact builder, not dispatch admission. Tensor kernels contain
runtime K loops and one 32-column tile per threadgroup. GELU is the application's
erf definition, expanded by the ordinary compiler, never a tanh replacement.
The final stage reuses the existing cooperative packed-parameter LayerNorm.
"""
import hashlib
import json
import math
from pathlib import Path
import struct

ROWS, WIDTH, HIDDEN = 32, 384, 1536


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def tensor_ir(stage):
    import g17ir as ir
    if stage not in ('expand', 'contract'):
        raise ValueError('refused: tensor stage must be expand or contract')
    n, k = (HIDDEN, WIDTH) if stage == 'expand' else (WIDTH, HIDDEN)
    a, weight, out = [ir.Buffer(name, slot, elem=elem) for name, slot, elem in
                      [('a', 1, ir.F16), ('weight', 2, ir.F16), ('output', 3, ir.F32)]]
    fn = ir.Function('resident_ffn_' + stage, [a, weight, out])
    b = ir.Builder(fn, fn.block('entry'))
    b.tensor_matmul(a, weight, out, M=ROWS, N=n, K=k,
                    kloop=True, grid_n=n // 32, kloop_chunk=64)
    b.ret()
    return fn


def pack_ir():
    """Narrow the complete row-major source in one bulk GPU launch."""
    import g17ir as ir
    source = ir.Buffer('source', 1, elem=ir.F32)
    out = ir.Buffer('half_source', 2, elem=ir.F16)
    fn = ir.Function('resident_ffn_pack', [source, out])
    b = ir.Builder(fn, fn.block('entry'))
    index = b.builtin('thread_position_in_grid')
    b.store_at(out, index, b.f32_to_f16_rte(b.load(source, index, width='word')), width='half')
    b.ret()
    return fn


def activation_ir():
    """FP32 expansion + bias -> FP32 erf GELU, before separate half packing."""
    import g17ir as ir
    source, bias = [ir.Buffer(n, i, elem=ir.F32) for i, n in enumerate(('expansion', 'bias'), 1)]
    fn = ir.Function('resident_ffn_bias_gelu', [source, bias])
    b = ir.Builder(fn, fn.block('entry'))
    column = b.builtin('thread_position_in_grid', axis='x')
    row = b.builtin('thread_position_in_grid', axis='y')
    index = b.add(b.mul(row, b.const(HIDDEN)), column)
    x = b.fadd(b.load(source, index, width='word'), b.load(bias, column, width='word'))
    e = b._def('erf', [b.fmul(x, b.const(bits(1 / math.sqrt(2))))], ir.F32)
    value = b.fmul(b.fmul(x, b.const(bits(.5))), b.fadd(e, b.const(bits(1.))))
    b.store_at(source, index, value, width='word')
    b.ret()
    return fn


def pack_hidden_ir():
    """Narrow validated FP32 GELU output through one full-buffer GPU launch."""
    import g17ir as ir
    source = ir.Buffer('activation', 1, elem=ir.F32)
    out = ir.Buffer('half_hidden', 2, elem=ir.F16)
    fn = ir.Function('resident_ffn_pack_hidden', [source, out])
    b = ir.Builder(fn, fn.block('entry'))
    index = b.builtin('thread_position_in_grid')
    b.store_at(out, index, b.f32_to_f16_rte(b.load(source, index, width='word')), width='half')
    b.ret()
    return fn


def layernorm_ir():
    import g17cooplayernorm
    return g17cooplayernorm.cooperative_layernorm_ir(ROWS, packed_parameters=True)


def residual_ir():
    """Round contraction+bias, then round the residual addition before LN."""
    import g17ir as ir
    # Contraction and norm_input already alias exactly. Represent that lifetime
    # in IR as one mutable buffer, not a redundant fourth descriptor. The old
    # four-binding pure controls returned callbacks but wrote 0/32 elements
    # (docs/archive/g17-pure-ffn-bridge.md); that admission remains unresolved.
    source, bias, residual = [ir.Buffer(n, i, elem=ir.F32) for i, n in enumerate(
        ('contraction', 'bias', 'residual'), 1)]
    fn = ir.Function('resident_ffn_bias_residual', [source, bias, residual])
    b = ir.Builder(fn, fn.block('entry'))
    column = b.builtin('thread_position_in_grid', axis='x')
    row = b.builtin('thread_position_in_grid', axis='y')
    index = b.add(b.mul(row, b.const(WIDTH)), column)
    value = b.fadd(b.load(source, index, width='word'), b.load(bias, column, width='word'))
    value = b.fadd(value, b.load(residual, index, width='word'))
    b.store_at(source, index, value, width='word')
    b.ret()
    return fn


# Exact full-buffer lengths; weights are K-by-N row-major half (transpose the
# application's N-by-K checkpoint matrix once at prepare, not each execution).
STAGES = {
    'pack': dict(buffers=['source', 'half_source'], grid=[ROWS * WIDTH, 1, 1], threadgroup=[32, 1, 1], binding_bytes=[ROWS * WIDTH * 4, ROWS * WIDTH * 2]),
    'expand': dict(buffers=['half_source', 'expand_weight', 'expansion'], grid=[32 * (HIDDEN // 32), 1, 1], threadgroup=[32, 1, 1], binding_bytes=[ROWS * WIDTH * 2, WIDTH * HIDDEN * 2, ROWS * HIDDEN * 4]),
    'activation': dict(buffers=['expansion', 'expand_bias'], output_buffer='expansion', grid=[HIDDEN, ROWS, 1], threadgroup=[32, 1, 1], binding_bytes=[ROWS * HIDDEN * 4, HIDDEN * 4]),
    'pack_hidden': dict(buffers=['expansion', 'half_hidden'], grid=[ROWS * HIDDEN, 1, 1], threadgroup=[32, 1, 1], binding_bytes=[ROWS * HIDDEN * 4, ROWS * HIDDEN * 2]),
    'contract': dict(buffers=['half_hidden', 'contract_weight', 'contraction'], grid=[32 * (WIDTH // 32), 1, 1], threadgroup=[32, 1, 1], binding_bytes=[ROWS * HIDDEN * 2, HIDDEN * WIDTH * 2, ROWS * WIDTH * 4]),
    'residual': dict(buffers=['contraction', 'contract_bias', 'source'], output_buffer='norm_input', grid=[WIDTH, ROWS, 1], threadgroup=[32, 1, 1], binding_bytes=[ROWS * WIDTH * 4, WIDTH * 4, ROWS * WIDTH * 4]),
    'layernorm': dict(buffers=['norm_input', 'norm_parameters', 'output'], grid=[32, ROWS, 1], threadgroup=[32, 1, 1], binding_bytes=[ROWS * WIDTH * 4, 2 * WIDTH * 4, ROWS * WIDTH * 4], threadgroup_bytes=128),
}


def programs():
    import g17cc
    return {name: g17cc.compile_function(fn()) for name, fn in
            [('pack', pack_ir), ('expand', lambda: tensor_ir('expand')),
             ('activation', activation_ir), ('pack_hidden', pack_hidden_ir), ('contract', lambda: tensor_ir('contract')),
             ('residual', residual_ir), ('layernorm', layernorm_ir)]}


def requirements(compiled):
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
                  stages=req, submission_count=7,
                  layernorm='Recompiled existing cooperative LayerNorm with gamma then beta in one FP32 parameter buffer',
                  layouts='All intermediates row-major; half weights K-by-N, checkpoint transpose at prepare',
                  scope='Compiler artifacts only. Resource/launch admission and hardware numerical validation remain required.')
    (destination / 'requirements.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    print(json.dumps(prepare(parser.parse_args().destination), indent=2))
