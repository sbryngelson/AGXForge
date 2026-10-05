"""Independent, bounded stage references for the resident seven-kernel MiniLM FFN.

The application reference is FP64 on the ORIGINAL FP32 checkpoint. The scheduled
reference additionally rounds source, weights and GELU output to half; this is a
separate numerical algorithm and is never substituted for the application check.
The default arithmetic model preserves the retained 64-K partial-fold schedule,
half MMA issue rounding, and explicit scalar order.
recip/exp2/rsqrt here are correctly rounded numpy estimates, NOT exact silicon
instruction models: compare their stage outputs within the preregistered bound.
"""
import math
import numpy as np

ROWS, WIDTH, HIDDEN = 32, 384, 1536
ERROR_FACTOR = 2e-5
SHAPES = dict(source=(ROWS, WIDTH), expand_weight=(HIDDEN, WIDTH),
              expand_bias=(HIDDEN,), contract_weight=(WIDTH, HIDDEN),
              contract_bias=(WIDTH,), gamma=(WIDTH,), beta=(WIDTH,))
STAGE_SHAPES = dict(pack=(ROWS, WIDTH), expand=(ROWS, HIDDEN),
                    activation=(ROWS, HIDDEN), pack_hidden=(ROWS, HIDDEN), contract=(ROWS, WIDTH),
                    residual=(ROWS, WIDTH), layernorm=(ROWS, WIDTH))


def require_arrays(arrays):
    if set(arrays) != set(SHAPES):
        raise ValueError('refused: resident FFN requires exactly the seven named input/parameter arrays')
    for name, shape in SHAPES.items():
        x = np.asarray(arrays[name])
        if x.dtype != np.float32 or x.shape != shape or not np.isfinite(x).all():
            raise ValueError('refused: finite FP32 array with measured shape required: ' + name)
        if name in ('source', 'expand_weight', 'contract_weight') and np.any(np.abs(x) > 65504):
            raise ValueError('refused: half transport overflow: ' + name)


def _f32(x):
    """FP32 scalar-ALU FTZ, preserving the sign of flushed values."""
    x = np.asarray(x, dtype=np.float32)
    return np.where(np.abs(x) < np.float32(2**-126), np.copysign(np.float32(0), x), x).astype(np.float32)


def _add(a, b):
    return _f32(_f32(a) + _f32(b))


def _mul(a, b):
    return _f32(_f32(a) * _f32(b))


def half_mma(a, weight):
    """N-by-K checkpoint weights; exact half products, measured 16-K reduction.

    P_i=RNE32(p_2i+p_2i+1); Q_j=RNE32(P_j+P_j+4); first
    issue starts Q0, later issues start their existing C, then Q0..Q3.
    No BLAS reduction or hardware output is used to predict the result.
    """
    a = np.asarray(a, dtype=np.float16).astype(np.float32)
    w = np.asarray(weight, dtype=np.float16).astype(np.float32)
    if a.ndim != 2 or w.ndim != 2 or a.shape[1] != w.shape[1] or a.shape[1] % 16:
        raise ValueError('refused: half MMA reference requires matrices and K multiple of 16')
    if not np.isfinite(a).all() or not np.isfinite(w).all():
        raise ValueError('refused: half MMA reference requires finite transported operands')
    out = np.empty((a.shape[0], w.shape[0]), dtype=np.float32)
    for n in range(0, w.shape[0], 256):
        acc = None
        for k in range(0, a.shape[1], 16):
            p = a[:, k:k+16, None] * w[n:n+256, k:k+16].T[None, :, :]
            pair = p[:, 0::2, :] + p[:, 1::2, :]
            q = pair[:, :4, :] + pair[:, 4:, :]
            if acc is None:
                acc = q[:, 0, :].copy()
                first = 1
            else:
                first = 0
            for j in range(first, 4):
                acc = acc + q[:, j, :]
        out[:, n:n+256] = acc
    return out


def legacy_block_mma(a, weight):
    """Existing diagnostic schedule: fresh 64-K GEMM, then explicit FP32 add.

    The runtime-K candidate chains every 16-K MMA directly. Those association
    orders differ; this model makes the old behavior available for comparison.
    """
    a, weight = np.asarray(a), np.asarray(weight)
    if a.shape[1] % 64:
        raise ValueError('refused: retained diagnostic FFN uses 64-K blocks')
    result = half_mma(a[:, :64], weight[:, :64])
    for k in range(64, a.shape[1], 64):
        result = _add(result, half_mma(a[:, k:k+64], weight[:, k:k+64]))
    return result


def gelu_model(x):
    """Scalar expansion of erf (A&S 7.1.26), every fmul/fadd rounded separately."""
    x = _f32(x)
    z = _mul(x, np.float32(1 / math.sqrt(2)))
    a = np.fmax(z, _mul(z, -1))
    t = _f32(np.float32(1) / _add(_mul(a, np.float32(.3275911)), 1))
    poly = np.float32(1.061405429)
    for coef in (-1.453152027, 1.421413741, -.284496736, .254829592):
        poly = _add(_mul(poly, t), np.float32(coef))
    poly = _mul(poly, t)
    exponent = _mul(_mul(a, a), np.float32(-1 / math.log(2)))
    e = _f32(np.exp2(exponent))
    y = _add(_mul(_mul(poly, -1), e), 1)
    sign = np.fmax(np.fmin(_mul(z, np.float32(1e30)), 1), -1)
    erf = _mul(y, sign)
    return _mul(_mul(x, .5), _add(erf, 1))


def layernorm_model(source, gamma, beta):
    """Match cooperative IR order without interpreting native instructions."""
    x = np.asarray(source, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != WIDTH:
        raise ValueError('refused: cooperative LayerNorm has 384 columns')
    shifted = _add(x, _mul(x[:, :1], -1))
    lanes = shifted.reshape(x.shape[0], 32, 12)
    def total(v):
        local = np.zeros(v.shape[:2], dtype=np.float32)
        for j in range(12):
            local = _add(local, v[:, :, j])
        result = np.zeros((v.shape[0], 1), dtype=np.float32)
        for lane in range(32):
            result = _add(result, local[:, lane:lane+1])
        return result
    mean = _mul(total(lanes), np.float32(1 / WIDTH))
    centered = _add(shifted, _mul(mean, -1))
    variance = _mul(total(_mul(centered, centered).reshape(x.shape[0], 32, 12)), np.float32(1 / WIDTH))
    inverse = _f32(np.float32(1) / np.sqrt(_add(variance, np.float32(1e-12))))
    return _add(_mul(_mul(centered, inverse), gamma), beta)


def references(arrays):
    """Return predictions plus independent original and quantized FP64 outputs."""
    require_arrays(arrays)
    import g17ffnchain
    source = arrays['source']
    params = {k: v for k, v in arrays.items() if k != 'source'}
    original = g17ffnchain.reference(source, params)
    pack = source.astype(np.float16)
    expand = legacy_block_mma(pack, arrays['expand_weight'])
    activation = gelu_model(_add(expand, arrays['expand_bias']))
    pack_hidden = activation.astype(np.float16)
    if not np.isfinite(pack_hidden).all():
        raise ValueError('refused: nonfinite activation transport')
    contract = legacy_block_mma(pack_hidden, arrays['contract_weight'])
    residual = _add(_add(contract, arrays['contract_bias']), source)
    output = layernorm_model(residual, arrays['gamma'], arrays['beta'])
    # Independent FP64 mathematical reference for the explicitly quantized schedule.
    import g17minilmffn
    ex64 = pack.astype(np.float64) @ arrays['expand_weight'].astype(np.float16).astype(np.float64).T + arrays['expand_bias'].astype(np.float64)
    act64 = g17minilmffn.gelu_reference(ex64).astype(np.float16).astype(np.float64)
    residual64 = act64 @ arrays['contract_weight'].astype(np.float16).astype(np.float64).T + arrays['contract_bias'].astype(np.float64) + source.astype(np.float64)
    centered = residual64 - residual64.mean(axis=1, keepdims=True)
    quantized = centered / np.sqrt(np.mean(centered * centered, axis=1, keepdims=True) + 1e-12) * arrays['gamma'].astype(np.float64) + arrays['beta'].astype(np.float64)
    stages = dict(pack=pack, expand=expand, activation=activation, pack_hidden=pack_hidden, contract=contract, residual=residual, layernorm=output)
    return dict(stages=stages, application_fp64=original['output'], quantized_fp64=quantized,
                application_error=compare(output, original['output']), quantized_error=compare(output, quantized),
                model_scope='MMA and scalar ordering; numpy recip/exp2/rsqrt are estimates, not exact silicon oracles')


def stagewise_references(arrays, readbacks):
    """Independent stage mathematics on ACTUAL GPU predecessors, labeled locally.

    This follows the retained diagnostic validation contract. A local pass does
    not establish end-to-end original-FP64 accuracy; report references() as well.
    Exact shapes/dtypes and finite values are mandatory before comparison.
    """
    require_arrays(arrays)
    if set(readbacks) != set(STAGE_SHAPES):
        raise ValueError('refused: seven actual stage readbacks required')
    for name, shape in STAGE_SHAPES.items():
        dtype = np.float16 if name in ('pack', 'pack_hidden') else np.float32
        value = np.asarray(readbacks[name])
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError('refused: stage readback shape/dtype/finiteness: ' + name)
    import g17minilmffn, g17layernorm
    p = readbacks
    biased = _add(p['expand'], arrays['expand_bias'])
    return dict(pack=arrays['source'].astype(np.float16),
                expand=p['pack'].astype(np.float64) @ arrays['expand_weight'].astype(np.float16).astype(np.float64).T,
                activation=g17minilmffn.gelu_reference(biased),
                pack_hidden=p['activation'].astype(np.float16),
                contract=p['pack_hidden'].astype(np.float64) @ arrays['contract_weight'].astype(np.float16).astype(np.float64).T,
                residual=_add(_add(p['contract'], arrays['contract_bias']), arrays['source']),
                layernorm=g17layernorm.reference(p['residual'], arrays['gamma'], arrays['beta']))


def compare(got, reference):
    got, ref = np.asarray(got), np.asarray(reference, dtype=np.float64)
    if got.shape != ref.shape:
        raise ValueError('readback shape differs from reference')
    error = np.abs(got.astype(np.float64) - ref)
    budget = ERROR_FACTOR * (1 + np.abs(ref))
    finite = np.isfinite(got) & np.isfinite(ref)
    return dict(passed=bool(np.all(finite & (error <= budget))), failures=int(np.count_nonzero(~finite | (error > budget))),
                max_abs_error=float(np.max(np.where(finite, error, np.inf))),
                max_error_over_budget=float(np.max(np.where(finite, error / budget, np.inf))),
                error_bound='2e-5 * (1 + abs(reference))')


def validation_contract():
    return dict(submissions=7, stages=list(STAGE_SHAPES), readonly=['source', 'expand_weight', 'expand_bias', 'contract_weight', 'contract_bias', 'gamma', 'beta'],
                guards='128 bytes before and after each resident allocation payload; unchanged after every execution',
                output_shape=[ROWS, WIDTH], output_dtype='float32',
                correctness='Original application FP64 AND explicitly quantized schedule reported separately; bound is never widened to hide half quantization',
                repeat='At least three different source inputs; parameters/code immutable; reset all writable output payloads to a sentinel before execution')
