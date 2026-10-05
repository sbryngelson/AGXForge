"""Typed model graphs and checkpoint metadata; no GPU admission or execution.

This frontend preserves model structure independently of native kernels. A
validated graph is not a compiled program. Hardware lowering must separately
admit every operation, numerical policy, allocation and launch contract.
"""
from dataclasses import asdict, dataclass
import json
import math


class Unsupported(ValueError):
    """A recognized model feature outside this frontend's domain."""


class InvalidCheckpoint(ValueError):
    """Malformed or contradictory checkpoint metadata."""


ELEMENT_BYTES = {'F32': 4, 'F16': 2, 'BF16': 2, 'I32': 4, 'I64': 8}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidCheckpoint('duplicate JSON key: ' + key)
        result[key] = value
    return result


@dataclass(frozen=True)
class Parameter:
    name: str
    dtype: str
    shape: tuple
    begin: int
    end: int

    @property
    def bytes(self):
        return self.end - self.begin


def checkpoint_header(raw, *, file_bytes):
    """Validate a safetensors header against the exact whole-file length.

    Only metadata is read. Contiguous payload coverage, unique names, dtype
    sizes and shapes are mandatory. This does not verify weight payload hashes.
    """
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= 4 * 1024 * 1024:
        raise InvalidCheckpoint('header must be 1..4 MiB of bytes')
    if type(file_bytes) is not int or file_bytes < 8 + len(raw):
        raise InvalidCheckpoint('invalid checkpoint byte length')
    try:
        data = json.loads(raw, object_pairs_hook=_unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCheckpoint('invalid header JSON') from exc
    if not isinstance(data, dict):
        raise InvalidCheckpoint('header must be a mapping')
    metadata = data.pop('__metadata__', {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                             for k, v in metadata.items()):
        raise InvalidCheckpoint('metadata must contain string pairs')
    parameters = {}
    for name, spec in data.items():
        if not name or not isinstance(spec, dict) or set(spec) != {'dtype', 'shape', 'data_offsets'}:
            raise InvalidCheckpoint('invalid tensor record: ' + name)
        dtype, shape, offsets = spec['dtype'], spec['shape'], spec['data_offsets']
        if not isinstance(dtype, str) or dtype not in ELEMENT_BYTES:
            raise Unsupported('refused: checkpoint dtype ' + repr(dtype))
        if not isinstance(shape, list) or len(shape) > 8 or any(type(d) is not int or d <= 0 for d in shape):
            raise InvalidCheckpoint('invalid shape: ' + name)
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(d) is not int or d < 0 for d in offsets):
            raise InvalidCheckpoint('invalid offsets: ' + name)
        begin, end = offsets
        if end - begin != math.prod(shape) * ELEMENT_BYTES[dtype]:
            raise InvalidCheckpoint('shape/dtype/byte extent disagreement: ' + name)
        parameters[name] = Parameter(name, dtype, tuple(shape), begin, end)
    cursor = 0
    for p in sorted(parameters.values(), key=lambda p: (p.begin, p.end)):
        if p.begin != cursor:
            raise InvalidCheckpoint('payload overlap or hole: ' + p.name)
        cursor = p.end
    if 8 + len(raw) + cursor != file_bytes:
        raise InvalidCheckpoint('payload extent does not match whole file')
    return parameters


def _positive(config, key):
    v = config.get(key)
    if type(v) is not int or v <= 0:
        raise Unsupported('refused: positive integer config field ' + key)
    return v


@dataclass(frozen=True)
class Profile:
    family: str
    width: int
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    vocabulary: int
    context: int
    epsilon: float
    rope_theta: float | None
    tied_embeddings: bool

    @property
    def head_width(self):
        return self.width // self.heads


def profile(config):
    if not isinstance(config, dict):
        raise Unsupported('refused: model config must be a mapping')
    family = config.get('model_type')
    if family not in ('bert', 'qwen2'):
        raise Unsupported('refused: model architecture ' + repr(family))
    expected_arch = ['BertModel'] if family == 'bert' else ['Qwen2ForCausalLM']
    if config.get('architectures') != expected_arch:
        raise Unsupported('refused: architecture class differs from selected model family')
    if config.get('quantization_config') is not None:
        raise Unsupported('refused: checkpoint quantization is not imported')
    width, heads = _positive(config, 'hidden_size'), _positive(config, 'num_attention_heads')
    kv = heads if family == 'bert' else _positive(config, 'num_key_value_heads')
    if width % heads or heads % kv:
        raise Unsupported('refused: integral head width and query/KV grouping required')
    if config.get('hidden_act') != ('gelu' if family == 'bert' else 'silu'):
        raise Unsupported('refused: activation outside the selected architecture')
    if family == 'bert':
        if config.get('position_embedding_type', 'absolute') != 'absolute' or config.get('is_decoder', False) or config.get('add_cross_attention', False):
            raise Unsupported('refused: BERT relative positions or decoder/cross attention')
        _positive(config, 'type_vocab_size')
        epsilon, theta = config.get('layer_norm_eps'), None
    else:
        if config.get('use_sliding_window', False) or config.get('rope_scaling') is not None:
            raise Unsupported('refused: sliding-window or scaled RoPE')
        if width // heads % 2:
            raise Unsupported('refused: RoPE requires an even head width')
        epsilon, theta = config.get('rms_norm_eps'), config.get('rope_theta')
        if type(theta) not in (float, int) or not math.isfinite(theta) or theta <= 0:
            raise Unsupported('refused: positive finite RoPE theta required')
    if type(epsilon) not in (float, int) or not math.isfinite(epsilon) or epsilon <= 0:
        raise Unsupported('refused: positive finite normalization epsilon required')
    tied = config.get('tie_word_embeddings', False)
    if type(tied) is not bool:
        raise Unsupported('refused: tie_word_embeddings must be boolean')
    return Profile(family, width, _positive(config, 'intermediate_size'),
                   _positive(config, 'num_hidden_layers'), heads, kv,
                   _positive(config, 'vocab_size'), _positive(config, 'max_position_embeddings'),
                   float(epsilon), None if theta is None else float(theta), tied)


class Builder:
    def __init__(self, params):
        self.params = params
        self.tensors = {}
        self.nodes = []
        self.used = set()

    def tensor(self, name, shape, dtype='F32', role='intermediate'):
        if name in self.tensors:
            raise ValueError('duplicate graph tensor: ' + name)
        self.tensors[name] = dict(shape=list(shape), dtype=dtype, role=role)
        return name

    def parameter(self, name, shape):
        if name not in self.params:
            raise InvalidCheckpoint('missing model parameter: ' + name)
        p = self.params[name]
        if p.shape != tuple(shape) or p.dtype not in ('F32', 'F16', 'BF16'):
            raise InvalidCheckpoint('model parameter shape/type mismatch: ' + name)
        if name not in self.tensors:
            self.tensor(name, shape, p.dtype, 'parameter')
            self.tensors[name]['checkpoint_payload_begin'] = p.begin
            self.tensors[name]['checkpoint_payload_end'] = p.end
        self.used.add(name)
        return name

    def op(self, name, kind, inputs, shape, **attrs):
        if any(i not in self.tensors for i in inputs):
            raise ValueError('undefined graph input for ' + name)
        self.tensor(name, shape)
        self.nodes.append(dict(name=name, op=kind, inputs=list(inputs), outputs=[name], attrs=attrs))
        return name

    def linear(self, name, source, prefix, output, bias=True):
        tokens, width = self.tensors[source]['shape']
        weight = self.parameter(prefix + '.weight', (output, width))
        inputs = [source, weight]
        if bias:
            inputs.append(self.parameter(prefix + '.bias', (output,)))
        return self.op(name, 'linear', inputs, (tokens, output), weight_layout='out_in',
                       compute_dtype='F32', transport_policy='not_selected')

    def norm(self, name, source, prefix, epsilon, rms=False):
        shape = self.tensors[source]['shape']
        inputs = [source, self.parameter(prefix + '.weight', (shape[-1],))]
        if not rms:
            inputs.append(self.parameter(prefix + '.bias', (shape[-1],)))
        return self.op(name, 'rms_norm' if rms else 'layer_norm', inputs, shape,
                       epsilon=epsilon, axis=-1)


def import_graph(config, params, *, tokens=32, cache_capacity=256, sentence_pooling=False):
    """Create a complete logical encoder or prefill/decode model graph.

    Fixed token count is a compilation input, not an inferred supported launch.
    KV states are explicit aliases. No CPU arithmetic fallback is provided.
    """
    p = profile(config)
    if type(tokens) is not int or not 1 <= tokens <= p.context:
        raise Unsupported('refused: token extent outside model context')
    if type(sentence_pooling) is not bool or (sentence_pooling and p.family != 'bert'):
        raise Unsupported('refused: sentence pooling applies only to the encoder')
    if p.family == 'qwen2' and (type(cache_capacity) is not int or not tokens <= cache_capacity <= p.context):
        raise Unsupported('refused: KV capacity outside token/model extent')
    b = Builder(params)
    ids = b.tensor('token_ids', (tokens,), 'I32', 'input')
    positions = b.tensor('position_ids', (tokens,), 'I32', 'input')
    mask = b.tensor('attention_mask', (tokens if p.family == 'bert' else cache_capacity,), 'I32', 'input')
    prefix = 'embeddings.word_embeddings' if p.family == 'bert' else 'model.embed_tokens'
    word = b.parameter(prefix + '.weight', (p.vocabulary, p.width))
    x = b.op('embedding.word', 'gather_rows', [word, ids], (tokens, p.width))
    if p.family == 'bert':
        segment = b.tensor('token_type_ids', (tokens,), 'I32', 'input')
        pos = b.parameter('embeddings.position_embeddings.weight', (p.context, p.width))
        seg = b.parameter('embeddings.token_type_embeddings.weight', (config['type_vocab_size'], p.width))
        pos = b.op('embedding.position', 'gather_rows', [pos, positions], (tokens, p.width))
        seg = b.op('embedding.segment', 'gather_rows', [seg, segment], (tokens, p.width))
        x = b.op('embedding.sum', 'add', [x, pos, seg], (tokens, p.width), ordered=True)
        x = b.norm('embedding.norm', x, 'embeddings.LayerNorm', p.epsilon)
    else:
        valid = b.tensor('cache_valid_length', (), 'I32', 'input')
    for layer in range(p.layers):
        n = 'layer.' + str(layer)
        base = ('encoder.layer.' if p.family == 'bert' else 'model.layers.') + str(layer)
        residual = x
        if p.family == 'qwen2':
            x = b.norm(n + '.attention_norm', x, base + '.input_layernorm', p.epsilon, rms=True)
        ap = base + ('.attention.self' if p.family == 'bert' else '.self_attn')
        q = b.linear(n + '.q', x, ap + ('.query' if p.family == 'bert' else '.q_proj'), p.width)
        k = b.linear(n + '.k', x, ap + ('.key' if p.family == 'bert' else '.k_proj'), p.kv_heads * p.head_width)
        v = b.linear(n + '.v', x, ap + ('.value' if p.family == 'bert' else '.v_proj'), p.kv_heads * p.head_width)
        if p.family == 'qwen2':
            q = b.op(n + '.rope_q', 'rotary', [q, positions], (tokens, p.width),
                     theta=p.rope_theta, heads=p.heads, head_width=p.head_width, convention='split_half')
            k = b.op(n + '.rope_k', 'rotary', [k, positions], (tokens, p.kv_heads * p.head_width),
                     theta=p.rope_theta, heads=p.kv_heads, head_width=p.head_width, convention='split_half')
            ck = b.tensor(n + '.cache_k', (p.kv_heads, cache_capacity, p.head_width), role='state')
            cv = b.tensor(n + '.cache_v', (p.kv_heads, cache_capacity, p.head_width), role='state')
            b.nodes.append(dict(name=n + '.cache_update', op='kv_cache_update',
                                inputs=[k, v, positions, valid, ck, cv], outputs=[],
                                attrs=dict(writes=[ck, cv], cache_layout='head_position_element',
                                           position_semantics='absolute', bounds_check_required=True)))
            attention_inputs = [q, ck, cv, mask, positions, valid]
        else:
            attention_inputs = [q, k, v, mask]
        x = b.op(n + '.attention', 'attention', attention_inputs, (tokens, p.width),
                 heads=p.heads, kv_heads=p.kv_heads, head_width=p.head_width,
                 scale=1.0 / math.sqrt(p.head_width), causal=p.family == 'qwen2',
                 mask_semantics='one_valid_zero_padding', all_masked_policy='refuse',
                 dropout=0.0, evaluation_only=True)
        if p.family == 'qwen2':
            b.nodes[-1]['control_dependencies'] = [n + '.cache_update']
        oprefix = base + ('.attention.output.dense' if p.family == 'bert' else '.self_attn.o_proj')
        x = b.linear(n + '.attention_output', x, oprefix, p.width, bias=p.family == 'bert')
        x = b.op(n + '.attention_residual', 'add', [x, residual], (tokens, p.width), ordered=True)
        if p.family == 'bert':
            x = b.norm(n + '.attention_norm', x, base + '.attention.output.LayerNorm', p.epsilon)
        residual = x
        if p.family == 'qwen2':
            x = b.norm(n + '.ffn_norm', x, base + '.post_attention_layernorm', p.epsilon, rms=True)
            gate = b.linear(n + '.gate', x, base + '.mlp.gate_proj', p.hidden, bias=False)
            up = b.linear(n + '.up', x, base + '.mlp.up_proj', p.hidden, bias=False)
            gate = b.op(n + '.silu', 'silu', [gate], (tokens, p.hidden))
            x = b.op(n + '.gated_hidden', 'multiply', [gate, up], (tokens, p.hidden))
            x = b.linear(n + '.down', x, base + '.mlp.down_proj', p.width, bias=False)
        else:
            x = b.linear(n + '.up', x, base + '.intermediate.dense', p.hidden)
            x = b.op(n + '.gelu', 'gelu_erf', [x], (tokens, p.hidden))
            x = b.linear(n + '.down', x, base + '.output.dense', p.width)
        x = b.op(n + '.ffn_residual', 'add', [x, residual], (tokens, p.width), ordered=True)
        if p.family == 'bert':
            x = b.norm(n + '.ffn_norm', x, base + '.output.LayerNorm', p.epsilon)
    if p.family == 'qwen2':
        x = b.norm('final_norm', x, 'model.norm', p.epsilon, rms=True)
        x = b.linear('logits', x, 'model.embed_tokens' if p.tied_embeddings else 'lm_head', p.vocabulary, bias=False)
    elif sentence_pooling:
        x = b.op('sentence.mean', 'masked_mean', [x, mask], (p.width,), denominator_min=1e-9)
        x = b.op('sentence.embedding', 'l2_normalize', [x], (p.width,), epsilon=1e-12)
    state_bytes = sum(math.prod(t['shape']) * ELEMENT_BYTES[t['dtype']]
                      for t in b.tensors.values() if t['role'] == 'state')
    unused = set(params) - b.used
    allowed_unused = {
        'embeddings.position_ids': ((1, p.context), {'I64', 'I32'}),
        'pooler.dense.weight': ((p.width, p.width), {'F32', 'F16', 'BF16'}),
        'pooler.dense.bias': ((p.width,), {'F32', 'F16', 'BF16'}),
    } if p.family == 'bert' else {}
    for name in unused:
        if name not in allowed_unused:
            raise Unsupported('refused: unconsumed checkpoint tensor ' + name)
        shape, dtypes = allowed_unused[name]
        if params[name].shape != shape or params[name].dtype not in dtypes:
            raise InvalidCheckpoint('unused buffer/pooler shape/type mismatch: ' + name)
    return dict(format='g17-inference-graph-v1', status='logical_graph_not_lowered',
                execution_order='listed nodes; preserve explicit state writes and control dependencies',
                gpu_admitted=False, profile=asdict(p), tokens=tokens,
                tensors=b.tensors, nodes=b.nodes, outputs=[x],
                required_parameter_bytes=sum(params[n].bytes for n in b.used),
                kv_state_bytes=state_bytes, used_parameters=sorted(b.used),
                unused_checkpoint_tensors=sorted(unused),
                numerical_policy='logical FP32 computation; native transport/rounding policy must be explicitly selected and validated',
                required_native_operations=sorted({n['op'] for n in b.nodes}))
