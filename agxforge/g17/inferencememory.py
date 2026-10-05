"""Logical resident tensor lifetimes; offsets are not AGX resource admission."""
import hashlib
import json
import math
from .inferencegraph import ELEMENT_BYTES, Unsupported


def plan(graph, *, alignment=256, guard_bytes=256):
    if graph.get('format') != 'g17-inference-graph-v1' or graph.get('gpu_admitted') is not False:
        raise Unsupported('refused: expected unlowered inference graph')
    if type(alignment) is not int or alignment < 16 or alignment & (alignment-1):
        raise Unsupported('refused: power-of-two alignment at least sixteen')
    if type(guard_bytes) is not int or guard_bytes < alignment or guard_bytes % alignment:
        raise Unsupported('refused: aligned guards required')
    tensors, nodes = graph['tensors'], graph['nodes']
    if not nodes or not graph['outputs'] or any(n not in tensors for n in graph['outputs']):
        raise Unsupported('refused: incomplete graph')
    sizes = {}
    birth, death = {}, {}
    for name, tensor in tensors.items():
        shape = tensor['shape']
        if tensor['dtype'] not in ELEMENT_BYTES or not isinstance(shape, list) or any(type(n) is not int or n <= 0 for n in shape):
            raise Unsupported('refused: tensor shape/dtype: ' + name)
        if tensor['role'] not in ('parameter', 'input', 'state', 'intermediate'):
            raise Unsupported('refused: tensor role: ' + name)
        sizes[name] = math.prod(shape) * ELEMENT_BYTES[tensor['dtype']]
        if tensor['role'] != 'intermediate':
            birth[name], death[name] = -1, len(nodes)
    seen = set()
    for step, node in enumerate(nodes):
        if node['name'] in seen or any(dep not in seen for dep in node.get('control_dependencies', [])):
            raise Unsupported('refused: duplicate node or unsatisfied control dependency')
        for name in node['inputs']:
            if name not in birth:
                raise Unsupported('refused: read before producer: ' + name)
            death[name] = max(death[name], step)
        for name in node.get('attrs', {}).get('writes', []):
            if name not in tensors or tensors[name]['role'] != 'state' or name not in node['inputs']:
                raise Unsupported('refused: only declared state inputs may be mutated')
        for name in node['outputs']:
            if name not in tensors or tensors[name]['role'] != 'intermediate' or name in birth:
                raise Unsupported('refused: duplicate or non-intermediate producer')
            birth[name], death[name] = step, step
        seen.add(node['name'])
    if set(birth) != set(tensors):
        raise Unsupported('refused: graph tensor without producer')
    for name in graph['outputs']:
        death[name] = len(nodes)
    def rounded(n):
        return (n + alignment - 1) // alignment * alignment
    arenas = {key: 0 for key in ('parameters', 'inputs', 'state', 'activations')}
    slots, regions = [], {}
    for name in sorted(tensors, key=lambda n: (birth[n], n)):
        role = tensors[name]['role']
        arena = {'parameter': 'parameters', 'input': 'inputs', 'state': 'state', 'intermediate': 'activations'}[role]
        capacity = rounded(sizes[name])
        required = capacity + 2 * guard_bytes
        available = [s for s in slots if s['last_use'] < birth[name] and s['capacity'] >= capacity] if arena == 'activations' else []
        if available:
            slot = min(available, key=lambda s: (s['capacity'], s['offset']))
            slot['last_use'] = death[name]
        else:
            slot = dict(offset=arenas[arena], capacity=capacity, last_use=death[name])
            arenas[arena] += required
            if arena == 'activations':
                slots.append(slot)
        regions[name] = dict(arena=arena, offset=slot['offset'] + guard_bytes,
                             bytes=sizes[name], capacity=slot['capacity'],
                             guard_before=slot['offset'], guard_after=slot['offset'] + guard_bytes + slot['capacity'],
                             guard_bytes=guard_bytes, birth=birth[name], last_use=death[name],
                             readonly=role in ('parameter', 'input'), persistent=role in ('parameter', 'state'))
    for name, region in regions.items():
        if tensors[name]['role'] == 'parameter':
            begin, end = tensors[name].get('checkpoint_payload_begin'), tensors[name].get('checkpoint_payload_end')
            if type(begin) is not int or type(end) is not int or begin < 0 or end - begin != sizes[name]:
                raise Unsupported('refused: checkpoint parameter extent: ' + name)
            region['checkpoint_payload_begin'], region['checkpoint_payload_end'] = begin, end
    return dict(format='g17-inference-memory-plan-v1', status='logical_offsets_not_allocated',
                gpu_admitted=False, graph_sha256=hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest(),
                alignment=alignment, arenas=arenas, regions=regions,
                activation_slots=len(slots),
                reset_state=[dict(name=n, offset=r['offset'], bytes=r['bytes']) for n,r in regions.items() if r['arena']=='state'],
                cache_control='reset valid length to zero; append/read bounds require runtime admission',
                execution_order=[n['name'] for n in nodes],
                scope='checkpoint dtypes retained; no quantization, resource addresses, submission or kernel temporaries admitted')
