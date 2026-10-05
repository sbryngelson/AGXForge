"""Independent allocation/launch requirements for isolated MiniLM FFN projections."""


def dimensions(kind, rows):
    if type(rows) is not int or rows not in (1, 32):
        raise ValueError('FFN projection rows must be 1 or 32')
    if kind == 'expand':
        return 384, 1536
    if kind == 'contract':
        return 1536, 384
    raise ValueError('unknown FFN projection')


def graph(kind, rows=32):
    ni, no = dimensions(kind, rows)
    allocations = {}
    for name, shape, role in [('source', [rows, ni], 'input'), ('weight', [no, ni], 'parameter'),
                               ('bias', [no], 'parameter'), ('output', [rows, no], 'output')]:
        elements = shape[0] * (shape[1] if len(shape) == 2 else 1)
        allocations[name] = dict(shape=shape, role=role, element_type='float',
            payload_bytes=4*elements, allocation_bytes=4*elements+256, offset=128)
    bindings = [dict(index=i+1, allocation=name, offset=128, length=a['payload_bytes'],
                     written=name == 'output') for i, (name, a) in enumerate(allocations.items())]
    return dict(format='g17-attention-graph-v1', status='proposed_not_executed', allocations=allocations,
                stages=[dict(name='output', program=kind, grid=[no, rows, 1], bindings=bindings)])


def requirements(kind, rows, abi, code_sha256):
    ni, no = dimensions(kind, rows)
    return {kind: dict(abi=abi, code_sha256=code_sha256, exact_grid=[no, rows, 1],
                       binding_elements=[rows*ni, no*ni, no, rows*no])}
