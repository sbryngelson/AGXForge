"""Launch and allocation requirements for the delivered elementwise FFN GELU.

The IR uses thread_position_in_grid.x directly as both load and store index.
The flat launch follows from that addressing, not merely from the presence of
SR160 in metadata. Diagnostic and full runs use identical program bytes.
"""

WIDTH = 1536


def elements(rows):
    if type(rows) is not int or rows not in (1, 32):
        raise ValueError('GELU rows must be 1 or 32')
    return rows * WIDTH


def graph(rows=32):
    count = elements(rows)
    allocations = {
        name: dict(shape=[rows, WIDTH], role=role, element_type='float',
                   payload_bytes=4*count, allocation_bytes=4*count+256, offset=128)
        for name, role in [('source', 'input'), ('output', 'output')]
    }
    bindings = [dict(index=index, allocation=name, offset=128,
                     length=4*count, written=name == 'output')
                for index, name in [(1, 'source'), (2, 'output')]]
    return dict(format='g17-attention-graph-v1', status='proposed_not_executed',
                allocations=allocations, stages=[dict(name='output', program='gelu',
                grid=[count, 1, 1], bindings=bindings)])


def requirements(rows, abi, code_sha256):
    count = elements(rows)
    if abi.get('system_registers') != [160]:
        raise ValueError('delivered GELU must declare only SR160')
    return {'gelu': dict(abi=abi, code_sha256=code_sha256,
                         exact_grid=[count, 1, 1], binding_elements=[count, count])}
