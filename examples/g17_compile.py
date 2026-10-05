"""Compile and author the retained n91 pressure kernel, without loading Metal.

From the checkout: python3 -m examples.g17_compile /tmp/g17-example
The output directory must not exist. This is image generation, not dispatch.
"""
import argparse
import hashlib
import json
from pathlib import Path

from agxforge.g17 import cc, ir, scanlink


def compile_image():
    """Ninety-one simultaneously live values exercise automatic scratch spilling.

    Each thread reads its own 91-word region and overwrites its first word with
    sum(input) + sum(range(91)), modulo 2**32. Scratch allocation requirements
    belong to the returned contract; an eventual runtime must satisfy them.
    """
    fn = ir.Function('disjoint_inplace_pressure', [ir.Buffer('IO', 1)])
    b = ir.Builder(fn, fn.block('entry'))
    tid = b.builtin('thread_position_in_grid', name='t')
    base = b.mul(tid, ir.Imm(91), name='base')
    values = [b.add(b.load(fn.buffers[0], base, offset=i, name='l%d' % i),
                    ir.Imm(i), name='v%d' % i) for i in range(91)]
    total = values[-1]
    for value in reversed(values[:-1]):
        total = b.add(total, value, name='s%d' % len(fn.blocks[0].ops))
    b.store_at(fn.buffers[0], base, total)
    b.ret()
    program = cc.compile_function(fn)
    return program, scanlink.author(program)



def compile_texture_pair(*, coordinate_inputs=False):
    """Compile two load-bearing FP32 texture reads; image/resource support is separate.

    One thread reads coordinate (0, 0) from each texture and writes each result
    to its own output slot. Keeping both results exposes binding substitution
    without relying on unmeasured arithmetic on texture-fetch results.
    """
    fn = ir.Function('texture_read', [ir.Buffer('o', 0)])
    b = ir.Builder(fn, fn.block('entry'))
    if coordinate_inputs:
        thread = b.builtin('thread_position_in_grid', name='t', axis='x')
    else:
        x = b.builtin('thread_position_in_grid', name='x', axis='x')
        y = b.builtin('thread_position_in_grid', name='y', axis='y')
    for texture, slot in ((0, 16), (1, 20)):
        if coordinate_inputs:
            # One admitted thread: the public buffer carries x/y at words 4..7.
            # Materialized indexed loads follow the measured texture load form.
            x = b.load(fn.buffers[0], b.add(thread, b.const(4 + 2 * texture)), name='x%d' % texture)
            y = b.load(fn.buffers[0], b.add(thread, b.const(5 + 2 * texture)), name='y%d' % texture)
        value = b.texture_read(x, y, tex=texture, type=ir.F32, name='tex%d' % texture)
        b.store_fetch(fn.buffers[0], ir.Imm(slot), value,
                      b.const(0, name='companion%d' % texture), components=1)
    b.ret()
    return cc.compile_function(fn)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    program, image = compile_image()
    files = {'program.bin': program.code, 'program.o': image.object,
             'program.lib.metallib': image.library, 'program.arc.metallib': image.archive}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (args.output / name).write_bytes(data)
    (args.output / 'contract.json').write_text(json.dumps(program.contract().to_dict(), indent=2) + '\n')
    print(json.dumps({'status': 'authored_not_dispatched',
                      'sha256': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}, indent=2))


if __name__ == '__main__':
    main()
