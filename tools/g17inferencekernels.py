"""Native inference IR kernels. Compiled artifacts are not launch admission."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agxforge.g17 import ir, cc


def gather_rows_ir(*, rows, width, vocabulary, dtype='F32'):
    """GPU embedding lookup; IDs must be validated before readonly upload.

    Grid [width, rows, 1] gives each output element one owner. Three bindings:
    I32 IDs, row-major FP32 table, FP32 output. No host embedding arithmetic.
    BF16 is read as little-endian packed uint32 pairs and widened by exact
    integer bit placement. Neither variant is hardware-admitted here.
    """
    if any(type(n) is not int or n <= 0 for n in (rows, width, vocabulary)):
        raise ValueError('refused: positive embedding extents required')
    if width % 32 or rows * width * 4 >= 2**32 or vocabulary * width * 4 >= 2**32:
        raise ValueError('refused: embedding grid alignment or 32-bit address extent')
    if dtype not in ('F32', 'BF16'):
        raise ValueError('refused: embedding lookup supports FP32/BF16 tables only')
    ids = ir.Buffer('token_ids', 1, elem=ir.I32)
    table = ir.Buffer('embedding_table', 2, elem=ir.F32 if dtype == 'F32' else ir.I32)
    output = ir.Buffer('embedding_output', 3, elem=ir.F32)
    fn = ir.Function('native_inference_gather_rows', [ids, table, output])
    b = ir.Builder(fn, fn.block('entry'))
    col = b.builtin('thread_position_in_grid', axis='x')
    row = b.builtin('thread_position_in_grid', axis='y')
    token = b.load(ids, row)
    if dtype == 'F32':
        source_index = b.add(b.mul(token, b.const(width)), col)
        output_index = b.add(b.mul(row, b.const(width)), col)
        value = b.load(table, source_index, type=ir.F32)
    else:
        # One aligned 32-bit load contains two BF16 elements. Column parity
        # selects the low/high half without a floating ALU or conversion.
        output_index = b.add(b.mul(row, b.const(width)), col)
        pair = b.shr(col, b.const(1))
        source_index = b.add(b.mul(token, b.const(width // 2)), pair)
        packed = b.load(table, source_index)
        parity = b._def('and', [col, b.const(1)])
        shift = b.shl(parity, b.const(4))
        bits16 = b._def('and', [b.shr(packed, shift), b.const(0xffff)])
        value = b.shl(bits16, b.const(16))
    b.store_at(output, output_index, value)
    b.ret()
    return fn


def prepare_gather(destination, *, rows, width, vocabulary, dtype='F32'):
    function = gather_rows_ir(rows=rows, width=width, vocabulary=vocabulary, dtype=dtype)
    program = cc.compile_function(function)
    result = dict(format='g17-native-inference-kernel-v1', operation='gather_rows',
                  status='compiled_not_executed', gpu_admitted=False,
                  code_hex=program.code.hex(), code_bytes=len(program.code), code_sha256=hashlib.sha256(program.code).hexdigest(),
                  abi=program.abi_plain(program.abi()), grid=[width, rows, 1],
                  threadgroup=[32, 1, 1], table_dtype=dtype, table_transport='FP32 scalars' if dtype=='F32' else 'little-endian BF16 pairs in uint32 words',
                  binding_bytes=[rows*4, vocabulary*width*(4 if dtype=='F32' else 2), rows*width*4],
                  preconditions=['every token ID is an integer in [0,vocabulary)',
                                 'IDs and table readonly throughout dispatch',
                                 'exact grid; no implicit row or column padding'],
                  readonly_bindings=[1, 2], output_binding=3,
                  numerical_policy='bit-preserving FP32 lookup' if dtype=='F32' else 'exact BF16-to-FP32 bit widening, including signed zero, infinities and NaN payload bits')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'program.bin').write_bytes(program.code)
    (destination / 'contract.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--rows', type=int, default=32)
    parser.add_argument('--width', type=int, default=384)
    parser.add_argument('--vocabulary', type=int, default=30522)
    parser.add_argument('--dtype', choices=('F32', 'BF16'), default='F32')
    args = parser.parse_args()
    print(json.dumps(prepare_gather(args.destination, rows=args.rows, width=args.width,
                                   vocabulary=args.vocabulary, dtype=args.dtype), indent=2))
