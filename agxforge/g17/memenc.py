"""Encoder for the per-lane vector load (op12709: 8, 10 and 14-byte forms) and store (op17256, 8-byte) from the
decoder-oracle field map (memmap.json), validated by round-tripping every load and store in this campaign's objects.
Same linear field model as mmaenc.py: decoded value = base + sum(weight_i * bit_i)."""
import json, os, sys
from pathlib import Path
# NO sys.path MANIPULATION. A module under agxforge/ must not touch sys.path: the library is
# imported as a package, and a module that edits the path on import decides what its
# callers can import. ROOT stays because these modules read repository data by path.
# Run as a script with `python -m agxforge.g17.memenc`.
ROOT = Path(__file__).resolve().parents[2]
from agxforge.g17 import model
# THE CENSUS LIVES UNDER isa/, NOT BESIDE THIS MODULE: agxforge/ carries no non-Python
# files, and a measured census is repository data rather than library code.
HERE = Path(__file__).resolve().parent
FM = json.loads((ROOT / 'isa/g17-tensor-mem-memmap.json').read_text())
names = model.registers(); INV = {v: k for k, v in names.items()}
FORMS = {(12709, 8): 'load8', (12709, 10): 'load10', (12709, 14): 'load14_disp', (17256, 8): 'store8', (17256, 14): 'store14', (17258, 10): 'mstore10', (17258, 16): 'mstore16',
         (12655, 10): 'load1w', (12674, 12): 'tload12', (12674, 16): 'tload16', (17257, 10): 'tstore10', (17257, 16): 'tstore16', (12675, 12): 'mload12', (12675, 16): 'mload16', (12655, 10): 'load1w', (12656, 12): 'tload1w12', (12656, 16): 'tload1w16', (12657, 12): 'mload1w12', (12657, 16): 'mload1w16'}
# the 14-byte load's word is not linear across bit 37 (group end): two references, chosen by that bit
def load14_form(word): return 'load14' if word >> 37 & 1 else 'load14_disp'

def _tables(form):
    f = FM[form]; ref = bytes.fromhex(f['reference'])
    dec = list(model.decode(ref, 0))[0]; vals = [(k, v) for k, v in dec.values if k != 'expr']
    fields = {}
    for k, spec in f['fields'].items():
        k = int(k); bits = []
        for r in spec['bits']:
            bit = r['bit']; setnow = (ref[bit // 8] >> (bit % 8)) & 1
            bits.append((bit, -r['delta'] if setnow else r['delta']))
        fields[k] = bits
    tpl = bytearray(ref); base = {}
    for k, bits in fields.items():
        v = vals[k][1]
        for bit, w in bits:
            if (tpl[bit // 8] >> (bit % 8)) & 1: tpl[bit // 8] &= ~(1 << (bit % 8)); v -= w
        base[k] = v
    return fields, bytes(tpl), base, vals

WIDTH_CODE = {16: 0, 1: 1, 4: 2, 8: 3}   # byte7[6:5]; code 3 is unwitnessed (assumed 8); width 2 = code 0 + byte8[7] in the 10-byte form

def encode(form, values, binding=0):
    """values: {operand index: decoded value}; register operands as register ids. Unlisted operands keep the reference's.
    binding: byte1, the buffer descriptor (4 x the binding rank; the decoder shows it only as an expr operand)."""
    fields, tpl, base, vals = _tables(form)
    out = bytearray(tpl); out[1] = binding
    width = values.get(7, vals[7][1])
    if not (form.startswith('mstore') or form.startswith('tload') or form.startswith('tstore') or form.startswith('mload') or form == 'load1w' or form.startswith('tload1w') or form.startswith('mload1w')):   # op12709/op17256: width code at byte7[6:5]; the others are linear
        out[7] &= ~0x60
        if width == 2:
            assert form == 'load10', 'width 2 needs the 10-byte form'
            out[8] |= 0x80
        else:
            out[7] |= WIDTH_CODE[width] << 5
            if form == 'load10': out[8] &= ~0x80
    for k, bits in fields.items():
        if k == 7 and not (form.startswith('mstore') or form.startswith('tload') or form.startswith('tstore') or form.startswith('mload') or form == 'load1w' or form.startswith('tload1w') or form.startswith('mload1w')): continue
        want = values.get(k, vals[k][1])
        b0 = base[k] + sum(w for _, w in bits if w < 0); rem = want - b0
        for bit, w in sorted(bits, key=lambda t: -abs(t[1])):
            on = rem >= abs(w)
            if on: rem -= abs(w)
            if w < 0: on = not on
            if on: out[bit // 8] |= 1 << (bit % 8)
        if rem != 0: raise ValueError('%s operand %d: %r not encodable (residual %r)' % (form, k, want, rem))
    out = bytes(out)
    # decode-back guard: Apple's decoder must read back exactly what was asked (an unrepresentable value that the linear
    # model happens to absorb, or a field the census missed, is caught here rather than in the hardware result)
    dec = list(model.decode(out, 0)); ref_op = list(model.decode(bytes.fromhex(FM[form]['reference']), 0))[0].opcode.id
    if not dec or dec[0].opcode is None or dec[0].opcode.id != ref_op or len(dec[0].raw) != len(out):
        raise ValueError('%s: encoding %s does not decode as op%d/%d bytes' % (form, out.hex(), ref_op, len(out)))
    got = [(k, v) for k, v in dec[0].values if k != 'expr']
    for k, want in values.items():
        if got[k][1] != want: raise ValueError('%s operand %d: asked %r, decodes as %r (%s)' % (form, k, want, got[k][1], out.hex()))
    if out[1] != binding: raise ValueError('binding byte')
    return out

def regid(base, n):
    return INV['_'.join('R%d' % (base + i) for i in range(n))] if n > 1 else INV['R%d' % base]

def load(dst, index, binding, disp=0, width=16, slot=7, group_end=False, index_last=False, form=None, wait_mask=0):
    """op12709: dst tuple base (4 words), index register number, displacement in bytes, width in bytes (scales the
    index), scoreboard slot 0..7 the load fills, group_end -> the 14-byte form carrying bit 37."""
    form = form or (('load14' if group_end else 'load14_disp') if (group_end or disp > 255) else 'load8')
    word = ((slot + 1) << 20) | (wait_mask << 24)
    if group_end: word |= 1 << 37
    v = {0: regid(dst, 4), 1: word, 4: regid(index, 1), 5: 16 if index_last else 0, 6: disp, 7: width}
    return 12709, encode(form, v, binding=4 * binding)

def store(src, index, binding, disp=0, width=16, src_last=True, index_last=False):
    v = {0: regid(src, 4), 1: 16 if src_last else 0, 4: regid(index, 1), 5: 16 if index_last else 0, 6: disp, 7: width}
    return 17256, encode('store14' if disp > 255 else 'store8', v, binding=4 * binding)

def mstore(src, index, binding, mask_reg, disp=0, width=4, src_last=True, index_last=False):
    """op17258 masked store of a 4-word tuple: mask_reg is a GPR16 register id (e.g. names inverse of 'R8L')"""
    form = 'mstore16' if disp > 255 else 'mstore10'
    v = {0: regid(src, 4), 1: (1 << 42) | (16 if src_last else 0), 4: regid(index, 1), 5: 16 if index_last else 0, 6: disp, 7: width, 8: mask_reg, 9: 0}
    return 17258, encode(form, v, binding=4 * binding)

def tload(dst, index, binding, disp=0, width=2, slot=7, mask=15, index_last=False, group_end=False, wait_mask=0):
    """op12674: the library's tensor load - two words (four halves) per lane at index x width + disp, element mask.
    wait_mask: eight-bit scoreboard mask (bits 24..31 of the word) the load waits on before issuing (e.g. its index register's load)"""
    form = ('tload16_end' if group_end else 'tload16') if (disp > 255 or group_end) else 'tload12'
    word = ((slot + 1) << 20) | (1 << 37 if group_end else 0) | (wait_mask << 24)
    v = {0: regid(dst, 2), 1: word, 4: regid(index, 1), 5: 16 if index_last else 0, 6: disp, 7: width, 8: mask}
    return 12674, encode(form, v, binding=4 * binding)

def mload(dst, index, binding, mask_reg, disp=0, width=2, slot=7, index_last=False, wait_mask=0):
    """op12675: the library's masked tensor load - two words per lane, per-lane element mask in a GPR16 (register id)"""
    form = 'mload16' if disp > 255 else 'mload12'
    word = ((slot + 1) << 20) | (wait_mask << 24)
    v = {0: regid(dst, 2), 1: word, 4: regid(index, 1), 5: 16 if index_last else 0, 6: disp, 7: width, 8: mask_reg, 9: 0}
    return 12675, encode(form, v, binding=4 * binding)

def load1w(dst, index, binding, disp=0, width=2, slot=7, index_last=False, wait_mask=0):
    """op12655: one 32-bit word per lane at index x width + disp (width 2 or 16; displacement up to 255 in this 10-byte form)"""
    word = ((slot + 1) << 20) | (wait_mask << 24)
    v = {0: regid(dst, 1), 1: word, 4: regid(index, 1), 5: 16 if index_last else 0, 6: disp, 7: width}
    return 12655, encode('load1w', v, binding=4 * binding)

def load1w(dst, index, binding, disp=0, width=2, slot=7, index_last=False, wait_mask=0):
    """op12655: one 32-bit word per lane at index x width + disp (width 2 or 16 in this form; eight-bit displacement)"""
    v = {0: INV['R%d' % dst], 1: ((slot + 1) << 20) | (wait_mask << 24), 4: regid(index, 1), 5: 16 if index_last else 0, 6: disp, 7: width}
    return 12655, encode('load1w', v, binding=4 * binding)

def tload1w(dst, index, binding, disp=0, width=1, slot=7, mask=15, index_last=False, wait_mask=0):
    """op12656: the library's ONE-WORD tensor load (four int8 elements per lane) at index x width + disp, width 1 = byte addressing, element mask"""
    form = 'tload1w16' if disp > 255 else 'tload1w12'
    word = ((slot + 1) << 20) | (wait_mask << 24)
    v = {0: regid(dst, 1), 1: word, 4: regid(index, 1), 5: 16 if index_last else 0, 6: disp, 7: width, 8: mask}
    return 12656, encode(form, v, binding=4 * binding)

def mload1w(dst, index, binding, mask_reg, disp=0, width=1, slot=7, index_last=False, wait_mask=0):
    """op12657: the masked one-word tensor load, per-lane element mask in a GPR16 (register id)"""
    form = 'mload1w16' if disp > 255 else 'mload1w12'
    word = ((slot + 1) << 20) | (wait_mask << 24)
    v = {0: regid(dst, 1), 1: word, 4: regid(index, 1), 5: 16 if index_last else 0, 6: disp, 7: width, 8: mask_reg, 9: 0}
    return 12657, encode(form, v, binding=4 * binding)

def tstore(src, index, binding, disp=0, width=4, mask=15, src_last=True, index_last=False):
    """op17257: the library's tensor store - four words per lane at index x width + disp, element mask"""
    form = 'tstore16' if disp > 255 else 'tstore10'
    v = {0: regid(src, 4), 1: (1 << 42) | (16 if src_last else 0), 4: regid(index, 1), 5: 16 if index_last else 0, 6: disp, 7: width, 8: mask}
    return 17257, encode(form, v, binding=4 * binding)

def roundtrip_all():
    ok = bad = 0; fails = []
    for n in sorted(d for d in os.listdir(HERE) if (HERE / d / 'code.bin').exists()):
        for i in model.decode((HERE / n / 'code.bin').read_bytes(), 0):
            if not i.opcode or (i.opcode.id, len(i.raw)) not in FORMS: continue
            vals = [(k, v) for k, v in i.values if k != 'expr']
            form = FORMS[(i.opcode.id, len(i.raw))]
            if form == 'load14_disp': form = load14_form(vals[1][1])
            if form == 'tload16' and vals[1][1] >> 37 & 1: form = 'tload16_end'
            try: enc = encode(form, {k: v[1] for k, v in enumerate(vals)}, binding=i.raw[1])
            except ValueError as e: bad += 1; fails.append((n, i.offset, i.raw.hex(), str(e))); continue
            if enc == i.raw: ok += 1
            else: bad += 1; fails.append((n, i.offset, i.raw.hex(), enc.hex()))
    return ok, bad, fails

if __name__ == '__main__':
    ok, bad, fails = roundtrip_all(); print('round trip: %d exact, %d differ' % (ok, bad))
    for f in fails[:12]: print('  ', f)
    (HERE / 'memenc_roundtrip.json').write_text(json.dumps(dict(exact=ok, differ=bad, failures=fails), indent=1) + '\n')
