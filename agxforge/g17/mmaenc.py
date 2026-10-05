"""Encoder for the accelerator MMA family (op5106/5107/5104/5098/10384), derived from the decoder-oracle field
map (fieldmap.json) and validated by round-tripping every MMA the compiler emitted in this campaign.

Field model: each decoded operand value = base + sum(weight_i * bit_i) over the bits the census found for it,
where base is the reference encoding's value with all of its variable bits cleared. Encoding is the reverse:
start from the reference with every variable bit cleared, then set the bits for the wanted values.
Operands (Apple's order): 0 D:tup8, 1 flags word, 2 imm, 3 A, 4 A-flags, 5 A-type, 6 B, 7 B-flags, 8 B-type,
[9 C:tup8, 10, 11 for the with-C forms]. D and C share bits (in-place accumulate).
"""
import json, sys
from pathlib import Path
# NO sys.path MANIPULATION. A module under agxforge/ must not touch sys.path: the library is
# imported as a package, and a module that edits the path on import decides what its
# callers can import. ROOT stays because these modules read repository data by path.
# Run as a script with `python -m agxforge.g17.mmaenc`.
ROOT = Path(__file__).resolve().parents[2]
from agxforge.g17 import model
# THE CENSUS LIVES UNDER isa/, NOT BESIDE THIS MODULE: agxforge/ carries no non-Python
# files, and a measured census is repository data rather than library code.
HERE = Path(__file__).resolve().parent
FM = json.loads((ROOT / 'isa/g17-tensor-mma-fieldmap.json').read_text())
names = model.registers()
REG_UNIT = {'GPR32tup8_alignedrc': 8, 'GPR32tup4_alignedrc': 4, 'GPR32tup2_alignedrc': 2}

def _tables(op):
    f = FM[str(op)]; ref = bytes.fromhex(f['reference'])
    fields = {}
    for k, spec in f['fields'].items():
        k = int(k); bits = []
        for r in spec['bits']:
            bit = r['bit']; setnow = (ref[bit // 8] >> (bit % 8)) & 1
            weight = -r['delta'] if setnow else r['delta']          # weight of the bit when it is 1
            bits.append((bit, weight))
        fields[k] = dict(kind=spec['kind'], operand=spec['operand'], bits=bits)
    # template: reference with every variable bit cleared; base values with those contributions removed
    tpl = bytearray(ref); base = {}
    dec = list(model.decode(ref, 0))[0]
    for k, spec in fields.items():
        v = dec.values[k][1]
        for bit, w in spec['bits']:
            if (tpl[bit // 8] >> (bit % 8)) & 1:
                tpl[bit // 8] &= ~(1 << (bit % 8)); v -= w
        base[k] = v
    return fields, bytes(tpl), base, dec

def encode(op, values):
    """values: {operand index: wanted decoded value} (register operands as register ids, e.g. names inverse)"""
    fields, tpl, base, dec = _tables(op)
    out = bytearray(tpl)
    for k, spec in fields.items():
        want = values.get(k, dec.values[k][1])
        bit36 = False
        if k == 1 and want & (1 << 36):       # byte4[6:5] is a 2-bit code: 01 -> 0, 00 -> 2^37, 10 -> 2^36, 11 invalid (probed)
            want -= 1 << 36; bit36 = True
        # inverted bits (negative weight w): value = base + w*bit = (base + w) + |w|*(1 - bit); encode the complement
        b0 = base[k] + sum(w for _, w in spec['bits'] if w < 0)
        rem = want - b0
        for bit, w in sorted(spec['bits'], key=lambda t: -abs(t[1])):
            if rem >= abs(w):
                rem -= abs(w); on = True
            else: on = False
            if w < 0: on = not on
            if on: out[bit // 8] |= 1 << (bit % 8)
        if rem != 0:
            raise ValueError('operand %d (%s): value %r not encodable (residual %r)' % (k, spec['operand'], want, rem))
        if bit36:
            out[4] = (out[4] & ~0x20) | 0x40
    out = bytes(out)
    dec = list(model.decode(out, 0))                      # decode-back guard
    if not dec or dec[0].opcode is None or dec[0].opcode.id != op or len(dec[0].raw) != 10: raise ValueError('op%d: %s does not decode' % (op, out.hex()))
    for k, want in values.items():
        if dec[0].values[k][1] != want: raise ValueError('op%d operand %d: asked %r, decodes as %r' % (op, k, want, dec[0].values[k][1]))
    return out

def roundtrip_all():
    """decode every MMA in every built object, re-encode from its decoded values, compare bytes"""
    import os
    ok = bad = 0; failures = []
    for n in sorted(d for d in os.listdir(HERE) if (HERE / d / 'code.bin').exists()):
        code = (HERE / n / 'code.bin').read_bytes()
        for i in model.decode(code, 0):
            if i.opcode and i.opcode.id in (5106, 5107, 5104, 5098, 10384, 5100, 5101, 5099, 5105, 10385):
                try:
                    enc = encode(i.opcode.id, {k: v[1] for k, v in enumerate(i.values)})
                except ValueError as e:
                    bad += 1; failures.append((n, i.offset, i.raw.hex(), str(e))); continue
                if enc == i.raw: ok += 1
                else: bad += 1; failures.append((n, i.offset, i.raw.hex(), enc.hex()))
    return ok, bad, failures

if __name__ == '__main__':
    ok, bad, failures = roundtrip_all()
    print('round trip: %d exact, %d differ' % (ok, bad))
    for f in failures[:20]: print('  ', f)
    (HERE / 'mmaenc_roundtrip.json').write_text(json.dumps(dict(exact=ok, differ=bad, failures=failures), indent=1) + '\n')

# ---------------------------------------------------------------- semantic layer
TYPE = {'half': 2, 'bfloat': 3, 'float': 1, 'int8': 75, 'uint8': 11}
REV = {v: k for k, v in TYPE.items()}
INV = {v: k for k, v in names.items()}
def regid(base, n):
    """register id of the aligned tuple R{base}..R{base+n-1}"""
    return INV['_'.join('R%d' % (base + i) for i in range(n))]
def mma(D, A, B, C=None, a_type='half', b_type='half', transA=False, transB=False, wait=False, tag=0, more=False, a_last=False, b_last=False,
        saturate=False):
    """Bytes of one accelerator MMA. D, A, B, C: base register numbers (D == C or C None -> the no-accumulator form).
    tag: 0 or a ring slot 1..7 (bit 24+tag-1 of the flags word). Returns (opcode, bytes).

    saturate: the int8 MMA with C that clips C + A.B to int32 (recon section 136 part 6), which is
    operand 2 = 9 + 32. Apple's compiler emits exactly that difference for the `_saturate` spelling
    (seta-int8-{plain,sat}-v1, compile only). The no-C form is refused: its saturating encoding
    differs in bytes 6 and 7, which this encoder does not reproduce, and it cannot saturate anyway
    (16 int8 products stay far inside int32)."""
    f32 = (a_type, b_type)
    if a_type in ('int8', 'uint8'):
        op = 10384 if C is not None else 10385
        wide = 2
    elif f32 == ('float', 'float'): op, wide = (5098 if C is not None else 5099), 8
    elif b_type == 'float': op, wide = (5104 if C is not None else 5105), 8
    elif a_type == 'float': op, wide = (5100 if C is not None else 5101), 8     # decoder-reachable, never compiler-emitted (family_exh.py)
    else: op, wide = (5106 if C is not None else 5107), 4
    if str(op) not in FM: raise ValueError('no field map for op%d (no reference encoding in this campaign)' % op)
    word = (wait << 31) | ((1 << (24 + tag - 1)) if tag else 0) | (0x20 if more else 0)
    a_n = 8 if a_type == 'float' else (2 if a_type in ('int8', 'uint8') else 4)
    b_n = 8 if b_type == 'float' else (2 if b_type in ('int8', 'uint8') else 4)
    vals = {0: regid(D, 8), 1: word, 3: regid(A, a_n), 4: 16 if a_last else 0, 5: TYPE[a_type] + (32 if transA else 0),
            6: regid(B, b_n), 7: 16 if b_last else 0, 8: TYPE[b_type] + (32 if transB else 0)}
    if C is not None:
        assert C == D, 'the compiler only emits in-place accumulation (D == C)'
        vals[9] = regid(C, 8)
    if saturate:
        if op != 10384:
            raise ValueError('refused: saturation is measured for the int8 MMA with C (op10384) only')
        vals[2] = SATURATE_OPERAND_2
    return op, encode(op, vals)

SATURATE_OPERAND_2 = 9 + 32      # plain op10384 carries 9 here; the `_saturate` spelling carries 41

def semantic_roundtrip():
    """decode -> semantic fields -> mma() -> bytes, for every compiler MMA"""
    import os
    ok = bad = 0; fails = []
    for n in sorted(d for d in os.listdir(HERE) if (HERE / d / 'code.bin').exists()):
        for i in model.decode((HERE / n / 'code.bin').read_bytes(), 0):
            if not (i.opcode and i.opcode.id in (5106, 5107, 5104, 5098, 10384, 5100, 5101, 5099, 5105, 10385)): continue
            v = i.values; w = v[1][1]
            D = int(names[v[0][1]].split('_')[0][1:]); A = int(names[v[3][1]].split('_')[0][1:]); B = int(names[v[6][1]].split('_')[0][1:])
            tagbits = [b for b in range(24, 31) if w >> b & 1]
            if len(tagbits) > 1 or (w & ~((1 << 31) | 0x7f000000 | 0x20)):
                bad += 1; fails.append((n, i.offset, i.raw.hex(), 'flag word outside the semantic model: %x' % w)); continue
            a_t = REV[v[5][1] & ~32]; b_t = REV[v[8][1] & ~32]
            try:
                op, enc = mma(D, A, B, D if len(v) > 9 else None, a_t, b_t, bool(v[5][1] & 32), bool(v[8][1] & 32), bool(w >> 31 & 1), (tagbits[0] - 23) if tagbits else 0, bool(w & 0x20), v[4][1] == 16, v[7][1] == 16)
            except Exception as e:
                bad += 1; fails.append((n, i.offset, i.raw.hex(), str(e))); continue
            if op == i.opcode.id and enc == i.raw: ok += 1
            else: bad += 1; fails.append((n, i.offset, i.raw.hex(), enc.hex()))
    return ok, bad, fails
if __name__ == '__main__':
    ok, bad, fails = semantic_roundtrip(); print('semantic round trip: %d exact, %d differ' % (ok, bad))
    for f in fails[:8]: print('  ', f)
    (HERE / 'mmaenc_roundtrip.json').write_text(json.dumps(dict(semantic_exact=ok, semantic_differ=bad, semantic_failures=fails), indent=1) + '\n')
