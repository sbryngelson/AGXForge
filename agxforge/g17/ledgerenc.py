"""Generic encoder from the repository's certified bit ledger (isa/g17-contract.jsonl): for an opcode, start from its
witness (all mapped bits cleared), write each operand's bits ('slot' operands in 16-bit register units: R n -> 2n,
R nL -> 2n, R nH -> 2n+1; 'raw'/'index' literal), then decode back with Apple's decoder and refuse any mismatch.
Only the operands the ledger maps can be set; unmapped bytes are the witness's."""
import json, sys
from pathlib import Path
# NO sys.path MANIPULATION. A module under agxforge/ must not touch sys.path: the library is
# imported as a package, and a module that edits the path on import decides what its
# callers can import. ROOT stays because these modules read repository data by path.
# Run as a script with `python -m agxforge.g17.ledgerenc`.
ROOT = Path(__file__).resolve().parents[2]
from agxforge.g17 import model
names = model.registers(); INV = {v: k for k, v in names.items()}
LEDGER = {}
for line in open(ROOT / 'isa/g17-contract.jsonl'):
    r = json.loads(line); LEDGER[r['opcode']] = r
def reg16(name):
    """'R8' -> 16, 'R8L' -> 16, 'R8H' -> 17"""
    n = int(''.join(c for c in name[1:] if c.isdigit())); return 2 * n + (1 if name.endswith('H') else 0)
def encode(op, values, witness=None):
    """values: {operand index: value}; slot operands take a register NAME ('R8', 'R3L') or an int in 16-bit units"""
    r = LEDGER[op]; e = r['encoding']; F = e['fields']
    wit = bytes.fromhex(witness or e['witness'])
    dec0 = list(model.decode(wit, 0)); assert dec0 and dec0[0].opcode and dec0[0].opcode.id == op, 'witness does not decode as op%d' % op
    L = len(dec0[0].raw); u = bytearray(wit[:L])
    for key, bits in F.items():
        for w, byte, bit, inv in bits:
            if byte < L: u[byte] &= ~(1 << bit)
    kinds = {int(k.split(':')[0]): (k, k.split(':')[1]) for k in F}
    for idx, val in values.items():
        key, kind = kinds[idx]
        if isinstance(val, tuple): v = val[0]                       # (raw field value, expected decoded register name)
        else: v = reg16(val) if (kind == 'slot' and isinstance(val, str)) else val
        for w, byte, bit, inv in F[key]:
            on = (v >> w) & 1
            if inv: on ^= 1
            if on: u[byte] |= 1 << bit
    for idx in kinds:                       # operands not given keep the witness's value: re-apply them
        if idx in values: continue
        key, kind = kinds[idx]; v = dec0[0].values[idx][1]
        for w, byte, bit, inv in F[key]:
            on = (v >> w) & 1
            if inv: on ^= 1
            if on: u[byte] |= 1 << bit
    u = bytes(u); d = list(model.decode(u, 0))
    if not d or d[0].opcode is None or d[0].opcode.id != op or len(d[0].raw) != L: raise ValueError('op%d: %s does not decode' % (op, u.hex()))
    for idx, val in values.items():
        got = d[0].values[idx]; key, kind = kinds[idx]
        if kind == 'slot':
            want = val[1] if isinstance(val, tuple) else (val if isinstance(val, str) else names.get(val))
            if names.get(got[1]) != want and not (isinstance(val, str) and names.get(got[1]) == val): raise ValueError('op%d operand %d: asked %s decodes %s' % (op, idx, val, names.get(got[1])))
        elif got[1] != val: raise ValueError('op%d operand %d: asked %r decodes %r' % (op, idx, val, got[1]))
    return u
if __name__ == '__main__':
    for op, vals in ((423, {0: 'R8', 2: 'R8', 4: 7}), (426, {0: 'R8', 2: 'R8L', 4: 1}), (17016, {0: 'R8', 3: 'R8L', 5: 2}), (14391, {0: 'R8', 3: 'R8', 5: 3}), (10279, {0: 'R8', 3: 'R8', 2: 5}), (14060, {0: 'R8L', 2: 45})):
        try: b = encode(op, vals); print(op, b.hex(), [(k, names.get(v) if k == 'reg' else v) for k, v in list(model.decode(b, 0))[0].values])
        except Exception as ex: print(op, 'ERR', ex)
