"""fadd (op998, 12-byte form) encoder from the repository's certified bit ledger (isa/g17-contract.jsonl: register
'slot' fields are bit indices in 16-bit register units, so bit w = 2^w half-registers = R(2^(w-1)); flags word bits
as on the MMA), validated by round-tripping every 12-byte op998 in this campaign's objects through Apple's decoder."""
import json, os, sys
from pathlib import Path
# NO sys.path MANIPULATION. A module under agxforge/ must not touch sys.path: the library is
# imported as a package, and a module that edits the path on import decides what its
# callers can import. ROOT stays because these modules read repository data by path.
# Run as a script with `python -m agxforge.g17.faddenc`.
ROOT = Path(__file__).resolve().parents[2]
from agxforge.g17 import model
HERE = Path(__file__).resolve().parent
names = model.registers(); INV = {v: k for k, v in names.items()}
LEDGER = {}
for line in open(ROOT / 'isa/g17-contract.jsonl'):
    r = json.loads(line)
    if r['opcode'] in (998, 10282): LEDGER[r['opcode']] = r
def encode(dest, src1, src2, word=0, f1=16, f2=16, op=998):
    """three-register ALU form (op998 fadd, op10282 integer add), 12 bytes. dest/src1/src2: register numbers (32-bit);
    word: flags (bit 31 = slot-7 wait, bits 24..30 slots 0..6); f1/f2: operand flags (16 = last use)"""
    F = LEDGER[op]['encoding']['fields']
    # the ledger's op10282 witness carries byte10 = 0x24 (an unmapped mode; its adds are 'add.block' forms); the form
    # matmul2d uses for its int32 C add has byte10 = 0x21 - taken from or_48x32x64_acc_int8.int8 at 0x5e4 - so that
    # instance is the template for the unmapped bits. op998's ledger witness is the fadd form the library uses.
    WIT = {998: LEDGER[998]['encoding']['witness'], 10282: '2f00849a2180a32229982100'}[op]
    u = bytearray(bytes.fromhex(WIT)[:12])
    for k, bits in F.items():
        for w, byte, bit, inv in bits:
            if byte < 12: u[byte] &= ~(1 << bit)
    def put(key, value):
        for w, byte, bit, inv in F[key]:
            on = (value >> w) & 1
            if inv: on ^= 1
            if on: u[byte] |= 1 << bit
    put('0:slot', dest * 2); put('2:slot', src1 * 2); put('4:slot', src2 * 2); put('1:raw', word); put('3:raw', f1); put('5:raw', f2)
    u = bytes(u)
    # decode-back guard (a register above the field's range would otherwise be silently truncated)
    dec = list(model.decode(u, 0))
    if not dec or dec[0].opcode is None or dec[0].opcode.id != op or len(dec[0].raw) != 12: raise ValueError('op%d: %s does not decode' % (op, u.hex()))
    v = dec[0].values; want = {0: 'R%d' % dest, 2: 'R%d' % src1, 4: 'R%d' % src2}
    for k, w in want.items():
        if names.get(v[k][1]) != w: raise ValueError('op%d operand %d: asked %s, decodes as %s' % (op, k, w, names.get(v[k][1])))
    if (v[1][1], v[3][1], v[5][1]) != (word, f1, f2): raise ValueError('op%d flags: asked %r decodes %r' % (op, (word, f1, f2), (v[1][1], v[3][1], v[5][1])))
    return u
def roundtrip_all():
    ok = bad = 0; fails = []
    for n in sorted(d for d in os.listdir(HERE) if (HERE / d / 'code.bin').exists()):
        for i in model.decode((HERE / n / 'code.bin').read_bytes(), 0):
            if not (i.opcode and i.opcode.id in (998, 10282) and len(i.raw) == 12): continue
            v = i.values
            reg = lambda k: int(names[v[k][1]][1:].split('_')[0]) if names[v[k][1]][1:2].isdigit() else None
            try: enc = encode(reg(0), reg(2), reg(4), v[1][1], v[3][1], v[5][1], op=i.opcode.id)
            except Exception as e: bad += 1; fails.append((n, i.offset, i.raw.hex(), str(e))); continue
            if enc == i.raw: ok += 1
            else: bad += 1; fails.append((n, i.offset, i.raw.hex(), enc.hex()))
    return ok, bad, fails
if __name__ == '__main__':
    ok, bad, fails = roundtrip_all(); print('fadd/iadd 12-byte round trip: %d exact, %d differ' % (ok, bad))
    for f in fails[:10]: print('  ', f)
    (HERE / 'faddenc_roundtrip.json').write_text(json.dumps(dict(exact=ok, differ=bad, failures=fails), indent=1) + '\n')
