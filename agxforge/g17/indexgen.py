"""Index-register prologue generated from the account: lane id from SR_SIMD_ELEM, then
    m   = 4 (lane >> 4) + ((lane >> 1) & 3)          col = (lane & 8) + ((lane & 1) << 2)
    idx = m * ld + col   for each leading dimension (shift-add decomposition of ld)
with ledger-encoded ALU forms whose semantics were measured in alusem.py: op14060 read_sr (4-byte form, census in
this file), op17016 shr16, op426 and16, op423 and32, op14391 shl, op10279 addi, op10282 add. The SR read fills
scoreboard slot 0 like a load; its first consumer waits on it."""
import sys
from pathlib import Path
# NO sys.path MANIPULATION. A module under agxforge/ must not touch sys.path - the library
# is imported as a package, and a module that edits the path changes the import
# behaviour of everything loaded after it. The originals inserted the checkout root so
# they could also be run as scripts; run them with `python -m agxforge.g17.<name>` instead.
from agxforge.g17 import model
from agxforge.g17 import ledgerenc, faddenc
names = model.registers()

def read_sr_lane(dest, slot=0):
    """4-byte op14060 reading SR_SIMD_ELEM into R<dest>L, filling scoreboard slot `slot` (census: dest bits byte0[4..7], byte2[6..7]; slot+1 at byte3[5..7])"""
    u = bytearray(bytes.fromhex('34821006')); u[0] &= 0x0f; u[2] &= 0x3f; u[3] &= 0x1f
    u[0] |= (dest & 15) << 4; u[2] |= ((dest >> 4) & 3) << 6; u[3] |= (slot & 7) << 5
    d = list(model.decode(bytes(u), 0)); assert d and d[0].opcode and d[0].opcode.id == 14060 and names[d[0].values[0][1]] == 'R%dL' % dest and names[d[0].values[2][1]] == 'SR_SIMD_ELEM' and d[0].values[1][1] == (slot + 1) << 20, d
    return bytes(u)
W = lambda wait: (1 << 24) if wait else 0
# The source-operand flag (operand 3 on and, operand 4 on the shifts and add-imm): 16 = LAST USE, releases the source
# register (srlanding.py: an op426 carrying 16 as the first reader of the SR-written lane register made every later
# reader see 0); 32 = live. Every form here keeps its source live unless told otherwise.
LIVE, LAST = 32, 16
def shr16(d, s, n, wait=False, last=False): return ledgerenc.encode(17016, {0: 'R%d' % d, 1: W(wait), 3: 'R%dL' % s, 4: LAST if last else LIVE, 5: n})
def shr32(d, s, n, last=False): return ledgerenc.encode(17013, {0: 'R%d' % d, 1: 0, 3: 'R%d' % s, 4: LAST if last else LIVE, 5: n})
def and16(d, s, imm, wait=False, last=False): return ledgerenc.encode(426, {0: 'R%d' % d, 1: W(wait), 2: 'R%dL' % s, 3: LAST if last else LIVE, 4: imm})
def and32(d, s, imm, last=False): return ledgerenc.encode(423, {0: 'R%d' % d, 1: 0, 2: 'R%d' % s, 3: LAST if last else LIVE, 4: imm})
def shl(d, s, n, last=False): return ledgerenc.encode(14391, {0: 'R%d' % d, 1: 0, 3: 'R%d' % s, 4: LAST if last else LIVE, 5: n})
def addi(d, s, imm, last=False): return ledgerenc.encode(10279, {0: 'R%d' % d, 1: 0, 3: 'R%d' % s, 4: LAST if last else LIVE, 2: imm})
def add(d, a, b): return faddenc.encode(d, a, b, word=0, f1=0, f2=0, op=10282)

def prologue(lane, m, col, t1, t2, idx_regs, lds):
    """returns bytes computing m, col and idx_regs[i] = m * lds[i] + col; lane, m, col, t1, t2: scratch 32-bit registers"""
    out = bytearray()
    out += read_sr_lane(lane, slot=0)
    out += shr16(t1, lane, 4, wait=True); out += shl(t1, t1, 2)          # 4 (lane >> 4)
    out += shr16(t2, lane, 1); out += and32(t2, t2, 3)                    # (lane >> 1) & 3
    out += add(m, t1, t2)
    out += and16(t1, lane, 8); out += and16(t2, lane, 1); out += shl(t2, t2, 2); out += add(col, t1, t2)
    for r, ld in zip(idx_regs, lds):
        bits = [k for k in range(31) if ld >> k & 1]; assert bits, 'ld must be positive'
        out += shl(r, m, bits[0])                                          # m << k0
        for k in bits[1:]:
            out += shl(t1, m, k); out += add(r, r, t1)
        out += add(r, r, col)
    return bytes(out)
if __name__ == '__main__':
    b = prologue(8, 9, 10, 11, 12, [13, 14, 15], [128, 64, 100])
    print(len(b), 'bytes:', [(i.opcode.id, len(i.raw)) for i in model.decode(b, 0)])
