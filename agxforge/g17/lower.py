"""Independent tensor lowering: a complete GEMM body for the M5 accelerator generated from the measured account.

What is generated (every byte from mmaenc.py / memenc.py field maps, none copied): the per-lane vector loads of
A, B (and C in accumulate mode) with scoreboard slot assignment, the MMA chain over K with in-place accumulation
(op5107 first, op5106 after; or op5106 from a loaded C), wait masks from the slot rule (recon section 28),
last-use flags, the stores, END (0e000000) and nop padding (0600, op13483 - the compiler's own filler).
What is inherited: the template object's prologue (argument fetch, lane index: everything before its first
load), its metadata (register count, slot 44) and container. The inherited byte range is reported.

Buffers are fragment-major (gen_gemm.py): A tile (mi,k) at (mi*KT+k)*512, B tile (ni,k) at (ni*KT+k)*512,
C tile (mi,ni) halves at ((mi*NT+ni)*2+h)*512, 16 bytes per lane each. Only the lane-index register (from the
template's first load) is used; tiles are addressed by displacement, so shapes are bounded by the 14-byte
form's displacement field (recon: up to 65535 bytes) and by the template's body room and register count.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
# NO sys.path MANIPULATION. A module under agxforge/ must not touch sys.path - the library
# is imported as a package, and a module that edits the path changes the import
# behaviour of everything loaded after it. The originals inserted the checkout root so
# they could also be run as scripts; run them with `python -m agxforge.g17.<name>` instead.
from agxforge.g17 import machobj, model, gpumd, obj as g17obj
from agxforge.g17 import mmaenc, memenc, faddenc
from agxforge.g17.registerdomain import registers_in_name_checked
HERE = Path(__file__).resolve().parent
NOP = bytes.fromhex('0600'); END = bytes.fromhex('0e000000')
A_BIND, B_BIND, C_BIND = 0, 1, 2

class Template:
    def __init__(self, name):
        self.name = name; d = HERE / name
        self.loc = machobj.locate(str(d / 'native.metallib'), str(d / 'extracted/object/0-0'))
        obj = self.loc['obj']; self.toff = self.loc['text_obj']; self.code = obj[self.toff:self.toff + self.loc['text_size']]
        ins = [i for i in model.decode(self.code, 0) if i.opcode]
        loads = [i for i in ins if i.opcode.id == 12709]
        self.body_start = loads[0].offset
        # index register per buffer binding (byte1 = 4 x rank), read from the template's own loads and stores
        self.index_reg = {}
        for i in ins:
            if i.opcode.id in (12709, 17256):
                self.index_reg.setdefault(i.raw[1] // 4, set()).add(names_of([v for v in i.values if v[0] != 'expr'], 4))
        # a template whose compiler spread a binding over several index registers offers no index register to inherit
        # (tlower.py computes its own); the packed/row-major paths need exactly one per binding and refuse otherwise
        self.index_reg = {k: (v.pop() if len(v) == 1 else None) for k, v in self.index_reg.items()}
        self.lane_reg = self.index_reg.get(1)
        # registers the prologue writes (definitions of instructions before the first load) stay untouched
        self.reserved = set()
        for i in ins:
            if i.offset >= self.body_start: break
            for k, (kind, v) in enumerate(i.values):
                if kind == 'reg' and k < i.opcode.ndefs:
                    self.reserved.update(registers_in_name_checked(NAMES.get(v, '')))
        self.reserved |= {r for r in self.index_reg.values() if r is not None}
        # every index register must be defined in the prologue; a template whose compiler computed one inside the body
        # (seen: the C index of a 4-simdgroup 64x64x64 template) is refused rather than inherited in part
        defined = set()
        for i in ins:
            if i.offset >= self.body_start: break
            for k, (kind, v) in enumerate(i.values):
                if kind == 'reg' and k < i.opcode.ndefs:
                    defined.update(registers_in_name_checked(NAMES.get(v, '')))
        missing = {b: r for b, r in self.index_reg.items() if r is not None and r not in defined}
        if missing: raise ValueError('template %s defines index register(s) %s inside the body, not the prologue: refused' % (name, missing))
        ends = [i for i in ins if i.opcode.id == 684 and i.offset > self.body_start]; self.end_offset = ends[-1].offset
        self.body_room = self.end_offset + 4 - self.body_start
        s, _ = g17obj.sections_of(obj); moff, msz = s['__GPU_METADATA,__compute']; md = obj[moff:moff + msz]
        pk = gpumd.kernel_table(md); p32 = gpumd._table_slot(md, pk, 32); p44 = gpumd._table_slot(md, pk, 44)
        # register count: the top byte of slot 32 when present (0x4d = 77 for the 32x32x64 template, max used R76),
        # else the top byte of slot 44 (0x11 = 17 for 16x16x32, max used R16) - recon section 8 read slot 44's top byte
        self.reg_count = md[p32 + 3] if p32 is not None else (md[p44 + 3] if p44 is not None else None)
        self.max_reg_used = max(max_reg(i.values) for i in ins)
        self.registers = max(self.reg_count or 0, self.max_reg_used + 1)
NAMES = model.registers(); INV = {v: k for k, v in NAMES.items()}
import re
def names_of(values, k):
    return registers_in_name_checked(NAMES[values[k][1]])[0]
def max_reg(values):
    m = -1
    for kind, v in values:
        if kind == 'reg' and v in NAMES:
            for x in registers_in_name_checked(NAMES[v]): m = max(m, x)
    return m

MASK_TABLE = 16384      # byte offset in the C buffer of the per-(tile, half) mask table (32 lanes x 16 bytes each); C tiles must end below it

def mask_table(Mp, Np, MT, NT):
    """host side: word 0 of lane l's 16-byte entry for (tile mi, ni, half h) = bits j where row < Mp and col+j < Np"""
    import numpy as np
    tbl = np.zeros((MT * NT * 2, 32, 4), np.uint32)
    for mi in range(MT):
        for ni in range(NT):
            for h in range(2):
                for l in range(32):
                    row = 16 * mi + 8 * (l >> 4) + 2 * ((l >> 1) & 3) + h; col = 16 * ni + 8 * ((l >> 3) & 1) + 4 * (l & 1)
                    tbl[(mi * NT + ni) * 2 + h, l, 0] = sum(1 << j for j in range(4) if row < Mp and col + j < Np)
    return tbl

def lower(t, M, N, K, accumulate=False, a_type='half', b_type='half', transA=False, transB=False, sg=1, partial=None):
    """Returns (body bytes, plan). accumulate: False (C = 0), 'first' (C loaded into the accumulator before the K
    chain - one MMA rounding chain, what the hardware offers) or True/'last' (Apple's matmul2d multiply_accumulate:
    the product from zero, then C added by fp32 fadd - matches the library bit-exactly)."""
    acc_last = accumulate in (True, 'last'); acc_first = accumulate == 'first'; accumulate = bool(accumulate)
    # partial tiles: (Mp, Np) are the true extents; M, N are rounded up to tiles and the boundary tiles are written with
    # masked stores (op17258) whose per-lane mask words the host tabulates (mask_table) in the C buffer at MASK_TABLE
    Mp, Np = partial if partial else (M, N)
    if partial: assert -(-Mp // 16) * -(-Np // 16) * 1024 <= MASK_TABLE, 'C tiles overlap the mask table'
    M, N = -(-Mp // 16) * 16, -(-Np // 16) * 16
    MT, NT, KT = M // 16, N // 16, K // 16
    assert K % 16 == 0
    # multi-simdgroup: the body is SIMD-uniform; simdgroup s owns M-tiles [s*MT/sg, (s+1)*MT/sg) through the template's A and
    # C index registers (lane + s * tiles-per-simdgroup offset, computed by the prologue), so the body is generated for MT/sg
    assert MT % sg == 0; MT = MT // sg
    ta = 8 if a_type == 'float' else (2 if a_type in ('int8', 'uint8') else 4)
    tb = 8 if b_type == 'float' else (2 if b_type in ('int8', 'uint8') else 4)
    # register pool: everything the template declares except its lane-index register, as aligned tuples. The load
    # and store register fields are seven bits (memmap.json: weights 1..64), so tuples they touch must lie below R128;
    # the MMA's fields reach R255 but accumulators are stored, so they are bound the same way.
    LIMIT = min(t.registers, 128)
    def allocator():
        free = [r for r in range(LIMIT) if r not in t.reserved]
        def take(n):
            for r in free:
                if r % n == 0 and all((r + i) in free for i in range(n)):
                    for i in range(n): free.remove(r + i)
                    return r
            raise ValueError('out of registers')
        return take
    tiles = [(mi, ni) for mi in range(MT) for ni in range(NT)]
    plan_found = None
    for sets in (2, 1):                              # double-buffered A/B tiles when the registers allow
        for per_group in range(len(tiles), 0, -1):   # accumulator tiles held at once; the rest in later groups
            try:
                take = allocator(); acc = {}
                for i in range(per_group): acc[tiles[i]] = take(8)
                abuf = [[take(max(ta, 4)) for mi in range(MT)] for s in range(sets)]
                bbuf = [[take(max(tb, 4)) for ni in range(NT)] for s in range(sets)]
                ctmp = take(8) if acc_last else None
                mtmp = [take(4), take(4)] if partial else None
                plan_found = (sets, per_group, acc, abuf, bbuf, ctmp, mtmp); break
            except ValueError: continue
        if plan_found: break
    if not plan_found: raise ValueError('no register plan fits %d declared registers (lane R%d)' % (t.registers, t.lane_reg))
    sets, per_group, acc0, abuf, bbuf, ctmp, mtmp = plan_found
    groups = [tiles[i:i + per_group] for i in range(0, len(tiles), per_group)]
    r_needed = 8 * per_group + sets * (MT * max(ta, 4) + NT * max(tb, 4)) + (8 if acc_last else 0)
    out = bytearray(); plan = dict(groups=[[list(x) for x in g] for g in groups], abuf=abuf, bbuf=bbuf, registers=r_needed, buffer_sets=sets, ops=[])
    def emit(op, b, what): out.extend(b); plan['ops'].append(dict(off=len(out) - len(b), op=op, what=what, hex=b.hex()))
    # bytes per lane per tile in the packed buffer: 16-bit and int8 fragments occupy one 16-byte lane slot (int8's 8 bytes
    # padded, so the unwitnessed width code for 8 is not needed); fp32 fragments occupy two 512-byte halves like C
    a_bytes = 16 * (ta // 4) if ta >= 4 else 16; b_bytes = 16 * (tb // 4) if tb >= 4 else 16
    slot_of = lambda k: k % 8
    for g, group in enumerate(groups):
        acc = {tile: acc0[tiles[i]] for i, tile in enumerate(group)}          # same accumulator registers, next tiles
        gm = sorted({mi for mi, ni in group}); gn = sorted({ni for mi, ni in group})
        if acc_first:
            for (mi, ni) in group:
                for h in range(2):
                    op, b = memenc.load(acc[mi, ni] + 4 * h, t.index_reg[C_BIND], C_BIND, disp=((mi * NT + ni) * 2 + h) * 512, slot=7)
                    emit(op, b, 'g%d C(%d,%d)h%d slot7' % (g, mi, ni, h))
        for k in range(KT):
            s = k % sets; sl = slot_of(k)
            for mi in gm:
                for q in range(a_bytes // 16 or 1):
                    op, b = memenc.load(abuf[s][mi] + 4 * q, t.index_reg[A_BIND], A_BIND, disp=(mi * KT + k) * 32 * a_bytes + 512 * q, width=16, slot=sl)
                    emit(op, b, 'g%d A(%d,k%d) slot%d' % (g, mi, k, sl))
            for ni in gn:
                for q in range(b_bytes // 16 or 1):
                    op, b = memenc.load(bbuf[s][ni] + 4 * q, t.index_reg[B_BIND], B_BIND, disp=(ni * KT + k) * 32 * b_bytes + 512 * q, width=16, slot=sl)
                    emit(op, b, 'g%d B(%d,k%d) slot%d' % (g, ni, k, sl))
            for (mi, ni) in group:
                first = (k == 0 and not acc_first)
                mask = (1 << sl) | ((1 << 7) if (k == 0 and acc_first) else 0)
                last_a = ni == max(n for m, n in group if m == mi); last_b = mi == max(m for m, n in group if n == ni)
                op, b = mma_masked(acc[mi, ni], abuf[s][mi], bbuf[s][ni], None if first else acc[mi, ni], a_type, b_type, transA, transB,
                                   mask, more=(k < KT - 1), a_last=last_a, b_last=last_b)
                emit(op, b, 'g%d MMA(%d,%d,k%d) mask %02x' % (g, mi, ni, k, mask))
        for (mi, ni) in group:
            if acc_last:                                   # Apple's order: product first, then C added in fp32
                for h in range(2):
                    op, b = memenc.load(ctmp + 4 * h, t.index_reg[C_BIND], C_BIND, disp=((mi * NT + ni) * 2 + h) * 512, slot=7)
                    emit(op, b, 'g%d C(%d,%d)h%d -> tmp slot7' % (g, mi, ni, h))
                for i in range(8):
                    aop = 10282 if a_type in ('int8', 'uint8') else 998
                    b = faddenc.encode(acc[mi, ni] + i, acc[mi, ni] + i, ctmp + i, word=(1 << 31) if i == 0 else 0, f1=16, f2=16, op=aop)
                    emit(aop, b, 'g%d %s C(%d,%d)[%d]%s' % (g, 'iadd' if aop == 10282 else 'fadd', mi, ni, i, ' wait slot7' if i == 0 else ''))
            for h in range(2):
                boundary = partial and (16 * mi + 16 > Mp or 16 * ni + 16 > Np)
                if boundary:                               # masked store: mask word for (tile, half) loaded per lane, slot 6
                    mt = mtmp[h]; entry = (mi * NT + ni) * 2 + h
                    op, b = memenc.load(mt, t.index_reg[C_BIND], C_BIND, disp=MASK_TABLE + entry * 512, slot=1)
                    emit(op, b, 'g%d mask(%d,%d)h%d -> R%d slot1' % (g, mi, ni, h, mt))
                    op, b = memenc.mstore(acc[mi, ni] + 4 * h, t.index_reg[C_BIND], C_BIND, INV['R%dL' % mt], disp=entry * 512, width=16)
                    vals = {k: v[1] for k, v in enumerate([(k, v) for k, v in list(model.decode(b, 0))[0].values if k != 'expr'])}
                    vals[1] |= 1 << 25                      # wait on slot 1 (the masked store's word carries bits 24, 25, 27 in its 10-byte form)
                    b = memenc.encode('mstore16' if entry * 512 > 255 else 'mstore10', vals, binding=4 * C_BIND)
                    emit(op, b, 'g%d masked store C(%d,%d)h%d mask R%dL' % (g, mi, ni, h, mt))
                else:
                    op, b = memenc.store(acc[mi, ni] + 4 * h, t.index_reg[C_BIND], C_BIND, disp=((mi * NT + ni) * 2 + h) * 512)
                    emit(op, b, 'g%d store C(%d,%d)h%d' % (g, mi, ni, h))
    emit(684, END, 'END')
    if len(out) > t.body_room: raise ValueError('body %d bytes exceeds template room %d' % (len(out), t.body_room))
    while len(out) < t.body_room: out.extend(NOP)
    plan['body_bytes'] = len(out); plan['inherited'] = dict(prologue=[0, t.body_start], template=t.name, index_registers=t.index_reg, reserved=sorted(t.reserved)); plan['simdgroups'] = sg
    return bytes(out), plan

def mma_masked(D, A, B, C, a_type, b_type, transA, transB, mask, more, a_last, b_last):
    """mmaenc.mma with an explicit eight-bit wait mask at flags bits 24..31 (slot 7 = byte0[3])."""
    op, b = mmaenc.mma(D, A, B, C, a_type, b_type, transA, transB, wait=bool(mask >> 7 & 1), tag=0, more=more, a_last=a_last, b_last=b_last)
    fields, tpl, base, dec = mmaenc._tables(op)
    vals = {k: v[1] for k, v in enumerate(list(model.decode(b, 0))[0].values)}
    vals[1] = (vals[1] & ~0x7f000000) | ((mask & 0x7f) << 24)
    return op, mmaenc.encode(op, vals)

def patched_archive(t, body, label):
    fat = bytearray(t.loc['fat']); at = t.loc['base'] + t.toff + t.body_start
    assert len(body) == t.body_room
    fat[at:at + len(body)] = body
    p = HERE / t.name / ('native.lowered-%s.metallib' % label); p.write_bytes(bytes(fat)); return p
