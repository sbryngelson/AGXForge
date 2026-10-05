"""Stage a tensor body's D fragment through the explicit imageblock, between two adjacent bodies.

Set A item 10b. Body 1 keeps its D tiles in registers (the register feed, tlower keep=True); this
block writes every D register to the lane's own imageblock element (member 32*t + 4*i for tile t,
word i), overwrites the D registers with zero, barriers, reads each word back into its register,
and copies it through an ALU that WAITS on the load (alu.12 byte0[3], the causally measured
load-use wait; ledger/g17-alu-load-use-wait.toml). Body 2 then takes those registers as its A.

Why the zeroing: without it a store and read that silently did nothing would leave body 1's values
in place and body 2 would still be right. With it, the only route by which body 2 can see D is the
imageblock. It is also the use case: between the stores and the reads the D registers hold other
data (here zero), so any code could have used them.

Every instruction is cc's own form, emitted by cc.emit from physical registers and read back by
cc.selfcheck (which includes the imageblock operand-1 readback), so no byte here is hand-encoded:
    read_sr.4 x2     the packed own-lane coordinate (SR_LOCAL_X low half, SR_LOCAL_Y high half)
    store.ib.32      one per D word; operand 1 is cc.IB_STORE_OP1, a wait on slot 0 where the
                     coordinate's read_sr lands (operand 1 bits 24-31 are the store's WAIT MASK; the
                     stored D words are MMA results, not loads, so no load slot needs naming. This
                     module was first written against the refuted "bit 31 = shared storage"
                     reading - MM section 25.104 - and demanded bit 31)
    movimm.8         zero into each D register (omitted when clobber=False)
    barrier          imageblock scope (Apple's op447, 0x4751)
    load.ib.32       one per D word, explicit own coordinate (omitted when read=False: the control)
    alu.12           the waiting in-place copy, one per word
"""
from agxforge.g17 import cc, model

READ_SR_X = cc.SR["thread_position_in_threadgroup"] + cc.SR_AXIS["x"]
READ_SR_Y = cc.SR["thread_position_in_threadgroup"] + cc.SR_AXIS["y"]


def element_bytes(acc):
    """The imageblock element: eight 32-bit words per staged tile."""
    return 32 * len(acc)


def _members(acc):
    return [(tile, i, 32 * n + 4 * i) for n, tile in enumerate(sorted(acc)) for i in range(8)]


def insts(acc, coord, *, clobber=True, read=True):
    """The staging MInsts for D tiles `acc` ({tile: first register}), physical registers throughout,
    `coord` a register in R0..R15 (read_sr's measured destination window)."""
    if not 0 <= coord <= 15:
        raise ValueError("refused: the coordinate is written by read_sr, whose destination is R0..R15")
    out = [cc.MInst("read_sr.4", 4, dict(sr=READ_SR_X, seq=0, half=0, _defs=[coord], _uses=[])),
           cc.MInst("read_sr.4", 4, dict(sr=READ_SR_Y, seq=0, half=1, _defs=[coord], _uses=[coord]))]
    words = _members(acc)
    for n, (tile, i, member) in enumerate(words):
        out.append(cc.MInst("store.ib.32", 14, dict(member=member, dx=0, dy=0, keep_coord=True,
                                                    _uses=[acc[tile] + i, coord], _defs=[])))
    if clobber:
        for tile, i, _m in words:
            out.append(cc.MInst("movimm.8", 8, dict(imm=0, _defs=[acc[tile] + i], _uses=[])))
    out.append(cc.MInst("barrier", 6, dict(scope="imageblock", _defs=[], _uses=[])))
    if read:
        for tile, i, member in words:
            out.append(cc.MInst("load.ib.32", 14, dict(member=member, dx=0, dy=0, explicit=True,
                                                       _defs=[acc[tile] + i], _uses=[coord])))
        for tile, i, _m in words:
            r = acc[tile] + i
            out.append(cc.MInst("alu.12", 12, dict(op=3, mode=1, imm=0, src1_w=1, dest_w=1, srcb_w=1,
                                                   scale_code=2, load_wait=1, b4_5=None, b0_5=None,
                                                   keep=0, _defs=[r], _uses=[r])))
    return out


def emit(acc, coord, *, clobber=True, read=True):
    """(bytes, layout) for the staging block, self-checked by cc and decoded back."""
    block = insts(acc, coord, clobber=clobber, read=read)
    code, layout = cc.emit(block)
    bad = cc.selfcheck(layout)
    if bad:
        raise ValueError("imageblock staging failed cc's selfcheck: %s" % bad[:3])
    decoded = [i for i in model.decode(code, 0) if i.opcode]
    stores = [i for i in decoded if i.opcode.id == 13075]
    loads = [i for i in decoded if i.opcode.id == 12151]
    if len(stores) != 8 * len(acc) or len(loads) != (8 * len(acc) if read else 0):
        raise ValueError("imageblock staging decodes to %d stores and %d loads" % (len(stores), len(loads)))
    if any([v for _k, v in s.values][1] != cc.IB_STORE_OP1 for s in stores):
        raise ValueError("imageblock staging store does not carry cc's operand 1 (a wait on slot 0)")
    return code, layout
