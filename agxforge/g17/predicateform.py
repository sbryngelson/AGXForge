"""The predicate form's encoder and its check.

All four of this module's definitions are production, so the whole of it moved; the
tools entry is a re-export.
"""


IMMEDIATES=frozenset((1,2,4,5,6,7))
def operands(defs,uses,imms):
    if len(defs)!=1 or len(uses)!=1:raise ValueError('op11452/10 requires one destination and one source')
    if any(type(r) is not int or not 0<=r<128 for r in (*defs,*uses)):
        raise ValueError('op11452/10 requires general register indices')
    if set(imms)!=IMMEDIATES or any(type(v) is not int for v in imms.values()):
        raise ValueError('op11452/10 requires all six explicit immediate operands')
    return {0:425+defs[0],3:105+uses[0],**imms}
def encode(defs,uses,imms,keeps):
    from agxforge.g17 import assembler as g17as
    values=operands(defs,uses,imms)
    if imms[4] not in (0,16,32) or (any(keeps or ()) and imms[4]==16):
        raise ValueError('op11452/10 requested source release conflicts with liveness')
    line='.form predicate op11452.l10\npredicate r%d, #%d, r%d, #%d, op1=#%d, op4=#%d, op6=#%d, op7=#%d'%tuple(
        values[i] for i in (0,2,3,5,1,4,6,7))
    raw=g17as.assemble(line).text
    check(raw,defs,uses,imms)
    return raw
def check(raw,defs,uses,imms):
    from agxforge.g17 import assembler as g17as
    if len(raw)!=10:raise ValueError('op11452/10 emitted length changed')
    for index,want in operands(defs,uses,imms).items():
        got=g17as.field_decode(11452,index,'reg' if index in (0,3) else 'imm',raw)
        if got!=want:raise ValueError('op11452/10 operand %d decoded %r, requested %r'%(index,got,want))
