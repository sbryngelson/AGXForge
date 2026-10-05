"""Derive the measured three-user-buffer tensor metadata layout.

No witness bytes are read during generation. The physical slot ordering and
two-byte vtable-start adjustment are measured class facts; relative references,
structure positions and trailing fill positions follow from the layout growth.
SR130's entry (52) and SR156's entry (0) are measured; a two-entry tensor
section is derived with the repository's measured slot-29 vector growth rule.
"""
import copy
from . import mdgen as M


def validate(abi,bindings):
    from collections.abc import Mapping
    if type(abi.get('abi_version')) is not int or abi['abi_version']!=5:
        raise ValueError('tensor metadata requires ABI v5')
    ex=abi.get('execution')
    if (not isinstance(ex,Mapping) or set(ex)!={'simd_width','tensor'} or
        type(ex['simd_width']) is not int or ex['simd_width']!=32 or ex['tensor'] is not True):
        raise ValueError('unmeasured tensor execution requirement')
    for key,want in [('uses_threadgroup',False),('has_stores',True),('writes_buffer',True),
                     ('writes_texture',False),('arch_flag',False)]:
        if abi.get(key) is not want:raise ValueError('tensor metadata class differs: '+key)
    if abi.get('threadgroup') is not None:raise ValueError('unmeasured tensor threadgroup declaration')
    pool=abi.get('constant_pool')
    if not isinstance(pool,(list,tuple)) or len(pool):
        raise ValueError('tensor metadata requires an explicit empty constant_pool')
    if type(abi.get('entry')) is not int or abi['entry']!=64:raise ValueError('unmeasured tensor entry')
    prologue=abi.get('prologue')
    if isinstance(prologue,str):prologue=bytes.fromhex(prologue)
    if prologue!=bytes.fromhex('0e000000')+bytes.fromhex('0600')*30:
        raise ValueError('tensor author requires declared END/filler prologue')
    for key in ('unswept_two_buffer','measured_class','reproduce_measured_class','pk_vectors',
                'pk_empty_vectors','slot2_extra','slot2_kind6','slot2_kind6_f2','constant_program',
                'v0_field2','v0_field3','ld_md_slots','ld_md_values'):
        if key in abi and abi[key] is not None and abi[key] is not False:
            raise ValueError('tensor class does not accept override '+key)
    if 'pk_slot1' in abi and (type(abi['pk_slot1']) is not int or abi['pk_slot1']!=8):
        raise ValueError('tensor pointer/constant block differs')
    values=abi.get('pk_values',{})
    if not isinstance(values,Mapping) or len(values)!=2 or any(type(v) is not int or v!=1 for v in values.values()):
        raise ValueError('tensor pk_values differ')
    if set(values) not in ({15,16},{'15','16'}):raise ValueError('tensor pk_values differ')
    if tuple(abi.get('pk_extra',()))!=(15,16):raise ValueError('tensor pk_extra differs')
    layout(abi.get('register_count'),abi.get('system_registers',()),
           abi.get('instruction_count'),abi.get('has_back_edge'))
    triples=[tuple(b[:3]) for b in bindings]
    if triples not in [[(1,0,False),(2,2,False),(3,4,True)],[(2,0,False),(4,2,False),(6,4,True)]]:
        raise ValueError('unmeasured indexed tensor binding signature')


def emit_contract(abi,bindings,ledger):
    validate(abi,bindings)
    # A straight-line program with an explicit imageblock carries slot 19 in its tail (layout above);
    # imageblock.declare then adds only the LD_MD and ARCH halves. The loop table is left to declare.
    ib_tail = abi.get('imageblock') is not None and not abi['has_back_edge']
    result=emit([b[:3] for b in bindings],register_count=abi['register_count'],
                system_registers=abi['system_registers'],instruction_count=abi['instruction_count'],
                has_back_edge=abi['has_back_edge'],imageblock=ib_tail)
    if ib_tail:
        ledger['per-kernel slot 19']=('MEASURED straight-line tensor tail 44,19,32 at tlen 64; whole section '
                                      'byte-identical to Apple k32_both (results/g17-tensor-ibfragment-v1/witness)')
    registers=tuple(abi.get('system_registers',()))
    # Historical singleton payloads must retain their exact ledger text. New wording belongs only
    # to the newly measured composed class; otherwise a byte-identical receipt fails rebuild.
    if registers == (130,):
        ledger['metadata class']='MEASURED: three indexed user buffers, tensor, singleton SR130, empty constant pool'
        ledger['metadata serializer']='g17mdgen.build; derived 488-byte layout with relocated fills'
    else:
        ledger['metadata class']='MEASURED: three indexed user buffers, tensor, SR130 + SR156, empty constant pool'
        ledger['metadata serializer']='g17mdgen.build; measured singleton layout with slot-29 vector growth rule for two entries'
    ledger['per-kernel slot 0']='COMPILER INPUT register_count'
    ledger['per-kernel slot 13']='COMPILER INPUT empty constant pool; measured eight-byte zero vector'
    ledger['per-kernel slot 29']=('MEASURED singleton SR130 encoded as entry52'
                                  if registers == (130,)
                                  else 'MEASURED map: SR130 -> 52, SR156 -> 0; vector derived from declared set')
    ledger['per-kernel slots 32/33/44']='Loop class from captured count/back-edge; measured tensor class flags'
    ledger['execution evidence']='Authored only; no complete common-path tensor image execution is established'
    if not abi['has_back_edge']:
        ledger['metadata serializer']=('g17mdgen.build; derived 484-byte layout with relocated fills'
                                      if registers == (130,)
                                      else 'g17mdgen.build; branchless tensor tail with measured slot-29 vector growth')
        # Further witness interpretation is retained in docs/archive/g17-tensor-common-handoff.md.
        # Keep the executed bundle's serialized evidence text stable.
        ledger['per-kernel slots 32/33/44']='PREDICTED branchless tail: slot32=1, slot33 absent, slot44=1; measured layouts do not establish the cause of slot33 presence'
    return result


def load_metadata(entry):
    """Measured tensor LD class: T3 has slots3/5 and a six-slot vtable.

    The table at44 still ends at50, where the main vtable begins. Its larger
    vtable pushes the preceding root and root vtable back four bytes; references
    are recomputed from their field positions. No witness is read here.
    """
    from . import ldmd as L
    out=bytearray(L.build(entry,restore=True))
    t3=44;t3vt=t3-(4+2*6);root=t3vt-12;rvt=root-(4+2*4)
    out[:50]=bytes(50)
    import struct
    struct.pack_into('<I',out,0,root)
    L._put_table(out,root,rvt,4,12,{0:(8,'<I',136-(root+8)),3:(4,'<I',t3-(root+4))})
    L._put_table(out,t3,t3vt,6,6,{3:(4,'<B',1),5:(5,'<B',1)})
    return bytes(out)


def arch_metadata():
    """Measured semantic-false tensor ARCH: a present empty subtable."""
    from . import schema as S
    sub=S.Table({}, {},4,4,name='archsub')
    root=S.Table({0:4},{0:('<I',S.Ref(sub))},8,8,name='archroot')
    doc=S.Doc(root,[root,sub]);S.place(doc)
    return S.emit(doc,size=32)


# THE TAIL IS THE VARIABLE, AND SLOT 33 IS THE PART OF IT THAT IS NOT MEASURED FOR A NEW PROGRAM.
# The per-kernel table carries a run of single-byte slots immediately below slot 16, ordered by
# DESCENDING slot number. SEPARATE leaves exactly two bytes of padding there, so a two-slot tail
# costs no growth and a three-slot tail rounds the table up by four. The rule is about the TABLE -
# its length and the tail's relative offsets - and four witnesses carry it exactly: tensor-k32 and
# tensor-k16 at tail (44,32), tlen 60, slot 0 at +56; tensor-k64 at (44,33,32), tlen 64, slot 0 at
# +60; and tensor-leading-buf at the same (44,33,32) tlen 64 with relative offsets 55/56/57/58/59/60,
# identical to k64's.
#
# THE SECTION SIZE IS NOT THE RULE, and saying "484 means a two-slot tail" is wrong. Size also
# tracks the BINDING COUNT: tensor-leading-buf has four bindings (slot 3 = 32) and is 484 bytes
# WITH the three-slot tail, the same size as the two-slot k32 at three bindings. 484 and 488 are
# consequences at three bindings only. This author refuses a four-binding tensor signature by name,
# so it cannot mis-author that witness - but the shorthand would have mis-read it.
#
# WHICH TAIL a program takes is where the measurement stops. Across the 86 distinct tensor
# sections available here, slot 33's presence coincides EXACTLY with four other facts at once -
# a back edge, op12675, slot 32 = 3, and (in the single slot-33-absent program) register count 45
# against 72. One branchless witness cannot separate four confounded candidates, so `tail_for`
# below is a RULE THAT REPRODUCES EVERY MEASURED SECTION, not a claim about what slot 33 means.
# A caller may state the tail explicitly; deriving it is a prediction and the ledger says so.
TAIL_LOOP = (44, 33, 32)
TAIL_LINEAR = (44, 32)
FREE_TAIL_BYTES = 2          # padding SEPARATE already leaves below slot 16


def tail_for(slot32):
    """The measured tail for a section whose slot 32 is `slot32`. A PREDICTION off the loop path."""
    return TAIL_LOOP if slot32 == 3 else TAIL_LINEAR


# (130, 164, 165): a tensor body plus an explicit imageblock, whose tile coordinate reads SR_LOCAL_X
# and SR_LOCAL_Y. Measured 2026-09-23 on Apple's own compile of a matmul2d kernel with an explicit
# imageblock (tools/g17tensorimageblockwitness.py, results/g17-tensor-imageblock-witness-v1): the
# slot-29 vector is [48, 49, 52], the tensor LD_MD differs from the imageblock-free one only at the
# declaration's two bytes, and the ARCH section is the imageblock form.
MEASURED_SETS = ((130,), (130, 156), (130, 133, 156), (130, 164, 165))


# SLOT 19 (THE IMAGEBLOCK DECLARATION) JOINS THE TAIL DIRECTLY ABOVE SLOT 44, and the table grows by
# the same rule as any other tail byte. Measured 2026-09-23 on Apple's own compiles of a straight-line
# matmul2d kernel with and without an explicit imageblock (results/g17-tensor-ibfragment-v1/witness):
#     29 insts, no slot 32:  tensor-only  tlen 60  44:53 16:54 15:55
#                            + imageblock tlen 60  44:52 19:53 16:54 15:55          (two bytes: no growth)
#     37 insts, slot 32 = 1: + imageblock tlen 64  44:55 19:56 32:57 16:58 15:59, slot 0 56 -> 60
# and it reproduces, unchanged, the loop-tail placement imageblock.declare already applies (44:54
# 19:55 33:56 32:57 at tlen 64; results/g17-tensor-imageblock-witness-v1). So the order from the
# bottom is 44, 19, then the remaining slots descending - NOT a plain descending sort, which would put
# 19 at the top. Only the straight-line table routes through here (imageblock=True); the loop table's
# declaration stays in imageblock.declare, where it was executed.
IB_SLOT = 19


def _tail_order(tail):
    rest = sorted((s for s in tail if s not in (44, IB_SLOT)), reverse=True)
    return [44] + ([IB_SLOT] if IB_SLOT in tail else []) + rest


def layout(register_count, system_registers, instruction_count, has_back_edge, tail=None, imageblock=False,
           indices=None, written=None, slot13_len=None, slot27_len=None):
    if type(register_count) is not int or not 1 <= register_count <= 128:
        raise ValueError('invalid tensor register count')
    registers=tuple(system_registers)
    # (130, 133, 156): a simdgroup-split body (SR_SIMD_GRP) with the SR156 epilogue. Its slot-29 vector
    # [0, 52, 53] is carried by 51 cached tensor objects, and 133 -> 53 is measured alone
    # (mdgen.SLOT29_BY_SYSTEM_REGISTER). A split body without SR156 would be (130, 133), which no
    # witness carries, so it stays refused.
    if registers not in MEASURED_SETS or any(type(v) is not int for v in registers):
        raise ValueError(
            'tensor metadata class requires SR130, optionally with SR156, or the simdgroup-split set '
            '(130, 133, 156); got %s' % (registers if registers else '()',))
    if type(instruction_count) is not int or instruction_count < 1 or type(has_back_edge) is not bool:
        raise ValueError('this tensor metadata class requires a stated instruction count and back edge')
    slot32 = M.slot32_for(instruction_count, has_back_edge)
    if slot32 is None:
        raise ValueError('a tensor program of %d instructions carries no slot 32; no witness has '
                         'that shape and its tail is unmeasured' % instruction_count)
    tail = tuple(tail) if tail is not None else tail_for(slot32)
    if tail not in (TAIL_LOOP, TAIL_LINEAR):
        raise ValueError('unmeasured tensor tail %s; the measured tails are %s and %s'
                         % (list(tail), list(TAIL_LOOP), list(TAIL_LINEAR)))
    if imageblock:
        if tail != TAIL_LINEAR or has_back_edge:
            raise ValueError('the imageblock tail is authored here only for the straight-line table; '
                             'the loop table is declared by imageblock.declare')
        tail = tail + (IB_SLOT,)
    # THE BINDING LAYOUT FOLLOWS THE CONTRACT'S INDICES AND WRITE MASK when the caller states them,
    # so a compiled (0,1,2) all-written program gets Apple's elided-buffer-0 layout (item 11). With
    # both absent this is exactly M.SEPARATE, so every existing output is byte-identical. The
    # delegated layout is one of SEPARATE's own family (same per-kernel table and tail), so the tail
    # growth below applies to it unchanged.
    base = M.SEPARATE
    if indices is not None and written is not None:
        selected = M.layout_for(indices, written=written, instructions=instruction_count,
                                back_edge=has_back_edge, system_registers=system_registers)
        if selected is not None:
            base = selected
    out = copy.deepcopy(base)
    old_end = out['pk'] + out['pk_tlen']
    new_vlen = 4 + 2*(44+1)
    new_vt = out['pkvt'] - 2
    new_pk = new_vt + new_vlen
    extra = len(tail) - FREE_TAIL_BYTES
    inline_growth = ((extra + 3) & ~3) if extra > 0 else 0
    new_tlen = out['pk_tlen'] + inline_growth
    growth = new_pk + new_tlen - old_end
    def moved(pos): return pos + growth if pos >= old_end else pos
    if 16 not in out['pk_slots']:                # tail placement is anchored on slot 16
        raise ValueError('this tensor metadata class has no per-kernel slot 16; its tail placement is '
                         'unmeasured (was a bare KeyError, "refused: 16", for e.g. ac-16x32x64)')
    base = out['pk_slots'][16] + inline_growth - len(tail)
    out.update(size=out['size']+growth,pk=new_pk,pkvt=new_vt,
               pk_vlen=new_vlen,pk_tlen=new_tlen,q=out['q']+inline_growth)
    out['pk_slots']={k:v+inline_growth if k in (0,15,16) else v
                     for k,v in out['pk_slots'].items()}
    out['pk_slots'].update({s: base+i for i,s in enumerate(_tail_order(tail))})
    out['pk_extra'].update({0:('<I',register_count)})
    for s in tail:
        out['pk_extra'][s]=('<B', slot32 if s==32 else 1)
    for key in ('vec_bind','vec2','vec0'): out[key]=moved(out[key])
    for key in ('name','cpname','v0'): out[key]=tuple(moved(v) for v in out[key])
    for key in ('bind','v2'):
        out[key]=[tuple(moved(v) if i<2 else v for i,v in enumerate(row)) for row in out[key]]
    out['ptrs']={k:moved(v) for k,v in out['ptrs'].items()}
    # Slot 29 is a vector whose entries now come from the measured register map. The old singleton
    # class represented its count and entry as fills; retaining those writes would overwrite the
    # generic vector writer and silently truncate the composed (130,156) declaration.
    out['slot29_vector'] = out['ptrs'][29]
    out['slot29_counts'] = (1, 2, 3)
    out['slot29_sets'] = MEASURED_SETS
    _base_slot29 = M.SEPARATE['ptrs'][29]
    out['fills']=tuple((moved(pos),fmt,value) for pos,fmt,value in out['fills']
                       if pos not in (_base_slot29, _base_slot29 + 4))
    # LENGTH-AWARE VECTOR FRAMING (item 11, Set C): slots 13/27 are variable-length byte vectors whose
    # content is the transplanted constant-program data. When a target byte length exceeds the class's
    # base length, the SECTION GROWS: everything after that vector's data shifts by the delta, exactly
    # like the tail growth above. Slot 27 sits before slot 13, so grow 27 first (its shift moves 13).
    # build() writes each vector at out['vector_lengths'][slot], and the pk13/pk27 transplant then fills
    # the content at the matching length. Positions are shifted here so build() recomputes every
    # flatbuffers offset from them; no per-field byte fixup.
    out['vector_lengths'] = {}
    for _slot, _tgt in ((27, slot27_len), (13, slot13_len)):
        if _tgt is None or _slot not in out['ptrs']:
            continue
        # A vector's COUNT WORD and its DATA byte length are the same only for slot 13, whose count is
        # a byte length. Slot 27's count is a WORD count, so its data is 4x the count word (0/1/2 buffer
        # selectors, one word each). _tgt is always the DATA byte length (len of the recovered vector);
        # the count word written into the section is that divided by the element width.
        w = 4 if _slot == 27 else 1
        vecpos = out['ptrs'][_slot]
        base_cw = next((v for p, f, v in out['fills'] if p == vecpos), 0)   # base COUNT WORD
        tgt_cw = _tgt // w                            # count word for the target data byte length
        out['vector_lengths'][_slot] = tgt_cw
        d = _tgt - base_cw * w                        # BYTE delta; grows if positive, SHRINKS if negative
        if d == 0:
            continue
        end = vecpos + 4 + base_cw * w                # the class base's data end is the shift anchor
        def _shift(pos, e=end, dd=d):
            return pos + dd if pos >= e else pos
        out['size'] += d
        # Q (per-kernel slots 6/8/10/12) is a byte-position reference into the region after the vector,
        # so it moves with the growth; slot 1 counts the vector in WORDS, so it moves by the word delta.
        out['q'] += d
        out['pk_extra'][1] = ('<I', out['pk_extra'].get(1, ('<I', 8))[1] + d // 4)
        if _slot == 13 and out.get('v2_vals'):     # the first v2 record's slot-2 counts the vector words
            _r = list(out['v2_vals'][0]); _r[1] += d // 4; out['v2_vals'][0] = tuple(_r)
        for key in ('vec_bind', 'vec2', 'vec0'):
            out[key] = _shift(out[key])
        for key in ('name', 'cpname', 'v0'):
            out[key] = tuple(_shift(v) for v in out[key])
        for key in ('bind', 'v2'):
            out[key] = [tuple(_shift(v) if i < 2 else v for i, v in enumerate(row)) for row in out[key]]
        out['ptrs'] = {k: _shift(v) for k, v in out['ptrs'].items()}
        out['slot29_vector'] = _shift(out['slot29_vector'])
        # shift fills after `end`, and set THIS vector's own count-word fill (at vecpos) to the target.
        # A class whose base vector is empty carries no fill at vecpos, so ADD one: without it the
        # count word stays 0, the pk transplant guard never matches, and the grown data is never written.
        _had = any(p == vecpos for p, f, v in out['fills'])
        out['fills'] = tuple((vecpos, f, tgt_cw) if p == vecpos else (_shift(p), f, v)
                             for p, f, v in out['fills'])
        if not _had:
            out['fills'] = out['fills'] + ((vecpos, '<I', tgt_cw),)
    return out


def emit(bindings, *, register_count, system_registers, instruction_count, has_back_edge, imageblock=False):
    bindings=[tuple(b) for b in bindings]
    if any(len(b)!=3 or type(b[0]) is not int or type(b[1]) is not int or type(b[2]) is not bool for b in bindings):
        raise ValueError('invalid tensor binding declaration')
    if bindings not in [[(1,0,False),(2,2,False),(3,4,True)],
                       [(2,0,False),(4,2,False),(6,4,True)]]:
        raise ValueError('unmeasured indexed tensor binding signature')
    measured=layout(register_count,system_registers,instruction_count,has_back_edge,imageblock=imageblock,
                    indices=[b[0] for b in bindings],written=[b[2] for b in bindings])
    return M.build([b[0] for b in bindings],layout=measured,
                   offsets=[b[1] for b in bindings], system_registers=system_registers,
                   register_count=register_count)
