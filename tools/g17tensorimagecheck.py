"""Read tensor resource declarations and memory ranks from delivered bytes.

These are structural checks for the measured empty-pool SR130 class. They do
not establish dynamic addresses, dependency behavior or numerical semantics.
No author or witness bytes are used by this reader.
"""
import re
import struct


def arch(data):
    # Present empty child, distinct from the scalar null-child false form.
    expected={0:('<I',12),4:('<H',0),6:('<H',6),8:('<H',8),10:('<H',4),
              12:('<i',6),16:('<I',8),20:('<H',4),22:('<H',4),
              24:('<i',4),28:('<I',0)}
    if len(data)!=32:raise ValueError('tensor ARCH size differs')
    for offset,(fmt,value) in expected.items():
        if struct.unpack_from(fmt,data,offset)[0]!=value:
            raise ValueError('tensor ARCH present-empty structure differs')
    return dict(serialized_flag=False,child='present_empty')


def resources(metadata,ld,abi,decoded,bindings):
    import g17gpumd as M
    def field(data,table,slot,fmt):
        pos=M._table_slot(data,table,slot)
        if pos is None:raise ValueError('missing tensor field '+str(slot))
        return struct.unpack_from(fmt,data,pos)[0]
    def child(data,table,slot):
        pos=M._table_slot(data,table,slot)
        if pos is None:raise ValueError('missing tensor reference '+str(slot))
        return pos+struct.unpack_from('<I',data,pos)[0]
    def vector(data,table,slot,width):
        pos=child(data,table,slot);count=struct.unpack_from('<I',data,pos)[0]
        if pos+4+count*width>len(data):raise ValueError('tensor vector exceeds section')
        return data[pos+4:pos+4+count*width]
    if abi.get('execution')!={'simd_width':32,'tensor':True} or abi.get('system_registers')!=[130]:
        raise ValueError('unmeasured tensor execution/register requirement')
    if abi.get('constant_pool')!=[] or abi.get('uses_threadgroup') is not False:
        raise ValueError('unmeasured tensor pool or threadgroup requirement')
    pk=M.kernel_table(metadata)
    cyclic=any(r[2]==458 for r in decoded)
    if not 31<=len(decoded)<=300:
        raise ValueError('unmeasured tensor instruction-count band')
    if not cyclic and any(r[2] in (450,462,577,578,579,582,583,10369,12675) for r in decoded):
        raise ValueError('unmodeled non-loop tensor control or masked-load class')
    if len(metadata)!=(488 if cyclic else 484) or pk is None:
        raise ValueError('unmeasured tensor metadata class')
    if not cyclic and M._table_slot(metadata,pk,33) is not None:
        raise ValueError('branchless tensor prediction requires absent slot 33')
    for slot in ((33,44) if cyclic else (44,)):
        if field(metadata,pk,slot,'<B')!=1:raise ValueError('tensor declaration differs: '+str(slot))
    if any(M._table_slot(metadata,pk,slot) is not None for slot in (18,28)):
        raise ValueError('unexpected tensor threadgroup storage')
    if vector(metadata,pk,29,4)!=struct.pack('<I',52):
        raise ValueError('tensor SR130 resource entry differs')
    if vector(metadata,pk,13,1)!=bytes(8):
        raise ValueError('tensor empty-pool representation differs')
    expected=3 if cyclic else 1
    if field(metadata,pk,32,'<B')!=expected:
        raise ValueError('tensor instruction-shape declaration differs')
    root=struct.unpack_from('<I',ld)[0];main=child(ld,root,0);flags=child(ld,root,3)
    if len(ld)!=216 or M._table_slot(ld,main,1) is not None:
        raise ValueError('tensor LD is not the measured first-compilation form')
    if field(ld,main,6,'<I')!=abi['entry']:
        raise ValueError('tensor LD entry differs from program')
    for slot in (3,5):
        if field(ld,flags,slot,'<B')!=1:raise ValueError('tensor LD declaration differs')
    resolved=[]
    for offset,length,opcode,tokens in decoded:
        if opcode not in (12674,12675,17257):continue
        match=re.fullmatch(r'expr:bin\(op0,const\((\d+)\),8\)',tokens[3])
        if not match or int(match[1])%4:raise ValueError('unmodeled tensor memory base')
        rank=int(match[1])//4
        if rank>=len(bindings):raise ValueError('tensor memory rank outside delivered bindings')
        index,position,written,kind=bindings[rank];store=opcode==17257
        if position!=2*rank or written!=store or kind!=('float' if store else 'half'):
            raise ValueError('tensor memory role/type differs from delivered binding')
        resolved.append(dict(offset=offset,opcode=opcode,rank=rank,binding_index=index,written=store))
    loads=sum(not row['written'] for row in resolved);stores=len(resolved)-loads
    if not loads or not stores:raise ValueError('tensor delivery has no checked loads or stores')
    return dict(status='structurally_checked',loads=loads,stores=stores,resolved=resolved,
        metadata_bytes=len(metadata),ld_main_slot1=0,serialized_system_register_entry=52,
        limits='Decoded base-to-binding agreement only; effective address units, lane ownership, '
               'wait/dependency semantics and arithmetic remain unvalidated.')
