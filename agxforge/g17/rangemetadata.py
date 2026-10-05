"""Measured two-buffer layouts with explicit register allocation, emitted by g17mdgen.

This is metadata reproduction, not an execution claim. What it now also pins is
the class boundary: this is the EMPTY-constant-pool form. Six Apple compiles of
the same binding shape emit the identical eight-byte constant program whatever
their literals, so the constant program is not a function of the program; the
section grows 380 -> 428 exactly when slot 13's vector stops being empty.
"""
import copy
from pathlib import Path
from . import mdgen as M

# parents[2], not parents[1]: this file sits two directories below the repository root in
# agxforge/g17/ where tools/ was one. A pathlib anchor fails the same silent way an os.path one
# does - the path simply does not exist and every retained witness read from it is missing.
ROOT=Path(__file__).resolve().parents[2]
NO_SR=dict(size=380,root=16,rvt=4,root_vlen=12,root_tlen=12,
    pk=124,pkvt=60,pk_vlen=64,pk_tlen=60,
    pk_slots={29:4,27:8,26:12,13:16,12:20,10:24,8:28,6:32,3:36,4:40,2:44,1:48,16:54,15:55,0:56},
    pk_extra={1:('<I',4),15:('<B',1),16:('<B',1)},
    vec_bind=288,bind=[(368,360,'short'),(344,332,'long')],
    vec2=300,v2=[(320,310,'short')],v2_vals=[(3,4,None)],
    vec0=192,v0=(212,200),q=128,name=(44,48),cpname=(236,240),
    ptrs={13:268,27:184,29:188},nametab=(36,28,8,8,{1:4},{1:('<I',4)}),fills=[(20,'<I',16)])
# Adding the measured SR160 entry widens the previously empty slot-29 vector by
# four bytes; every subsequent record moves by four, confirmed on the paired source.
WITH_SR=copy.deepcopy(NO_SR)
WITH_SR.update(size=384,vec_bind=292,bind=[(372,364,'short'),(348,336,'long')],
    vec2=304,v2=[(324,314,'short')],vec0=196,v0=(216,204),q=132,cpname=(240,244),
    ptrs={13:272,27:184,29:188},slot29_vector=188)


# Exact source syn-se8dbb86316, object f83ea2bc4ba5cfe7... retained in
# results/g17-source-admission-v1/conversion-image-witness.json. Its first
# binding elides index zero; the second carries public index 7 at pointer offset
# 2. The long record's inline extent is 18, abutting the elided record's vtable.
# All other positions equal WITH_SR. The sole SR is 156, whose existing measured
# slot-29 value is zero. This is a separate named class, not a widened old default.
INDEXED_ZERO_SEVEN_CLASS = 'indexed-0-7-sr156-v1'
INDEXED_ZERO_CLASS = 'indexed-zero-sr156-v1'
INDEXED_ZERO_SEVEN = copy.deepcopy(WITH_SR)
INDEXED_ZERO_SEVEN.update(size=380,bind=[(372,366,'elided'),(348,336,'long',18)],
                         slot29_counts=(1,))


# THE SAME ELIDED-FIRST TWO-BUFFER SHAPE WITH NO SYSTEM REGISTER, measured from two CPU controls
# compiled for this boundary from the actual resource signatures - one float read at buffer 0 and one
# half written at buffer 1, and the same pair with the write at buffer 0. Both carry the FULL slot
# set this family has, 13/27/29 included and empty; the twelve-slot shape the generic serializer
# emits for these contracts matches no measured class and is not reproduced here.
#
# ONE FACT AT A TIME, and each difference is a single field:
#   SR156 write-1 380 -> no-SR write-1 376   the slot-29 vector loses its one entry, 4 bytes
#   no-SR  write-1 376 -> no-SR write-0 380  the written record changes shape, elided->elided_written
#                                            and long->mid, and slot 2's inline length 12->14
# Both reproduce their control byte for byte from the contract alone.
INDEXED_ZERO_NOSR_CLASS = 'indexed-zero-nosr-v1'
_NOSR_COMMON = dict(vec_bind=288, vec2=300, vec0=192, v0=(212, 200), cpname=(236, 240),
                    ptrs={13: 268, 27: 184, 29: 188},
                    arch32=bytes.fromhex('0c0000000000060008000400060000000c00000008000800'
                                         '00000700080000000000000100000000'))
INDEXED_ZERO_NOSR_WRITE1 = copy.deepcopy(INDEXED_ZERO_SEVEN)
INDEXED_ZERO_NOSR_WRITE1.pop('slot29_counts', None)
INDEXED_ZERO_NOSR_WRITE1.pop('slot29_vector', None)
INDEXED_ZERO_NOSR_WRITE1.update(_NOSR_COMMON, size=376, v2=[(320, 310, 'short')],
                                bind=[(368, 362, 'elided'), (344, 332, 'long', 18)])
INDEXED_ZERO_NOSR_WRITE0 = copy.deepcopy(INDEXED_ZERO_NOSR_WRITE1)
INDEXED_ZERO_NOSR_WRITE0.update(size=380, v2=[(320, 310, 'short', 14)],
                                bind=[(372, 360, 'elided_written'), (344, 334, 'mid', 16)])
for _k in (INDEXED_ZERO_NOSR_WRITE1, INDEXED_ZERO_NOSR_WRITE0):
    _k['q'] = _k['vec_bind'] - (_k['pk'] + _k['pk_slots'][4]) + 4


def build_indexed_zero_nosr(register_count, output_index, written_index):
    """The no-SR elided-first two-buffer class, selected by WHICH buffer is written.

    `written_index` is 0 or the output's public index; only those two are witnessed. A contract
    writing neither, or both, has no control here and is refused rather than served by whichever
    shape happens to be nearer.
    """
    if type(register_count) is not int or not 1 <= register_count <= 0xffffffff:
        raise ValueError('register allocation must be a positive uint32')
    if type(output_index) is not int or not 1 <= output_index <= 30:
        raise ValueError('indexed-zero output index must be a nonzero public buffer index1..30')
    if written_index == output_index:
        layout = INDEXED_ZERO_NOSR_WRITE1
    elif written_index == 0:
        layout = INDEXED_ZERO_NOSR_WRITE0
    else:
        raise ValueError('indexed-zero-nosr is measured for the write at buffer 0 or at the '
                         'output index; %r is witnessed by neither control' % (written_index,))
    return M.build([0, output_index], layout=layout, offsets=[0, 2],
                   register_count=register_count)


def build_indexed_zero_seven(register_count,system_registers):
    return build_indexed_zero(register_count,system_registers,7)


def build_indexed_zero(register_count,system_registers,output_index):
    """Same elided-first class, with the output's declared public index.

    Exact Apple syn-s6bedd68422 (0/1) and syn-se8dbb86316 (0/7) metadata
    differ only at byte352, the second binding's public-index field. Their
    complete380-byte sections, constant programs and pointer offsets agree.
    No layout or constant-program fact is inferred from the element type.
    """
    if type(register_count) is not int or not 1<=register_count<=0xffffffff:
        raise ValueError("register allocation must be a positive uint32")
    if type(output_index) is not int or not 1<=output_index<=30:
        raise ValueError('indexed-zero output index must be a nonzero public buffer index1..30')
    if tuple(system_registers)!=(156,):
        raise ValueError('indexed-zero class is measured only for system register 156')
    return M.build([0,output_index],layout=INDEXED_ZERO_SEVEN,offsets=[0,2],
                   register_count=register_count,system_registers=(156,))


# THE WITNESSED SETS FOR THIS CLASS, AND THE ONLY PLACE THEY ARE STATED. Every caller that
# decides which sets are admitted reads this name; it was written down four times and the copies
# drifted, which is what refused the delivered 178-byte packer's [160, 161] after the class itself
# had been witnessed for it. (160,) is the executed one. (160, 161) is measured twice over, by two
# compiles of DIFFERENT programs whose 388-byte sections are byte-identical: this side's
# measured/xy-2buf (float, o[t.y+t.x]=a[t.y+t.x]*2) and the compiler's xy-add (uint,
# y[4]=a[t.y+t.x]+1). The section is a function of the class, the register count and the register
# set, not of the program.
#
# The layout needed no new form. with_system_registers already widens the slot-29 vector and moves
# every later record by four, which is why the only thing that had to change was the gate.
MEASURED_REGISTER_SETS=((),(160,),(160,161))

def build(register_count,system_registers):
    if type(register_count) is not int or not 1<=register_count<=0xffffffff:
        raise ValueError('register allocation must be a positive uint32')
    registers=tuple(system_registers)
    # THE SET IS DECIDED BY THE SHARED MAP, NOT BY A COPY OF IT. slot29_entries exists so the
    # four-buffer and three-buffer serializers "cannot drift apart on which sets are admitted";
    # this was a third serializer that never called it. Membership above is this CLASS's witness
    # question; the map's coverage is a separate one, and an unmapped register refuses BY NAME.
    if registers not in MEASURED_REGISTER_SETS:
        raise ValueError('unmeasured system-register set for this two-buffer layout: %s; this class '
                         'is witnessed for %s' % (list(registers),
                                                  [list(r) for r in MEASURED_REGISTER_SETS]))
    if registers:
        M.slot29_entries(registers)
    return M.build([1,2],layout=WITH_SR if registers else NO_SR,offsets=[0,2],
        register_count=register_count,system_registers=registers)


WITNESSES=[('g17-range-class-volatile-witness-v1','volatile',()),
           ('g17-range-class-indexed-witness-v1','volatile-indexed',(160,)),
           ('g17-abi-class-probes-indexed','affine',(160,)),
           ('g17-range-class-pool-witness-v1','cp-plain',()),
           ('g17-range-class-pool-witness-v1','big-main-nosr',()),
           ('g17-range-class-pool-witness-v1','big-main-sr',(160,)),
           # 'x' declares uint2 but reads only t.x: 384 bytes, ONE entry. The declaration does
           # not widen slot 29; the use does. 'xy-add' and 'measured/xy-2buf' are the two-entry
           # vector, compiled independently on either side from different programs, byte-identical.
           # All three are compile-only and were never dispatched.
           ('g17-two-buffer-xy-witness-v1','x',(160,)),
           ('g17-two-buffer-xy-witness-v1','xy-add',(160,161)),
           ('g17-two-buffer-xy-witness-v1/measured','xy-2buf',(160,161))]
# The same kernels with pooled literals. They are NOT this class, and saying so is half the
# boundary: if one of them ever reproduced here, the class would be selecting on nothing.
POOLED=[('g17-range-class-pool-witness-v1',name) for name in ('cp-bigconst','cp-manyconst','cp-loop')]+[('g17-two-buffer-xy-witness-v1','xy')]


def _section(folder,name):
    from . import obj as g17obj
    raw=(ROOT/'results'/folder/(name+'.o')).read_bytes()
    sections,_=g17obj.sections_of(raw);off,size=sections['__GPU_METADATA,__compute']
    return raw[off:off+size]


def _pool_length(metadata):
    import struct
    from . import gpumd as g17gpumd
    field=g17gpumd._table_slot(metadata,g17gpumd.kernel_table(metadata),13)
    return struct.unpack_from('<I',metadata,field+struct.unpack_from('<I',metadata,field)[0])[0]


def check():
    from . import gpumd as g17gpumd
    from . import scanlink as g17scanlink
    observations=[]
    for folder,name,registers in WITNESSES:
        metadata=_section(folder,name)
        count=g17gpumd.register_count(metadata)
        if count is None:raise ValueError('witness carries no register allocation: '+name)
        if build(count,registers)!=metadata:raise ValueError('metadata reproduction differs: '+name)
        if g17scanlink.binding_records(metadata)!=[(1,0,False),(2,2,True)]:raise ValueError('witness binding list differs')
        if _pool_length(metadata):raise ValueError('this class is the empty-pool form: '+name)
        observations.append(dict(witness=name,bytes=len(metadata),register_count=count,
                                 system_registers=list(registers),constant_pool_words=0,differing_bytes=0))
    excluded=[]
    for folder,name in POOLED:
        metadata=_section(folder,name)
        words=_pool_length(metadata)
        if not words:raise ValueError('expected a non-empty constant pool: '+name)
        if len(metadata) in (380,384,388):raise ValueError('a pooled program landed in the empty-pool sizes: '+name)
        if build(g17gpumd.register_count(metadata),())==metadata:
            raise ValueError('the empty-pool class reproduced a pooled program: '+name)
        excluded.append(dict(witness=name,bytes=len(metadata),constant_pool_words=words))
    return dict(status='passed',gpu_dispatched=False,observations=observations,excluded=excluded,
        scope=('Nine metadata sections reproduce byte-exact across seven distinct programs at '
               'register counts 1 and 2 and all three measured register sets - none, (160,) and '
               '(160, 161), the last of these measured twice from different programs; four pooled '
               'programs are checked to fall outside. The two-coordinate witnesses were compiled '
               'for this class and never dispatched. Native prologue compatibility and GPU '
               'execution are not established by this check.'))


def main(argv=None):
    import json
    print(json.dumps(check(), indent=2))


# CPU oracle controls in results/g17-resident-ffn-metal-metadata-v1, 2026-09-29.
# Exact indices1/2 write-first/read-second, XY, empty pool. Short392B and
# long396B reproduce completely, without residual bytes or copied section bytes.
# Long source4ecbbe54 / object982d0d5a / metadata216fb031; exactly55 instructions.
WRITE_FIRST_XY = copy.deepcopy(WITH_SR)
WRITE_FIRST_XY.update(size=388, bind=[(376,364,'written'),(348,338,'mid',16)],
                      v2=[(324,314,'short',14)])
WRITE_FIRST_XY_LONG = copy.deepcopy(WRITE_FIRST_XY)
WRITE_FIRST_XY_LONG.update(size=392,pk=128,pkvt=58,pk_vlen=70,pk_tlen=60,
    pk_slots={0:56,1:48,2:44,3:36,4:40,6:32,8:28,10:24,12:20,13:16,
              15:55,16:54,26:12,27:8,29:4,32:53},
    pk_extra={1:('<I',4),15:('<B',1),16:('<B',1),32:('<B',1)},
    vec_bind=296,bind=[(380,368,'written'),(352,342,'mid',16)],
    vec2=308,v2=[(328,318,'short',14)],vec0=200,v0=(220,208),q=132,
    cpname=(244,248),ptrs={13:276,27:188,29:192},slot29_vector=192)

def build_write_first_xy(register_count, instructions, back_edge=False):
    """Measured empty-pool write-first XY family, bounded by the55-instruction control."""
    if type(register_count) is not int or not 1 <= register_count <= 126:
        raise ValueError('write-first XY requires register_count1..126')
    if back_edge or type(instructions) is not int or not 1 <= instructions <= 55:
        raise ValueError('write-first XY is measured only for straight-line1..55 instructions')
    layout = WRITE_FIRST_XY if M.slot32_for(instructions,False) is None else WRITE_FIRST_XY_LONG
    return M.build([1,2],layout=layout,offsets=[0,2],register_count=register_count,
                   system_registers=(160,161))

if __name__ == "__main__":
    main()
