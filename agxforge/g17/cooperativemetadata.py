"""Emit the measured cooperative resource class - empty pool, and now the pool-carrying one.

    python3 tools/g17cooperativemetadata.py check <delivery>   rebuild every retained witness

WHAT WAS OPEN. The empty-pool cooperative class reproduced its 516-byte witness, and the
pool-carrying members did not: 126 of 624 bytes differed in two structures - the root table's
tlen and one value, and slot 13's vector, which in this class carries 128 bytes of real content
where every earlier class has a zero vector. Both are closed here, and NEITHER is a donor copy.

  THE ROOT FORM IS A RULE, measured over the census rather than read off a witness. The root table
  takes exactly two shapes in 5,691 distinct sections: 12 bytes with slot 3 = 16 (4,999) and 14
  bytes with slot 3 = 20 (692). Slot 28 - the static threadgroup size - is present in all 692 of
  the second and in none of the first. So the form follows a fact the contract already states.

  SLOT 13 IS THE CONSTANT POOL, and its content is COMPILE-TIME DATA. Decoded as halves, all 64
  values in coop4-1c's vector are literals of its own source's fma chain (0.0, 16.5, 17.5, ...).
  Nothing derives it; the contract supplies it or this refuses. `field2` of the kind-6 record is
  then the pool length over four - the split that closed at 24,975 of 24,975 with field4 elided.

  AND ONE CANDIDATE IS REFUTED RATHER THAN ADOPTED. Both pool values fit `slot 1 = 8 + pool/4`
  (12 at a 16-byte pool, 40 at 128), which is two points and a line. Over the census it holds on
  154 of 5,685 sections, so slot 1 stays a contract input and this refuses without it.

WHAT IS REPRODUCED. With the pool bytes, pk_slot1, register_count and the system registers
supplied, coop4-1c (624, one register), coop4-n185 (628, two) and coop4-flat-a (628) are built
BYTE-EXACT - zero differing bytes, including both out-of-sample witnesses integration named.
"""
from collections.abc import Mapping as _Mapping
import copy
import hashlib
import json
import os
import struct
import sys

from . import mdgen as M
from . import gpumd as GM

SIGNATURE=((1,0,False),(2,2,False),(3,4,False),(4,6,True))

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WITNESSES = os.path.join(ROOT, "results", "g17-cooperative-class-census-v1", "measured")

# THE POOL-CARRYING CLASS's layout, measured in results/g17-cooperative-class-census-v1 and kept
# here as the one-register form; g17mdgen widens it for a second system register itself.
POOL_LAYOUT = {
    "size": 624, "pk": 136, "pkvt": 66, "pk_vlen": 70, "pk_tlen": 64,
    "pk_slots": {0: 60, 1: 48, 2: 44, 3: 36, 4: 40, 6: 32, 8: 28, 10: 24, 12: 20, 13: 16,
                 15: 59, 16: 58, 18: 57, 26: 12, 27: 8, 28: 52, 29: 4, 32: 56},
    "vec_bind": 436, "q": 264, "vec2": 456, "vec0": 212,
    "bind": [(612, 604, "short"), (588, 578, "mid", 16), (560, 550, "mid", 18),
             (532, 520, "long", 18)],
    "v2": [(504, 492, "long"), (480, 470, "short")],
    "ptrs": {13: 288, 27: 200, 29: 204},
    "name": (52, 56), "cpname": (256, 260), "slot29_vector": 204,
}

# The root form, and the fact that selects it. Measured over 5,691 distinct census sections:
# slot 28 present in 692 of 692 carrying the 14/20 form and in 0 of the 4,999 carrying 12/16.
ROOT_FORMS = {True: {"root_tlen": 14, "slot3_value": 20},
              False: {"root_tlen": 12, "slot3_value": 16}}

POOL_WORD_BYTES = 4          # the kind-6 field2 unit, and the pool's own alignment

# Named here because a caller has to know them, and because a field that is supplied must never
# be mistaken for one that is derived.
CONTRACT_INPUTS = ("constant_pool (the pool's own bytes - compile-time data)",
                   "pk_slot1 (slot 1; the pool-length candidate is refuted at 154 of 5,685)",
                   "register_count (slot 0)", "system_registers (the slot-29 vector)",
                   "threadgroup.static_memory_bytes (slot 28)",
                   "instruction_count and has_back_edge (slot 32, program-dependent)")


# THE THREE-BINDING THREADGROUP SIGNATURES, each witnessed by its own CPU control compiled from a
# delivered source with all declarations kept active. They are the same family as SIGNATURE at three
# records rather than four, so they route here; the write position selects the class, because only
# a record shape carrying slot 3 marks a write and that shape sits where the write is.
THREE_SIGNATURES={((0,0,False),(1,2,True),(2,4,False)):1,
                  ((0,0,True),(1,2,False),(2,4,False)):0}


def _three_binding_write(bindings):
    """The written buffer index for a witnessed three-binding threadgroup signature, else None."""
    return THREE_SIGNATURES.get(tuple(tuple(b[:3]) for b in bindings))


# THE THREE-BINDING CLASS'S OWN MEASURED CONSTANTS. Neither is a default and neither is the
# four-binding class's: slot 1 is 8 at an empty pool in all five controls, where the four-binding
# class carries 12 for the same empty pool, and every control was compiled at SR156/SR164. The
# emitter used to read `abi.get('pk_slot1', 8)`, which would have served a contract that stated
# nothing the same bytes as one that stated 8 - a default wearing a measurement's clothes.
# threadgroup sizes this class has executed bit-exact on hardware (MM 25.140.3; wider sizes MM 25.141.4)
MEASURED_SIZES = {(32, 1, 1),
                  # build_rmsnorm_wide at 1,024 threads: 32 simdgroups publish partials through the scratchpad across
                  # one threadgroup barrier and every lane reads all 32 - bit-exact against its reference, both norms
                  (1024, 1, 1),
                  # build_qmv2 split-K (MM 25.141.15): 2, 4 and 8 simdgroups publish row partials through the scratchpad
                  # across one threadgroup barrier and every simdgroup sums all of them - bit-exact on hardware at q4
                  # 2048 x 8192, rows 1 and 2
                  (64, 1, 1), (128, 1, 1), (256, 1, 1)}

THREE_SLOT1 = 8
THREE_REGISTERS = (156, 164)
FOUR_SLOT1 = 12   # the four-binding class's own value at the same empty pool


def _class_of(bindings):
    """Which measured cooperative class this binding signature selects, or None for neither.

    The SIGNATURE selects, never the record count: the three-binding shapes below are the same
    family at three records rather than four, and the write position selects between them.
    """
    signature = tuple(tuple(b[:3]) for b in bindings)
    if signature == SIGNATURE:
        return 'four'
    if signature in THREE_SIGNATURES:
        return 'three'
    return None


def empty_pool_slot1(bindings):
    """Slot 1 for an empty external constant pool, for whichever cooperative class this selects.

    MEASURED PER CLASS, not shared: 12 in the four-binding class's witnesses and 8 in all five of
    the three-binding class's. A caller that fills in one class's value for the other authors a
    section whose pointer block does not match the class it just selected.
    """
    return THREE_SLOT1 if _class_of(bindings) == 'three' else FOUR_SLOT1


def validate(abi,bindings):
    """THE GATE for every cooperative contract: shared invariants first, then the class's own.

    BOTH CLASSES ARE REACHED FROM HERE, and that is the repair. `g17authorobj.author` calls this -
    and only this - before it emits anything, so routing the three-binding class inside `emit`
    alone left it unreachable from the public author: the gate refused first, with the FOUR-binding
    class's message, for a contract this side had measured. The only test that exercised the route
    called a private entry point and so could not see it.

    The split also fixes what the early route skipped. A contract that OMITTED
    static_memory_alignment authored anyway against a silent 4, and one that ADDED a dynamic_memory
    block authored anyway with the block ignored. Those checks live in the shared half now, so
    neither class can be reached without them.
    """
    if abi.get('abi_version')!=4:raise ValueError('cooperative author requires ABI v4')
    which=_class_of(bindings)
    if which is None:
        raise ValueError('unmeasured cooperative binding class')
    _validate_shared(abi,bindings)
    if which=='four':
        _validate_four(abi,bindings)
    else:
        _validate_three(abi,bindings)


def _validate_shared(abi,bindings):
    """What EVERY cooperative contract must state, whichever class its signature selects.

    Nothing here is defaulted: a fact that is absent refuses rather than taking the value the
    witnessed contracts happen to carry.
    """
    for key,want in [('uses_threadgroup',True),('has_stores',True),('writes_buffer',True),('writes_texture',False),('arch_flag',False)]:
        if abi.get(key) is not want:raise ValueError('cooperative class differs: '+key)
    if abi.get('constant_pool') is None:raise ValueError('cooperative author requires compiler constant_pool declaration')
    pool=_pool_bytes(abi)
    if pool is None:
        raise ValueError('constant_pool must be a list of byte values or a hex string')
    if len(pool)%POOL_WORD_BYTES:
        raise ValueError('constant_pool is %d bytes; the kind-6 field2 unit is %d and an unaligned '
                         'pool has no measured encoding'%(len(pool),POOL_WORD_BYTES))
    # A NON-EMPTY POOL NEEDS SLOT 1 STATED. It is 12 at an empty pool and 40 at the 128-byte one,
    # which fits 8 + pool/4 on both points and on 154 of 5,685 census sections - a two-point line,
    # not a rule. So the contract states it or this refuses; it is never taken from a witness.
    # THE ONE EXCEPTION IS MEASURED, not derived: the three-binding class's mask pool (three_mask_pool),
    # whose witnesses carry slot 1 = 8 at 0, 1, 2 and 3 masks alike (MM 25.141.16).
    if (len(pool)!=16 and abi.get('pk_slot1') is None
            and not (_class_of(bindings)=='three' and three_mask_pool(pool))):
        raise ValueError('a non-empty constant pool requires the contract to state pk_slot1: it is '
                         'not derivable from the pool length (8 + pool/4 holds on 154 of 5,685 '
                         'census sections)')
    tg=abi.get('threadgroup')
    if (not isinstance(tg,dict) and not hasattr(tg,'items')):raise ValueError('missing threadgroup block')
    # EVERY FIELD IS STATED. The field-set check is the one the early three-binding route bypassed;
    # with it, a contract missing static_memory_alignment or carrying a dynamic_memory block is
    # refused here instead of being authored against a default or an ignored declaration.
    required=('required_size','static_memory_bytes','static_memory_alignment','dynamic_memory')
    if set(tg)!=set(required):
        missing=[k for k in required if k not in tg];extra=sorted(set(tg)-set(required))
        raise ValueError('unsupported threadgroup fields: the block states %s and this author '
                         'requires exactly %s%s%s'
                         %(sorted(tg),list(required),
                           '; missing '+', '.join(missing) if missing else '',
                           '; unsupported '+', '.join(extra) if extra else ''))
    got=tg['required_size']
    # THE SIZE IS A LAUNCH BOUNDARY, not a byte of this section (see the ledger line below), so a wider
    # threadgroup authors the same metadata. What a wider size needs is a MEASUREMENT that the class's barrier and
    # scratchpad work across simdgroups: MEASURED_SIZES holds the ones executed bit-exact on hardware, and
    # G17_COOP_SIZE_EXPERIMENT=1 admits any multiple of 32 up to 1024 so that measurement can be made.
    ok=(isinstance(got,(list,tuple)) and len(got)==3 and all(type(n) is int for n in got)
        and (tuple(got) in MEASURED_SIZES
             or (os.environ.get('G17_COOP_SIZE_EXPERIMENT')=='1' and got[1]==got[2]==1
                 and got[0]%32==0 and 32<=got[0]<=1024)))
    if not ok:
        raise ValueError('unmeasured cooperative required_size')
    got=tg['dynamic_memory']
    # NO DYNAMIC STORAGE POLICY IS MEASURED on either class. Every witness declares none, so a
    # contract that asks for dynamic threadgroup memory is refused rather than authored with the
    # request dropped - the section would then describe a program the contract does not describe.
    if not isinstance(got,(list,tuple)) or tuple(got)!=():
        raise ValueError('unmeasured cooperative dynamic_memory')
    size=tg['static_memory_bytes'];align=tg['static_memory_alignment']
    if type(size) is not int or size<=0:
        raise ValueError('cooperative author requires a declared positive static threadgroup '
                         'extent; the contract states %r'%(size,))
    if type(align) is not int or align<=0:
        raise ValueError('cooperative author requires the contract to state '
                         'static_memory_alignment; it states %r'%(align,))
    if size%align:
        raise ValueError('declared threadgroup extent %d is not a multiple of its stated '
                         'alignment %d'%(size,align))
    registers=tuple(abi.get('system_registers',()))
    # THE ORDER IS CHECKED BEFORE THE LOOKUP, because the lookup's refusal names the set and a
    # caller reads that as "add this set". g17authorobj requires system_registers to be SORTED;
    # the witnessed keys are sorted too, and within one base that is invisible - the slot-29
    # entries ascend with the registers, so a vector read back out of a witness is already in
    # register order. Read back out of a CROSS-BASE witness it is not: coop4-tg's vector is
    # [48, 80, 81], which maps to (164, 160, 161), and a caller handing that straight back got
    # "no reproduced cooperative witness for ... [164, 160, 161]" - a set named in an order this
    # author would have rejected two frames later. Two refusals, and the misleading one came
    # first.
    if registers!=tuple(sorted(set(registers))):
        raise ValueError('system_registers must be sorted and unique before this class is looked '
                         'up; got %s, which sorts to %s. A vector read back from a cross-base '
                         'witness arrives in ENTRY order and has to be sorted into register order '
                         'first.'%(list(registers),sorted(set(registers))))
    count=abi.get('instruction_count');back=abi.get('has_back_edge')
    # SLOT 32 IS PROGRAM-DEPENDENT and its ABSENCE is a different layout - four bytes shorter, with
    # every structure after it moved - so both classes need these two facts STATED, and each one
    # decides for itself which side of the boundary it is witnessed on. Neither may be inferred:
    # the early three-binding route read `abi.get('has_back_edge', False)`, which would have given
    # a looping contract the flat table.
    if type(count) is not int or type(back) is not bool:
        raise ValueError('this cooperative author requires the compiler instruction count and '
                         'back-edge fact for slot 32')
    if abi.get('entry')!=64:raise ValueError('unmeasured cooperative entry')
    prologue=abi.get('prologue')
    if isinstance(prologue,str):prologue=bytes.fromhex(prologue)
    if prologue!=bytes.fromhex('0e000000')+bytes.fromhex('0600')*30:
        raise ValueError('native cooperative author requires declared END/filler prologue')
    if type(abi.get('register_count')) is not int or not 1<=abi['register_count']<=0xffffffff:
        raise ValueError('invalid cooperative register allocation')
    # These facts are represented by this measured layout, not caller overrides.
    for key in ('pk_vectors','pk_empty_vectors','slot2_extra','slot2_kind6','slot2_kind6_f2',
                'constant_program','v0_field2','v0_field3','measured_class','reproduce_measured_class'):
        if key in abi and abi[key] is not None and abi[key] is not False:
            raise ValueError('cooperative class does not accept override '+key)
    if dict(abi.get('pk_values',{}))!={15:1,16:1,18:1}:
        raise ValueError('cooperative pk_values differ from measured class')
    if tuple(abi.get('pk_extra',()))!=(15,16,18):
        raise ValueError('cooperative pk_extra differs from measured class')


def _validate_four(abi,bindings):
    """The four-binding cooperative class's own facts, unchanged from the accepted baseline."""
    tg=abi['threadgroup'];pool=_pool_bytes(abi)
    if tg['static_memory_bytes']!=128:raise ValueError('unmeasured cooperative static_memory_bytes')
    if tg['static_memory_alignment']!=4:raise ValueError('unmeasured cooperative static_memory_alignment')
    registers=tuple(abi.get('system_registers',()))
    count=abi['instruction_count'];back=abi['has_back_edge']
    # THE (POOL LENGTH, REGISTER SET) PAIRS WITH A WITNESS, and no others. The slot-29 vector's
    # length moves every structure after it, so a set admitted by analogy from another pool length
    # would yield a section of an entirely plausible size that decodes as if nothing were wrong.
    # THE CONTRACT'S BACK-EDGE FACT CHOOSES THE TABLE, and it has to: a back edge is a different
    # CLASS - slot 32 = 3 and per-kernel slot 33, a per-kernel table four bytes longer with
    # everything after it moved - so a form witnessed flat says nothing about the same form with a
    # loop, and vice versa.
    #
    # THIS CHECKED THE FLAT TABLE FIRST FOR EVERY CONTRACT and then the loop table as well. Both
    # loop forms happen to be witnessed flat too, so it worked - by coincidence. A loop form
    # without a flat counterpart would have been refused by a message listing the FLAT forms,
    # which is a refusal naming the wrong table, and a reader would have gone and added it there.
    table, which = (WITNESSED_LOOP_FORMS, 'loop') if back else (WITNESSED_FORMS, 'flat')
    if (len(pool), registers) not in table:
        raise ValueError('no reproduced cooperative witness for a %d-byte pool with system '
                         'registers %s and %s; the witnessed %s forms are %s'
                         %(len(pool),list(registers) or 'none',
                           'a back edge' if back else 'no back edge', which,
                           ', '.join('%d/%s'%(n,list(r)) for n,r in sorted(table)) or 'none'))
    # A LOOPING PROGRAM WITH NO SLOT 32 IS A THIRD CLASS. slot32_for returns None below thirty-one
    # instructions; with a back edge that is the coop4-tg2 shape - slot 33 present, slot 32 absent,
    # a per-kernel table four bytes shorter. One form of it is reproduced byte-exact
    # (results/g17-coop-noslot32-v1); every other is refused rather than served by a neighbour.
    if M.slot32_for(count,back) is None:
        if not (back and (len(pool), registers) in WITNESSED_NO32_FORMS):
            raise ValueError('a program of %d instruction(s) carries no slot 32, which is a '
                             'different measured layout than this class. The reproduced '
                             'no-slot-32 loop forms are %s; this contract is a %d-byte pool with '
                             'system registers %s and %s'
                             %(count,
                               ', '.join('%d/%s'%(n,list(r)) for n,r in sorted(WITNESSED_NO32_FORMS))
                               or 'none', len(pool), list(registers) or 'none',
                               'a back edge' if back else 'no back edge'))
    if len(pool)==16 and abi.get('pk_slot1') not in (None,12):
        raise ValueError('cooperative pointer/constant block differs')


def _validate_three(abi,bindings):
    """The three-binding threadgroup class's own facts.

    Its extent is its own - 1024 and 256 bytes in the two deliveries, where the four-binding class
    is measured at 128 - so the extent is the one threadgroup value this class does not pin. The
    facts it DOES pin are the ones five CPU controls establish.
    """
    tg=abi['threadgroup'];size=tg['static_memory_bytes']
    # THE SECTION STATES THE EXTENT AS A COUNT OF FIXED 4-BYTE SCRATCH WORDS, so an extent that is
    # not a whole number of them has no encoding here rather than a rounded one.
    if size%M.SCRATCH_WORD_BYTES:
        raise ValueError('this class states the declared threadgroup extent as a count of %d-byte '
                         'scratch words; %d bytes is not a whole number of them'
                         %(M.SCRATCH_WORD_BYTES,size))
    registers=tuple(abi.get('system_registers',()))
    if registers!=THREE_REGISTERS:
        raise ValueError('every control for the three-binding threadgroup class was compiled at '
                         'system registers %s; %s is witnessed by none of them, and the slot-29 '
                         'vector moves every structure after it'
                         %(list(THREE_REGISTERS),list(registers) or 'none'))
    pool=_pool_bytes(abi)
    if (len(pool)!=16 or any(pool)) and not three_mask_pool(pool):
        raise ValueError('this class is witnessed at an empty pool and at the 8-byte mask pool (halfword '
                         '0 zero, one to three 16-bit masks after it, MM 25.141.16); a %d-byte pool of '
                         'another shape is unwitnessed here'%len(pool))
    if abi.get('pk_slot1') not in (None,THREE_SLOT1):
        raise ValueError('slot 1 is %d in every control for this class at an empty pool; the '
                         'contract states %r'%(THREE_SLOT1,abi.get('pk_slot1')))
    # A BACK EDGE IS A DIFFERENT MEASURED LAYOUT - slots 32 = 3 and 33 = 1 and a longer per-kernel table -
    # WITNESSED since MM 25.140.4 by one-edit loop variants of both controls (results/g17-coop3-loopclass-v1,
    # byte-identical through g17mdgen.threadgroup_three(..., back_edge=True)). Only that form: a looping
    # program below the slot-32 boundary (slot 33 without slot 32) has no control and stays refused, and a
    # straight-line program is still witnessed below the boundary only.
    if abi['has_back_edge']:
        if M.slot32_for(abi['instruction_count'],True)!=3:
            raise ValueError('the three-binding loop class is witnessed at slot 32 = 3 (31 or more '
                             'instructions); a looping program of %d instructions carries no slot 32, '
                             'a shape no control has' % abi['instruction_count'])
    elif M.slot32_for(abi['instruction_count'],abi['has_back_edge']) is not None:
        raise ValueError('the three-binding threadgroup class is witnessed below the slot-32 '
                         'instruction boundary; this contract crosses it at %d instructions'
                         %abi['instruction_count'])
    # THE LAUNCH FACTS ARE STATED, not inferred from the storage declaration. This class is new, so
    # requiring them costs no existing caller; the four-binding class's accepted baseline is left
    # exactly as it was rather than tightened underneath the payloads it has already authored.
    launch=abi.get('launch')
    if (not isinstance(launch,_Mapping) or set(launch)!={'bounds_checked','exact_grid_required'}
            or launch.get('bounds_checked') is not False
            or launch.get('exact_grid_required') is not True):
        raise ValueError('the three-binding threadgroup class requires the contract to state its '
                         'launch facts as bounds_checked false and exact_grid_required true; it '
                         'states %r'%(launch,))


EMPTY_POOL_BYTES=bytes(16)   # what "no external constants" means in this class: a zero vector


def three_mask_pool(pool):
    """True for the three-binding class's witnessed MASK POOL (MM 25.141.16): 8 bytes, halfword 0 zero, then
    one to three non-zero 16-bit masks and zero padding - what cc's and16(pool=True) states.

    WITNESSED, content-only: tgctl3-u64b and its loop control (MM 25.140.4) recompiled with one, two and three
    16-bit field masks keep their sections (472 and 484 bytes, slot 1 = 8, slot 32 and 33 unchanged) and differ
    from the empty-pool witness ONLY in these 8 bytes, so g17mdgen's layout plus the bytes reproduces Apple's
    section byte for byte. Eight masks is a different, 488/492-byte layout with slot 1 = 12 and stays refused."""
    if pool is None or len(pool)!=8:return False
    halves=[pool[i]|pool[i+1]<<8 for i in range(0,8,2)]
    if halves[0]!=0:return False
    used=[h for h in halves[1:] if h]
    return bool(used) and halves[1:]==used+[0]*(3-len(used))


def _pool_bytes(abi):
    """The contract's constant pool as bytes, or None if it is not a shape this accepts.

    A contract declaring NO external constants gets the measured 16-byte zero vector, which is
    what this class has always emitted for it. A contract that states 16 bytes gets those bytes:
    the vector's content is not always zero - coop4-flat-b carries 00 00 80 01 there - so an empty
    declaration and a stated all-but-zero vector are different inputs and stay different.
    """
    pool=abi.get('constant_pool')
    if isinstance(pool,(list,tuple)) and not pool:return EMPTY_POOL_BYTES
    if isinstance(pool,(bytes,bytearray)) and not pool:return EMPTY_POOL_BYTES
    if isinstance(pool,str) and not pool:return EMPTY_POOL_BYTES
    if isinstance(pool,str):
        try:return bytes.fromhex(pool)
        except ValueError:return None
    if isinstance(pool,(bytes,bytearray)):return bytes(pool)
    if isinstance(pool,(list,tuple)):
        if all(type(v) is int and 0<=v<256 for v in pool):return bytes(pool)
        return None
    return None


# ONE MEASURED LAYOUT PER POOL LENGTH. The pool's bytes sit inside the section, so its length is
# a layout fact, not a payload one: 16 bytes is the 516-byte class and 128 the 624-byte one. A
# length with no witness is refused rather than served by the nearer layout, which would write the
# pool over whatever follows it and return a section of an entirely plausible size.
POOL_LENGTHS = (16, 128)

# Every (pool length, system-register set) this author accepts, each because a witness of that
# form is reproduced BYTE-EXACT by this file's own build - see RETAINED and the delivery's report.
# THE THREE-REGISTER PAIR IS WITNESSED NOW, and by a compile made for the purpose rather than by
# analogy. coop4-flat-tg is coop4-flat-b with TWO source edits - the threadgroup-position attribute
# and the index it feeds - so the register set is the only fact that moves, and it authors
# byte-for-byte at 520 bytes (results/g17-coop-tg-witness-v1).
#
# WHAT IT DOES NOT ADMIT. The four coop4-tg* witnesses stay refused, and not for their registers:
# coop4-tg-nolt declares the already-witnessed (160, 161) and still misses by 189 bytes. They carry
# slot 33 and slot 32 = 3, which travel with the LOOP - coop4-flat-tg has the same registers and
# neither. The pair (16, (160, 164)) also stays unwitnessed.
# THE LOOP FORMS ARE THEIR OWN TABLE. A form witnessed without a back edge says nothing about the
# same form with one: coop4-flat-tg authors (16, (160,161,164)) flat, and coop4-tg carries that
# same pair WITH a loop and is not reproduced. Keeping one table would have admitted it.
# NO NEW LAYOUT FOR THE THREE-REGISTER LOOP. The extra slot-29 entry is what
# g17mdgen.with_system_registers already derives from the two-register class, and the byte check
# against coop4-loop-tg confirms that derivation rather than a second layout literal transcribed
# beside the first. Admitting the FORM is the whole change (results/g17-coop-loop3-v1).
# THE NO-SLOT-32 LOOP FORMS ARE THEIR OWN TABLE TOO, for the same reason the loop forms are: a
# form witnessed WITH slot 32 says nothing about the same form without it - the table is four bytes
# shorter and everything after it moves.
WITNESSED_NO32_FORMS = {(16, (160, 161, 164)):
                            "coop4-tg2 and cn32-trip31, 524 bytes, slot 33 = 1, no slot 32"}

# THE SLOT-29 VECTOR'S WITNESSED LENGTHS FOR THIS CLASS. The module default is (1, 2, 3); a fourth
# entry is witnessed here and nowhere else - cgb-group-4, 524 bytes, entries [0, 48, 80, 81], which
# is the three-register class plus one four-byte entry (results/g17-coop-group-base-v1). Stating it
# on the class rather than widening the module constant keeps the two serializers that have never
# witnessed a fourth entry refusing one.
SLOT29_COUNTS = (1, 2, 3, 4)

WITNESSED_LOOP_FORMS = {(16, (160, 161)): "coop4-loop-nolt, 524 bytes, slot 32 = 3, slot 33 = 1",
                        (16, (160, 164)):
                            "coop4-tg-x, 524 bytes, slot 32 = 3, slot 33 = 1",
                        (16, (160, 161, 164)):
                            "coop4-loop-tg, 528 bytes, slot 32 = 3, slot 33 = 1"}

# THE GROUP BASE NEEDED NO LAYOUT, ONLY A WITNESS. cgb-group-x is coop4-flat-x with the local
# register replaced by the group register, and its section differs in FOUR BYTES - one slot-29
# entry, 48 to 0 - because the vector's LENGTH is what moves structures. Its y sibling was
# predicted before compiling and confirmed. The z axis is deliberately absent: two axes of a base
# do not witness the third (results/g17-coop-group-base-v1).
WITNESSED_FORMS = {(16, (156, 160)): "cgb-group-x, 516 bytes, slot 32 = 1, no slot 33",
                   (16, (157, 160)): "cgb-group-y, 516 bytes, slot 32 = 1, no slot 33",
                   (16, (156, 160, 161, 164)):
                       "cgb-group-4, 524 bytes, four slot-29 entries, slot 32 = 1, no slot 33",
                   (16, (160, 161)): "coop4-flat-b, 516 bytes, slot 32 = 1",
                   (16, (160, 164)): "coop4-flat-x, 516 bytes, slot 32 = 1, no slot 33",
                   (16, (160, 161, 164)): "coop4-flat-tg, 520 bytes, slot 32 = 1, no slot 33",
                   (128, (160,)): "coop4-1c, 624 bytes",
                   (128, (160, 161)): "coop4-n185 and coop4-flat-a, 628 bytes"}


def _layout_for(pool,abi):
    """The class layout, chosen by the pool's LENGTH - and the root form with it."""
    if len(pool) not in POOL_LENGTHS:
        raise ValueError('no measured cooperative layout for a %d-byte constant pool; the '
                         'witnessed lengths are %s'%(len(pool),', '.join(map(str,POOL_LENGTHS))))
    # SELECTED BY THE CONTRACT'S OWN BACK-EDGE FACT, not by a buffer or register count and not by
    # a fallback: the loop class exists only for a contract that states has_back_edge.
    if abi.get('has_back_edge') and len(pool)==16:
        # The no-slot-32 loop class is selected by the contract's own instruction count, through
        # the same measured law that decides the value when there is one.
        layout=copy.deepcopy(M.FOUR_THREADGROUP_LOOP_NO32
                             if M.slot32_for(abi['instruction_count'],True) is None
                             else M.FOUR_THREADGROUP_LOOP)
    else:
        layout=copy.deepcopy(M.FOUR_THREADGROUP if len(pool)==16 else dict(M.FOUR_THREADGROUP,**POOL_LAYOUT))
    layout['slot29_counts']=SLOT29_COUNTS
    layout['pk_extra']=dict(layout['pk_extra'])
    layout['pk_extra'][28]=('<I',abi['threadgroup']['static_memory_bytes'])
    slot32=M.slot32_for(abi['instruction_count'],abi['has_back_edge'])
    if slot32 is None:
        layout['pk_extra'].pop(32,None)
    else:
        layout['pk_extra'][32]=('<B',slot32)
    # THE ROOT FORM FOLLOWS SLOT 28, and this class always declares it. Written from the rule
    # rather than carried in the layout literal so the fact is stated once and checkable.
    form=ROOT_FORMS[abi['threadgroup']['static_memory_bytes'] is not None]
    layout['root_tlen']=form['root_tlen']
    layout['fills']=tuple((pos,fmt,form['slot3_value'] if pos==0x14 else val)
                          for pos,fmt,val in layout['fills'])
    # DERIVED: kind-6 field2 is the pool length over four, with field4 elided. That is the split
    # this side closed at 24,975 of 24,975 sections across two corpora.
    layout['v2_vals']=[(6,len(pool)//POOL_WORD_BYTES,8),(3,8,None)]
    return layout


def _emit_three(abi,bindings,ledger,written):
    """The three-binding threadgroup class, from the contract's own declaration.

    `validate` has already run every shared invariant and every fact specific to this class, so
    nothing is defaulted here: the extent, the alignment, the register set, the pool, the launch
    facts and the two instruction facts were all STATED by the contract or the contract was
    refused. This function reads them and serializes.
    """
    tg=abi['threadgroup'];size=tg['static_memory_bytes'];align=tg['static_memory_alignment']
    words=size//M.SCRATCH_WORD_BYTES
    layout=M.threadgroup_three(size,written,instructions=abi['instruction_count'],
                               back_edge=bool(abi['has_back_edge']))
    result=M.build([b[0] for b in bindings],layout=layout,offsets=[b[1] for b in bindings],
        register_count=abi['register_count'],system_registers=abi['system_registers'])
    result=_place_pool(result,_pool_bytes(abi),THREE_SLOT1)
    ledger['metadata class']=('MEASURED three-binding threadgroup class: %d bytes, declared static '
        'extent %d bytes stated as %d 4-byte scratch words, write at buffer %d'
        %(len(result),size,words,written))
    ledger['per-kernel slot 28 and the name table']=('DERIVED FROM THE DECLARED EXTENT ALONE: slot '
        '28 is the extent and the name table states a fixed 4-byte word with extent/4 of them. '
        'Five CPU controls separate that reading from the source element count (half[512] in 1024 '
        'bytes states 256, not 512) and from extent/alignment (an aligned(16) float[256] control '
        'is BYTE-IDENTICAL to the aligned(4) one), so the declared alignment %d is checked against '
        'the extent and is not encoded in this section.'%align)
    ledger['per-kernel slot 1']=('MEASURED for this class at an empty pool: %d. The four-binding '
        'cooperative class carries 12 for the same empty pool, which is why this is a class '
        'constant here and never a default.'%THREE_SLOT1)
    ledger['launch requirement']=('LAUNCH BOUNDARY, not encoded in this section: required size %s, '
        '%r. Stated by the contract and enforced at dispatch, not read back out of the storage '
        'declaration.'%(list(tg['required_size']),abi.get('launch')))
    return result


def emit(abi,bindings,ledger):
    # VALIDATE FIRST, FOR BOTH CLASSES. The three-binding route used to run BEFORE `validate` and
    # so skipped every shared invariant; it was also unreachable from `g17authorobj.author`, which
    # calls `validate` itself and refused these contracts with the four-binding class's message
    # before `emit` ran at all. `validate` now dispatches on the same signature this does, so the
    # gate and the emitter cannot disagree about which class a contract selects.
    validate(abi,bindings)
    if _class_of(bindings)=='three':
        return _emit_three(abi,bindings,ledger,_three_binding_write(bindings))
    pool=_pool_bytes(abi)
    layout=_layout_for(pool,abi)
    result=M.build([b[0] for b in bindings],layout=layout,offsets=[b[1] for b in bindings],
        register_count=abi['register_count'],system_registers=abi['system_registers'])
    result=_place_pool(result,pool,abi.get('pk_slot1',12))
    ledger['metadata class']='MEASURED: four indexed user buffers, 128-byte static threadgroup storage, SR160/SR161, empty external constant pool'
    ledger['metadata serializer']='g17mdgen.build; 516-byte register-chain witness reproduced from measured layout'
    ledger['per-kernel slot 0']='COMPILER INPUT: register_count, never the witness allocation'
    ledger['per-kernel slots 18 and 28']='COMPILER INPUT: uses_threadgroup and static_memory_bytes, checked against the measured class'
    ledger['per-kernel slot 32']='DERIVED from captured compiler instruction count and op458 presence; long straight-line class = 2'
    ledger['per-kernel slot 13']=('COMPILER INPUT: the constant pool is compile-time data - in '
        'coop4-1c every one of its 64 half values is a literal of its own source. %d byte(s) '
        'supplied; an empty pool keeps the measured 16-byte zero vector'%len(pool or b''))
    ledger['root table']=('DERIVED from slot 28: a section declaring static threadgroup memory '
        'carries the 14-byte root with slot 3 = 20, and one that does not carries 12 and 16 - '
        '692 of 692 against 4,999 of 4,999 over 5,691 distinct census sections')
    ledger['slot-2 kind-6 field2']=('DERIVED: the constant-pool length over four with field4 '
        'elided, the split closed at 24,975 of 24,975 across two corpora')
    if pool:
        ledger['per-kernel slot 1']=('CONTRACT INPUT: not derivable from the pool length - '
            '8 + pool/4 fits both cooperative points and 154 of 5,685 census sections')
    ledger['constant-program record']='MEASURED PRESENT: descriptor retained; actual prologue is compiler-declared END/filler'
    return result


def _place_pool(section,pool,slot1):
    """Write the supplied pool and the stated slot 1 into the built section.

    LOCATED IN THE BUILT BYTES, not at the one-register layout's offset: a second system register
    adds an entry to the slot-29 vector and moves every structure after it, so writing the pool at
    a fixed 288 silently corrupts the two-register form - which is exactly how the first attempt
    at this failed, on the two out-of-sample witnesses and nowhere else.
    """
    out=bytearray(section)
    pk=GM.kernel_table(bytes(out))
    slots,_=GM.table_at(bytes(out),pk)
    struct.pack_into("<I",out,pk+slots[1],slot1)
    at=pk+slots[13]
    vector=at+struct.unpack_from("<I",bytes(out),at)[0]
    if vector+4+len(pool)>len(out):
        raise ValueError('the constant pool does not fit this measured class: %d byte(s) at %d in '
                         'a %d-byte section'%(len(pool),vector,len(out)))
    struct.pack_into("<I",out,vector,len(pool))
    out[vector+4:vector+4+len(pool)]=pool
    return bytes(out)


# The witnesses this class is reproduced against, with the contract inputs each one supplies.
# register_count, pk_slot1 and the system registers are read from the compiled program in the
# real flow; here they are the retained contract for a witness, and they are INPUTS either way.
RETAINED = {
    "coop4-1c": {"register_count": 33, "pk_slot1": 40, "system_registers": [160],
                 "instruction_count": 400, "has_back_edge": False},
    "coop4-n185": {"register_count": 34, "pk_slot1": 40, "system_registers": [160, 161],
                   "instruction_count": 400, "has_back_edge": False},
    "coop4-flat-a": {"register_count": 36, "pk_slot1": 40, "system_registers": [160, 161],
                     "instruction_count": 400, "has_back_edge": False},
    "coop4-flat-b": {"register_count": 36, "pk_slot1": 12, "system_registers": [160, 161],
                     "instruction_count": 200, "has_back_edge": False},
}


def _contract(name,inputs,pool):
    """The delivered-contract shape this author accepts, for one retained witness."""
    return {"abi_version": 4, "uses_threadgroup": True, "has_stores": True, "writes_buffer": True,
            "writes_texture": False, "arch_flag": False, "entry": 64,
            "prologue": (bytes.fromhex('0e000000')+bytes.fromhex('0600')*30).hex(),
            "constant_pool": pool.hex(), "pk_slot1": inputs["pk_slot1"],
            "register_count": inputs["register_count"],
            "system_registers": inputs["system_registers"],
            "instruction_count": inputs["instruction_count"],
            "has_back_edge": inputs["has_back_edge"],
            "pk_values": {15: 1, 16: 1, 18: 1}, "pk_extra": [15, 16, 18],
            "threadgroup": {"required_size": [32, 1, 1], "static_memory_bytes": 128,
                            "static_memory_alignment": 4, "dynamic_memory": []}}


def _witness_pool(section):
    """The pool a witness carries, read from its own slot 13 - the input a contract must supply."""
    pk=GM.kernel_table(section)
    slots,_=GM.table_at(section,pk)
    at=pk+slots[13]
    vector=at+struct.unpack_from("<I",section,at)[0]
    count=struct.unpack_from("<I",section,vector)[0]
    return section[vector+4:vector+4+count]


def build_delivery(destination):
    """Retain each witness's section, its contract inputs, and the build's own verdict."""
    os.makedirs(destination,exist_ok=True)
    members=[]
    for name,inputs in sorted(RETAINED.items()):
        path=os.path.join(WITNESSES,name+".metadata.bin")
        section=open(path,"rb").read()
        pool=_witness_pool(section)
        contract=_contract(name,inputs,pool)
        bindings=[(1,0,False),(2,2,False),(3,4,False),(4,6,True)]
        ledger={}
        built=emit(contract,bindings,ledger)
        differing=[i for i in range(min(len(built),len(section))) if built[i]!=section[i]]
        differing+=list(range(min(len(built),len(section)),max(len(built),len(section))))
        open(os.path.join(destination,name+".metadata.bin"),"wb").write(section)
        with open(os.path.join(destination,name+".contract.json"),"w") as handle:
            json.dump(contract,handle,indent=1,sort_keys=True)
        members.append({"name":name,"section_bytes":len(section),
                        "section_sha256":hashlib.sha256(section).hexdigest(),
                        "built_bytes":len(built),
                        "built_sha256":hashlib.sha256(built).hexdigest(),
                        "differing_bytes":len(differing),
                        "pool_bytes":len(pool),
                        "ledger":{k:v for k,v in ledger.items()}})
    document={
        "status":"the_pool_carrying_cooperative_class_reproduces_from_its_contract",
        "gpu_dispatched":False,
        "wrong_builds":sum(1 for m in members if m["differing_bytes"]),
        "derived":{"root table":"14 bytes with slot 3 = 20 when slot 28 is declared, 12 and 16 "
                                "when it is not - 692 of 692 against 4,999 of 4,999 over 5,691 "
                                "distinct census sections",
                   "slot-2 kind-6 field2":"the constant-pool length over four with field4 elided"},
        "contract_inputs":list(CONTRACT_INPUTS),
        "refuted":"slot 1 = 8 + pool/4 fits both cooperative points (12 at 16, 40 at 128) and "
                  "holds on 154 of 5,685 census sections. Two points and a line; slot 1 stays an "
                  "input and this author refuses a non-empty pool without it.",
        "slot13_is_compile_time_data":"decoded as halves, all 64 values of coop4-1c's vector are "
                                      "literals of its own source's fma chain (0.0, 16.5, 17.5, "
                                      "...). It is supplied, never derived and never donated.",
        "still_not_derivable":"slot 32 is program-dependent (instruction count and back edge), and "
                              "slot 1, slot 0 and the system-register set are contract inputs.",
        "members":members,
    }
    with open(os.path.join(destination,"cooperative.json"),"w") as handle:
        json.dump(document,handle,indent=1,sort_keys=True)
    return document


def _from_json(contract):
    """JSON turns integer mapping keys into strings; the author compares them as integers."""
    out=dict(contract)
    if isinstance(out.get("pk_values"),dict):
        out["pk_values"]={int(k):v for k,v in out["pk_values"].items()}
    return out


def check(destination):
    """Rebuild every retained witness from its retained contract and refuse anything that differs."""
    document=json.load(open(os.path.join(destination,"cooperative.json")))
    findings=[]
    bindings=[(1,0,False),(2,2,False),(3,4,False),(4,6,True)]
    for member in document["members"]:
        name=member["name"]
        section_path=os.path.join(destination,name+".metadata.bin")
        contract_path=os.path.join(destination,name+".contract.json")
        if not (os.path.exists(section_path) and os.path.exists(contract_path)):
            findings.append("%s: the retained section or contract is missing"%name)
            continue
        section=open(section_path,"rb").read()
        actual=hashlib.sha256(section).hexdigest()
        if actual!=member["section_sha256"]:
            findings.append("%s: the retained section hashes to %s"%(name,actual[:12]))
            continue
        contract=_from_json(json.load(open(contract_path)))
        try:
            built=emit(dict(contract),bindings,{})
        except ValueError as error:
            findings.append("%s: the retained contract no longer authors - %s"%(name,error))
            continue
        if len(built)!=len(section):
            findings.append("%s: built %d bytes against %d"%(name,len(built),len(section)))
            continue
        differing=[i for i in range(len(section)) if built[i]!=section[i]]
        if differing:
            findings.append("%s: %d byte(s) differ from the witness, first at %d"
                            %(name,len(differing),differing[0]))
        if member["differing_bytes"]:
            findings.append("%s: the retained report admits %d differing byte(s); a partially "
                            "reproduced class must not be reported as built"
                            %(name,member["differing_bytes"]))
        # THE POOL IS AN INPUT, and a contract without it must refuse rather than inherit one.
        without=dict(contract); without.pop("constant_pool")
        try:
            emit(without,bindings,{})
            findings.append("%s: a contract with no constant_pool authored anyway"%name)
        except ValueError:
            pass
        # SCOPED TO WHERE IT IS REQUIRED. The 16-byte form derives slot 1 as 12 from its own
        # measured class, so dropping the statement there is not an error; only a pool length with
        # no such derivation must refuse.
        if member["pool_bytes"]!=16:
            unstated=dict(contract); unstated.pop("pk_slot1")
            try:
                emit(unstated,bindings,{})
                findings.append("%s: a non-empty pool authored without a stated pk_slot1"%name)
            except ValueError:
                pass
    if document.get("wrong_builds"):
        findings.append("the retained report records %d wrong build(s)"%document["wrong_builds"])
    if not document["members"]:
        findings.append("no witness is retained, so nothing here is reproduced")
    return findings


def main(argv):
    if len(argv)!=3 or argv[1] not in ("build","check"):
        print("usage: g17cooperativemetadata.py build|check <delivery>")
        return 2
    if argv[1]=="build":
        document=build_delivery(argv[2])
        for member in document["members"]:
            print("%-14s %4d bytes  pool %-4d  differing %d"
                  %(member["name"],member["section_bytes"],member["pool_bytes"],
                    member["differing_bytes"]))
        print("wrong builds: %d"%document["wrong_builds"])
        return 0
    findings=check(argv[2])
    for finding in findings:
        print("FINDING: %s"%finding)
    document=json.load(open(os.path.join(argv[2],"cooperative.json")))
    print("%d witness(es) rebuilt from their contracts; %d finding(s)"
          %(len(document["members"]),len(findings)))
    return 1 if findings else 0


if __name__=="__main__":
    sys.exit(main(sys.argv))
