#!/usr/bin/env python3
"""ONE-STORE PROBE PROGRAMS for the execution oracle - NOT the compiler's ladder.

They lived in g17ladder first, and g17scorecard.programs() counts every public function there as
"the compiler's own corpus", so twelve probes silently grew two regression populations (emitted
opcodes on single-witness map fields, unsettled scalar-debt bits). A probe is not a rung. They are
resolved by g17oracle.ladder_program when g17ladder has no program of that name, which keeps the
dispatched records that name them (isa/g17-execution-onestore/-storereread/-storelifetime/-storekeep)
rebuilding.
"""
import os, sys
_T = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _T)
import g17ir as ir
# ONE-STORE PROGRAMS, the isolated records for a store form. A store has no destination, so the
# oracle's one-instruction scaffold cannot hold its result; the compiler's own store, alone in a
# program, can - its landing word is the observable, against the harness's 0xDEADBEEF fill. Each
# program contains exactly ONE instance of the store under test, at the length the compiler picks
# for that shape (measured: an ALU or constant value -> 8 bytes, a value straight from a load ->
# 10, a displacement outside eight bits -> 14), and g17oracle refuses a record naming an opcode the
# program does not contain exactly once at its declared length. Deliberately NOT in LADDER: these
# are probes for the execution oracle, not rungs of the compiler's progress metric.
STORE_WORD = 0x13572468

def _one_store(name):
    f = ir.Function(name, [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    return f, ir.Builder(f, f.block("entry"))

def store_word_8():
    """op17235 at 8 bytes: a constant word into C[20]."""
    f, b = _one_store("store_word_8")
    b.store(f.buffers[1], ir.Imm(20), b.const(STORE_WORD, name="v"), reserve_companion=False)
    b.ret(); return f

def store_word_14():
    """op17235 at 14 bytes: the same constant into C[70], a displacement outside eight bits."""
    f, b = _one_store("store_word_14")
    b.store(f.buffers[1], ir.Imm(70), b.const(STORE_WORD, name="v"), reserve_companion=False)
    b.ret(); return f

def store_word_10():
    """op17235 at 10 bytes: B[0] straight from a load into C[40] (the store carries the wait)."""
    f, b = _one_store("store_word_10")
    t = b.builtin("thread_position_in_grid", name="t")
    b.store(f.buffers[1], ir.Imm(40), b.load(f.buffers[0], t, name="v"), reserve_companion=False)
    b.ret(); return f

def store_half_8():
    """op17199 at 8 bytes: (B[0] + 1) as a half into half-element 40, the LOW half of C[20]."""
    f, b = _one_store("store_half_8")
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.load(f.buffers[0], t, name="v")
    b.store(f.buffers[1], ir.Imm(40), b.add(v, ir.Imm(1), name="h", type=ir.I16), width="half")
    b.ret(); return f

def store_half_14():
    """op17199 at 14 bytes: (B[0] + 3) as a half into half-element 200, the LOW half of C[100]."""
    f, b = _one_store("store_half_14")
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.load(f.buffers[0], t, name="v")
    b.store(f.buffers[1], ir.Imm(200), b.add(v, ir.Imm(3), name="h", type=ir.I16), width="half")
    b.ret(); return f

def store_half_10():
    """op17199 at 10 bytes: B's first half straight from a half load into half-element 41, the
    HIGH half of C[20]."""
    f, b = _one_store("store_half_10")
    t = b.builtin("thread_position_in_grid", name="t")
    h = b.load(f.buffers[0], t, name="hl", type=ir.I16, width="half")
    b.store(f.buffers[1], ir.Imm(41), h, width="half")
    b.ret(); return f


# THE SAME STORES WITH THE STORED VALUE READ AGAIN AFTERWARDS. This backend does not reorder, and
# the liveness pass writes the store's source lifetime as 16 - RELEASE - even here (decoded), where
# Apple never writes 32 on these forms and schedules the store last instead. So these programs ask
# the question mem.device.halfelement's spec calls unmeasured: does a value a store has released
# still read correctly afterwards? The observable is the second word, written by the default word
# store (op17244), so the program still holds exactly ONE instance of the store under test.

def reread_word_8():
    f, b = _one_store("reread_word_8")
    v = b.const(STORE_WORD, name="v")
    b.store(f.buffers[1], ir.Imm(20), v, reserve_companion=False)
    b.store(f.buffers[1], ir.Imm(30), b.add(v, ir.Imm(1), name="w"))
    b.ret(); return f

def reread_word_14():
    f, b = _one_store("reread_word_14")
    v = b.const(STORE_WORD, name="v")
    b.store(f.buffers[1], ir.Imm(70), v, reserve_companion=False)
    b.store(f.buffers[1], ir.Imm(80), b.add(v, ir.Imm(1), name="w"))
    b.ret(); return f

def reread_word_10():
    f, b = _one_store("reread_word_10")
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.load(f.buffers[0], t, name="v")
    b.store(f.buffers[1], ir.Imm(40), v, reserve_companion=False)
    b.store(f.buffers[1], ir.Imm(50), b.add(v, ir.Imm(1), name="w"))
    b.ret(); return f

def _reread_half(name, slot, from_load):
    f, b = _one_store(name)
    t = b.builtin("thread_position_in_grid", name="t")
    if from_load:
        h = b.load(f.buffers[0], t, name="h", type=ir.I16, width="half")
    else:
        h = b.add(b.load(f.buffers[0], t, name="v"), ir.Imm(1), name="h", type=ir.I16)
    b.store(f.buffers[1], ir.Imm(slot), h, width="half")
    b.store(f.buffers[1], ir.Imm(slot // 2 + 10), b.add(h, ir.Imm(1), name="w", type=ir.I32))
    b.ret(); return f

def reread_half_8():
    return _reread_half("reread_half_8", 40, False)

def reread_half_14():
    return _reread_half("reread_half_14", 200, False)

def reread_half_10():
    return _reread_half("reread_half_10", 41, True)



# A THREADGROUP ROUND TRIP WITH NO BARRIER. A threadgroup store cannot be observed from the host
# except through a threadgroup load, so the smallest observable for either is the pair. The
# harness launches ONE thread, which reads back its own write; the ladder's
# threadgroup_constant_roundtrip carries an op447 barrier, which the loads/stores authorization
# does not cover, and this program has none. It emits exactly one op13288 (10 bytes) and one
# op12364 (14 bytes) - the two forms compiler.gap.mem.threadgroup names.
TG_WORD = 0x5A5A1234

def tg_roundtrip_nobarrier():
    f = ir.Function("tg_roundtrip_nobarrier", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    f.declare_threadgroup(32, size=(32, 1, 1), alignment=4)
    b = ir.Builder(f, f.block("entry"))
    z = b.const(0, name="z")
    b.store_tg(b.const(TG_WORD, name="k"), z)
    b.store(f.buffers[1], ir.Imm(20), b.load_tg(z, name="r"))
    b.ret(); return f


# THE REMAINING DEVICE STORE AND LOAD FORMS, one per program. compiler.gap.mem.device.store and
# .load name them; each is the compiler's own lowering of the shape shown, measured to select it.
RANGE_BASE = 0x11110000

def _range(name, n, slot):
    f, b = _one_store(name)
    vals = [b.const(RANGE_BASE + j, name="v%d" % j) for j in range(n)]
    b.store_range(f.buffers[1], ir.Imm(slot), vals)
    b.ret(); return f

def range2_8():  return _range("range2_8", 2, 40)    # op17244 at 8 bytes
def range2_14(): return _range("range2_14", 2, 90)   # op17244 at 14 bytes
def range3_8():  return _range("range3_8", 3, 40)    # op17253 at 8
def range3_14(): return _range("range3_14", 3, 90)   # op17253 at 14
def range4_8():  return _range("range4_8", 4, 40)    # op17262 at 8
def range4_14(): return _range("range4_14", 4, 90)   # op17262 at 14

def indexed_word():
    """op17229 at 8 bytes: a constant stored at the lane's own index, C[t] (t = 0, one thread)."""
    f, b = _one_store("indexed_word")
    t = b.builtin("thread_position_in_grid", name="t")
    b.store_at(f.buffers[1], t, b.const(STORE_WORD, name="v"))
    b.ret(); return f

def indexed_half():
    """op17193 at 14 bytes: (B[0] + 1) as a half at the lane's own half index - the low half of C[0]."""
    f = ir.Function("indexed_half", [ir.Buffer("B", 1), ir.Buffer("C", 2, elem=ir.F16)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.load(f.buffers[0], t, name="v")
    b.store_at(f.buffers[1], t, b.add(v, ir.Imm(1), name="h", type=ir.I16), width="half")
    b.ret(); return f

def load_word_8():
    """op12682 at 8 bytes (form_length=8): B[0] into C[40]."""
    f, b = _one_store("load_word_8")
    t = b.builtin("thread_position_in_grid", name="t")
    b.store(f.buffers[1], ir.Imm(40), b.load(f.buffers[0], t, form_length=8, name="v"),
            reserve_companion=False)
    b.ret(); return f

def load_half_10():
    """op12646 at 10 bytes (form_length=10): B's first half, converted f16 -> f32, into C[0]."""
    f = ir.Function("load_half_10", [ir.Buffer("B", 1, elem=ir.F16), ir.Buffer("C", 2, elem=ir.I32)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.load(f.buffers[0], t, type=ir.I16, width="half", form_length=10, name="v")
    b.store_at(f.buffers[1], t, b.f16_to_f32(v, name="z"), width="word")
    b.ret(); return f


# THE HALF-VECTOR RANGE STORES, one per program: op17226 (four 16-bit members), op17217 (three),
# op17208 (two), at 14 bytes - the forms compiler.gap.mem.device.halfvector names. The same member
# values as g17endtoend._halfvec_ir, (B[k] + k + 1) & 0xffff; the three-member store writes six
# bytes, so its second word's HIGH half must keep the harness fill.
def _halfvec(name, n, slot):
    f, b = _one_store(name)
    t = b.builtin("thread_position_in_grid", name="t")
    loads = [b.load(f.buffers[0], t, offset=k, name="b%d" % k) for k in range(n)]
    b.store_range(f.buffers[1], ir.Imm(slot),
                  [b.add(loads[k], ir.Imm(k + 1), name="q%d" % k, type=ir.I16) for k in range(n)],
                  width="half")
    b.ret(); return f

def halfvec4(): return _halfvec("halfvec4", 4, 14)   # op17226/14
def halfvec3(): return _halfvec("halfvec3", 3, 20)   # op17217/14
def halfvec2(): return _halfvec("halfvec2", 2, 24)   # op17208/14


# THE REGISTER COPY compiler.gap.reg.move names at 32 bits, op586 at 4 bytes. The compiler emits it
# only when a range store names one value twice: the second member needs its own register, since
# the range form writes consecutive registers. One copy, feeding C[41]; C[40] is the original.
def move_word():
    f, b = _one_store("move_word")
    v = b.const(STORE_WORD, name="v")
    b.store_range(f.buffers[1], ir.Imm(40), [v, v])
    b.ret(); return f


# CONTROL-FLOW AND BARRIER PROBES, each holding its form exactly once as written, on the same
# B (slot 1) / C (slot 2) binding as the store probes above.
def tg_roundtrip_barrier():
    """op447 at 6 bytes between a threadgroup store and load, one thread: C[20] = 0x5A5A1234."""
    f = ir.Function("tg_roundtrip_barrier", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    f.declare_threadgroup(32, size=(32, 1, 1), alignment=4)
    b = ir.Builder(f, f.block("entry"))
    z = b.const(0, name="z")
    b.store_tg(b.const(TG_WORD, name="k"), z)
    b.barrier()
    b.store(f.buffers[1], ir.Imm(20), b.load_tg(z, name="r"))
    b.ret(); return f

def cond_not_taken():
    """op577/op582 exec mask around a store the one thread skips (t = 0 is not > 8): C[20] keeps
    the fill, and C[21] - stored after the join - is written."""
    f = ir.Function("cond_not_taken", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    e = f.block("entry"); th = f.block("then"); jn = f.block("join")
    b = ir.Builder(f, e)
    t = b.builtin("thread_position_in_grid", name="t")
    b.br_cond(b.cmp(t, 8, "gt", name="p"), th, jn)
    b.at(th); b.store(f.buffers[1], ir.Imm(20), b.const(STORE_WORD, name="v"), reserve_companion=False); b.br(jn)
    b.at(jn); b.store(f.buffers[1], ir.Imm(21), b.const(RANGE_BASE, name="w"), reserve_companion=False)
    b.ret(); return f

def counted_loop4():
    """op458 at 10 bytes: a back edge whose trip count the compiler proves (i = 0 .. 3), then C[20] = 4."""
    f = ir.Function("counted_loop4", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    pre = f.block("pre"); hdr = f.block("header"); ex = f.block("exit")
    b = ir.Builder(f, pre)
    zero = b.const(0, name="zero")
    b.br(hdr)
    b.at(hdr)
    i = b.phi(zero, name="i")
    nxt = b.add(i, ir.Imm(1), name="i_next")
    ir.Builder.phi_latch(i, nxt)
    b.br_cond(b.cmp(nxt, 4, "lt", name="p"), hdr, ex)
    b.at(ex); b.store(f.buffers[1], ir.Imm(20), nxt, reserve_companion=False); b.ret()
    return f


# REQUESTED-LENGTH FORMS on CONSTANT sources, so no source waits on a load (op3290/4 read before
# its loads arrived when fed from them directly). Values are float32 bit patterns.
F1_5, F2_25, F0_5 = 0x3FC00000, 0x40100000, 0x3F000000

def movimm2():
    """op11842 at 2 bytes (length=2): the seven-bit immediate 0x55 into C[20]."""
    f, b = _one_store("movimm2")
    v = b._def("const", [ir.Imm(0x55)], type=ir.I32, name="v", length=2)
    b.store(f.buffers[1], ir.Imm(20), v, reserve_companion=False)
    b.ret(); return f

def _fconst_op(name, kind, length, nsrc):
    f, b = _one_store(name)
    srcs = [b.const(bits, type=ir.F32, name="c%d" % i) for i, bits in enumerate((F1_5, F2_25, F0_5)[:nsrc])]
    v = b._def(kind, srcs, type=ir.F32, name="v", length=length)
    b.store(f.buffers[1], ir.Imm(20), v, reserve_companion=False)
    b.ret(); return f

def fadd6(): return _fconst_op("fadd6", "fadd", 6, 2)     # op998/6: 1.5 + 2.25
def fadd4(): return _fconst_op("fadd4", "fadd", 4, 2)     # op998/4
def ffma6(): return _fconst_op("ffma6", "fma", 6, 3)      # op2190/6: 1.5 * 2.25 + 0.5
def ffma4(): return _fconst_op("ffma4", "fma", 4, 3)      # op2190/4
