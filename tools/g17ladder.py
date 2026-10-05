#!/usr/bin/env python3
"""THE COMPILER TEST LADDER - Track A's progress metric.

The mission says to drive the compiler with execution tests rather than with the Apple corpus,
because corpus completeness measures the DESCRIPTION and this measures the CAPABILITY. Each rung
is a kernel written in the IR. A rung reports one of:

    COMPILES     selection, allocation and emission all succeeded and the bytes decode back
    BLOCKED      an honest Unsupported: the IR is fine, the backend cannot lower it yet, and the
                 message names the missing ISA semantics - which is how the compiler tells the
                 ISA work what to recover next, rather than the other way round
    BROKEN       something raised that should not have

BLOCKED is the useful state. It converts "what should I reverse engineer?" from a corpus-frequency
question into a compiler-need question, which is exactly the reversal the mission asks for.

Execution is a SEPARATE step and is not run from here: these programs are instruction streams, and
splicing one into a container and dispatching it is a GPU operation with its own risks.
"""
import os, sys, traceback
_T = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _T)
import g17ir as ir, g17cc

def vector_add():
    f = ir.Function("vector_add", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="tid")
    x = b.load(f.buffers[0], t, name="x"); y = b.load(f.buffers[1], t, name="y")
    b.store(f.buffers[2], ir.Imm(40), b.add(x, y, name="s")); b.ret()
    return f

def affine_index():
    """Address arithmetic authored from semantics: t*3 + 7, then a scaled shift-add."""
    f = ir.Function("affine_index", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    k = b.const(7, name="k")
    n = b.add(b.mul(t, ir.Imm(3), name="m"), k, name="n")
    v = b.load(f.buffers[0], b.shiftadd(t, 4, n, name="sa"), name="v")
    b.store(f.buffers[1], ir.Imm(8), b.sub(v, ir.Imm(1), name="w")); b.ret()
    return f

def tiled_pointer():
    """Tiled pointer arithmetic: row = tid, col offset by a scaled stride, two loads combined."""
    f = ir.Function("tiled_pointer", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    g = b.builtin("threadgroup_position_in_grid", name="g")
    base = b.shiftadd(g, 8, t, name="base")
    a = b.load(f.buffers[0], base, name="a")
    c = b.load(f.buffers[1], b.add(base, ir.Imm(1), name="base1"), name="c")
    b.store(f.buffers[2], ir.Imm(16), b.add(a, c, name="sum")); b.ret()
    return f

def wide_constant():
    """A 32-bit constant materialised into a register - mov.imm.wide, all 32 bits authored."""
    f = ir.Function("wide_constant", [ir.Buffer("C", 0)])
    b = ir.Builder(f, f.block("entry"))
    b.store(f.buffers[0], ir.Imm(4), b.const(0xDEADBEEF, name="k")); b.ret()
    return f

def bitwise():
    """EXPECTED TO BLOCK, with a sharper reason since 2026-09-04. The size IS now established -
    8 bytes, causally - and the immediate is recovered. What is missing is the OPERAND SELECTOR:
    six source expressions and two destinations leave the eight bytes unchanged, and the varying
    2-byte instruction before it is unread. Without it no register can be allocated.
    ledger/g17-bitwise-size-resolved.toml"""
    f = ir.Function("bitwise", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    x = b.load(f.buffers[0], t, name="x")
    b.store(f.buffers[1], ir.Imm(4), b.__getattribute__("and")(x, ir.Imm(15), name="m")); b.ret()
    return f

def conditional():
    """Single-level if-then with explicit reconvergence. The relation must be the template's own
    'gt': cmp.pair.imm's relation encoding is unresolved, so the compiler can author the
    IMMEDIATE and the branch DISPLACEMENT but not which comparison is performed."""
    f = ir.Function("conditional", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    e = f.block("entry"); th = f.block("then"); jn = f.block("join")
    b = ir.Builder(f, e)
    t = b.builtin("thread_position_in_grid", name="t")
    b.br_cond(b.cmp(t, 8, "gt", name="p"), th, jn)
    b.at(th); b.store(f.buffers[1], ir.Imm(4), b.load(f.buffers[0], t, name="x")); b.br(jn)
    b.at(jn); b.ret()
    return f

def conditional_lt():
    """EXPECTED TO BLOCK, and documents WHY rather than leaving the gap implicit: the same kernel
    with '<' cannot be emitted, because the relation field is not recovered."""
    f = ir.Function("conditional_lt", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    e = f.block("entry"); th = f.block("then"); jn = f.block("join")
    b = ir.Builder(f, e)
    t = b.builtin("thread_position_in_grid", name="t")
    b.br_cond(b.cmp(t, 8, "lt", name="p"), th, jn)
    b.at(th); b.store(f.buffers[1], ir.Imm(4), b.load(f.buffers[0], t, name="x")); b.br(jn)
    b.at(jn); b.ret()
    return f

def nested_conditional():
    """EXPECTED TO BLOCK: nesting needs a reconvergence stack, the next CFG rung."""
    f = ir.Function("nested", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    bs = [f.block(n) for n in ("entry", "t1", "t2", "j2", "j1")]
    b = ir.Builder(f, bs[0])
    t = b.builtin("thread_position_in_grid", name="t")
    b.br_cond(b.cmp(t, 8, "gt", name="p"), bs[1], bs[4])
    b.at(bs[1]); b.br_cond(b.cmp(t, 4, "gt", name="q"), bs[2], bs[3])
    b.at(bs[2]); b.store(f.buffers[1], ir.Imm(4), b.load(f.buffers[0], t, name="x")); b.br(bs[3])
    b.at(bs[3]); b.br(bs[4])
    b.at(bs[4]); b.ret()
    return f

def tensor_matmul():
    """A tensor matmul at a shape with a reference sequence. The compiler authors the dtypes,
    the k_slice and enable bits, and the N extent; the mac COUNT and ORDER are inherited."""
    f = ir.Function("tensor_matmul", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2], M=32, N=32, K=64, sequence_experiment=True,
                    a_dtype="half", b_dtype="half")
    b.ret()
    return f

def mixed_scalar_tensor():
    """Scalar address arithmetic and a tensor matmul in one kernel, from one IR."""
    f = ir.Function("mixed", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    b.store(f.buffers[2], ir.Imm(64), b.add(t, ir.Imm(5), name="idx"))
    b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2], M=16, N=16, K=64, sequence_experiment=True,
                    a_dtype="bfloat", b_dtype="half")
    b.ret()
    return f

def tensor_unseen_shape():
    """EXPECTED TO BLOCK. 48x48x64 has no reference sequence, and the tensor ISA states that
    composing a novel instruction sequence is unsolved - so the compiler must refuse rather than
    extrapolate a mac count from the shape."""
    f = ir.Function("tensor_unseen", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    b.tensor_matmul(f.buffers[0], f.buffers[1], f.buffers[2], M=48, N=48, K=64, sequence_experiment=True)
    b.ret()
    return f

def reused_operand():
    """Exposes the lifetime bug: y is operand B of the first add AND is used again afterwards.

    With the old defaults every ALU released operand B after reading it, so y would be read
    correctly by the first add and then be gone for the second - the precise failure that led to
    byte8[5] being recovered. The compiler must now emit keep for y and release for values that
    really are dead.
    """
    f = ir.Function("reused_operand", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    x = b.load(f.buffers[0], t, name="x")
    y = b.load(f.buffers[0], t, name="y")
    s1 = b.add(x, y, name="s1")          # y is operand B here ...
    s2 = b.add(s1, y, name="s2")         # ... and still needed here
    b.store(f.buffers[1], ir.Imm(4), s2); b.ret()
    return f

def range_store():
    """A REGISTER-RANGE store: four values to four consecutive slots, which the ISA encodes as
    one instruction writing r<src>..r<src+3>. Exercises store.14's n field and its high slot bits,
    both of which the rest of the ladder leaves at one value
    (ledger/g17-track-a-null-control.toml)."""
    f = ir.Function("range_store", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    vs = [b.add(t, ir.Imm(i + 1), name="v%d" % i) for i in range(4)]
    b.store_range(f.buffers[1], ir.Imm(96), vs); b.ret()
    return f

def store_widths():
    """Stores of 1, 2 and 3 registers at slots spanning the 6-bit boundary, so the n field and
    byte13 both take several values."""
    f = ir.Function("store_widths", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    a = b.add(t, ir.Imm(1), name="a"); c = b.add(t, ir.Imm(2), name="c")
    d = b.add(t, ir.Imm(3), name="d")
    b.store_range(f.buffers[1], ir.Imm(8), [a, c])
    b.store_range(f.buffers[1], ir.Imm(130), [c, d])
    b.store(f.buffers[1], ir.Imm(200), t)
    b.ret()
    return f

def multi_buffer():
    """Loads from three different buffers with different element offsets, so load.14's base and
    offset fields vary instead of sitting at one value."""
    f = ir.Function("multi_buffer", [ir.Buffer("A", 0), ir.Buffer("B", 1),
                                     ir.Buffer("C", 2), ir.Buffer("D", 3)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    x = b.load(f.buffers[0], t, name="x")
    y = b.load(f.buffers[1], t, name="y")
    z = b.load(f.buffers[3], t, name="z")
    b.store_range(f.buffers[2], ir.Imm(12), [b.add(x, y, name="s"), b.add(z, ir.Imm(1), name="u")])
    b.ret()
    return f

def addressing_modes():
    """Every recovered load addressing mode in one kernel: element offset, the second
    displacement, a scaled index, a byte-width load and a shift-16 load. All five are causal in
    the ISA and were unreachable from the IR until now, which is why load.14 exercised 5 of its
    38 owned bits."""
    f = ir.Function("addressing_modes", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    a = b.load(f.buffers[0], t, name="a", offset=5)
    c = b.load(f.buffers[0], t, name="c", disp=2)
    d = b.load(f.buffers[0], t, name="d", scale=2)
    e = b.load(f.buffers[0], t, name="e", width="byte")
    g = b.load(f.buffers[0], t, name="g", shift16=True, offset=130)
    b.store_range(f.buffers[1], ir.Imm(20),
                  [b.add(a, c, name="p"), b.add(d, e, name="q")])
    b.store(f.buffers[1], ir.Imm(24), g)
    b.ret()
    return f

def counted_loop():
    """A structured loop: body, then a compare and a BACK EDGE to the header.

        header:  ...body...
                 cmp t > K
                 br header          <- back edge, branch.cond.back
        exit:    store; ret

    SAFETY: this is the first construct the compiler emits that can fail to TERMINATE. Every
    branch before it was forward-only, and "a forward branch only moves the PC forward" was the
    argument that made authored control flow safe to dispatch. That argument does not survive a
    back edge.
    """
    f = ir.Function("counted_loop", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    pre = f.block("pre"); hdr = f.block("header"); ex = f.block("exit")
    b = ir.Builder(f, pre)
    # A REAL INDUCTION VARIABLE. The previous version computed acc = t + 1 inside the header,
    # which recomputes the same value every iteration - it has no loop-carried dependence at all,
    # so the comparison never changes state and the only two outcomes are "runs once" and "runs
    # forever". That was a defect in the rung, not in the branch: no back-edge semantics could
    # have made it terminate. A peer session then checked every loop in Apple's corpus and found
    # a self-referencing definition in 515 of 515 - the machine expresses a loop-carried value as
    # a register that is both source and destination of one instruction, which is what a phi
    # coalesces to.
    zero = b.const(0, name="zero")
    b.br(hdr)
    b.at(hdr)
    i = b.phi(zero, name="i")
    nxt = b.add(i, ir.Imm(1), name="i_next")
    ir.Builder.phi_latch(i, nxt)
    b.br_cond(b.cmp(nxt, 4, "lt", name="p"), hdr, ex)
    b.at(ex); b.store(f.buffers[1], ir.Imm(4), nxt); b.ret()
    return f

def with_barrier():
    """A barrier between two phases. All 225 barriers in Apple's corpus are byte-identical, so
    this form is emitted as a constant with no template and no unresolved bits - the only form in
    the compiler of which that is true."""
    f = ir.Function("with_barrier", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    b.store(f.buffers[1], ir.Imm(4), b.add(t, ir.Imm(1), name="p"))
    b.barrier()
    b.store(f.buffers[1], ir.Imm(8), b.load(f.buffers[0], t, name="x"))
    b.ret()
    return f

def wide_registers():
    """A 24-value accumulate chain. Impossible before the allocator learned the per-form field
    widths and started commuting commutative operands: every value sat in operand B, a 4-bit
    field, so all of them were confined to r0..r15 and the allocator ran out at twelve.

    Now the accumulator goes to operand B and the values reach r16..r31, which also exercises the
    ALU destination's high bits - byte0[7] and byte7[5] - that no other rung reaches.
    """
    f = ir.Function("wide_registers", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    vs = [b.add(t, ir.Imm(i + 1), name="v%d" % i) for i in range(24)]
    acc = vs[0]
    for v in vs[1:]: acc = b.add(acc, v, name="acc")
    b.store(f.buffers[1], ir.Imm(4), acc); b.ret()
    return f

LADDER = [vector_add, affine_index, tiled_pointer, wide_constant, bitwise, reused_operand,
          wide_registers,
          with_barrier,
          counted_loop,
          range_store, store_widths, multi_buffer, addressing_modes,
          conditional, conditional_lt, nested_conditional,
          tensor_matmul, mixed_scalar_tensor, tensor_unseen_shape]

# --- WIDENING THE PROGRAM SET ------------------------------------------------------------------
#
# The nineteen rungs above are a hand-picked sample, and every defect the compiler's scorecard
# found was invisible until the population changed: store.14's template could not move its source,
# `narrow` and `hi16` were written on a form that has neither, decode_movimm read four destination
# bits where the encoder wrote eight. Chasing the last rung on nineteen programs optimises the
# sample. These exercise what the sample does not - the IR builder's csel, icmp, fma, madd, faddi,
# shiftadd and store_at, threadgroup memory, and enough simultaneously live values to make the
# allocator work.

def atomic_counter():
    """An ATOMIC read-modify-write that returns the old value - the first one this compiler emits.

    o[t] = atomic_fetch_add(&a[t], t*3+1). The addend is COMPUTED rather than a literal: a literal
    sets operand 1's 2^24 bit, which the operand map cannot yet tell from 2^25, so selection refuses
    it. See ledger/g17-operand-1-has-three-bits-nobody-mapped.toml.

    Lowered through g17as from the operand maps rather than from a modal template, so no bit in this
    instruction is inherited from a witness.
    """
    f = ir.Function("atomic_counter", [ir.Buffer("A", 0), ir.Buffer("U", 1), ir.Buffer("O", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.add(b.mul(t, ir.Imm(3), name="m"), ir.Imm(1), name="v")
    old = b.atomic_add(f.buffers[0], t, v, name="old")
    b.store_at(f.buffers[2], t, old)
    b.ret()
    return f


def select_ternary():
    """csel: two candidate values and a comparison choosing between them."""
    f = ir.Function("select_ternary", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    x = b.add(t, ir.Imm(7), name="x"); y = b.add(t, ir.Imm(9), name="y")
    r = b.csel(x, y, x, y, rel="gt", name="r")
    b.store(f.buffers[2], ir.Imm(8), r); b.ret()
    return f


def integer_compare():
    """icmp between two computed values, stored as its own result."""
    f = ir.Function("integer_compare", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    x = b.add(t, ir.Imm(3), name="x"); y = b.mul(t, ir.Imm(2), name="y")
    p = b.icmp(x, y, rel="eq", name="p")
    b.store(f.buffers[1], ir.Imm(10), p); b.ret()
    return f


def fused_multiply_add():
    """fma and madd side by side - the float and integer fused forms."""
    f = ir.Function("fused_multiply_add", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    a = b.add(t, ir.Imm(1), name="a"); c = b.add(t, ir.Imm(2), name="c")
    m = b.madd(a, c, t, name="m")
    b.store(f.buffers[1], ir.Imm(12), m); b.ret()
    return f


def float_add_imm():
    """faddi: the float add-immediate form, which no other rung reaches."""
    f = ir.Function("float_add_imm", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.faddi(t, 1.5, ty="f32", name="v")
    b.store(f.buffers[1], ir.Imm(14), v); b.ret()
    return f


def shift_and_add():
    """shiftadd: base + (index << scale), the addressing primitive on its own."""
    f = ir.Function("shift_and_add", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    x = b.add(t, ir.Imm(5), name="x")
    r = b.shiftadd(x, 2, t, name="r")
    b.store(f.buffers[1], ir.Imm(16), r); b.ret()
    return f


def computed_store_index():
    """store_at: a store whose index is a REGISTER, so each thread writes its own element."""
    f = ir.Function("computed_store_index", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.add(t, ir.Imm(11), name="v")
    b.store_at(f.buffers[1], t, v); b.ret()
    return f


def register_pressure():
    """Twelve values live at once, then summed - enough to make the allocator choose."""
    f = ir.Function("register_pressure", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    vs = [b.add(t, ir.Imm(i + 1), name="v%d" % i) for i in range(12)]
    acc = vs[0]
    for i, v in enumerate(vs[1:]):
        acc = b.add(acc, v, name="s%d" % i)
    b.store(f.buffers[1], ir.Imm(18), acc); b.ret()
    return f


def threadgroup_roundtrip():
    """Threadgroup memory written and read back across a barrier."""
    f = ir.Function("threadgroup_roundtrip", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    f.declare_threadgroup(32, size=(32, 1, 1), alignment=4)   # ABI v4: the scratchpad is stated
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    v = b.add(t, ir.Imm(21), name="v")
    # THE INDEX IS A REGISTER, not a slot: the threadgroup forms take it in an operand the
    # allocator colours, so a constant index has to be materialised. Written the other way first,
    # and the allocator reported `use before def of 0` - which ir.verify now catches by name.
    z = b.const(0, name="z")
    b.store_tg(v, z)
    b.barrier()
    r = b.load_tg(z, name="r")
    b.store(f.buffers[1], ir.Imm(20), r); b.ret()
    return f


def two_conditionals():
    """Two independent conditionals in sequence, not nested - a shape nested_conditional misses."""
    f = ir.Function("two_conditionals", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    x = b.add(t, ir.Imm(1), name="x")
    p = b.cmp(x, 4, pred="lt", name="p")
    then1, join1 = f.block("t1"), f.block("j1")
    b.br_cond(p, then1, join1)
    b.at(then1); b.store(f.buffers[1], ir.Imm(22), x); b.br(join1)
    b.at(join1)
    y = b.add(t, ir.Imm(2), name="y")
    q = b.cmp(y, 9, pred="lt", name="q")
    then2, join2 = f.block("t2"), f.block("j2")
    b.br_cond(q, then2, join2)
    b.at(then2); b.store(f.buffers[1], ir.Imm(24), y); b.br(join2)
    b.at(join2); b.ret()
    return f


def many_buffers_mixed():
    """Four buffers, loads from three of them, one store - wider than multi_buffer."""
    f = ir.Function("many_buffers_mixed", [ir.Buffer("A", 0), ir.Buffer("B", 1),
                                           ir.Buffer("C", 2), ir.Buffer("D", 3)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    x = b.load(f.buffers[0], t, name="x")
    y = b.load(f.buffers[1], t, name="y")
    z = b.load(f.buffers[3], t, name="z")
    s = b.add(b.add(x, y, name="xy"), z, name="s")
    b.store(f.buffers[2], ir.Imm(26), s); b.ret()
    return f


LADDER += [atomic_counter, select_ternary, integer_compare, fused_multiply_add, float_add_imm, shift_and_add,
           computed_store_index, register_pressure, threadgroup_roundtrip, two_conditionals,
           many_buffers_mixed]


# --- THE OP FAMILIES NOTHING REACHES -----------------------------------------------------------
#
# The twenty-nine rungs above still touch 26 opcodes, and the IR builder offers thirteen unary
# operations and twenty arithmetic ones. Most of them have never been selected once. A rung that
# COMPILES here widens the certified set; a rung that comes back BLOCKED is the compiler naming
# what the ISA work should recover next, which is the reversal the mission asks for and the only
# reason to write a program the backend will refuse.

def unary_rounding():
    """floor, ceil, trunc and rint - four opcodes in one program, none of them ever selected."""
    f = ir.Function("unary_rounding", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    x = b.load(f.buffers[0], b.builtin("thread_position_in_grid", name="t"), name="x")
    for i, k in enumerate(("floor", "ceil", "trunc", "rint")):
        b.store(f.buffers[1], ir.Imm(8 + 2 * i), getattr(b, k)(x, name=k))
    b.ret()
    return f


def transcendentals():
    """recip, rsqrt, exp2, log2 - the one-source float unit."""
    f = ir.Function("transcendentals", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    x = b.load(f.buffers[0], b.builtin("thread_position_in_grid", name="t"), name="x")
    for i, k in enumerate(("recip", "rsqrt", "exp2", "log2")):
        b.store(f.buffers[1], ir.Imm(16 + 2 * i), getattr(b, k)(x, name=k))
    b.ret()
    return f


def bit_manipulation():
    """not, msb and reverse - integer unary, which nothing above selects either."""
    f = ir.Function("bit_manipulation", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    x = b.load(f.buffers[0], b.builtin("thread_position_in_grid", name="t"), name="x")
    for i, k in enumerate(("not", "msb", "reverse")):
        b.store(f.buffers[1], ir.Imm(24 + 2 * i), getattr(b, k)(x, name=k))
    b.ret()
    return f


def saturating_arith():
    """addsat and subsat - the saturating family, whose form takes an immediate shift amount."""
    f = ir.Function("saturating_arith", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    x = b.load(f.buffers[0], t, name="x")
    b.store(f.buffers[1], ir.Imm(30), b.addsat(x, t, name="as"))
    b.store(f.buffers[1], ir.Imm(32), b.subsat(x, t, name="ss"))
    b.ret()
    return f


def shift_family():
    """shl, shr and sar by an immediate, and sarv by a register - four opcodes, two operand modes."""
    f = ir.Function("shift_family", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    x = b.load(f.buffers[0], t, name="x")
    b.store(f.buffers[1], ir.Imm(34), b.shl(x, ir.Imm(3), name="l"))
    b.store(f.buffers[1], ir.Imm(36), b.shr(x, ir.Imm(3), name="r"))
    b.store(f.buffers[1], ir.Imm(38), b.sar(x, ir.Imm(3), name="a"))
    b.store(f.buffers[1], ir.Imm(40), b.sarv(x, t, name="v"))
    b.ret()
    return f


def full_bitwise():
    """The six negated bitwise opcodes, none of which the `bitwise` rung reaches."""
    f = ir.Function("full_bitwise", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    x = b.load(f.buffers[0], t, name="x"); y = b.load(f.buffers[1], t, name="y")
    for i, k in enumerate(("nand", "andn", "nor", "orn", "xnor")):
        b.store(f.buffers[2], ir.Imm(42 + 2 * i), getattr(b, k)(x, y, name=k))
    b.ret()
    return f


def float_arith():
    """fadd, fmul and fma on registers - the float three, one of them three-source."""
    f = ir.Function("float_arith", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    x = b.load(f.buffers[0], t, name="x"); y = b.load(f.buffers[1], t, name="y")
    b.store(f.buffers[2], ir.Imm(52), b.fadd(x, y, name="fa"))
    b.store(f.buffers[2], ir.Imm(54), b.fmul(x, y, name="fm"))
    b.store(f.buffers[2], ir.Imm(56), b.fma(x, y, t, name="ff"))
    b.ret()
    return f


def saturate_clamp():
    """fsat - clamp(x, 0, 1), the opcode measured through its source's negate modifier."""
    f = ir.Function("saturate_clamp", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    x = b.load(f.buffers[0], b.builtin("thread_position_in_grid", name="t"), name="x")
    b.store(f.buffers[1], ir.Imm(58), b.fsat(x, name="s")); b.ret()
    return f


def narrow_types():
    """i16 values through the ALU. The three widths are INDEPENDENT operands - src1, operand B and
    dest each carry their own - so a program in which they disagree is the one that says whether
    the backend writes all three or inherits any."""
    f = ir.Function("narrow_types", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", type=ir.I16, name="t")
    x = b.add(t, ir.Imm(5), type=ir.I16, name="x")
    w = b.add(x, ir.Imm(1), type=ir.I32, name="w")
    b.store(f.buffers[1], ir.Imm(60), w); b.ret()
    return f


def store_range_four():
    """The range store at its widest: n=4, which is the top of the form's two-bit count."""
    f = ir.Function("store_range_four", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    vs = [b.add(t, ir.Imm(i + 1), name="v%d" % i) for i in range(4)]
    b.store_range(f.buffers[1], ir.Imm(62), vs); b.ret()
    return f


LADDER += [unary_rounding, transcendentals, bit_manipulation, saturating_arith, shift_family,
           full_bitwise, float_arith, saturate_clamp, narrow_types, store_range_four]


# --- PROGRAMS SHAPED FOR DISPATCH --------------------------------------------------------------
#
# The rungs above are written to exercise the compiler. These two are written to be DISPATCHED, and
# the difference is entirely about what a many-lane grid does to an observable: a program where
# every lane writes the same output slot has a racy result, three runs need not agree, and agree()
# refuses it - correctly, because a value that depends on which lane won is not a measurement.

def indexed_store_lane():
    """op17229: every lane writes its OWN element, so the result is a function of the lane.

    The index is offset by 32 to clear the harness's own slots - the canary at 6 and the case slots
    from 8 - because an indexed store that lands on the canary makes "did the program finish" and
    "did the store work" the same question, and they have to be separable.
    """
    f = ir.Function("indexed_store_lane", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    b.store_at(f.buffers[1], b.add(t, ir.Imm(32), name="idx"), b.add(t, ir.Imm(11), name="v"))
    b.ret()
    return f


def threadgroup_constant_roundtrip():
    """op13288 + op12364 + op447: a threadgroup round trip whose answer does not depend on the lane.

    threadgroup_roundtrip sends each lane's own t through the exchange, so the single output slot
    holds whatever lane wrote last. Sending a CONSTANT through instead makes every lane agree on the
    answer while still proving both halves: the value only reaches the output if the store put it in
    threadgroup memory and the load took it out again.
    """
    f = ir.Function("threadgroup_constant_roundtrip", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    # ABI v4: a program that touches threadgroup memory states its scratchpad, or the compiler
    # refuses it (it cannot bound a register-indexed scratchpad from the program). One word is
    # used; 32 are declared so the block matches the dispatch group this ladder runs under.
    f.declare_threadgroup(32, size=(32, 1, 1), alignment=4)
    b = ir.Builder(f, f.block("entry"))
    z = b.const(0, name="z")
    b.store_tg(b.const(0x5A5A1234, name="k"), z)
    b.barrier()
    b.store(f.buffers[1], ir.Imm(20), b.load_tg(z, name="r"))
    b.ret()
    return f


LADDER += [indexed_store_lane, threadgroup_constant_roundtrip]


# --- THE EXEC MASK, SHAPED SO ITS EFFECT IS SEPARABLE -------------------------------------------
#
# op582 exec.mask, op577 exec.restore and op10369 cmp cannot be proved the way an ALU opcode is:
# cmp writes a FLAGR, which no store can read, and the exec pair produce no value at all. Their only
# observable is whether a store INSIDE the masked region fires. One program cannot say that - a slot
# left at its fill is equally consistent with "the mask suppressed the store" and "the store never
# worked" - so this is three programs, and the third is the control that separates them.

def exec_taken():
    """Condition TRUE: 5 < 9, so the guarded store must fire."""
    f = ir.Function("exec_taken", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    x = b.add(b.builtin("thread_position_in_grid", name="t"), ir.Imm(5), name="x")
    p = b.cmp(x, 9, pred="lt", name="p")
    then, join = f.block("then"), f.block("join")
    b.br_cond(p, then, join)
    b.at(then); b.store(f.buffers[1], ir.Imm(24), b.const(0x11111111, name="k")); b.br(join)
    b.at(join); b.ret()
    return f


def exec_not_taken():
    """Condition FALSE: 5 < 1 is not, so the guarded store must NOT fire.

    This is the record that carries the finding, and it is worthless alone: an unwritten slot is
    what a broken store looks like too. exec_ungated is what makes it evidence."""
    f = ir.Function("exec_not_taken", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    x = b.add(b.builtin("thread_position_in_grid", name="t"), ir.Imm(5), name="x")
    p = b.cmp(x, 1, pred="lt", name="p")
    then, join = f.block("then"), f.block("join")
    b.br_cond(p, then, join)
    b.at(then); b.store(f.buffers[1], ir.Imm(26), b.const(0x22222222, name="k")); b.br(join)
    b.at(join); b.ret()
    return f


def exec_ungated():
    """THE NULL CONTROL: the same store with no mask around it. It must fire."""
    f = ir.Function("exec_ungated", [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    b = ir.Builder(f, f.block("entry"))
    b.store(f.buffers[1], ir.Imm(28), b.const(0x33333333, name="k"))
    b.ret()
    return f


LADDER += [exec_taken, exec_not_taken, exec_ungated]


def bitwise_registers():
    """`x & y` with BOTH operands in registers - which nothing else in this ladder writes.

    `bitwise` uses immediates and `full_bitwise` reaches nand/andn/nor/orn/xnor through the generic
    authoring path, so the plain register-register and/or/xor was never selected by any rung. It
    does not compile: op424, op13575 and op17771 have no entry in g17asm.ALU_FORM, so selfcheck
    cannot decode what selection emitted. Kept as a rung so the gap stays visible and reports
    BLOCKED with the reason rather than being invisible.
    """
    f = ir.Function("bitwise_registers", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    # THROUGH AN ALU FIRST. The four-byte form has no load-wait - byte0[3] is the bit every other
    # authored form uses and it was measured not to work here - so a value taken straight from a
    # load is read before it lands.
    x = b.add(b.load(f.buffers[0], t, name="x0"), ir.Imm(0), name="x")
    y = b.add(b.load(f.buffers[1], t, name="y0"), ir.Imm(0), name="y")
    b.store(f.buffers[2], ir.Imm(30), getattr(b, "and")(x, y, name="r"))
    b.ret()
    return f


LADDER += [bitwise_registers]


def main():
    print("%-16s %-10s %s" % ("rung", "status", "detail"))
    n_ok = 0
    for k in LADDER:
        try:
            p = g17cc.compile_function(k())
            n_ok += 1
            print("%-16s %-10s %d bytes, %d instructions" % (k.__name__, "COMPILES",
                                                             len(p.code), len(p.layout)))
        except g17cc.Unsupported as e:
            print("%-16s %-10s %s" % (k.__name__, "BLOCKED", e))
        except Exception:
            print("%-16s %-10s %s" % (k.__name__, "BROKEN",
                                      traceback.format_exc().strip().splitlines()[-1]))
    print("\n%d/%d rungs compile" % (n_ok, len(LADDER)))
    if "--image" in sys.argv:
        # The image view: how much of a real container the compiler actually generates. This is
        # the number "complete native program image" reduces to, and it is small on purpose -
        # naming it is what lets it be driven up.
        import os, g17image
        ref = os.path.expanduser("~/.cache/agxforge/agx/ac2-32x32x64")
        print()
        for k in LADDER:
            try:
                p = g17cc.compile_function(k())
            except Exception:
                continue
            try:
                img = p.to_image(ref, at=0x4fa)
            except ValueError as e:
                # Reported, not skipped: a program that does not fit the window is a real
                # constraint of splicing into someone else's container, and it is one of the
                # reasons to stop splicing.
                print("  %-20s %5d bytes  DOES NOT FIT: %s" % (k.__name__, len(p.code), e))
                continue
            gen = len(p.code); tot = len(img.text)
            print("  %-20s %5d generated of %5d __text bytes  %4.1f%%   container %.2f%%"
                  % (k.__name__, gen, tot, 100.0 * gen / tot,
                     100.0 * gen / len(img.loc["fat"])))

def imageblock_roundtrip():
    """Read one imageblock member, add to it, write it back, and copy it to a device buffer.

    THE FOURTH ADDRESSING SHAPE after load, store and atomic, and the first whose address is
    neither a buffer index nor a literal: an imageblock is indexed by this thread's own tile
    coordinate, delivered in one register as a packed ushort2 whose halves the backend fills from
    SR_LOCAL_X and SR_LOCAL_Y. `member` is the field's byte offset inside the imageblock struct.

    NOT EXECUTION-PROVEN AND IT CANNOT BE from this harness: an imageblock is tile memory a render
    pass allocates, and the compute dispatch path has no tile to hand it. It is here because it
    compiles and is byte-identical to Apple's own instruction for the same registers, and because
    leaving it out would hide a capability from the only place that counts them.
    ledger/g17-the-imageblock-coordinate-is-a-packed-register.toml
    """
    f = ir.Function("imageblock_roundtrip", [ir.Buffer("U", 0), ir.Buffer("O", 1)])
    b = ir.Builder(f, f.block("entry"))
    t = b.builtin("thread_position_in_grid", name="t")
    old = b.imageblock_read(member=12, name="old")
    b.imageblock_write(b.add(old, b.load(f.buffers[0], t, name="u"), name="sum"), member=12)
    b.store_at(f.buffers[1], t, old)
    b.ret()
    return f


LADDER += [imageblock_roundtrip]


def wave_aggregate():
    """APPLE'S WAVE-AGGREGATED ATOMIC, whole: vote, elect, one RMW for the group, redistribute.

    Five things this backend could not express a day ago, in one program - the two simd vote
    registers read into the halves of one register, an exec-gated election, the UNIFORM atomic
    form, the simd broadcast that carries the elected lane's result back, and the prefix that gives
    each lane a different answer.

    op10094's operand 6 is a source lifetime and the form's default is 16, which authors
    byte-exactly and writes nothing. ledger/g17-the-uniform-atomic-needed-one-lifetime-bit.toml
    """
    f = ir.Function("wave_aggregate", [ir.Buffer("A", 0), ir.Buffer("C", 2)])
    e = f.block("entry"); one = f.block("one"); join = f.block("join")
    b = ir.Builder(f, e)
    t = b.builtin("thread_position_in_grid", name="t")
    w = b.simd_vote_pair(name="w")
    total = b.shr(w, ir.Imm(16), name="total")
    prefix = b.shr(b.shl(w, ir.Imm(16), name="lo"), ir.Imm(16), name="prefix")
    b.br_cond(b.cmp(prefix, 1, "lt", name="p"), one, join)
    b.at(one)
    old = b.atomic_uniform("add", f.buffers[0], total, name="old")
    b.br(join)
    b.at(join)
    bc = b.simd_broadcast_first(old, name="bc")
    b.store_at(f.buffers[1], t, b.add(bc, prefix, name="mine"))
    b.ret()
    return f


LADDER += [wave_aggregate]


def threadgroup_atomic():
    """A THREADGROUP ATOMIC COUNTER - op11765, the twin of the device uniform form.

    It carries NO address operand at all, so it names one implicit location per threadgroup. Two
    things about it were measured rather than read off the mnemonic: it SUBTRACTS, because the
    device three-bit operation field's middle carrier is an opcode bit here and the operation is
    fixed by the form; and it needs the threadgroup region that an ordinary load or store
    allocates, without which it authors correctly and writes nothing.

    The store below is therefore load-bearing, not decoration.
    ledger/g17-the-threadgroup-atomic-subtracts-and-needs-a-region.toml
    """
    f = ir.Function("threadgroup_atomic", [ir.Buffer("B", 1), ir.Buffer("C", 2)])
    f.declare_threadgroup(32, size=(32, 1, 1), alignment=4)   # ABI v4: the scratchpad is stated
    e = f.block("entry"); one = f.block("one"); join = f.block("join")
    b = ir.Builder(f, e)
    t = b.builtin("thread_position_in_grid", name="t")
    w = b.simd_vote_pair(name="w")
    total = b.shr(w, ir.Imm(16), name="total")
    prefix = b.shr(b.shl(w, ir.Imm(16), name="lo"), ir.Imm(16), name="prefix")
    b.store_tg(total, t)
    b.barrier()
    b.br_cond(b.cmp(prefix, 1, "lt", name="p"), one, join)
    b.at(one)
    o1 = b.atomic_tg_uniform("sub", total, name="o1")
    o2 = b.atomic_tg_uniform("sub", total, name="o2")
    d = b.sub(o1, o2, name="d")
    b.br(join)
    b.at(join)
    b.store_at(f.buffers[1], t, b.add(b.simd_broadcast_first(d, name="bc"), prefix, name="mine"))
    b.ret()
    return f


LADDER += [threadgroup_atomic]



# ---------------------------------------------------------------------------------------------
# THE COMPOSITION RUNGS - programs whose answer is about the RELATION between two instructions.
#
# Every other rung asks whether the backend can LOWER something. These four ask what one
# instruction leaves for the next, which is the half of the machine nothing here measures. They
# are designed in docs/archive/g17-composition-batch.md and turned into a dispatch plan by
# tools/g17composition.py; that document holds the candidate answers and why these shapes tell
# them apart. They live HERE rather than in the generator because spike/accel/re/oracle.py
# dispatches a whole-program record in its own child process, where g17oracle.ladder_program
# resolves the name against this module - a rung the generator installed at import time would not
# exist in that child, and the record would die mid-batch.
#
# THE TWO-BUFFER SHAPE IS NOT A STYLE CHOICE. g17oracle._wrap hardcodes `buffers=[1, 2]` and says
# in as many words that this is "correct by coincidence rather than by derivation: every ladder
# program it wraps declares exactly those two slots". Every rung that has actually been dispatched
# - exec_ungated, indexed_store_lane, threadgroup_constant_roundtrip - declares A at 0 and C at 1
# and stores to f.buffers[1]. A composition rung declaring three would build an image whose
# bindings do not match the code compiled against them, which tools/g17endtoend.py describes
# exactly: "a store at a rank the image does not declare, which returns the fill value at status 0
# and looks exactly like a broken opcode".
#
# AND THEY DO NOT LOAD. Which host buffer a ladder program's rank-0 binding receives has never
# been established - no dispatched rung has ever read an input - so a load here would make a wrong
# answer about a hazard indistinguishable from a wrong answer about a binding. The memory-hazard
# questions are a separate tranche and the design document says what has to be measured first.

COMP_SENTINEL = 0x0BEE      # 3054: nonzero, under 16 bits, and 3054/3053/0/0xBEEF are distinct
COMP_SLOT_MASKED = 40       # what the reader INSIDE the region stored, or the fill if suppressed
COMP_SLOT_FIRST = 52        # what the first reader of y computed
COMP_SLOT_REREAD = 54       # y READ AGAIN afterwards - 0 here means something released it


def _comp_function(name):
    f = ir.Function(name, [ir.Buffer("A", 0), ir.Buffer("C", 1)])
    return f, ir.Builder(f, f.block("entry"))


def comp_reread_unmasked():
    """y is read by a sub, then read AGAIN and stored: the release channel with no mask near it.

    Unpatched, the liveness pass writes KEEP on the sub's source because y has a later reader, so
    nothing releases anything and slot 54 must hold y. Patched to release - op11666's whole
    lifetime operand, 16 rather than 32 - it must hold 0. That pair is the positive and negative
    control for the observable every composition record reads, and it is what makes "slot 54 held
    y" mean "nothing released it" rather than "the patch never reached the bytes".
    """
    f, b = _comp_function("comp_reread_unmasked")
    t = b.builtin("thread_position_in_grid", name="t")
    y = b.add(t, ir.Imm(COMP_SENTINEL), name="y")
    b.store(f.buffers[1], ir.Imm(COMP_SLOT_FIRST), b.sub(y, ir.Imm(1), name="w"))
    b.store(f.buffers[1], ir.Imm(COMP_SLOT_REREAD), b.add(y, ir.Imm(0), name="z"))
    b.ret()
    return f


def comp_reread_masked_off():
    """The same reader, moved INSIDE an exec region whose condition is FALSE.

    5 < 1 is not, so op582/op577 gate a region that does not run - the same condition
    exec_not_taken uses, which is the program that established that the mask suppresses a store.
    The question this asks is the next one: the read inside the region is still ISSUED and its
    lifetime operand still says release, so does a value die in a branch that was never taken? A
    liveness pass over a CFG has to know, and it fails silently in one of the two directions.
    """
    f, b = _comp_function("comp_reread_masked_off")
    t = b.builtin("thread_position_in_grid", name="t")
    y = b.add(t, ir.Imm(COMP_SENTINEL), name="y")
    g = b.add(t, ir.Imm(5), name="g")
    p = b.cmp(g, 1, pred="lt", name="p")
    then, join = f.block("then"), f.block("join")
    b.br_cond(p, then, join)
    b.at(then)
    b.store(f.buffers[1], ir.Imm(COMP_SLOT_MASKED), b.sub(y, ir.Imm(1), name="m"))
    b.br(join)
    b.at(join)
    b.store(f.buffers[1], ir.Imm(COMP_SLOT_REREAD), b.add(y, ir.Imm(0), name="z"))
    b.ret()
    return f


def comp_reread_masked_on():
    """Byte-identical to comp_reread_masked_off apart from the compared immediate: 5 < 9 is true.

    The region runs, so a release inside it must fire. Without this record a patched
    comp_reread_masked_off that returned y would be ambiguous between "the mask predicated the
    release" and "being inside an exec region is what stopped the patch doing anything" - which is
    the third-program argument ledger/g17-the-mask-suppresses-a-store.toml already had to make.
    """
    f, b = _comp_function("comp_reread_masked_on")
    t = b.builtin("thread_position_in_grid", name="t")
    y = b.add(t, ir.Imm(COMP_SENTINEL), name="y")
    g = b.add(t, ir.Imm(5), name="g")
    p = b.cmp(g, 9, pred="lt", name="p")
    then, join = f.block("then"), f.block("join")
    b.br_cond(p, then, join)
    b.at(then)
    b.store(f.buffers[1], ir.Imm(COMP_SLOT_MASKED), b.sub(y, ir.Imm(1), name="m"))
    b.br(join)
    b.at(join)
    b.store(f.buffers[1], ir.Imm(COMP_SLOT_REREAD), b.add(y, ir.Imm(0), name="z"))
    b.ret()
    return f


def comp_store_reread():
    """y is STORED and then read again - the STORE's own lifetime operand, not an ALU's.

    The backend emits store.8 here (op17244 at EIGHT bytes) and g17auth.length(17244) is 14, so
    the authoring table's field map belongs to the other form and g17oracle.program REFUSES an
    operand patch on it by name. That refusal is right, and it is why this rung carries no
    separator: without the patch both "the store releases its value" and "the operand is inert on
    this form" predict the same slot 54, because the compiler's own liveness is the only thing
    writing the field and it writes it correctly.

    It ships as a rung anyway because it is the program the question needs on the day (17244, 8)
    gets an authoring entry, and because compiling it is what shows the field moving with liveness
    at all: the two stores differ in exactly the bits a lifetime would occupy. The design document
    says what that is and is not evidence of - those values were read through the OTHER form's
    map, which g17auth.decode, unlike g17oracle.program, does not refuse.
    """
    f, b = _comp_function("comp_store_reread")
    t = b.builtin("thread_position_in_grid", name="t")
    y = b.add(t, ir.Imm(COMP_SENTINEL), name="y")
    b.store(f.buffers[1], ir.Imm(COMP_SLOT_FIRST), y)
    b.store(f.buffers[1], ir.Imm(COMP_SLOT_REREAD), b.add(y, ir.Imm(0), name="z"))
    b.ret()
    return f


COMPOSITION = [comp_reread_unmasked, comp_reread_masked_off, comp_reread_masked_on,
               comp_store_reread]
LADDER += COMPOSITION


if __name__ == "__main__":
    main()
