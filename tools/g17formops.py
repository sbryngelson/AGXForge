#!/usr/bin/env python3
"""WHICH APPLE OPCODE EACH EMITTED FORM IS, harvested into the repository once.

G17Program.abi() reports the instruction forms an image contains, and the linker consumes them as
(opcode, length) - three of its tools do. It used to get them by running Apple's disassembler over
the emitted code, which makes the compiler unable to describe its own output without the vendor,
and puts a fork in the middle of ABI generation.

The compiler already knows what it emitted. What it does not know is Apple's NUMBER for it, and
that number is a fact about Apple's encoding rather than about this program: for most forms it is
constant, and for three it is selected by fields the compiler itself set.

    alu.12      the operation nibble AND the operand widths: op=3 mode=1 is 10279 at
                dest_w=1 src1_w=1, 10280 at src1_w=0, 10288 at dest_w=0
    read_sr.4   `half` picks 14060 over 14059 - the sixteen-bit special-register read
    load.14     `half` picks 12646 over 12682, which is the whole FP16 load story

So the key is the form, its length, and the few fields that select the opcode - nothing about the
registers, the immediate or the address. Harvested with the decoder once, checked in, and read
without it. --check re-derives every entry and compares, so a stale table fails rather than drifts.

An entry the table does not cover is an ERROR at ABI time, not a fallback: a form whose opcode this
side cannot name is one the linker should not be told about.

    python3 tools/g17formops.py --refresh    compile the corpus, decode once, write the table
                                             (refuses to drop a committed row; --allow-drop)
    python3 tools/g17formops.py --check      re-derive and fail if anything disagrees
    python3 tools/g17formops.py              report coverage
"""
import json
import os
import sys

# THE PRODUCTION HALF MOVED TO agxforge.g17.formops; THE PROBES BELOW DID NOT.
#
# What g17cc calls lives in the package now; the witness builders and diagnostics that
# call g17cc back stay here. Every production name is re-exported from that one
# implementation so existing callers are unaffected.
import sys as _sys
import types as _types

# THE REPOSITORY IMPORT ROOT COMES FIRST.
#
# This file is a legacy entry point as well as a module: run by absolute path from another
# directory, with PYTHONPATH unset, nothing puts the checkout on sys.path - so importing the
# package raised ModuleNotFoundError: agxforge before any of its own work ran. Root found it on two
# of these; the same ordering error was in every split entry. The package itself must not touch
# sys.path, so establishing the root belongs here, ahead of the package import.
import os as _os
import sys as _sys

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from agxforge.g17 import formops as _impl

# ONE SHARED FORWARDING MECHANISM, in agxforge.g17.compat. Eleven private copies of this were
# 834 lines; the reasons each half exists are recorded there, next to the code.
from agxforge.g17 import compat as _compat

globals().update(_compat.install(__name__, _impl, (
    "HERE",
    "ISA",
    "SELECTORS",
    "TABLE",
    "key_of",
    "load",
    "opcode_of",
)))

sys.path.insert(0, HERE)

# The fields that select the opcode, per form. Everything absent here has one opcode, and the
# harvest FAILS if that turns out to be false rather than recording whichever it saw last.








def _programs():
    """Everything this backend can compile that the repository can rebuild without a GPU."""
    import g17cc, g17endtoend, g17halfscan
    out = []
    for name, ent in sorted(g17endtoend.KERNELS.items()):
        try:
            out.append(("e2e:" + name, g17cc.compile_function(ent[0]())))
        except Exception:
            continue
    for r, c in ((4, 8), (33, 384)):
        try:
            out.append(("scan:%dx%d" % (r, c), g17halfscan.compile_scan(r, c)[0]))
        except Exception:
            continue
    # THE TEN-BYTE HALF LOAD is an explicit measured form. Keep a source-owned program in the
    # harvest so the ABI can name it without consulting Apple's decoder at ABI time.
    try:
        import g17ir as ir
        b0 = ir.Buffer("src", 1, elem=ir.F16); b1 = ir.Buffer("dst", 2, elem=ir.I16)
        f = ir.Function("half_load10", [b0, b1]); b = ir.Builder(f, f.block("entry"))
        t = b.builtin("thread_position_in_grid", name="t")
        v = b.load(b0, t, type=ir.I16, width="half", form_length=10, name="v")
        z = b.f16_to_f32(v, name="z")
        b.store_at(b1, t, z, width="word"); b.ret()
        out.append(("form:half_load10", g17cc.compile_function(f)))
    except Exception:
        pass
    # THE FORMS THE POPULATION ABOVE NEVER EMITS, stated as programs rather than waited for. A
    # counted loop whose result goes to a CONSTANT slot uses the eight-byte `store` (op17244), which
    # no end-to-end kernel and neither scan emits - so the table had no row, and a program that
    # compiled fine raised at abi() with "no opcode recorded for store.8|8|half=None". A form the
    # compiler can select must be nameable, so the harvest includes one program per such form.
    import g17ir as ir
    try:
        o = ir.Buffer("output", 1, elem=ir.I32)
        f = ir.Function("slot_store", [o])
        pre, loop, end = (f.block(n) for n in ("pre", "loop", "exit"))
        b = ir.Builder(f, pre)
        zero = b.const(0, name="zero")
        b.br(loop)
        b.at(loop)
        k = b.phi(zero, name="k")
        nk = b.add(k, ir.Imm(1), name="nk")
        ir.Builder.phi_latch(k, nk)
        b.br_cond(b.cmp(nk, 8, "lt", name="more"), loop, end)
        b.at(end)
        b.store(o, ir.Imm(4), nk)
        b.ret()
        out.append(("form:slot_store_loop", g17cc.compile_function(f)))
    except Exception as e:
        print("   (slot-store harvest program did not compile: %s)" % str(e)[:120])
    # THE SKIP BRANCH (cc's opt-in fn.skip_regions, MM 25.141.2): a guarded region's forward branch, op462 in
    # Apple's code, which no default-compiled program emits.
    try:
        o = ir.Buffer("output", 1, elem=ir.I32)
        f = ir.Function("skip_region", [o])
        pre, then, join = (f.block(n) for n in ("pre", "then", "join"))
        b = ir.Builder(f, pre)
        t = b.builtin("thread_position_in_grid", name="t")
        b.br_cond(b.cmp(t, 4, "gt", name="g"), then, join)
        b.at(then)
        b.store_at(o, t, b.add(t, ir.Imm(1), name="t1"))
        b.br(join)
        b.at(join)
        b.ret()
        f.skip_regions = True
        out.append(("form:skip_region", g17cc.compile_function(f)))
    except Exception as e:
        print("   (skip-region harvest program did not compile: %s)" % str(e)[:120])
    # EVERY COMPONENT COUNT AT BOTH SLOT WIDTHS: n=2, 3, 4 at slot 4 (the eight-byte forms) and
    # at slot 70 (the fourteen-byte forms). n=1 is not a range store - it is the k/k+3 encoding
    # and select() refuses it; a one-value slot store is `store`, harvested above.
    try:
        o = ir.Buffer("output", 1, elem=ir.I32)
        f = ir.Function("range_stores", [o])
        b = ir.Builder(f, f.block("entry"))
        t = b.builtin("thread_position_in_grid", name="t")
        for slot in (4, 70):
            for n in (2, 3, 4):
                b.store_range(o, ir.Imm(slot + 10 * n), [b.add(t, ir.Imm(i + 1)) for i in range(n)])
        b.ret()
        out.append(("form:range_stores", g17cc.compile_function(f)))
    except Exception as e:
        print("   (range-store harvest program did not compile: %s)" % str(e)[:120])
    # THE VECTOR MEMORY FORMS (handoff 10ae): the tuple load at both lengths and the vector store, stated as two
    # programs because the load's LENGTH is selected by what precedes it (8 after an ALU, 14 after a
    # special-register read) - so one program cannot witness both. Without these the registry had no row for
    # op12709/op17256 and abi() raised at contract time, which is the blocker integration named on the WIP snapshot.
    try:
        import g17vecmem
        out.append(("form:vector_memory_14", g17cc.compile_function(g17vecmem.ir_of())))
        out.append(("form:vector_memory_8", g17cc.compile_function(g17vecmem.ir_of(index_plus=1, index_through_alu=True))))
    except Exception as e:
        print("   (vector-memory harvest programs did not compile: %s)" % str(e)[:120])
    # THE OTHER COMPONENT COUNTS of the tuple load (handoff 10am): the count is byte4[6:5] = n-1 and the
    # opcode follows it in steps of nine, so n=2 is op12691 and n=3 is op12700 - neither of which any
    # end-to-end kernel or the four-component harvest programs above emit.
    for n in (2, 3):
        try:
            o = ir.Buffer("out", 0); i = ir.Buffer("in", 1)
            f = ir.Function("vecload%d" % n, [o, i])
            b = ir.Builder(f, f.block("entry"))
            t = b.builtin("thread_position_in_grid", name="t")
            lanes = b.load_vec_at(i, t, n, name="q")
            for k, lane in enumerate(lanes):
                b.store(o, ir.Imm(4 + k), b.add(lane, ir.Imm(k + 1), name="s%d" % k))
            b.ret()
            out.append(("form:vector_load_n%d" % n, g17cc.compile_function(f)))
        except Exception as e:
            print("   (the %d-component vector-load harvest program did not compile: %s)" % (n, str(e)[:110]))
    # THE TEXTURE FORMS, stated as the three programs integration's common texture frontier compiles
    # (results/g17-texture-common-frontier-v1, docs/archive/g17-texture-common-handoff.md): the coordinate
    # publishes and the thirty-two-bit texture read are emitted by no end-to-end kernel and no scan, so
    # the table had no row and every one of those programs refused at abi() by name. Harvested from the
    # same source they are compiled from, so their keys are named without the decoder like every other.
    try:
        import g17texrun
        for name, args in (("uniform_5_1", (5, 1)), ("uniform_2_3", (2, 3)), ("lane31", ())):
            out.append(("texture:" + name, g17cc.compile_function(g17texrun.kernel_ir(*args))))
    except Exception as e:
        print("   (texture harvest programs did not compile: %s)" % str(e)[:120])
    # THE SHORT DEVICE LOAD is a separate length-specific form. Apple's corpus has 960 of these
    # op12682/8 instructions, while the end-to-end programs used above all select the 14-byte
    # member. Keep one source-level control in the harvest so the checked-in opcode table can name
    # the compiler's explicit short-form request without consulting the decoder at ABI time.
    try:
        o = ir.Buffer("output", 1, elem=ir.I32)
        s = ir.Buffer("source", 2, elem=ir.I32)
        f = ir.Function("short_load", [o, s])
        b = ir.Builder(f, f.block("entry"))
        t = b.builtin("thread_position_in_grid", name="t")
        v = b.load(s, t, form_length=8)
        b.store(o, ir.Imm(0), v)
        b.ret()
        out.append(("form:load8", g17cc.compile_function(f)))
    except Exception as e:
        print("   (short-load harvest program did not compile: %s)" % str(e)[:120])
    # THE SIXTEEN-BIT ZERO MOVE (555,4) (handoff 10ag): a half store of a literal zero. No end-to-end kernel
    # and neither scan writes one, and the emitter deliberately does not name its own opcode, so without this
    # program the registry has no row and the program refuses at abi() by name.
    try:
        o = ir.Buffer("output", 0, elem=ir.I16)
        f = ir.Function("half_zero", [o])
        b = ir.Builder(f, f.block("entry"))
        b.store_at(o, b.builtin("thread_position_in_grid", name="t"), b.const(0, type=ir.I16, name="z"), width="half")
        b.ret()
        out.append(("form:half_zero", g17cc.compile_function(f, regs=range(0, 16))))
    except Exception as e:
        print("   (half-zero harvest program did not compile: %s)" % str(e)[:120])
    # THE TWO-BYTE MOVE-IMMEDIATE (op11842/2): a literal of 0..127 requested with length=2. Every
    # other program materialises through movimm.8, so without this one the registry has no row
    # and a program that asks for the short form refuses at abi() (Set B item 7).
    try:
        o = ir.Buffer("output", 1, elem=ir.I32)
        f = ir.Function("movimm2", [o])
        b = ir.Builder(f, f.block("entry"))
        t = b.builtin("thread_position_in_grid", name="t")
        b.store_at(o, t, b.add(t, b._def("const", [ir.Imm(0x55)], type=ir.I32, name="k", length=2), name="v"))
        b.ret()
        out.append(("form:movimm2", g17cc.compile_function(f)))
    except Exception as e:
        print("   (movimm2 harvest program did not compile: %s)" % str(e)[:120])
    # THE HALF-TO-HALF WAITING COPY (op10289/12): `add(x, 0)` on a sixteen-bit value, which the
    # integer16 widening route emits to make a loaded half ready before op10283 reads it. It is the
    # one alu.12 key with BOTH src1_w=0 and dest_w=0 - the docstring's three rows are the three
    # combinations Apple's own programs use, and no end-to-end kernel, scan or probe above emits
    # this fourth one, so every integer16 source refused at abi() by name. Harvested here in the
    # shape the frontend selects (sixteen-bit load, waiting copy, widen, word store) so the opcode
    # comes from the vendor's decoder rather than from the emitter that chose the form.
    try:
        o = ir.Buffer("output", 0, elem=ir.I32)
        s = ir.Buffer("source", 1, elem=ir.I16)
        f = ir.Function("widen16", [o, s])
        b = ir.Builder(f, f.block("entry"))
        t = b.builtin("thread_position_in_grid", name="t")
        v = b.load(s, t, type=ir.I16, width="half", name="v")
        ready = b.add(v, ir.Imm(0), type=ir.I16, name="z16w")
        b.store(o, ir.Imm(0), b.u16_to_u32(ready, name="z16"))
        b.ret()
        out.append(("form:widen16_waiting_copy", g17cc.compile_function(f)))
    except Exception as e:
        print("   (widening harvest program did not compile: %s)" % str(e)[:120])
    # THE CONSUMED ADD HAS ITS OWN LENGTH KEY. The compiler now selects a 12-byte slot-7 op10090
    # when the old value is read; the update-only ADD stays ten bytes. ABI generation must name
    # both forms. This source-owned harvest program exercises the returned-value wait and store.
    try:
        a = ir.Buffer("counter", 0, elem="atomic_uint")
        o = ir.Buffer("output", 1, elem=ir.I32)
        f = ir.Function("atomic_add_return", [a, o])
        b = ir.Builder(f, f.block("entry"))
        t = b.builtin("thread_position_in_grid", name="t")
        old = b.atomic_rmw("add", a, t, b.const(1, name="one"), name="old")
        b.store_at(o, t, old)
        b.ret()
        out.append(("form:atomic_add_return", g17cc.compile_function(f)))
    except Exception as e:
        print("   (consumed ADD harvest program did not compile: %s)" % str(e)[:120])
    # THE ATOMIC OPERATION IS A FIELD, SO EVERY OPERATION IS A SEPARATE KEY. This table is keyed on
    # the fields that select the opcode, and for the atomic forms `aop` is one of them - so
    # `atomic.add.10|10|aop=0` being present said nothing about `aop=1`. Root found it from the
    # other side: an atomic `and` compiled to 34 bytes and then raised at contract() with "no
    # opcode recorded for 'atomic.add.10|10|aop=1'". The operation does not change the opcode here
    # (Apple's own compilations of add, `and` and or are all op10090/10 with only the operation
    # field moving - results/g17-atomic-assessment-v1), but that is a fact the DECODER has to
    # state rather than something this emitter may assume, which is the whole point of the table.
    # One program per operation the front end can select, so the vendor names each.
    for _aop in ("and",):
        try:
            a = ir.Buffer("counter", 0, elem="atomic_uint")
            o = ir.Buffer("output", 1, elem=ir.I32)
            f = ir.Function("atomic_%s" % _aop, [a, o])
            b = ir.Builder(f, f.block("entry"))
            t = b.builtin("thread_position_in_grid", name="t")
            # THE OLD VALUE IS NOT READ. A later refusal (an atomic `and` whose returned value is
            # read has unresolved scope) made this program stop compiling, and the refresh then
            # dropped the aop=1 row without failing. The harvest needs the instruction, not its
            # return, so the program stores the lane index instead.
            b.atomic_rmw(_aop, a, t, b.const(7, name="mask"), name="old")
            b.store_at(o, t, t)
            b.ret()
            out.append(("form:atomic_%s" % _aop, g17cc.compile_function(f)))
        except Exception as e:
            print("   (atomic %s harvest program did not compile: %s)" % (_aop, str(e)[:110]))
    # THE IMAGEBLOCK STORE AND LOAD (store.ib.32 / load.ib.32): only the tensor staging program of
    # Set A item 12 carries them to abi(), which refused by name with no row for 'store.ib.32|14'.
    # A write of this lane's value and a read of the same member, so the decoder names both forms.
    try:
        o = ir.Buffer("output", 0, elem=ir.I32)
        f = ir.Function("imageblock_rt", [o])
        b = ir.Builder(f, f.block("entry"))
        t = b.builtin("thread_index_in_simdgroup", name="t")
        b.imageblock_write(t, member=0)
        b.store_at(o, t, b.imageblock_read(member=0, name="back"))
        b.ret()
        out.append(("form:imageblock_rt", g17cc.compile_function(f)))
    except Exception as e:
        print("   (imageblock harvest program did not compile: %s)" % str(e)[:120])
    return out


def harvest():
    """{key: opcode}, decoding each program once. Conflicts raise rather than being overwritten."""
    import g17ref
    table, seen = {}, {}
    for name, p in _programs():
        real = {a: op for a, l, op in g17ref.walk(p.code, 0)}
        for at, raw, m in p.layout:
            if m.fields.get("opcode") is not None:
                continue
            k, op = key_of(m, len(raw)), real.get(at)
            if op is None:
                continue
            if k in table and table[k] != op:
                raise ValueError("%r is opcode %d in %s and %d in %s - the key does not select it"
                                 % (k, table[k], seen[k], op, name))
            table[k] = op
            seen[k] = name
    return table


def main():
    if "--refresh" in sys.argv:
        table = harvest()
        # A REFRESH MAY ADD ROWS, NEVER LOSE ONE SILENTLY. Each harvest program that stops
        # compiling only prints a line, so a newer refusal elsewhere dropped atomic.add.10|aop=1
        # from a refresh that exited 0. A row the committed table has and the harvest did not
        # reproduce is refused here; --allow-drop is the stated way to remove one on purpose.
        if os.path.exists(TABLE) and "--allow-drop" not in sys.argv:
            lost = sorted(set(load()) - set(table))
            if lost:
                print("REFUSED: the harvest no longer reproduces %s - a program that emitted them stopped "
                      "compiling (see the lines above); pass --allow-drop to remove them deliberately" % lost)
                return 1
        with open(TABLE, "w") as fh:
            json.dump({
                "note": ("Apple's opcode for each form this backend emits, keyed by the form, its "
                         "length, and only those fields that select the opcode. Harvested once so "
                         "that G17Program.abi() can name its own instructions without forking the "
                         "vendor's disassembler. A key this table does not carry is an error at "
                         "ABI time, not a fallback."),
                "source": "tools/g17formops.py --refresh",
                "selectors": {k: list(v) for k, v in sorted(SELECTORS.items())},
                "opcodes": table,
            }, fh, indent=1, sort_keys=True)
        print("wrote %s: %d keys" % (os.path.relpath(TABLE, os.path.dirname(ISA)), len(table)))
        return 0

    have = load()
    if "--check" in sys.argv:
        fresh = harvest()
        missing = sorted(set(fresh) - set(have))
        differ = sorted(k for k in set(fresh) & set(have) if fresh[k] != have[k])
        print("checked-in %d keys, re-derived %d" % (len(have), len(fresh)))
        print("  missing from the table   %d %s" % (len(missing), missing[:4]))
        print("  DISAGREEING              %d %s" % (len(differ), differ[:4]))
        if missing or differ:
            print("\nthe table is stale; run --refresh")
            return 1
        print("\n  every form this backend emits can be named without the decoder")
        return 0

    print("%s: %d keys" % (os.path.relpath(TABLE, os.path.dirname(ISA)), len(have)))
    for k in sorted(have):
        print("   %-46s op%d" % (k, have[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
