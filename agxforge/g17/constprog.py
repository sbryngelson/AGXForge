"""The constant program's footprint in a tensor object's metadata (goal item 11, Set A's slice).

What section 25.113 called the "spill record" is not register spill. Every one of the 726 Apple MMA
programs carries a second program, `agc.main.constant_program` (`...constant_program.cfg` when it
branches), and in the 78 whose constant program writes uniforms the metadata grows these,
each a function of that program or of the argument map (docs/g17-tensorops-machine-model.md 25.120):

  per-kernel slot 31      present iff the constant program writes >= 1 uniform          (L4)
                          value 16 + 16 x ceil(buffers / 2): a SCALAR, not an offset     (S31)
  kind-3 record field 0   8 + 4 x len(argument map)                                     (L1)
  kind-3 record field 2   the constant program's register count, or 1 if it writes none (L2)
  kind-6 record field 3   the uniform HIGH-WATER MARK: 1 + the highest uniform written.
                          A uniform write is ANY constant-program instruction whose first
                          operand Apple's decoder prints as bin(op0,const(K),S) - op592 in
                          both forms AND the ALU forms that write their result straight to
                          a uniform (op10298, op10306, op11185, op1038, op10883, op17789,
                          ...) - and it writes uniform K // 2 (K counts 16-bit halves).
                          Not 6 + writes: a program may skip uniforms (oracle W4 writes
                          u8-u23, field 3 = 24). With no writes it is the MAIN program's
                          floor (6, or 4 in 18 corpus programs), not this program's.
                          Counting op592 alone (the first statement of this law) was
                          refuted by held-out Y2, whose top uniform u10 is written by an
                          op10298; the ALU writers are counted since round 4, which held on
                          all 11 of its held-out compiles and refuted op592-only on 4 (L3)
  per-kernel slot 1       kind-6 field 2 + the high-water mark (field 2 is 0 when the
                          kind-6 record is absent); kind-6 field 2 is the MAIN program's (S1)
  argument map            [4b, 4b+1, 4b+2, 4b+3] for each buffer b whose binding-table
                          entry the constant program reads; 2 op14061 reads per buffer  (L5)

The map's CONTENTS are not in the constant program's bytes - two programs with byte-identical
op14061 reads list different buffers (ac2-64x64x64 [0..3], rep5_3 [8..11]) - so a compiler states
them from its own knowledge of which buffers it pre-computes, and a reader of Apple's object takes
them from the metadata (`metadata_facts`).

op592's 8-byte form writes one uniform, K // 2, like every other writer; K is uniformpreload.CONST9
(nine bits). Apple uses it for K >= 128, beyond the 4-byte form's seven bits, and for some last
writes. Held on held-out D1-D3 (K 16, 50, 134) and X1-X6, and on 261 of the 275 corpus programs
whose top writer it is.

WHERE THE HIGH-WATER LAW STOPS. Over the 2,720 corpus programs with a kind-6 record and a
constant-program write it fits 2,650. The 70 misses all UNDER-predict, and in 50 of them the main
program reads op0 uniforms up to exactly Apple's mark: the main program can raise field 3 too, by a
rule not pinned here (main reads also exceed the mark in 81 programs the law fits). All 78 MMA
programs with a write fit.
"""
import re
import subprocess
import tempfile

UNIFORM_WRITE = 592          # op592: the constant program's uniform-register write
ARGUMENT_READ = 14061        # op14061: one 32-bit half of a buffer's binding-table entry
RESERVED_UNIFORMS = 6        # u0-u5 in the MMA corpus; outside it a constant program may write lower
UNIFORM_BASE = 0             # the expression base Apple's decoder prints for a uniform: bin(op0, ...)
_EXPR = re.compile(r"^expr:bin\(op(\d+),const\((-?\d+)\),(\d+)\)$")
SYMBOLS = ("_agc.main.constant_program", "_agc.main.constant_program.cfg")


def span(sections, symbols):
    """The constant program's (start, end) inside __TEXT,__text, or None. It ends at `_agc.main`
    when main follows it, else at the section's end."""
    name = next((s for s in SYMBOLS if s in symbols), None)
    if name is None:
        return None
    start, main = symbols[name], symbols["_agc.main"]
    return start, (main if main > start else sections["__TEXT,__text"][1])


def first_operand_exprs(raws):
    """Apple's decoder's (base, const, scale) for each instruction's FIRST operand, or None where
    it is not an expression. One decoder call for all of them (agx3dis --expr, a 64-byte stride:
    each instruction alone in its slot, so a decode never reads a neighbour)."""
    from agxforge.g17 import ref as g17ref
    raws = [bytes(r) for r in raws]
    if not raws:
        return []
    with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
        for r in raws:
            if len(r) > 32:
                raise ValueError("a %d-byte instruction does not fit the 32-byte decode slot" % len(r))
            fh.write(r.ljust(64, b"\x00"))
        fh.flush()
        out = subprocess.run([g17ref.binary(), fh.name, "0", str(64 * len(raws)), "--pc", "0",
                              "--stride", "64", "--expr"], capture_output=True, text=True).stdout
    got = [None] * len(raws)
    for line in out.splitlines():
        p = line.split()
        if len(p) < 4 or p[1] == "bad":
            continue
        k, off = divmod(int(p[0], 16), 64)
        m = _EXPR.match(p[3])
        if off == 0 and k < len(raws) and int(p[1]) == len(raws[k]) and m:
            got[k] = tuple(int(g) for g in m.groups())
    return got


def uniform_index(raw, long_form=False):
    """The uniform one uniform-writing instruction writes: const // 2 of its first operand when
    Apple's decoder prints that operand as bin(op0, const, S), else None. Any opcode, any form.
    For 4-byte op592 this is its bytes' reading (bits 4-7 of byte 0 the low nibble, bits 6-7 of
    byte 2 the next two: u16 = 0b 49, u32 = 0b 89); for the 8-byte form const is CONST9. The
    decoder, not a byte reading, is the source, so a form nobody has mapped is still read right.
    `long_form` is accepted for old callers and ignored."""
    e, = first_operand_exprs([bytes(raw)])
    return e[1] // 2 if e and e[0] == UNIFORM_BASE else None


def uniform_writes(constant_program_instructions):
    """[(instruction, uniform)] for every instruction that writes a uniform: first operand an
    op0-based expression (op592 in both forms and the ALU uniform-destination forms)."""
    ins = [i for i in constant_program_instructions
           if i.opcode and i.values and i.values[0][0] == "expr"]
    exprs = first_operand_exprs([i.raw for i in ins])
    return [(i, e[1] // 2) for i, e in zip(ins, exprs) if e and e[0] == UNIFORM_BASE]


def code_facts(constant_program_instructions, registers=None):
    """What the constant program's own bytes determine: argument reads, uniform writes (by every
    writer, `writers` counting them by opcode), the high-water mark, registers."""
    from collections import Counter
    from agxforge.g17 import model, registerdomain
    names = registers or model.registers()
    ins = [i for i in constant_program_instructions if i.opcode]
    regs = [x for i in ins for k, v in i.values if k == "reg"
            for x in registerdomain.registers_in_name(names.get(v, ""))]
    writes = uniform_writes(ins)
    return dict(argument_reads=sum(1 for i in ins if i.opcode.id == ARGUMENT_READ),
                uniform_writes=len(writes),
                writers=dict(Counter(i.opcode.id for i, _u in writes)),
                high_water=max(u for _i, u in writes) + 1 if writes else None,
                registers=max(regs, default=-1) + 1)


def slot31(argument_map):
    """Per-kernel slot 31's value (S31): 16 + 16 per started pair of buffers the map lists."""
    buffers = len(argument_map) // 4
    return 16 + 16 * ((buffers + 1) // 2)


def argument_map(buffers):
    """The kind-3 record's argument map for the buffers the constant program reads (L5)."""
    return [4 * b + w for b in sorted(buffers) for w in range(4)]


def predict(facts, buffers, main_uniforms=0):
    """The metadata fields the constant program determines (L1-L5, S31, S1), from its code facts,
    the buffers it reads and kind-6 field 2 (`main_uniforms`, the main program's; 0 when the record
    is absent). Returns the values Apple's serializer writes; slot31 is None when absent, and
    kind6_field3 and slot1 are None when the program writes no uniform (the main program's floor
    decides them)."""
    writes = facts["uniform_writes"]
    amap = argument_map(buffers) if writes else []
    if writes and 2 * len(set(buffers)) != facts["argument_reads"]:
        raise ValueError("argument reads (%d) are not two per listed buffer (%s)"
                         % (facts["argument_reads"], sorted(buffers)))
    hw = facts["high_water"]
    return dict(slot31=slot31(amap) if writes else None, kind3_field0=8 + 4 * len(amap),
                kind3_field2=facts["registers"] if writes else 1, kind6_field3=hw,
                slot1=None if hw is None else main_uniforms + hw, argument_map=amap)


def metadata_facts(md):
    """What Apple's metadata states about the constant program: per-kernel slot 31 and slot 1, the
    kind-3 record's field 2 and argument map (the vector after its body, count at record + 20), and
    kind-6 fields 2 and 3 (None when the record is absent)."""
    import struct
    from agxforge.g17 import mdgen
    desc = mdgen.describe(md)
    fields = {int(p): (t, {int(k): v[2] for k, v in t["fields"].items()}) for p, t in desc["tables"].items()}
    pk = next(f for t, f in fields.values() if t["vlen"] >= 60)
    k3 = next(p for p, (t, _f) in fields.items() if (t["vlen"], t["tlen"]) == (12, 20))
    n = struct.unpack_from("<I", md, k3 + 20)[0]
    kind6 = next(((f.get(2), f.get(3)) for t, f in fields.values()
                  if (t["vlen"], t["tlen"]) == (12, 16) and f.get(0) == 6), None)
    return dict(slot31=pk.get(31), slot1=pk.get(1), kind3_field2=fields[k3][1].get(2),
                argument_map=list(struct.unpack_from("<%dI" % n, md, k3 + 24)), kind6=kind6)
