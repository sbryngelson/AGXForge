"""A decoded, schedulable view of a tensor body, and the hazard check any reordering must pass.

tlower emits bytes, not an instruction list, so a scheduler has nothing to reorder until the bytes
are read back. This reads them the way tensorlife does - through the reference decoder, not
through tlower's own reasoning - and gives each instruction what a scheduler needs:

    kind       load / mma / store / sr / alu / ctrl
    defs/uses  architectural registers (a tuple operand expands to its members)
    fills      the scoreboard slot a load or system-register read publishes (operand 1 bits 20-23,
               slot + 1; memenc.load and tlower's `(slot + 1) << 20`)
    waits      the slot mask it waits on before issuing: operand 1 bits 24-31 on a load or an ALU
               word, the flags word's bits 24-30 plus bit 31 (slot 7) on an MMA (mmaenc.mma)

`hazards(view)` is the order check. A register a load wrote is LATE until an instruction at or
after the load waits on the load's slot; reading a late register is a race (machine model section
6: a consumer that does not name the slot reads zeros). A reordered body is admissible only if it
has no more hazards than the body it came from.

A SLOT COUNTS OUTSTANDING LOADS; it is not one load's. The first version also flagged a load
filling a slot whose previous load was unwaited, and it fired on all 31 executed tlower bodies:
every load of a K slice fills the same slot and one MMA wait covers them all (Apple does the same).
Refilling a slot is therefore not a race - one wait then covers both fills - and is not checked.

SCOPE, MEASURED AGAINST EXECUTED BODIES. Every retained tlower body (the generic, fp8, epilogue,
int8, grid and imageblock bundles - all dispatched bit-exact) has zero hazards, which is the
ground truth a guard must accept first. Bodies from other emitters (the encoder, FFN and
projection images) are flagged, at op17013 and op14391 readers: those ALU forms wait through a
field this view does not model, so their flags are this view's blind spot, not races. The
transforms here are for tlower bodies.
"""
from agxforge.g17 import model, tensorlife

MMA = tensorlife.MMA
LOADS = {12709, 12674, 12675, 12655, 12656, 12657, 12691, 12700, 12682}
STORES = {17256, 17257, 17258, 17229, 17235, 13075}
# ATOMICS THAT RETURN LATE. op10094 (uniform device) and op11765 (uniform threadgroup) publish their
# old value on the slot in operand 1 bits 20-23, as a load does: in Apple's corpus all 34 first
# consumers wait on exactly that slot, slot 1 included (tools/g17waitlaw.py census). cc read
# op10094's result without a wait until #171. The per-lane op14157 is NOT here: its bits 20-23 are
# 0 and none of its 19 consumers waits, so the reading would not transfer.
ATOMICS = {10094, 11765}
SR = {14060, 14059}
_NAMES = model.registers()


class Ins:
    __slots__ = ("index", "offset", "raw", "opcode", "kind", "defs", "uses", "fills", "waits")

    def __repr__(self):
        return "<%d op%s %s d%s u%s f%s w%s>" % (self.index, self.opcode, self.kind, sorted(self.defs),
                                                  sorted(self.uses), self.fills, bin(self.waits))


def _regs(name):
    """Register HALVES, as 2r (low) and 2r + 1 (high): a full register or tuple member covers both,
    'R2L' only the low. Merging halves made a read of R2H wait on the slot that filled R2L."""
    out = set()
    for part in str(_NAMES.get(name, name)).split("_"):
        if not part.startswith("R"):
            continue
        body = part[1:]
        if body.isdigit():
            out |= {2 * int(body), 2 * int(body) + 1}
        elif body[:-1].isdigit() and body[-1] in "LH":
            out.add(2 * int(body[:-1]) + (body[-1] == "H"))
    return out


def view(code):
    out, off = [], 0
    for j, ins in enumerate(model.decode(code, 0)):
        i = Ins()
        i.index, i.offset, i.raw = j, off, bytes(ins.raw)
        off += len(ins.raw)
        i.opcode = ins.opcode.id if ins.opcode else None
        vals = list(ins.values)
        regs = [(k, v) for k, v in enumerate(vals) if v[0] == "reg"]
        word = vals[1][1] if len(vals) > 1 and vals[1][0] == "imm" else 0
        i.fills, i.waits = None, 0
        if i.opcode in MMA:
            i.kind = "mma"
            i.waits = (word >> 24) & 0x7F | ((word >> 31) & 1) << 7
        elif i.opcode in LOADS or i.opcode in ATOMICS:
            i.kind = "load" if i.opcode in LOADS else "atomic"
            i.fills = ((word >> 20) & 0xF) - 1 if (word >> 20) & 0xF else None
            i.waits = (word >> 24) & 0xFF
        elif i.opcode in SR:
            i.kind = "sr"
            i.fills = ((word >> 20) & 0xF) - 1 if (word >> 20) & 0xF else None
        elif i.opcode in STORES:
            i.kind = "store"
            i.waits = (word >> 24) & 0xFF
        elif i.opcode is None or not regs:
            i.kind = "ctrl"
        else:
            i.kind = "alu"
            # ASSUMED, NOT MEASURED PER FORM: bits 24-31 of operand 1 as tlower writes them (the fadd
            # of a C accumulate carries 0x80 there, op998 at 0x2280000000). A wrong reading here
            # could credit an ALU instruction with a wait it lacks and hide a race, so the
            # transforms built on this view move loads and MMAs only, never ALU instructions.
            i.waits = (word >> 24) & 0xFF
        if i.kind == "store":
            i.defs, i.uses = set(), set().union(*[_regs(v) for _k, (_t, v) in regs]) if regs else set()
        else:
            i.defs = _regs(regs[0][1][1]) if regs else set()
            i.uses = set().union(*[_regs(v) for _k, (_t, v) in regs[1:]]) if len(regs) > 1 else set()
        out.append(i)
    return out


def _step(i, late):
    """Advance the pending-register map over instruction i (its waits, then its writes)."""
    for s in range(8):
        if i.waits >> s & 1:
            for r in [r for r, t in late.items() if t == s]:
                del late[r]
    if i.fills is not None:
        for r in i.defs:
            late[r] = i.fills
    else:
        for r in i.defs:
            late.pop(r, None)


def entry_states(v):
    """{instruction index: pending map on entry}, with every counted loop iterated to a fixpoint:
    the state at a loop's back edge also flows into its first instruction. A linear scan misses
    that a loop's first reader can read what the previous iteration's last load wrote."""
    spans = loops(v) if any(i.opcode == 458 for i in v) else []
    head = {f: b for f, b in spans}
    extra = {}
    for _ in range(8):
        states, late = {}, {}
        for i in v:
            if i.index in extra:
                for r, t in extra[i.index].items():
                    late.setdefault(r, t)
            states[i.index] = dict(late)
            _step(i, late)
        grew = False
        for f, bk in spans:
            at_back = dict(states[bk])
            _step(v[bk], at_back)
            merged = dict(extra.get(f, {}))
            for r, t in at_back.items():
                if r not in merged:
                    merged[r] = t
                    grew = True
            extra[f] = merged
        if not grew:
            return states
    raise ValueError("pending-register propagation did not settle")


def hazards(v):
    """[(index, what)] for every read of a register whose load's slot has not been waited on -
    across loop back edges too."""
    states, bad = entry_states(v), []
    for i in v:
        late = dict(states[i.index])
        for s in range(8):
            if i.waits >> s & 1:
                for r in [r for r, t in late.items() if t == s]:
                    del late[r]
        for r in sorted(i.uses & set(late)):
            bad.append((i.index, "reads r%d%s before slot %d is waited on" % (r // 2, "LH"[r % 2], late[r])))
    return bad


def encode(v):
    return b"".join(i.raw for i in v)


# ---------------------------------------------------------------- measurement transforms
# rotate() rewrites a body for a TIMING question, not a correct answer. The re-encoders below it
# are not used by rotate (which only moves instructions); they are the verified path the real
# pipeliner needs to redirect a load's destination or rewrite an MMA's waits, and each first
# re-encodes an instruction with its OWN operands and must reproduce its original bytes, so a
# field misread fails there, before any dispatch.

_TLOAD_FORM = {(12674, 12): "tload12", (12674, 16): "tload16"}


def loops(v):
    """[(first, backedge)] instruction indices of every counted loop: an op458 whose decoded
    ten-byte displacement lands on an instruction start."""
    from agxforge.g17 import asm
    start = {i.offset: i.index for i in v}
    out = []
    for i in v:
        if i.opcode == 458 and len(i.raw) == 10:
            tgt = i.offset + asm.decode_branch10(i.raw)
            if tgt in start and start[tgt] < i.index:
                out.append((start[tgt], i.index))
    return out


def _values(raw):
    ins = list(model.decode(raw + b"\x0e\x00\x00\x00", 0))[0]
    return {k: val for k, (_t, val) in enumerate(ins.values)}


def _reencode_load(i, dst=None):
    """Through memenc, whose operand numbers skip the decoder's expression operand: memenc's
    {0 dst, 1 word, 4 index, 5 index lifetime, 6 displacement, 7 width, 8 mask} are the decoder's
    positions {0, 1, 5, 6, 7, 8, 9}. The identity check is what proves that mapping."""
    from agxforge.g17 import memenc
    form = _TLOAD_FORM.get((i.opcode, len(i.raw)))
    if form is None:
        raise ValueError("no load re-encoder for op%d/%d" % (i.opcode, len(i.raw)))
    d = _values(i.raw)
    if form == "tload16" and d[1] >> 37 & 1:
        form = "tload16_end"
    vals = {0: d[0], 1: d[1], 4: d[5], 5: d[6], 6: d[7], 7: d[8], 8: d[9]}
    enc = lambda v: bytes(memenc.encode(form, v, binding=i.raw[1]))
    if enc(dict(vals)) != i.raw:
        raise ValueError("op%d at %d does not re-encode to itself" % (i.opcode, i.index))
    if dst is not None:
        vals[0] = dst
    return enc(vals)


def _reencode_mma(i, waits=None):
    from agxforge.g17 import mmaenc
    vals = _values(i.raw)
    enc = lambda v: bytes(mmaenc.encode(i.opcode, v))
    if enc(dict(vals)) != i.raw:
        raise ValueError("op%d at %d does not re-encode to itself" % (i.opcode, i.index))
    if waits is not None:
        w = vals[1] & ~(0xFF << 24)
        vals[1] = w | ((waits & 0x7F) << 24) | ((waits >> 7 & 1) << 31)
    return enc(vals)


def rotate(code):
    """The loop body with its MMAs moved ahead of its loads: the software-pipelined SHAPE.

    Each MMA still waits on its slot, but that slot now holds the loads issued at the end of the
    PREVIOUS iteration, which had a whole iteration to land; this iteration's loads then refill the
    registers behind the MMAs. So load latency overlaps MMA issue, and at most one iteration's
    loads are ever outstanding. The answer is NOT the GEMM (each MMA reads the previous slice's
    fragments, and the first reads slice 0's twice) - the arm exists to be timed against the body
    as emitted, and the difference bounds what overlapping loads with MMAs can win.

    A first version instead pointed the loads at spare registers and cleared every MMA wait; it
    was not dispatched, because nothing then waits on the loop's loads and hundreds of fills would
    stay outstanding on one slot, a state no measurement covers. Instructions move; none is
    re-encoded, and the back edge still lands on the loop's first byte (same body length)."""
    v = view(code)
    spans = loops(v)
    if not spans:
        raise ValueError("no counted loop in this body")
    raws = [i.raw for i in v]
    for first, back in spans:
        body = v[first:back]
        loads = [i for i in body if i.kind == "load"]
        mmas = [i for i in body if i.kind == "mma"]
        if not loads or not mmas or max(i.index for i in loads) > min(i.index for i in mmas):
            raise ValueError("loop at %d is not loads-then-MMAs; nothing to rotate" % first)
        moved = {i.index for i in mmas}
        order = mmas + [i for i in body if i.index not in moved]
        raws[first:back] = [i.raw for i in order]
    new = b"".join(raws)
    if len(new) != len(code) or loops(view(new)) != spans:
        raise ValueError("rotation moved a loop's boundary")
    return new


def _tuple_id(base, n):
    from agxforge.g17 import mmaenc
    return mmaenc.regid(base, n)


def add_waits(code):
    """Every instruction that reads a register still pending on a slot gets that slot added to its
    operand 1 bits 24-31 - the wait mask, which the corpus census shows those bits are on 126 forms
    (tools/g17waitlaw.py). Rewritten through the assembler text and decoded back: only operand 1
    may move. For MEASUREMENT programs whose chains must wait without the extra instruction a
    waiting copy costs (a pointer chase); the compiler's own rule is the waiting copy."""
    from agxforge.g17 import assembler as A
    import g17packedcheck as D
    for _ in range(4):
        v = view(code)
        states, out = entry_states(v), []
        for i in v:
            late = states[i.index]
            need = 0
            for r in i.uses:
                if r in late:
                    need |= 1 << late[r]
            raw = i.raw
            if need & ~i.waits:
                before = D.decode(raw + b"\x0e\x00\x00\x00")[0][3]
                old = int(before[1].split(":")[1])
                new = old | ((need & 0xFF) << 24)
                line = [x for x in A.render(raw, [(0, len(raw), i.opcode)]).splitlines()
                        if ("@%d" % i.opcode) in x or x.strip().startswith(("load", "store"))]
                if not line:
                    raise ValueError("op%d at %d does not render" % (i.opcode, i.index))
                text = line[0].split(" / ")[0].strip().replace("#%d" % old, "#%d" % new, 1)
                raw = bytes(A.assemble(text).text)[:len(i.raw)]
                after = D.decode(raw + b"\x0e\x00\x00\x00")[0][3]
                if after[1] != "imm:%d" % new or [x for k, x in enumerate(before) if k != 1] != \
                        [x for k, x in enumerate(after) if k != 1]:
                    raise ValueError("op%d at %d: the wait could not be written alone" % (i.opcode, i.index))
            out.append(raw)
        new_code = b"".join(out)
        if new_code == code:
            break
        code = new_code
    out = [code]
    if hazards(view(code)):
        raise ValueError("waits added, hazards remain: %s" % hazards(view(code))[:3])
    return code
