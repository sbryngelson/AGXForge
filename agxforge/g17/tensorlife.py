"""A decode-level check on tensor bodies: no MMA may read an A or B register that an earlier MMA
released (lifetime operand 16) unless something wrote that register in between.

tlower's release rule was once per accumulator group, and a register-fed A (never reloaded) was
released by one group and read by the next: Set C's stages run found the tile all zero. This scans
the emitted bytes, so it does not share tlower's reasoning about what is live.
"""
from agxforge.g17 import model

# Runtime latches compare the counter with a register-held cap, unlike the
# constant latch's eight-bit immediate. The decoded-byte proof below checks
# the runtime cap explicitly. Stationary-B trials hold B at one address,
# so larger bounds do not also expand the GPU memory footprint.
TENSOR_LOOP_MAX_RUNTIME_TRIPS = 8388608

MMA = {5098, 5099, 5100, 5101, 5104, 5105, 5106, 5107, 10384, 10385}
RELEASE = 16
_NAMES = model.registers()


def _regs(name):
    """'R72_R73' -> {72, 73}; 'R12L' -> {12}; system registers -> empty."""
    out = set()
    for part in str(name).split("_"):
        digits = part[1:].rstrip("LH") if part.startswith("R") else ""
        if digits.isdigit():
            out.add(int(digits))
    return out


def released_reads(code):
    """[(instruction index, register)] for every MMA operand read of a released, unwritten register."""
    released, bad = set(), []
    for j, ins in enumerate(model.decode(code, 0)):
        if ins.opcode is None:
            continue
        vals = list(ins.values)
        regs = [(i, _NAMES.get(v, v)) for i, (k, v) in enumerate(vals) if k == "reg"]
        if ins.opcode.id in MMA:
            # operands: D, then A with (lifetime, width), then B with (lifetime, width)[, C ...]
            # every read of the instruction happens before its releases: Apple's own code reads one
            # group as both A and B and releases it at both (ac2-32x32x32 and four others)
            for pos, name in regs[1:3]:
                bad.extend((j, r) for r in sorted(_regs(name) & released))
            for pos, name in regs[1:3]:
                if vals[pos + 1][1] == RELEASE:
                    released |= _regs(name)
        if regs:                                     # the first register operand is the destination
            released -= _regs(regs[0][1])
    return bad


# STORE FORMS (Piece B's list, cc's duplicate-operand keep is scoped to the same set).
STORES = {17229, 17235, 17256, 17257, 17258, 13075, 17193, 17199}


def aliased_store_releases(code):
    """[(instruction index, register)] for every store that names its VALUE register again in a later
    register slot (the index) and releases it there.

    Measured on op17229 (MM 25.117, isa/g17-execution-sr-latency-results.json): decoded values
    [R16, 16, 2066, expr, 0, R16, 16] - value R16, index R16, index lifetime 16 - stored 0 on every
    lane at 1 and 8,192 threadgroups; clearing ONLY the index slot's release stored R16's value, and
    clearing only the value's did not. So the index slot's release clears the register before the
    value is read. Apple's same-register stores release neither slot.

    Scoped to stores on purpose: an ALU form naming one register in two released slots (op9700,
    R40 = f(R40[16], R34[32], R40[16], R34[32])) is bit-exact in four executed bodies, so a check on
    every repeated source would refuse programs proven correct."""
    bad = []
    for j, ins in enumerate(model.decode(code, 0)):
        if ins.opcode is None or ins.opcode.id not in STORES:
            continue
        vals = list(ins.values)
        regs = [(i, _NAMES.get(v, v)) for i, (k, v) in enumerate(vals) if k == "reg"]
        if not regs:
            continue
        value = _regs(regs[0][1])
        for pos, name in regs[1:]:
            life = vals[pos + 1][1] if pos + 1 < len(vals) and vals[pos + 1][0] == "imm" else None
            if life == RELEASE:
                bad.extend((j, r) for r in sorted(_regs(name) & value))
    return bad


# THE COUNTED KEY-BLOCK LOOP'S STATIC LATCH CHECK (MM 25.114.5; the discipline of 25.116). A loop whose
# counter is released, or rewritten by anything but its increment, never exits: that ran away on
# 2026-09-24, its host was killed, the kernel kept the GPU at 100% and only a reboot cleared it. So a
# tensor loop is checked from its DECODED BYTES before anything may dispatch it, not from what the
# compiler meant. The shape is cc's counted-loop lowering, the one every cc constant-bound loop emits:
#     cnt = 0 ... [body] ... cnt = cnt + 1 (op10279) ; pop (op577) ; FLAG = cnt < TRIPS (op10369) ;
#     op582 FLAG ; back edge (op458)
LOOP_CONTROL = {577, 578, 579, 582, 450, 458, 10369, 10370}
_LT = 9                    # op10369's relation value for `lt`, as cc emits `i + 1 < N` in every counted loop


def loop_carried_releases(code):
    """[(loop first index, instruction index, register)]: a register a loop CARRIES - read in the body
    before the body writes it - released (a source lifetime of 16) after the body's last write of it.
    Per section 25.95 a release clears the lane's copy, so the next trip reads 0 (the generic form of
    g17residency.static_checks' counter refusal; tools/g17ccfuzz.latch_releases is the same rule)."""
    from agxforge.g17 import tensorview as TV
    v = TV.view(code)
    ins = list(model.decode(code, 0))
    out = []
    for first, back in TV.loops(v):
        written, carried, last_write = set(), set(), {}
        for n in range(first, back + 1):
            carried |= v[n].uses - written
            for r in v[n].defs:
                written.add(r)
                last_write[r] = n
        for n in range(first, back + 1):
            if not ins[n].opcode:
                continue
            vals = list(ins[n].values)
            regs = [(k, x) for k, (kind, x) in enumerate(vals) if kind == "reg"]
            for k, x in (regs if v[n].kind == "store" else regs[1:]):
                if not (k + 1 < len(vals) and vals[k + 1] == ("imm", RELEASE)):
                    continue
                hit = TV._regs(x) & carried
                if hit and n > max(last_write.get(h, -1) for h in hit):
                    out.append((first, n, str(_NAMES.get(x, x))))
    return out


def counted_loop_check(code, trips, carried=(), runtime=False):
    """Refuse (ValueError 'refused: ...') a tensor loop program that cannot be proved, from its bytes,
    to run exactly `trips` trips and exit; else a dict describing what was proved.

      1. exactly one back edge, landing on an earlier instruction (one counted loop, not nested);
      2. the body's control forms are exactly the tail's pop, compare, op582 and back edge. A
         constant latch uses kept `cnt < trips` with 1 <= trips <= 255 (op10369's 8-bit immediate);
         a runtime latch proves `cnt < n && cnt < cap` from the emitted bytes, with cap <= 8388608;
      3. the counter is named in the body ONLY by its kept increment `cnt = cnt + 1` (just before the
         tail) and the compare;
      4. the last instruction before the loop that names the counter sets it to 0 (op11842);
      5. `carried` (the stream's B index registers): written in the body only by a self-add
         (op10282 r = r + s, or op10279 r = r + imm), and never released anywhere in the program;
      6. no carried register released after the body's last write of it (loop_carried_releases);
      7. no unwaited load read, across the back edge too (tensorview.hazards), no released MMA
         operand read and no aliased store release."""
    from agxforge.g17 import tensorview as TV
    max_trips = TENSOR_LOOP_MAX_RUNTIME_TRIPS if runtime else 255
    if not isinstance(trips, int) or isinstance(trips, bool) or not 1 <= trips <= max_trips:
        raise ValueError("refused: %r trips; %s" %
                         (trips, "the runtime tensor-loop cap is 1..%d" % max_trips if runtime
                          else "the compare's immediate is 8 bits, 1..255"))
    v = TV.view(code)
    ins = list(model.decode(code, 0))
    backs = [i.index for i in v if i.opcode == 458]
    spans = TV.loops(v)
    if len(backs) != 1 or len(spans) != 1 or spans[0][1] != backs[0]:
        raise ValueError("refused: %d back edges and %d counted loops; the key-block loop is exactly one"
                         % (len(backs), len(spans)))
    first, back = spans[0]
    ctrl = [n for n in range(first, back + 1) if v[n].opcode in LOOP_CONTROL]
    if [v[n].opcode for n in ctrl] != [577, 10369, 582, 458] or ctrl != list(range(back - 3, back + 1)):
        raise ValueError("refused: the loop body's control forms are %s, not the counted tail pop, compare, "
                         "op582, back edge" % [(n, v[n].opcode) for n in ctrl])
    if runtime:
        cnt, inc, creg_name, bound_reg = _runtime_tail(v, ins, first, back, trips)
    else:
        cvals = list(ins[back - 2].values)
        creg = [(k, x) for k, (kind, x) in enumerate(cvals) if kind == "reg"]
        if (len(creg) != 2 or cvals[2] != ("imm", _LT) or cvals[-1] != ("imm", trips) or
                cvals[creg[1][0] + 1] != ("imm", 0)):
            raise ValueError("refused: the trip compare is %s, not a kept `cnt < %d`" % (cvals, trips))
        cnt = TV._regs(creg[1][1])
        creg_name = creg[1][1]
        inc = back - 4
        ivals = list(ins[inc].values) if ins[inc].opcode else []
        if (v[inc].opcode != 10279 or v[inc].defs != cnt or v[inc].uses != cnt or len(ivals) != 5 or
                ivals[2] != ("imm", 1) or ivals[4] != ("imm", 0)):
            raise ValueError("refused: the instruction before the tail is %s, not the kept increment cnt = cnt + 1"
                             % (ivals,))
        naming = [n for n in range(first, back + 1) if (v[n].defs | v[n].uses) & cnt]
        if naming != [inc, back - 2]:
            raise ValueError("refused: the counter %s is named in the body at %s, not only by its increment and "
                             "the compare" % (creg[1][1], [(n, v[n].opcode) for n in naming]))
    before = [n for n in range(first) if (v[n].defs | v[n].uses) & cnt]
    if not before:
        raise ValueError("refused: nothing sets the counter before the loop")
    init = before[-1]
    ivals = list(ins[init].values)
    if v[init].opcode != 11842 or v[init].defs != cnt or ivals[-1] != ("imm", 0):
        raise ValueError("refused: the counter's last mention before the loop is %s, not cnt = 0" % (ivals,))
    advances = {}
    for r in carried:
        halves = TV._regs("R%d" % r)
        for n in range(first, back + 1):
            if not v[n].defs & halves:
                continue
            regs = [x for kind, x in ins[n].values if kind == "reg"]
            self_add = (v[n].opcode in (10282, 10279) and v[n].defs == halves and len(regs) > 1 and
                        TV._regs(regs[1]) == halves)
            if not self_add:
                raise ValueError("refused: carried register R%d is written in the body by op%s, not a self-add"
                                 % (r, v[n].opcode))
            advances[r] = advances.get(r, 0) + 1
        for i in ins:
            if not i.opcode:
                continue
            vals = list(i.values)
            for k, (kind, x) in enumerate(vals):
                if (kind == "reg" and k > 0 and TV._regs(x) & halves and k + 1 < len(vals) and
                        vals[k + 1] == ("imm", RELEASE)):
                    raise ValueError("refused: op%d releases the stream index register R%d" % (i.opcode.id, r))
    rel = loop_carried_releases(code)
    if rel:
        raise ValueError("refused: loop-carried registers released after their last write: %s" % rel[:4])
    hz_all = TV.hazards(v)
    hz = [h for h in hz_all if not _sr_slot0_read(v, h)]
    if hz:
        raise ValueError("refused: %d unwaited load reads, first %s" % (len(hz), hz[:2]))
    if released_reads(code) or aliased_store_releases(code):
        raise ValueError("refused: a released MMA operand read or an aliased store release")
    out = dict(back_edges=1, trips=trips, body_instructions=back - first + 1,
               body_bytes=v[back].offset + len(v[back].raw) - v[first].offset,
               counter=str(_NAMES.get(creg_name, creg_name)), counter_init_index=init,
               advances={"R%d" % r: advances.get(r, 0) for r in carried}, hazards=0, latch_releases=0)
    if len(hz_all) > len(hz):
        out["sr_slot0_reads_admitted"] = len(hz_all) - len(hz)
    if runtime:
        out.update(runtime=True, cap=trips, bound=str(_NAMES.get(bound_reg, bound_reg)))
    return out


_GT = 10                   # op10369's relation value for `gt`: the lowered runtime latch is `p > 0`
ICMP = 11372               # the value compare (0 or 1) the runtime latch computes its predicate with
MOVIMM = 11842
AND = 424


def _sr_slot0_read(v, hazard):
    """True when `hazard` (tensorview.hazards' (index, 'reads rNX before slot S ...')) reads a register whose
    latest write before the read is a system-register read published on slot 0. MM 25.117 / 25.121: such a value
    is correct at distance 0 and the hardware does not need the wait (Apple's wait there is a zero-cost uniform
    policy); tools/g17ccfuzz.py admits the same class (--admit-sr-hazards). Anything else stays a hazard."""
    from agxforge.g17 import tensorview as TV
    import re
    idx, what = hazard
    m = re.match(r"reads r(\d+)([LH]) before slot (\d+)", what)
    if not m or int(m.group(3)) != 0:
        return False
    half = 2 * int(m.group(1)) + (1 if m.group(2) == "H" else 0)
    for n in range(idx - 1, -1, -1):
        if half in v[n].defs:
            return v[n].opcode in TV.SR and v[n].fills == 0
    return False


def _last_def(v, reg, lo, hi):
    """Index of the last instruction in [lo, hi) that writes any half of `reg`, or None."""
    for n in range(hi - 1, lo - 1, -1):
        if v[n].defs & reg:
            return n
    return None


def _runtime_tail(v, ins, first, back, cap):
    """The capped runtime latch, proved from the bytes (MM 25.144.8). cc lowers `nxt < n, cap=K` to

        cnt = cnt + 1            op10279, kept: the counter's only write in the body
        p1  = icmp.ult(cnt, n)   op11372
        k   = K                  op11842 (in the body, before its compare)
        p2  = icmp.ult(cnt, k)   op11372
        p   = p1 & p2            op424, just before the tail
        pop; cmp p > 0; op582; back edge

    TERMINATION needs only the counter, the cap and the and: cnt rises by exactly 1 per trip from 0
    and the loop continues only while cnt < K, so it ends within K trips WHATEVER n is. n matters for
    uniformity, which cc proves in the IR (_simdgroup_uniform) before it compiles the loop. From the
    bytes, n is either kept from before the loop (then never written or released in the body) or
    reloaded in the body before its compare every trip. -> (counter halves, increment index,
    counter name, bound name)."""
    from agxforge.g17 import tensorview as TV
    cvals = list(ins[back - 2].values)
    creg = [(k, x) for k, (kind, x) in enumerate(cvals) if kind == "reg"]
    if len(creg) != 2 or cvals[2] != ("imm", _GT) or cvals[-1] != ("imm", 0):
        raise ValueError("refused: the runtime latch compare is %s, not `p > 0`" % (cvals,))
    p = TV._regs(creg[1][1])
    a = back - 4
    if v[a].opcode != AND or v[a].defs != p:
        raise ValueError("refused: the predicate is not an and of two compares (op%s)" % v[a].opcode)
    avals = list(ins[a].values)
    srcs = [TV._regs(x) for k, (kind, x) in enumerate(avals) if kind == "reg" and k > 0]
    if len(srcs) != 2:
        raise ValueError("refused: the predicate's and has %d register sources" % len(srcs))
    cmps = []
    for s in srcs:
        d = _last_def(v, s, first, a)
        if d is None or v[d].opcode != ICMP:
            raise ValueError("refused: a latch operand is not an icmp in the body (op%s)" % (v[d].opcode if d is not None else None))
        cmps.append(d)
    cnt = None
    kparts = []
    for d in cmps:
        vals = list(ins[d].values)
        regs = [(k, x) for k, (kind, x) in enumerate(vals) if kind == "reg"]
        if len(regs) != 3 or vals[2] != ("imm", _LT):
            raise ValueError("refused: the latch compare %s is not icmp.ult" % (vals,))
        c = TV._regs(regs[1][1])
        if cnt is None:
            cnt = c
        elif c != cnt:
            raise ValueError("refused: the two latch compares test different counters")
        kparts.append((d, TV._regs(regs[2][1]), regs[2][1], regs[2][0], vals))
    writes = [n for n in range(first, back + 1) if v[n].defs & cnt]
    if len(writes) != 1:
        raise ValueError("refused: the counter is written %d times in the body; the increment is its only write" % len(writes))
    inc = writes[0]
    ivals = list(ins[inc].values) if ins[inc].opcode else []
    if (v[inc].opcode != 10279 or len(ivals) != 5 or ivals[2] != ("imm", 1) or ivals[4] != ("imm", 0)
            or v[inc].defs != v[inc].uses or v[inc].defs != cnt or inc > min(cmps)):
        raise ValueError("refused: the counter's write is %s, not the kept cnt = cnt + 1 before the compares" % (ivals,))
    naming = [n for n in range(first, back + 1) if (v[n].defs | v[n].uses) & cnt]
    if naming != sorted([inc] + cmps):
        raise ValueError("refused: the counter is named in the body at %s, not only by its increment and "
                         "the two latch compares" % [(n, v[n].opcode) for n in naming])
    capped = bound = None
    for d, reg, name, k, vals in kparts:
        kd = _last_def(v, reg, first, d)
        if kd is not None and v[kd].opcode == MOVIMM and list(ins[kd].values)[-1] == ("imm", cap):
            if capped is not None:
                raise ValueError("refused: two cap compares")
            capped = d
            continue
        if bound is not None:
            raise ValueError("refused: the latch has no compare against the cap %d" % cap)
        bound = (d, reg, name, k, vals, kd)
    if capped is None or bound is None:
        raise ValueError("refused: the latch is not `cnt < n && cnt < %d` (no compare against the cap)" % cap)
    d, breg, bname, k, vals, kd = bound
    if kd is None:
        # KEPT from before the loop: never written in the body, never released in it
        for n in range(first, back + 1):
            nv = list(ins[n].values) if ins[n].opcode else []
            for j, (kind, x) in enumerate(nv):
                if kind == "reg" and j > 0 and TV._regs(x) & breg and k_released(nv, j):
                    raise ValueError("refused: the runtime bound %s is kept from before the loop but released "
                                     "inside it (op%s); the next trip would read a released register" % (bname, v[n].opcode))
    return cnt, inc, [x for kind, x in ins[inc].values if kind == "reg"][0], bname


def k_released(vals, k):
    """True when register operand k's lifetime modifier is RELEASE."""
    return k + 1 < len(vals) and vals[k + 1] == ("imm", RELEASE)
