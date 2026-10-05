#!/usr/bin/env python3
"""A released register must not be read again: checked on EMITTED G17 bytes, across loops and masks.

WHY. A source whose lifetime operand says RELEASE (16) frees that lane's copy of the register; a later
read of it returns 0 - sometimes, not always, so one passing hardware check proves nothing. cc shipped
this twice, both times through liveness that did not see a loop edge:

  * f453622af (MM 25.141.4): build_rmsnorm_wide's scale-loop phi started from the thread index t and took
    t's register; t's last mention BEFORE the loop released it, so the loop's first trip read 0 and every
    thread stored element 0. It passed one bit-exact check.
  * a64e5b34e: a runtime-bound latch's cap compare released the loop counter; next trip read 0 and the
    loop never exited on the GPU.

`asm.read_after_release` already checks this on delivered bytes, for straight-line programs over 13
measured forms, and refuses anything with a back edge. This is the same question over the CONTROL FLOW:

  THE MODEL IS ONE LANE. Measured (isa/g17-execution-release-mask-results.json, tools/g17releasemask.py):
  a release under a partial mask zeroes only the lanes that executed it. So a lane's path through the
  program decides everything: an instruction the lane executes may release, write and read; one it does
  not execute does nothing to it. The analysis walks every path a lane can take:
    - the lane's execution counter (active iff 0), stepped by the exec forms as _step_mask documents;
      a condition is a per-lane FLAG value, chosen once per path and remembered until a compare rewrites
      it, and equality compares against an immediate decide later ones on the same unchanged register
      (_decide); `FLAGTRUE` always holds;
    - branches: a back edge (op458) and op450 may or may not be taken; a forward op462 is taken only
      when NO lane is active, so only by a path on which this lane is inactive; op684 ends the lane;
    - state: the set of register HALVES that may be released on some path to here, with where they
      were released. Joined by union and iterated to a fixed point, so a value released after its
      last mention in the body reaches the loop's next trip (memory: linear liveness is wrong for
      loops).

  WHAT COUNTS AS A READ, A RELEASE AND A WRITE (register halves, as tensorview._regs: 2r low, 2r+1
  high; a tuple operand covers its members):
    - release: a register operand k whose lifetime operand is k + 1 per the certified field map
      (auth.lifetime_operand) and whose value has bit 4 set and bit 5 clear (16/18/20, with any high
      index-family bits);
    - read: register operands k >= 1, and operand 0 of a store (tensorview.STORES);
    - write: operand 0 of every other form;
    - after an executed instruction: released = (released | its releases) - its writes (a destination
      may be the register a source just released: the write makes a new value).
  NO FALSE POSITIVE BY CONSTRUCTION WHERE UNSURE. An instruction the decoder cannot name writes every
  register it mentions (ends any release) and reads none; a form with no certified lifetime operand
  releases nothing. Both only hide findings, never invent one, and both are COUNTED in the result, so a
  clean answer states how much it could not see.

  WHAT IS A FINDING, AND WHAT IS ONLY RECORDED (check() documents each):
    - finding: an active lane reads an operand EVERY half of which it released on a path along which
      it stayed active from the release to the read;
    - partial: only some halves released (Apple's zero-extension idiom: a half released, the other live);
    - conditional: the path also needs the lane to have been inactive in between (to have skipped a
      definition), which only data can rule out;
    - mma_accumulator: a simdgroup MMA's accumulator (Apple zero-initialises it by reading it released);
    - cross-lane: simd.shuffle's value operand is another lane's register and is not a read here.

GROUND TRUTH FIRST (memory: a guard must first accept ground truth). Every program in
isa/g17-corpus-programs.jsonl is Apple's compiler output, so every finding there is a false positive of
this checker by construction. `--corpus` runs it over all 6,594 of them; test_g17releasecheckcorpus
requires zero. Each rule above was added for a named false positive there, in this order: the 6-byte
op582 (cf-loop2), the execution counter (338 do-while programs), flag values per path (mx-precise.pow),
whole-operand reads (b-swap, bhet, cm-simd2 and 14 more), simd.shuffle (setup_indirect_update_mapping),
the MMA accumulator (mm-f32.tg), straight paths (setup_indirect_update_mapping's flag-guarded reload)
and equality facts (blit_fast_clear's switch). test_g17releasecheck shows the check still fires on
both cc bugs with all of them in place.

    python3 tools/g17releasecheck.py --corpus            the Apple ground truth
    python3 tools/g17releasecheck.py --builders          every shipped cc builder this knows
    python3 tools/g17releasecheck.py --reproduce         the two fixed bugs, rebuilt on the unfixed cc
"""
import heapq

EXEC = {582: "if", 583: "if", 575: "else", 576: "else", 577: "pop", 578: "while", 579: "while"}
BACK, FWD_NONE_ACTIVE, OTHER_BRANCH, END = 458, 462, 450, 684
ALWAYS_TRUE_PREDS = ("FLAGTRUE",)   # cf.PRED_TABLE entries 6-7 as the decoder names them: Apple's unconditional push
MAX_DEPTH = 64              # a counter bound; Apple nests 6 levels in its deepest probe


class Ins:
    __slots__ = ("index", "offset", "raw", "opcode", "read_ops", "releases", "writes", "flag_write", "eq_origin",
                 "exec", "target", "unknown")


# EQUALITY COMPARES AGAINST AN IMMEDIATE decide later ones on the same unchanged register. The flag-compare
# condition code is `8 | signed << 2 | gt << 1 | lt` (ledger/g17-all-ten-relations-from-three-codes.toml,
# measured on execution), so 8 and 12 are integer equality. Apple's blit_fast_clear kernels are a switch:
# one region under `R4H == 1`, the next under `R4H == 2`, R4H unchanged between them. Taken as independent
# conditions, a lane could be in both, and a register released in the first region read "straight" in the
# second - five of the last false positives. Knowing R == 1 on a path decides R == 2 as false.
FLAG_COMPARES = {10369, 10370, 10372, 10378, 10381}
EQUALITY_CODES = (8, 12)


# CROSS-LANE OPERANDS: the register named is read in ANOTHER lane, so this lane's release state says nothing
# about it. simd.shuffle's operand 2 is the value it fetches from the source lane; its index is per-lane
# and stays checked. Found on Apple's code (setup_indirect_update_mapping: an atomic's old value released
# by the lane that fetched it, then broadcast to the others by op14158).
CROSS_LANE_OPERANDS = {14158: {2}}
# THE ACCUMULATOR OF A SIMDGROUP MMA (its last register source, the same tuple as its destination) is a
# cooperative read, and Apple zero-initialises it by reading it RELEASED: mm-f32.tg-1/2/4 release R0 and
# R1 (address arithmetic) and then issue op2842 with C = R0_R1. Recorded as `mma_accumulator`, never a
# finding. The A and B operands stay checked: tlower's released register-fed A was a real bug (memory:
# a release scoped to a group needs a reload).
SIMDGROUP_MMA = {838, 2842, 2862, 2902, 2905}


def _halves_of(reg, names):
    from agxforge.g17 import tensorview as TV
    return TV._regs(names.get(reg, reg) if isinstance(reg, int) else reg)


_LIFE = {}


# LIFETIME OPERANDS MEASURED ON HARDWARE that the certified field map does not mark (its bits 4/5 are not both
# mapped): op10094's value (operand 8) carries its lifetime in operand 9 - pinned at 16, a second uniform atomic
# sharing the value read 0 (MM 25.144.6, fuzz seed 5012).
MEASURED_LIFETIME = {(10094, 8): 9}


def _lifetime_at(opcode, k):
    """auth.lifetime_operand (or MEASURED_LIFETIME), memoised for the process."""
    from agxforge.g17 import auth
    key = (opcode, k)
    if key not in _LIFE:
        if key in MEASURED_LIFETIME:
            _LIFE[key] = MEASURED_LIFETIME[key]
            return _LIFE[key]
        try:
            _LIFE[key] = auth.lifetime_operand(opcode, k)
        except Exception:
            _LIFE[key] = None
    return _LIFE[key]


def decode_many(codes, chunk=400):
    """[decoded instruction list per program], decoding CHUNKS of programs as one byte stream: the reference
    decoder is a subprocess, one per call, and that was most of a corpus run. Instructions are self-
    delimiting, so the concatenated stream decodes to the programs' instructions back to back; a program
    whose instructions do not end exactly at its boundary is decoded again on its own."""
    from agxforge.g17 import model
    out = []
    for c0 in range(0, len(codes), chunk):
        part = codes[c0:c0 + chunk]
        dec = list(model.decode(b"".join(part), 0))
        k, pos = 0, 0
        for code in part:
            got, n = [], 0
            while k < len(dec) and n < len(code):
                got.append(dec[k]); n += len(dec[k].raw); k += 1
            if n != len(code):
                # misaligned: resynchronise the stream at the next program and decode this one alone
                got = list(model.decode(code, 0))
                pos += len(code)
                acc = 0
                k = 0
                while k < len(dec) and acc < pos:
                    acc += len(dec[k].raw); k += 1
                if acc != pos:
                    raise ValueError("batch decode lost the program boundary")
            else:
                pos += len(code)
            out.append(got)
    return out


def decode(code, decoded=None):
    """[Ins] for the program bytes `code` (entry at 0); `decoded` is its model.decode list if already made."""
    from agxforge.g17 import model, asm, tensorview as TV
    names = model.registers()
    out, off = [], 0
    lifetime_at = _lifetime_at

    for j, d in enumerate(decoded if decoded is not None else model.decode(code, 0)):
        i = Ins()
        i.index, i.offset, i.raw = j, off, bytes(d.raw)
        off += len(d.raw)
        i.opcode = d.opcode.id if d.opcode else None
        i.read_ops, i.releases, i.writes, i.flag_write, i.eq_origin = [], set(), set(), None, None
        i.exec, i.target, i.unknown = None, None, False
        vals = list(d.values)
        regs = [(k, v) for k, (t, v) in enumerate(vals) if t == "reg"]
        if i.opcode is None:
            i.unknown = True
            for _k, v in regs:
                i.writes |= _halves_of(v, names)
            out.append(i)
            continue
        if i.opcode in EXEC:
            # READ FROM THE DECODED OPERANDS, NOT cf._decode: that reads the 4-byte form only, and Apple
            # also emits a 6-byte op582 (an unconditional two-level push, `FLAGTRUE`, count 2). Missing
            # it left every later pop misaligned (cf-loop2's false positive). Operands: [aux, pred, count]
            # for if/else/while, [aux, count] for pop.
            count = vals[-1][1] if vals and vals[-1][0] == "imm" else 1
            pred = next((names.get(v, v) for t, v in vals if t == "reg"), None)
            i.exec = (EXEC[i.opcode], count, pred, i.opcode in (583, 576, 579))
            out.append(i)
            continue
        if i.opcode in (BACK, FWD_NONE_ACTIVE, OTHER_BRANCH) and len(i.raw) == 10:
            i.target = i.offset + asm.decode_branch10(i.raw)
        store = i.opcode in TV.STORES
        for k, v in regs:
            nm = names.get(v, v)
            h = _halves_of(v, names)
            if k == 0 and not store:
                i.writes |= h
                if isinstance(nm, str) and nm.startswith("FLAG"):
                    i.flag_write = nm
                    if (i.opcode in FLAG_COMPARES and len(vals) == 6 and vals[2] == ("imm", vals[2][1])
                            and vals[2][1] in EQUALITY_CODES and vals[3][0] == "reg" and vals[5][0] == "imm"):
                        i.eq_origin = (frozenset(_halves_of(vals[3][1], names)), vals[5][1])
                continue
            if h and k not in CROSS_LANE_OPERANDS.get(i.opcode, ()):
                i.read_ops.append((k, frozenset(h)))
            la = lifetime_at(i.opcode, k)
            if la is not None and la < len(vals) and vals[la][0] == "imm" and (vals[la][1] & 0x30) == 0x10:
                i.releases |= h
        if i.opcode in SIMDGROUP_MMA and i.read_ops:
            k, h = i.read_ops[-1]
            i.read_ops[-1] = (("mma_accumulator", k), h)
        out.append(i)
    return out


def _decide(flags, pred):
    """[(value of `pred` on this path, flags after deciding it)]. `flags` is (values, facts): values maps a
    flag to True / False / ('eq', halves, constant) - an equality compare not yet decided; facts are
    (halves, constant, holds) this path has already established about unchanged registers."""
    values, facts = flags
    vd = dict(values)
    v = vd.get(pred)
    if isinstance(v, bool):
        return [(v, flags)]
    if isinstance(v, tuple):
        _eq, h, c = v
        for fh, fc, holds in facts:
            if fh == h and holds:
                return [(fc == c, flags)]
            if fh == h and fc == c and not holds:
                return [(False, flags)]
        out = []
        for b in (True, False):
            nv = tuple(sorted(dict(vd, **{pred: b}).items(), key=repr))
            nfacts = tuple(sorted(set(facts) | {(h, c, b)}, key=repr))
            out.append((b, (nv, nfacts)))
        return out
    return [(b, (tuple(sorted(dict(vd, **{pred: b}).items(), key=repr)), facts)) for b in (True, False)]


def _write_flags(flags, i):
    """Flags after an executed instruction: its flag write, and its register writes invalidating facts
    and undecided equality origins about the registers it changed."""
    values, facts = flags
    if not (i.flag_write or i.writes):
        return flags
    vd = dict(values)
    if i.flag_write:
        vd.pop(i.flag_write, None)
        if i.eq_origin is not None:
            vd[i.flag_write] = ("eq",) + i.eq_origin
    if i.writes:
        w = i.writes
        vd = {k: v for k, v in vd.items() if not (isinstance(v, tuple) and v[1] & w)}
        facts = tuple(f for f in facts if not (f[0] & w))
    return (tuple(sorted(vd.items(), key=repr)), facts)


def _step_mask(counter, flags, ex):
    """Successor (counter, flags) pairs after an exec instruction, for ONE lane. The lane is active iff
    its counter is 0 - the AGX per-lane nesting counter (Asahi/Mesa call it r0l), not a stack of levels.
    That model is what balances Apple's code: 338 corpus programs run `while 2 / back edge / pop 2`
    with no push at all (a do-while at top level), which a push/pop stack reads as popping levels
    that were never pushed.
        if n     active: stays 0 if the condition holds, else n;   inactive: += n
        else n   0 -> n;   n -> 0 if the condition holds;          anything else unchanged
        pop n    max(0, counter - n)
        while n  active and the condition fails -> n (waits for its `pop n`);  else unchanged
    `FLAGTRUE` (cf.PRED_TABLE entries 6-7) always holds; the inverting forms (583/576/579) negate it.

    A CONDITION IS A PER-LANE FLAG VALUE, AND IT IS REMEMBERED. A compare writes FLAGk; every exec that
    reads FLAGk before the next write sees the SAME value in this lane. Treating each exec's condition as
    independent walked paths no lane can take - active under FLAG0 and then again under NOT FLAG0 - and
    produced mx-precise.pow's false positives. `flags` holds the values this path has already chosen."""
    kind, count, pred, invert = ex
    if pred in ALWAYS_TRUE_PREDS:
        choices = [(not invert, flags)]
    elif pred is None:
        choices = [(True, flags)]
    else:
        choices = [(v != invert, f) for v, f in _decide(flags, pred)]

    def res(fn):
        return sorted({(fn(c), f) for c, f in choices})
    if kind == "if":
        if counter > 0:
            return [(min(counter + count, MAX_DEPTH), flags)]
        return res(lambda c: 0 if c else count)
    if kind == "pop":
        return [(max(0, counter - count), flags)]
    if kind == "else":
        if counter == 0:
            return [(count, flags)]
        if counter == count:
            return res(lambda c: 0 if c else count)
        return [(counter, flags)]
    if kind == "while":
        if counter == 0:
            return res(lambda c: 0 if c else count)
        return [(counter, flags)]
    return [(counter, flags)]


def _successors(i, at, n):
    out = [i.index + 1] if i.opcode != END and i.index + 1 < n else []
    if i.target is not None and i.target in at:
        out.append(at[i.target])
    return out


def _flag_liveness(ins, at):
    """{index: flags some exec may read at or after this instruction before a write of that flag}."""
    n = len(ins)
    live = [frozenset()] * n
    succ = [_successors(i, at, n) for i in ins]
    changed = True
    while changed:
        changed = False
        for i in reversed(ins):
            s = set()
            for j in succ[i.index]:
                s |= live[j]
            if i.flag_write:
                s.discard(i.flag_write)
            if i.exec and i.exec[2] and i.exec[2].startswith("FLAG") and i.exec[2] not in ALWAYS_TRUE_PREDS:
                s.add(i.exec[2])
            s = frozenset(s)
            if s != live[i.index]:
                live[i.index] = s
                changed = True
    return live


def _fact_liveness(ins, at):
    """{index: register-half sets H whose facts may still be consulted before H is written}. A fact is
    CONSULTED at the exec that reads the flag (after the compare that wrote it), so an exec on FLAGk keeps
    alive every H an equality compare writing FLAGk tests - pruning at the compare alone dropped the fact
    one step before the exec needed it."""
    n = len(ins)
    live = [frozenset()] * n
    by_flag = {}
    for i in ins:
        if i.eq_origin is not None:
            by_flag.setdefault(i.flag_write, set()).add(i.eq_origin[0])
    succ = [_successors(i, at, n) for i in ins]
    changed = True
    while changed:
        changed = False
        for i in reversed(ins):
            s = set()
            for j in succ[i.index]:
                s |= live[j]
            if i.writes:
                s = {h for h in s if not (h & i.writes)}
            if i.eq_origin is not None:
                s.add(i.eq_origin[0])
            if i.exec and i.exec[2] in by_flag:
                s |= by_flag[i.exec[2]]
            s = frozenset(s)
            if s != live[i.index]:
                live[i.index] = s
                changed = True
    return live


def check(code, decoded=None):
    """Released-register reads on some lane path. Returns a dict:
    findings  [(read offset, 'rNL/H', [release offsets])]: an instruction this lane executes reads a
              register operand ALL of whose halves may have been released on some path to it
    partial   the same where only SOME of the operand's halves are released - Apple's zero-extension
              idiom (a half released, the other half live, the whole register read), not reported as a
              finding (see WHOLE_OPERAND below)
    states    (instruction, counter, flags) states explored
    unknown   instructions the decoder could not name (their registers were treated as rewritten)

    WHOLE_OPERAND. Apple's compiler reads a register whose one half it released and whose other half is
    live, in 17 of the 27 corpus programs this first flagged (and16/shl/store/add/publish/funnel readers:
    b-swap releases R0L then shifts R0 whose high half it just wrote). A released half evidently reads
    as zero there and the idiom depends on it. Both cc bugs this exists for released a WHOLE register,
    so a finding requires every half the read operand covers to be released; the partial case is
    counted, not reported.

    STRAIGHT PATHS ONLY. A release reaches a read "straight" when the lane stays ACTIVE from the release
    to the read: it skipped nothing, so it really executes the release and then the read. If the lane
    went inactive in between, the path also assumes the lane skipped a definition, which only data can
    rule out - Apple's setup_indirect_update_mapping reloads a pointer under one flag inside a loop and
    reads it under another, and a lane that reads it has always loaded it on some earlier trip. Such
    reads are listed as `conditional`, not reported: this check reports only what it can show. Both cc
    bugs it exists for are straight (the wide norm's t released before the loop and read on the first
    trip; the latch's counter released in the body and read on the next)."""
    ins = decode(code, decoded)
    at = {i.offset: i.index for i in ins}
    n = len(ins)
    live_flags = _flag_liveness(ins, at)
    # an equality origin is worth remembering only for a register TWO OR MORE equality compares test (a
    # switch); every other origin is left undecided, so programs without one keep their state count
    from collections import Counter
    eq_regs = Counter(i.eq_origin[0] for i in ins if i.eq_origin is not None)
    for i in ins:
        if i.eq_origin is not None and eq_regs[i.eq_origin[0]] < 2:
            i.eq_origin = None
    live_facts = _fact_liveness(ins, at)
    start = (0, 0, ((), ()))
    # released: half -> (straight release offsets, conditional release offsets)
    state = {start: {}}
    # IN PROGRAM ORDER (a heap on the instruction index): a state is processed after everything that
    # flows into it along forward edges, so it is re-queued only by back edges. LIFO order re-walked the
    # 40 KB driver programs many times (126 s for one).
    work = [start]
    queued = {start}
    findings, partial, conditional, mma = {}, {}, {}, {}
    while work:
        key = heapq.heappop(work)
        queued.discard(key)
        idx, counter, flags = key
        rel = state[key]
        if idx >= len(ins):
            continue          # a successor past the last instruction (bytes that do not end in a return): exit
        i = ins[idx]
        active = counter == 0
        out = rel
        if active:
            for k, hs in i.read_ops:
                hit = hs & set(rel)
                if not hit:
                    continue
                if isinstance(k, tuple):
                    dest = mma
                elif hit != hs:
                    dest = partial
                elif all(rel[h][0] for h in hs):
                    dest = findings
                else:
                    dest = conditional
                for h in hit:
                    st, cd = rel[h]
                    dest.setdefault((i.offset, h), set()).update(st if dest is findings else st | cd)
            if i.releases or i.writes:
                out = dict(rel)
                for h in i.releases:
                    st, cd = out.get(h, (frozenset(), frozenset()))
                    out[h] = (st | {i.offset}, cd)
                for h in i.writes:
                    out.pop(h, None)
        elif rel:
            # this lane skips this instruction: every release it carries now depends on a skipped path
            out = {h: (frozenset(), st | cd) for h, (st, cd) in rel.items()}
        nf = _write_flags(flags, i) if active else flags
        if i.opcode == END:
            continue
        nexts = _step_mask(counter, nf, i.exec) if i.exec else [(counter, nf)]
        succ = [(idx + 1, c, f) for c, f in nexts if idx + 1 < n]
        if i.target is not None and i.target in at:
            t = at[i.target]
            if i.opcode != FWD_NONE_ACTIVE or not active:
                succ.append((t, counter, nf))
        # keep only the flag values some exec still reads before a rewrite: dead knowledge only
        # multiplies states (the 40 KB driver programs did not finish without this)
        succ = [(j, c, (tuple((k, v) for k, v in f[0] if k in live_flags[j]),
                         tuple(x for x in f[1] if x[0] in live_facts[j]))) for j, c, f in succ]
        for sk in succ:
            prev = state.get(sk)
            if prev is None:
                state[sk] = dict(out)
                heapq.heappush(work, sk)
                queued.add(sk)
                continue
            grew = False
            for h, (st, cd) in out.items():
                old = prev.get(h)
                if old is None or not (st <= old[0] and cd <= old[1]):
                    prev[h] = (st | old[0], cd | old[1]) if old else (st, cd)
                    grew = True
            if grew and sk not in queued:
                heapq.heappush(work, sk)
                queued.add(sk)

    def fmt(d):
        return [(o, "r%d%s" % (h // 2, "LH"[h % 2]), sorted(r)) for (o, h), r in sorted(d.items())]
    return dict(findings=fmt(findings), partial=fmt(partial), conditional=fmt(conditional),
                mma_accumulator=fmt(mma), states=len(state), instructions=n,
                unknown=sum(1 for i in ins if i.unknown))
