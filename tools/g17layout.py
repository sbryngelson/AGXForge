#!/usr/bin/env python3
"""Slot-aware field layout: what an opcode's bits encode, and what is still unexplained.

tools/g17fields.py correlates instruction bits against the operand VALUES Apple's decoder
reports, which are MCRegister ids. But the encoding does not carry an MCRegister id. It carries
a SLOT, slot = hardware_register * 2 + half, which the encode side established causally. A bit
carrying slot bit 3 will not match MCRegister bit 3, so every register-bearing opcode was being
under-credited: its slot bits looked unexplained when they are the best-understood bits it has.

On op10282, add reg,reg and one of the most common instructions in the corpus:

    explained against raw MCRegister values   14 bits
    explained against slot-encoded values     21 bits
    union                                     35 of 41 varying
    unexplained                               16 -> 6

This does not replace causal evidence and says nothing about what an instruction DOES. It says
which bits are accounted for under Apple's own description of its own encoding, which is the
authorability question: a form can be authored when nothing in it has to be inherited.

FROZEN IS NOT THE SAME AS SAFE, and conflating them caused a wrong answer on hardware. The encode
side authored the 4-byte register-register bitwise form with ZERO unexplained bits by the operand
test, executed it, and `and` returned the XOR of its operands. The operation selector byte2[2:0]
is CONSTANT within each opcode, so it never varies, never appears as unexplained, and was
inherited from a template that happened to be an xor.

So an opcode's frozen bits split into two categories that the operand test cannot separate:

    SELECTOR    constant within this opcode but DIFFERENT in a sibling opcode - a sibling being
                one with the same operand signature and length. A selector is semantic: the
                compiler must author it, and inheriting it from the wrong sibling's template
                produces a well-formed instruction that computes something else.
    STRUCTURAL  constant here AND identical in every sibling. Nothing distinguishes it, so
                copying it inherits no choice.

    python3 tools/g17layout.py 10282        one opcode
    python3 tools/g17layout.py --selectors  classify frozen bits against sibling opcodes
    python3 tools/g17layout.py --handoff    regenerate isa/g17-handoff.toml
"""
import collections, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "spike", "accel", "re"))
import g17target
import g17fields, g17model, g17slice

_NAME_RE = re.compile(r"R(\d+)([LH]?)$")

# Bits the encode side established CAUSALLY, by execution against preregistered values. They are
# authored from semantics and have no decoder operand to correlate with, so the operand test
# counts them as unexplained when they are the opposite: known, and known by stronger evidence
# than anything structural. Subtracting them gives NET RESIDUE - bits the compiler can neither
# author nor attribute, which is the number that actually gates authorability.
#
# Sources are ledger entries on the encode side: the ALU liveness bit, the operand-width bits,
# and the fused-shift scale whose omission silently doubled every operand for a while.
CAUSAL_FIELDS = {
    (0, 3): "load-use wait",
    (3, 1): "operand width, dest",
    (4, 3): "liveness marker",
    (8, 3): "operand width, operand B",
    (9, 7): "operand width, src1",
    (10, 0): "fused-shift scale",
    (10, 1): "fused-shift scale",
    (10, 2): "fused-shift scale",
    (11, 0): "fused-shift scale",
}

# Bits that were EXECUTED and found to change nothing. This is a different status from
# CAUSAL_FIELDS and the difference matters to a composer: a causal field must be AUTHORED from
# semantics, an inert bit may be written either way and the program is still correct.
#
# An inert bit is not an understood bit. byte4[5] correlates with the previous instruction at
# 93.5% across 283 distinct objects and does nothing when inverted, which is what compiler
# metadata looks like - Apple's scheduler writing a note to itself. It stops being an OBSTACLE to
# authorability, which is the only thing this table measures, and it stays an open question
# everywhere else.
INERT_FIELDS = {
    (4, 5): "measured inert - every program the encoder can execute was run with it inverted "
            "(ten ALU ops, reconvergence, predication at 16 widths, slot A at r125, the 16-bit "
            "truncation, the load-use hazard program), all correct. Positive control: byte0[3] "
            "cleared on that same hazard program returns 100 instead of 1100",
    # b0[5] and b6[5] are ONE bit written twice: equal in 7135 of 7135 op10279, 361 of 361
    # op11666, 248 of 248 op14391, 1203 of 1203 op17013, and 7019 of 7031 op10282 where the twelve
    # exceptions are exactly the twelve with b6[6] set. An instruction that separates them does not
    # occur, so a composer that writes one must write the other.
    (0, 5): "measured inert - inverted as a PAIR with b6[5] across the same battery, all correct",
    (6, 5): "measured inert - inverted as a PAIR with b0[5] across the same battery, all correct",
}



# FIELDS RECOVERED BY MUTATION. Bits that correlation cannot place because the corpus never varies
# the operand far enough - an immediate's bit 21 when every kernel uses small values - but which
# Apple's own decoder attributes when the bit is flipped and the operand read back.
#
# The criterion is strict, and op12691's b5[5], b5[6], b8[6] and b10[7] fail it: their deltas are
# 2^24 in some samples and 2^25 or 2^32 in others, so the attribution is not a single field and
# they stay unexplained. Only a UNANIMOUS verdict with ONE power-of-two delta across every sampled
# instance is credited.
#
# Keyed (opcode, form length) -> {(byte, bit): (operand, value bit)}. tools/g17bitprobe.py.
# CODE TABLES ESTABLISHED BY MUTATION. A field can select an operand value through a table rather
# than carry its bits, and where the corpus exercises only part of the table, correlation gets only
# part of the field. op11491's byte7[7:5] is the case: sweeping all eight codes through Apple's
# decoder gives the whole map, of which the corpus shows four.
#
#     0 -> 0x00000000   1 -> 0x01000000   2 -> 0x02000000   3 -> 0x04000000
#     4 -> 0x08000000   5 -> 0x10000000   6 -> 0x20000000   7 -> 0x03000000
#
# Codes 0..6 are zero and the one-hot bits 24..29; code 7 is bits 24 and 25 together. The corpus
# contains only 0, 2, 3 and 4, which is why byte7[6] - bit 1 of the code - was the last residual
# bit in the whole corpus.
MUTATION_CODE_FIELDS = {
    (11491, 10): {(7, 5), (7, 6), (7, 7)},   # operand 1 bits 24..31, table above
    # THE ATOMIC OPERATION, a four-bit code table shared by every member of the family. It is a
    # SELECTOR, not an operand, so no correlation against operand values can reach it - and the
    # solver was right to report it as blocked before the table existed. Now it is authorable:
    # isa/g17-atomic-family.toml has all sixteen combinations, of which eleven are op10094, one is
    # op10095, and four do not decode. The control is that the table reproduces all ten encodings
    # Apple's own compiler emits.
    (10094, 10): {(4, 5), (5, 3), (6, 3), (7, 7)},
    (10094, 12): {(4, 5), (5, 3), (6, 3), (7, 7)},
    (10095, 10): {(4, 5), (5, 3), (6, 3), (7, 7)},
    (10022, 10): {(4, 5), (5, 3), (6, 3), (7, 7)},
    (10022, 12): {(4, 5), (5, 3), (6, 3), (7, 7)},
    (10023, 10): {(4, 5), (5, 3), (6, 3), (7, 7)},
    # The 12-byte atomic form carries a fifth code bit. Flipping it moves operand 2 by +7 in three
    # instances, +1 in three and stops the decode in six - not a field bit, which is exactly what
    # a code-table entry looks like, and operand 2 is the operation code whose table is known.
    (10094, 12): {(4, 5), (5, 3), (6, 3), (7, 7), (11, 4)},
    # op9706's condition code. Flipping b14[6] moves operand 2 by -1 in eight instances, +1 in
    # three and +2 in two - a code, and the code space is the float comparison table established
    # by compilation in isa/g17-condition-codes.toml: eq 0, lt 1, gt 2, ge 5, le 6.
    (9706, 16): {(14, 6)},
}

MUTATION_FIELDS = {
    (592, 8): {(2, 6): (2, 30)},     # unanimous over all 7 corpus instances, delta exactly 2^30
    (595, 8): {(5, 5): (2, 21), (5, 6): (2, 22)},
    (2190, 4): {(3, 4): (6, 3), (3, 5): (6, 4)},
    (12652, 10): {(4, 4): (1, 21)},
    (12697, 14): {(4, 4): (1, 21), (6, 4): (1, 22)},
}


def slot(mcreg):
    """Hardware slot for an MCRegister id, or None if it is not a plain R register.

    R_n = MCRegister 105 + n, and the halves are their own ids. A slot addresses a 16-bit half:
    slot = n*2 for the low half, n*2+1 for the high half. A full 32-bit register names the pair
    and encodes as the low half's slot.
    """
    name = g17model.registers().get(mcreg, "")
    # A REGISTER TUPLE names its members, "R8_R9_R10_R11...", and the encoding cannot carry
    # twelve register numbers - it carries the BASE. Matching only the R<n> form left every
    # tuple-bearing opcode uncredited: op5106, the tensor MAC, has four tuple operands and all
    # four were being correlated against MCRegister table indices, which are allocation order and
    # encode nothing. Take the first member.
    m = _NAME_RE.match(name.split("_")[0]) if name else None
    if not m:
        return None
    n, half = int(m.group(1)), m.group(2)
    return n * 2 + (1 if half == "H" else 0)


_FLAG_RE = re.compile(r"FLAG(\d+)$")


def flag_index(mcreg):
    """Apple's COMPACT flag numbering, which is neither the MCRegister id nor the class index.

    FLAGR's member list runs FLAG0..FLAG14, FLAGTRUE, FLAGFALSE, so FLAGTRUE's class index is 15.
    The instructions encode it as 6, in three bits: Apple's own decoder resolves the field value 6
    to FLAGTRUE on op582, and op575 - which reads FLAGTRUE and nothing else - carries those three
    bits frozen at exactly 6. Two independent opcodes, one from variation and one from a constant.

    So a flag operand is encoded as a 3-bit compact index and no correlation against ids or slots
    can see it. FLAGFALSE is presumed 7 and does not occur in this corpus.
    """
    name = g17model.registers().get(mcreg, "")
    m = _FLAG_RE.match(name)
    if m:
        return int(m.group(1))
    return {"FLAGTRUE": 6, "FLAGFALSE": 7}.get(name)


def class_index(opcode, k, mcreg):
    """The CLASS-RELATIVE MEMBER INDEX of an operand's register, or None.

    A register operand is not always encoded as its slot. Apple's tuple classes come in an
    overlapping form and an ALIGNED form, and the tensor MAC uses the aligned ones: op5106's tiles
    are GPR32tup4_alignedrc, whose member N is R(4N)..R(4N+3), and its accumulator is
    GPR32tup8_alignedrc, member N = R(8N)..R(8N+7). The encoding carries N.

    Reading such an operand as slot(base) gives 8N where the field holds N, so the correlation
    still finds the bits and attributes them THREE POSITIONS TOO HIGH - a field map that is wrong
    in exactly the way a composer would not notice until the register came out eight times too
    far along. Verified by mutation on op5106: every bit flip XORs the member index by one power
    of two, five bits per operand.

    This resolves through the same member list that made the FLAGR selector readable.
    """
    d = _INSTRS.get(opcode) if _INSTRS is not None else None
    if d is None:
        d = _load_instrs().get(opcode)
    if not d or k >= len(d):
        return None
    cname = d[k]
    if cname is None:
        return None
    members = class_members().get(cname)
    if not members:
        return None
    try:
        return members.index(mcreg)
    except ValueError:
        return None


_INSTRS = None


def _load_instrs():
    """{opcode: [class name or None per operand]} from Apple's MCInstrDesc."""
    global _INSTRS
    if _INSTRS is None:
        import g17opclass
        cls, desc = g17opclass.classes(), g17opclass.instrs()
        _INSTRS = {o: [cls[rc][0] if rc in cls else None for rc, _, _ in d["operands"]]
                   for o, d in desc.items()}
    return _INSTRS


def class_members():
    """{class name: [MCRegister ids in class order]} from Apple's MCRegisterClass RegsBegin."""
    global _MEMBERS
    if _MEMBERS is None:
        _MEMBERS = {}
        import subprocess
        r = subprocess.run([os.path.join(HERE, "agx3meta"), "members"],
                           capture_output=True, text=True)
        byname = {v: k for k, v in g17model.registers().items()}
        for line in r.stdout.splitlines():
            if line.startswith("#") or "|" not in line:
                continue
            head, rest = line.split("|", 1)
            parts = head.split()
            if len(parts) < 3:
                continue
            _MEMBERS[parts[1]] = [byname.get(n) for n in rest.split()]
    return _MEMBERS


_MEMBERS = None


def relocated_register_bits(rows):
    """Bits carrying a register operand that Apple's PRINTER renders as a relocation.

    The memory family declares operand 3 as GPR32tup2 - the base address pair - and the printer
    emits an `expr` for it in about 97% of instances, because the base is a binding the linker
    resolves. So the operand VALUE is unavailable exactly where the operand matters, and no amount
    of correlating against printed values can see the field. byte1[5:1] was the largest unexplained
    region in the corpus for that reason alone: 16 opcodes, 16k instructions.

    The minority that DO print a register are the way in. Over 794 reg-printed instances of six
    opcodes, byte1[5:1] equals slot[5:1] in 100% of them, and decoding the other 17564 that way
    names a register DEFINED earlier in the program 97.7% of the time - mostly by op12682, op11452
    and the add forms, which is what a base pointer's producer should be.

    Only what is visible is credited: a bit is explained when it matches the slot of an operand the
    printer sometimes shows, on every instance where it shows it.
    """
    out = set()
    if not rows:
        return out
    # THE FAMILY RULE. Where the printer does show the base register, slot bit j is byte1 bit j -
    # 794 instances across six opcodes, 100%, no counterexample. An opcode whose own visible subset
    # has too few distinct slots to confirm that (op17257 shows one register, ever) is still
    # encoding the same field, so the rule is applied whenever ANY position is a declared register
    # class that the printer renders as a relocation. Bits that do not vary are not credited.
    if any(ops[k][0] == "expr" for _, ops in rows for k in range(min(4, len(ops)))):
        # SIX BITS, NOT FIVE. byte1[5:1] was established by correlation on the reg-printed
        # minority, where no base ever reaches slot 64, so slot bit 6 agreed VACUOUSLY. Settled by
        # mutation instead: flipping byte1[6] on a reg-printed instance and re-decoding moves the
        # base slot by exactly +64 = 2^6, on op12674, op12675, op12682 and op17257 alike.
        # tools/g17bitprobe.py. That closes the last residual bit of the tensor loads and store.
        out |= {(1, j) for j in range(1, 7)}
    nops = min(len(ops) for _, ops in rows)
    ln = len(rows[0][0])
    for k in range(nops):
        vis = [(b, ops) for b, ops in rows if ops[k][0] == "reg"]
        if len(vis) < 8:
            continue
        slots = [slot(ops[k][1]) for _, ops in vis]
        if any(x is None for x in slots) or len(set(slots)) < 2:
            continue
        cols = {(by, i): [(b[by] >> i) & 1 for b, _ in vis]
                for by in range(ln) for i in range(8)}
        for j in range(10):
            bits = [(x >> j) & 1 for x in slots]
            if len(set(bits)) < 2:
                continue
            for key, col in cols.items():
                if col == bits:
                    out.add(key)
    return out


def _encode(enc, opcode, k, mcreg):
    """The value a field would carry for this operand under one of the three encodings."""
    if enc == "raw":
        return mcreg
    if enc == "slot":
        return slot(mcreg)
    return class_index(opcode, k, mcreg) if opcode is not None else None


def field_map(rows, kinds, ln, opcode=None):
    """{(operand, encoding): [(value bit, byte, bit, inverted)]} for one solved form.

    Kept separate per encoding: a register operand can be carried as its raw MCRegister value or as
    its slot, and mixing the two in one list makes a map that decodes 7.8% of the form it came
    from. That was a real bug here, caught by asking the map to decode its own instances.
    """
    cols = {(by, i): [(b[by] >> i) & 1 for b, _ in rows]
            for by in range(ln) for i in range(8)}
    varying = {k: v for k, v in cols.items() if len(set(v)) > 1}
    out = {}
    for k, kind in enumerate(kinds):
        if kind == "expr":
            continue
        # THREE ENCODINGS, kept apart. A register operand can be carried as its raw MCRegister id,
        # as its hardware slot, or as its CLASS-RELATIVE MEMBER INDEX - the last being what the
        # aligned tuple classes use, where slot() is eight times the encoded value.
        for enc in (("raw", "slot", "index") if kind == "reg" else ("raw",)):
            vals = [_encode(enc, opcode, k, ops[k][1]) for _, ops in rows]
            if any(v is None for v in vals) or len(set(vals)) < 2:
                continue
            got = []
            for j in range(32):
                bits = [(v >> j) & 1 for v in vals]
                if len(set(bits)) < 2:
                    continue
                hit = [key for key, c in varying.items() if c == bits]
                inv = [key for key, c in varying.items() if c == [1 - x for x in bits]]
                if len(hit) == 1:
                    got.append((j, hit[0][0], hit[0][1], False))
                elif not hit and len(inv) == 1:
                    got.append((j, inv[0][0], inv[0][1], True))
            if got:
                out[(k, enc)] = got
    return out


def decodes(fmap, rows, opcode=None):
    """How many of `rows` a field map decodes exactly. This is a PREDICTION, not an assumption:
    a thin form counts as covered only when every one of its instances comes out right."""
    good = 0
    for b, ops in rows:
        ok = True
        for (k, enc), bits in fmap.items():
            if k >= len(ops) or ops[k][0] == "expr":
                continue
            want = _encode(enc, opcode, k, ops[k][1])
            if want is None:
                continue
            for j, by, i, inverted in bits:
                if by >= len(b):
                    # The map came from a longer form. A field position is not portable across
                    # lengths, so this is a failed prediction and not a skipped one.
                    ok = False
                    break
                got = (b[by] >> i) & 1
                if inverted:
                    got ^= 1
                if got != ((want >> j) & 1):
                    ok = False
                    break
            if not ok:
                break
        good += ok
    return good


def subform_split(opcode, rows, all_rows, varying):
    """A bit that splits this form into two SUB-FORMS, each of which solves clean.

    (signature, length) is not always fine enough to name a form. op10369's 10-byte form has nine
    residual bits as a whole and ZERO on each side of byte1[0]; op10370's has three and zero. The
    residue was the artefact of merging two encodings that Apple's tables do not distinguish,
    because they share an operand signature and a length.

    The criterion is deliberately severe: BOTH sides must have at least eight instances and BOTH
    must come out with no residue at all. A bit that merely reduces the residue does not count,
    which is what keeps this from being a search for any convenient partition.
    """
    for cand in sorted(varying):
        sides = []
        for side in (0, 1):
            part = [r for r in rows if ((r[0][cand[0]] >> cand[1]) & 1) == side]
            # SIDE SIZE SET BY THE SHUFFLE CONTROL. At eight the criterion finds 7 splits in the
            # real data and 1 with operands randomly reassigned - a 7 to 1 ratio, below what every
            # other detector here reaches. At sixteen it finds 4 and ZERO, and stays at zero for
            # 24 and 32. Four splits with no false positives is the correct trade against seven
            # with one.
            if len(part) < 16:
                sides = None
                break
            sides.append(part)
        if not sides:
            continue
        if all(_residue_of(opcode, p, all_rows) == set() for p in sides):
            return cand
    return None


def _residue_of(opcode, rows, all_rows):
    r = analyse(opcode, rows, all_rows=all_rows, _split=False)
    return r[6] if r else {(0, 0)}


def analyse(opcode, rows, all_rows=None, _split=True):
    """(varying, explained, unexplained) bit sets for the modal form of `rows`.

    `all_rows` is every instance of the opcode, used only to recover operands that THIS form's
    printer renders as a relocation but a SIBLING form shows as a register. Without it, analysing
    one form at a time hides exactly the evidence that resolves it: op10283's 12-byte form with an
    expr at operand 4 has residue b8[6] and b9[0:3], and its 732-instance sibling at the same
    length puts that operand's slot bits at precisely b8[6], b8[7], b9[0], b9[1], b9[2], b9[3].
    """
    if len(rows) < 8:
        return None
    all_rows = rows if all_rows is None else all_rows
    sig = collections.Counter(tuple(k for k, _ in ops) for _, ops in rows)
    kinds, _ = sig.most_common(1)[0]
    rows = [r for r in rows if tuple(k for k, _ in r[1]) == kinds]
    lens = collections.Counter(len(b) for b, _ in rows)
    ln, _ = lens.most_common(1)[0]
    rows = [r for r in rows if len(r[0]) == ln]
    if len(rows) < 8:
        return None
    cols = {(b, i): [(r[0][b] >> i) & 1 for r in rows]
            for b in range(ln) for i in range(8)}
    varying = {k: v for k, v in cols.items() if len(set(v)) > 1}

    def match(values):
        got = set()
        if values is None or len(set(values)) < 2:
            return got
        for j in range(40):
            bits = [(v >> j) & 1 for v in values]
            if len(set(bits)) < 2:
                continue
            inv = [1 - b for b in bits]
            for key, col in varying.items():
                # INVERTED FIELDS ARE FIELDS. src2_keep is already known to be stored as NOT
                # byte8[5], established causally, and tools/g17fields.py reports inverted matches
                # while this solver only ever tested equality - so every negated field in the
                # corpus was counted as unexplained.
                if col == bits or col == inv:
                    got.add(key)
        return got

    def table_bits(values):
        """Bits that are a FUNCTION of an operand without being a bit of its value.

        Some register classes are encoded through an opaque code table rather than by id, slot or
        class index. The special registers are: over op14059 and op14060, byte1[5:0] maps
        SR_SIMD_ELEM to 2, SR_TG_X_SIZE/Y/Z to 24/25/26, SR_TG_X/Y/Z to 28/29/30 and
        SR_TP_IN_GRID_X/Y/Z to 32/33/34 - systematic, 1:1, and consistent across both opcodes, but
        equal to nothing in the MCRegister id or the class index.

        A bit is credited when it is CONSTANT within every distinct operand value and still varies
        across them. Requiring four distinct values keeps a two-valued operand from explaining an
        arbitrary bit by coincidence.
        """
        got = set()
        # THE GUARD MEASURES CONSTRAINT, NOT CARDINALITY. "Constant within each distinct value" is
        # free when the groups are singletons and overwhelming when they are large, so the count
        # that matters is how many rows are constrained: sum(size - 1) over the groups. A first
        # version demanded four distinct values and missed op577, whose three bits are an exact
        # function of an operand taking THREE values in groups of 1331, 591 and 6 - a coincidence
        # with probability around 2^-1925.
        groups = collections.defaultdict(list)
        for i, v in enumerate(values):
            groups[v].append(i)
        if len(groups) < 2 or sum(len(g) - 1 for g in groups.values()) < 20:
            return got
        for key, col in varying.items():
            if all(len({col[i] for i in idx}) == 1 for idx in groups.values()):
                got.add(key)
        return got

    explained = set()
    for k in range(len(kinds)):
        # An `expr` operand is NOT a value the instruction encodes. Apple's printer emits an MCExpr
        # for the base operand and this harness reads the POINTER: on tg-16x48x64 those values
        # advance by exactly 72 per instance while the instructions alternate 12 and 16 bytes, so
        # the stride is the decoder's allocator, not the program. The compiler agent confirmed the
        # objects carry ZERO Mach-O relocation entries - nreloc is 0 across 389 objects - so there
        # is no relocation behind the expression either.
        #
        # Matching instruction bits against a heap pointer credits bits to noise. The real field at
        # that operand is recovered by relocated_register_bits() from the instances where the
        # printer does show a register.
        if kinds[k] == "expr":
            continue
        # VALUE TRANSFORMS WERE TRIED AND REMOVED. A field need not hold the operand as the
        # printer reports it - src2_keep is stored as NOT byte8[5], established by execution - so
        # biased, incremented, negated and complemented encodings were worth testing. Over 282
        # forms the branch ran at 1104 operand positions and credited 3168 bits beyond plain
        # matching, and corpus coverage did not move: 89.4% and 88 residual bits before and after.
        # Every bit it found was already explained by another mechanism.
        #
        # So the transforms add no explanatory power and 3168 fresh chances to attribute a bit to
        # the wrong operand. A mechanism that explains nothing new does not earn its risk, and it
        # is deleted rather than left in as a harmless-looking extra.
        explained |= match([ops[k][1] for _, ops in rows])
        # A PACKED IMMEDIATE is encoded through a table too, not only a register class. op11375's
        # operand 1 takes 32, 0x1000020, 0x2000020 and 0x4000020, and byte5[6:5] selects among them
        # as 0, 1, 2, 3 - so no bit of the value equals any bit of the field, and matching on value
        # bits cannot see it. This was reaching register operands only.
        explained |= table_bits([ops[k][1] for _, ops in rows])
        if kinds[k] == "reg":
            slots = [slot(ops[k][1]) for _, ops in rows]
            if all(s is not None for s in slots):
                explained |= match(slots)
            # A flag operand is a compact 3-bit index, not an id and not a slot. Without this the
            # selector shows as unexplained on every opcode that reads a flag - eight of them.
            flags = [flag_index(ops[k][1]) for _, ops in rows]
            if all(f is not None for f in flags):
                explained |= match(flags)
            # And the class-relative member index, which is what the ALIGNED tuple classes encode.
            idx = [class_index(opcode, k, ops[k][1]) for _, ops in rows]
            if all(i is not None for i in idx):
                explained |= match(idx)

    # TIED OPERANDS. A short form can carry more register operands than it has bits for, because
    # one bit says an operand is a COPY of another and its field is then unused. op11437's 8-byte
    # form sets byte4[0] in 166 of 183 instances and in every one of those operand 8 is the same
    # register as operand 0; in the 17 where it is clear, operand 8's slot bits 1..5 sit at
    # byte5[1..5]. op11365's 8-byte form does the same with byte5[1..6] and byte4[6].
    #
    # Both halves are credited only when the tie bit predicts the equality EXACTLY. That is the
    # verification: a bit that merely correlates with two operands being equal is not a tie bit.
    reg_ops = [k for k, kind in enumerate(kinds) if kind == "reg"]
    for a in reg_ops:
        for c in reg_ops:
            if a >= c:
                continue
            tied = [1 if ops[a][1] == ops[c][1] else 0 for _, ops in rows]
            if len(set(tied)) < 2 or min(tied.count(0), tied.count(1)) < 4:
                continue
            marks = [key for key, col in varying.items()
                     if col == tied or col == [1 - x for x in tied]]
            if not marks:
                continue
            explained |= set(marks)
            free = [(b, ops) for (b, ops), t in zip(rows, tied) if not t]
            if len(free) < 4:
                continue
            sub_cols = {key: [(b[key[0]] >> key[1]) & 1 for b, _ in free] for key in varying}
            sub_varying = {k2: v for k2, v in sub_cols.items() if len(set(v)) > 1}
            slots = [slot(ops[c][1]) for _, ops in free]
            if any(x is None for x in slots):
                continue
            for j in range(10):
                bits = [(x >> j) & 1 for x in slots]
                if len(set(bits)) < 2:
                    continue
                hit = [k2 for k2, col in sub_varying.items()
                       if col == bits or col == [1 - x for x in bits]]
                if len(hit) == 1:
                    explained.add(hit[0])

    # CONDITIONAL OPERAND PRESENCE, the generalisation of the tie above. An operand's field can be
    # active only in part of a form: some bit partitions the instances into two operand SCHEMAS,
    # and the field resolves inside one partition while meaning nothing in the other. Analysed
    # whole, such a field looks like noise, because half its rows are unrelated to the operand.
    #
    # The tie is the special case where the inactive schema makes the operand a copy. This looks
    # for the general one: partition by each varying bit, and inside each side ask whether an
    # as-yet-unexplained operand resolves UNIQUELY. Both sides must have at least eight instances
    # and the operand must actually vary within the partition, or the match is free.
    for cond in list(varying):
        col = varying[cond]
        for side in (0, 1):
            part = [r for r, c in zip(rows, col) if c == side]
            # GUARDS SET BY THE SHUFFLE CONTROL, not by judgement. This search has many degrees of
            # freedom - every varying bit, two sides, every operand, twelve value bits - and at
            # part>=8 with 2 distinct values it credits 373 bits to randomly reassigned operands
            # against 3722 real, a ratio of only 10 to 1. Tightening moves it:
            #
            #     part>=8  distinct>=2    3722 real,  373 shuffled    10 to 1
            #     part>=16 distinct>=4    3042        89              34
            #     part>=24 distinct>=6    2648        32              83
            #     part>=32 distinct>=8    2321        16             145
            #     part>=40 distinct>=10   2026        10             203
            #
            # The other detectors here run at 500 and 800 to 1, so the last row is the one that
            # meets the same standard. It credits fewer bits and that is the correct trade.
            if len(part) < 40 or len(rows) - len(part) < 40:
                continue
            sub_cols = {key: [(b[key[0]] >> key[1]) & 1 for b, _ in part] for key in varying}
            sub_varying = {k2: v for k2, v in sub_cols.items() if len(set(v)) > 1}
            for k in range(len(kinds)):
                if kinds[k] == "expr":
                    continue
                for values in ([ops[k][1] for _, ops in part],
                               [slot(ops[k][1]) for _, ops in part]
                               if kinds[k] == "reg" else None):
                    if values is None or any(v is None for v in values) or len(set(values)) < 10:
                        continue
                    for j in range(12):
                        bits = [(v >> j) & 1 for v in values]
                        if len(set(bits)) < 2:
                            continue
                        hit = [k2 for k2, c2 in sub_varying.items()
                               if c2 == bits or c2 == [1 - x for x in bits]]
                        if len(hit) == 1 and hit[0] not in explained:
                            explained.add(hit[0])
                            explained.add(cond)

    # THE HIDDEN OPERAND, RECOVERED BY MUTATION RATHER THAN CORRELATION. relocated_register_bits
    # below solves an expr-printed operand from reg-printed siblings by correlation, which needs
    # eight of them and enough variation. Some forms have one. Mutation needs one:
    # flip a bit on a reg-printed instance and see whether that operand's SLOT moves by 2^j.
    #
    # This matters because the freedom test that calls a bit "outside the encoding" EXCLUDES expr
    # operands - Apple's printer reallocates the MCExpr on every decode, so their value changes
    # under any mutation. A bit encoding the expr operand's register is therefore invisible to it
    # and was being reported as free. op437's b8[6] through b9[3] are operand 4's slot bits 0..5,
    # op10829's b8[7] through b9[4] are its slot bits 1..6, and op10837's b8[6] is slot bit 0.
    # Thirteen bits that were called degrees of freedom are register fields.
    if all_rows is not None:
        import g17bitprobe as _probe
        for k in range(len(kinds)):
            if kinds[k] != "expr":
                continue
            visible = [(b, ops) for b, ops in all_rows
                       if len(b) == ln and len(ops) > k and ops[k][0] == "reg"]
            for bit in sorted(set(varying) - explained):
                deltas = set()
                for b, ops in visible[:8]:
                    base = _probe.decode(bytes(b))
                    mut = bytearray(b)
                    mut[bit[0]] ^= 1 << bit[1]
                    got = _probe.decode(bytes(mut))
                    if not base or not got or base[0] != got[0] or base[1] != got[1]:
                        deltas.add(None)
                        continue
                    if base[2][k][0] != "reg" or got[2][k][0] != "reg":
                        deltas.add(None)
                        continue
                    a0, a1 = slot(base[2][k][1]), slot(got[2][k][1])
                    deltas.add(abs(a1 - a0) if a0 is not None and a1 is not None else None)
                if deltas and None not in deltas and len(deltas) == 1:
                    d = deltas.pop()
                    if d and (d & (d - 1)) == 0:
                        explained.add(bit)

    # An operand the printer renders as a relocation is invisible to the loop above, so solve it
    # from the instances where the printer does show it, across every signature of this opcode at
    # this length - the reg-printed ones usually sit in a different signature.
    explained |= relocated_register_bits([r for r in all_rows if len(r[0]) == ln])
    unexplained = set(varying) - explained
    net = unexplained - set(CAUSAL_FIELDS) - set(INERT_FIELDS)
    net -= set(MUTATION_FIELDS.get((opcode, ln), {}))
    net -= MUTATION_CODE_FIELDS.get((opcode, ln), set())
    # If what is left is only the seam between two sub-forms, it is not residue.
    if net and _split:
        cand = subform_split(opcode, rows, all_rows, varying)
        if cand is not None:
            net = set()
    return len(rows), ln, kinds, varying, explained, unexplained, net


def frozen_profile(rows):
    """(length, operand kinds, {bit: value}) for the modal form's frozen bits."""
    sig = collections.Counter(tuple(k for k, _ in ops) for _, ops in rows)
    kinds, _ = sig.most_common(1)[0]
    rows = [r for r in rows if tuple(k for k, _ in r[1]) == kinds]
    lens = collections.Counter(len(b) for b, _ in rows)
    ln, _ = lens.most_common(1)[0]
    rows = [r for r in rows if len(r[0]) == ln]
    if len(rows) < 8:
        return None
    frozen = {}
    for b in range(ln):
        for i in range(8):
            col = {(r[0][b] >> i) & 1 for r in rows}
            if len(col) == 1:
                frozen[(b, i)] = col.pop()
    return ln, kinds, frozen


def selectors(rows_all, named):
    """Split every opcode's frozen bits into SELECTOR and STRUCTURAL by comparing siblings."""
    prof = {}
    for op, rows in rows_all.items():
        p = frozen_profile(rows)
        if p:
            prof[op] = p
    families = collections.defaultdict(list)
    for op, (ln, kinds, _) in prof.items():
        families[(ln, kinds)].append(op)
    out = {}
    for key, members in families.items():
        if len(members) < 2:
            for op in members:
                out[op] = (set(), set(prof[op][2]), members)
            continue
        for op in members:
            sel, struct = set(), set()
            for bit, val in prof[op][2].items():
                differs = any(bit in prof[o][2] and prof[o][2][bit] != val
                              for o in members if o != op)
                (sel if differs else struct).add(bit)
            out[op] = (sel, struct, members)
    return out


HANDOFF = os.path.join(os.path.dirname(HERE), "isa", "g17-handoff.toml")


def forms(rows):
    """[(signature, length, rows)] for every distinct form of one opcode, biggest first.

    A FORM is the unit the codegen side must author against, not an opcode. Field positions are
    not stable across forms of the same operation: the flag selector sits at byte0[7:5] in the
    SIX-byte compares and is unresolved at ten bytes, and applying one form's layout to another
    produced an instruction that wrote the wrong flag AND corrupted its own compared register.
    Reporting only an opcode's modal form is what let that happen.
    """
    by = collections.defaultdict(list)
    for b, ops in rows:
        by[(tuple(k for k, _ in ops), len(b))].append((b, ops))
    return sorted(by.items(), key=lambda kv: -len(kv[1]))


def write_handoff(recs, sel, objs):
    """Regenerate isa/g17-handoff.toml, the authorability contract the codegen side reads."""
    safe = [r for r in recs if r[2] and not r[2][6] and not sel.get(r[0], (set(),))[0]]
    bound = [r for r in recs if r[2] and not r[2][6] and sel.get(r[0], (set(),))[0]]
    blocked = [r for r in recs if r[2] and r[2][6]]
    thin = [r for r in recs if not r[2]]
    out = [open(HANDOFF).read().split("[meta]")[0].rstrip(), "", "[meta]",
           'chip = "%s"' % g17target.CHIP, 'arch = "%s"' % g17target.ARCH,
           "objects_scanned = %d" % objs, "named = %d" % len(recs),
           "template_safe = %d" % len(safe), "selector_bound = %d" % len(bound),
           "blocked = %d" % len(blocked), "unsolvable_here = %d" % len(thin), ""]
    for op, name, r in recs:
        out.append("[[opcode]]")
        out.append("id = %d" % op)
        out.append('name = "%s"' % name)
        if not r:
            out.append('status = "too few instances here"')
            out.append("")
            continue
        n, ln, kinds, varying, explained, unexp, net = r
        out.append("# field positions are stated for THIS form only; siblings differ")
        s_, st, members = sel.get(op, (set(), set(), [op]))
        status = "blocked" if net else ("selector-bound" if s_ else "template-safe")
        out.append('status = "%s"' % status)
        out.append("instances = %d" % n)
        out.append("bytes = %d" % ln)
        out.append("operands = [%s]" % ", ".join('"%s"' % k for k in kinds))
        out.append("varying_bits = %d" % len(varying))
        out.append("unexplained_bits = %d" % len(net))
        if net:
            out.append("unexplained = [%s]"
                       % ", ".join('"b%d[%d]"' % b for b in sorted(net)))
        dropped = sorted(unexp & set(CAUSAL_FIELDS))
        if dropped:
            out.append("causally_known = [%s]"
                       % ", ".join('"b%d[%d]"' % b for b in dropped))
        inert = sorted(unexp & set(INERT_FIELDS))
        if inert:
            out.append("measured_inert = [%s]"
                       % ", ".join('"b%d[%d]"' % b for b in inert))
        if s_:
            out.append("selector_bits = [%s]"
                       % ", ".join('"b%d[%d]"' % b for b in sorted(s_)))
        out.append("family_size = %d" % len(members))
        out.append("")
    open(HANDOFF, "w").write("\n".join(out))
    return len(safe), len(bound), len(blocked), len(thin)


def main():
    named = dict(g17slice.KNOWN_OPS)
    want = {int(a) for a in sys.argv[1:] if a.isdigit()}
    # The sample used to be capped at 300 objects, which is why 17 named opcodes reported "too few
    # instances here": not a property of the opcode, a property of the cap. --objects overrides it
    # and the handoff regeneration uses the whole corpus.
    cap = None
    if "--objects" in sys.argv:
        cap = int(sys.argv[sys.argv.index("--objects") + 1])
    elif not want and "--handoff" not in sys.argv:
        cap = 300
    if "--handoff" in sys.argv:
        rows_all, objs = g17fields.instances(None, limit_objects=cap)
        sel = selectors(rows_all, named)
        recs = [(op, name, analyse(op, rows_all.get(op, []))) for op, name in sorted(named.items())]
        a, b, c, d = write_handoff(recs, sel, objs)
        print("isa/g17-handoff.toml: %d objects, %d named - template-safe %d, selector-bound %d, "
              "blocked %d, too few %d" % (objs, len(recs), a, b, c, d))
        for op, name, r in recs:
            if r and r[6]:
                print("   blocked op%-7d %-10s %s"
                      % (op, name, " ".join("b%d[%d]" % k for k in sorted(r[6]))))
        return
    if "--selectors" in sys.argv:
        rows_all, objs = g17fields.instances(None, limit_objects=cap)
        sel = selectors(rows_all, named)
        print("%-8s %-11s %-8s %-10s %-10s %s"
              % ("opcode", "name", "family", "SELECTOR", "structural", "selector bits"))
        for op in sorted(named):
            if op not in sel:
                continue
            s_, st, members = sel[op]
            print("%-8d %-11s %-8d %-10d %-10d %s"
                  % (op, named[op], len(members), len(s_), len(st),
                     " ".join("b%d[%d]" % b for b in sorted(s_))[:44]))
        return
    rows_all, objs = g17fields.instances(want or set(named), limit_objects=cap)
    if want:
        for op in sorted(want):
            r = analyse(op, rows_all.get(op, []))
            if not r:
                print("op%d: too few instances" % op)
                continue
            n, ln, kinds, varying, explained, unexp, net = r
            print("op%-6d %-11s %d instances, %d bytes, operands %s"
                  % (op, named.get(op, "?"), n, ln, " ".join(kinds)))
            print("  varying %d   explained %d   unexplained %d  %s"
                  % (len(varying), len(explained), len(unexp),
                     " ".join("b%d[%d]" % k for k in sorted(unexp))))
            causal = sorted(unexp & set(CAUSAL_FIELDS))
            if causal:
                print("    of which causally known: %s"
                      % ", ".join("b%d[%d] %s" % (b, i, CAUSAL_FIELDS[(b, i)]) for b, i in causal))
            inert = sorted(unexp & set(INERT_FIELDS))
            if inert:
                print("    of which measured INERT: %s"
                      % ", ".join("b%d[%d]" % b for b in inert))
            print("  NET RESIDUE %d  %s"
                  % (len(net), " ".join("b%d[%d]" % k for k in sorted(net)) or "none"))
        return

    recs = []
    for op, name in sorted(named.items()):
        r = analyse(op, rows_all.get(op, []))
        recs.append((op, name, r))
    ready = sum(1 for _, _, r in recs if r and not r[6])
    blocked = sum(1 for _, _, r in recs if r and r[6])
    thin = sum(1 for _, _, r in recs if not r)
    print("%-8s %-11s %-7s %-8s %-9s %-9s %s"
          % ("opcode", "name", "insts", "varying", "unexpl", "NET", "status"))
    for op, name, r in recs:
        if not r:
            print("%-8d %-11s %-7s %-8s %-9s %-9s %s" % (op, name, "-", "-", "-", "-", "too few here"))
            continue
        n, ln, kinds, varying, explained, unexp, net = r
        print("%-8d %-11s %-7d %-8d %-9d %-9d %s"
              % (op, name, n, len(varying), len(unexp), len(net),
                 "AUTHORABLE" if not net else "blocked"))
    print("\nnamed %d   authorable %d   blocked %d   too few here %d"
          % (len(recs), ready, blocked, thin))


if __name__ == "__main__":
    main()
