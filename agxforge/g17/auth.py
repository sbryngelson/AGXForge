#!/usr/bin/env python3
"""Author ANY admitted opcode from its recovered field map.

Until now this backend could emit an instruction only if someone had written an encoder for its
form: fourteen scalar operations, the tensor load/store/MAC, the branch and the compare. That is
a per-form cost, and the mission's measure is the fraction of the instruction set the backend can
SELECT - so a per-form cost is the wrong shape of work entirely.

The peer's isa/g17-authoring.jsonl carries, for each of 6,684 admitted opcodes, a witness encoding
and a complete field map: for every operand, which instruction bit carries which bit of its value,
and with what polarity. This module is the codec that makes that table executable. One code path
encodes and decodes all of them.

    import g17auth
    g17auth.encode(3978, {0: 14, 2: 10})        # sqrt: dest slot 14, source slot 10
    g17auth.decode(3978, bytes_)                # -> {0: 14, 1: 32, 2: 10, 3: 32}

WHAT A VALUE MEANS depends on the operand's DOMAIN, which the table names per operand and which
this module does not try to hide:

    raw     the value Apple's decoder prints - immediates and modifier words
    slot    slot = hardware_register * 2 + half, for ordinary registers
    index   the class-relative member index, for the aligned tuple classes

THE LIMITS, because an encoder that lies about its reach is worse than one that refuses. The
table's own verdict is carried through as `certified(opcode)`:

    operand   every operand accepts arbitrary values           3,058 opcodes
    register  registers free, immediates only from a table     2,765
    narrow    the form reaches only part of the register class   616   - clamped, not refused
    none      every operand is an expr or a constant             302
    partial   a mapped bit does not round-trip                   132
    gap       a value bit is MISSING from the map                 39   - refused

`encode` refuses an opcode whose map is incomplete, and refuses a packed immediate on an opcode
certified only for registers, rather than emitting bytes whose meaning is not established. Bits
the map does not name are the witness's, and `residue(opcode)` counts them so the inheritance is
visible rather than silent.

    python3 tools/g17auth.py --gate [N]     author and read back through Apple's decoder
    python3 tools/g17auth.py --summary      what the table admits, by verdict and by length
"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# THE ANCHOR MOVED WITH THE FILE, and one of the things it locates is a native binary. Under tools/
# this module sat beside agx3dis and one level below the checkout; under agxforge/g17/ it is two levels
# below and the helper does not move - tools/agx3dis stays where the Makefile builds it. Resolving
# from the checkout root keeps the decoder, the length table and the lifetime table addressable
# without putting tools/ on the library's import path.
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS = os.path.join(ROOT, "tools")
TABLE = os.path.join(ROOT, "isa", "g17-authoring.jsonl")
DIS = os.path.join(TOOLS, "agx3dis")
REG0 = 105                      # the decoder prints register n as MCRegister id 105 + n
_INDEX = None


def load(path=TABLE):
    """{opcode: record}. Parsed once; the file is 8.5MB and the parse is a second."""
    global _INDEX
    if _INDEX is None:
        _INDEX = {}
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    _INDEX[r["opcode"]] = r
    return _INDEX


def record(opcode):
    r = load().get(opcode)
    if r is None:
        raise KeyError("opcode %d is not in the authoring table" % opcode)
    return r


# The table's verdict is a dict; these are its levels, shortened to the words this module uses.
LEVEL = {"certified": "operand",
         "registers certified, table-encoded immediates": "register",
         "narrow field - the operand cannot reach every register": "narrow",
         "bit-certified only": "bit",
         "field map incomplete": "partial",
         "field map has a GAP - a bit is missing": "gap",
         "no mapped operand": "none"}


def certified(opcode):
    """'operand' | 'register' | 'bit' | 'partial' | 'none' - how far the table's own round-trip
    got. `operand_fail` names the operands that failed the multi-bit check, so an opcode certified
    only for registers still authors its registers freely."""
    return LEVEL.get(record(opcode)["certified"]["level"], "partial")


def failing_operands(opcode):
    return set(record(opcode)["certified"].get("operand_fail") or ())


def gapped_operands(opcode):
    """Operands with a MISSING value bit - the table's own verdict, and a refusal.

    A LADDER FAILURE IS NOT ALWAYS A HOLE. When the register ladder first ran here it refused 573
    operands, and the peer's re-derivation separates them: 616 are NARROW - the form encodes only
    part of the register class, so high registers are unreachable and nothing is missing - and 39
    have a genuine gap, a bit skipped below the top of the map. Refusing both would have cost 616
    opcodes that author perfectly inside their range, so a narrow field is CLAMPED by the ordinary
    width check and only a gap is refused.
    """
    return set(record(opcode)["certified"].get("gap") or ())


def reach(opcode, operand):
    """The largest value an operand's mapped bits can hold, in its own domain."""
    domain, bits = fields(opcode)[operand]
    return (1 << (max(j for j, _, _, _ in bits) + 1)) - 1


def fields(opcode):
    """{operand_index: (domain, [(value_bit, byte, bit, inverted), ...])}."""
    out = {}
    for key, bits in record(opcode)["fields"].items():
        idx, domain = key.split(":")
        out[int(idx)] = (domain, [tuple(b) for b in bits])
    return out


def operand_classes(opcode):
    return record(opcode)["operands"]


def witness(opcode):
    return bytes.fromhex(record(opcode)["witness"])


def _shared(opcode):
    """(byte, bit) -> how many operands claim that encoding bit."""
    n = {}
    for _, bits in fields(opcode).values():
        for _, by, bi, _ in bits:
            n[(by, bi)] = n.get((by, bi), 0) + 1
    return n


def carriers(opcode, operand):
    """{value_bit: [(byte, bit, inverted), ...]} - every encoding bit that carries each value bit.

    A value bit can have MORE THAN ONE carrier and they are not interchangeable. op3290's operand 3
    lists value bit 4 twice, at byte6[3] inverted and at byte8[2]; byte8[2] moves the operand's
    value cleanly and byte6[3] also turns operand 2 from a register into an EXPRESSION. 1,031 of
    9,282 operands sampled have such a bit, so which carrier to write is a real question and not a
    curiosity - and the only honest answer is to try each and ask Apple's decoder.
    """
    out = {}
    for j, by, bi, inv in fields(opcode)[operand][1]:
        out.setdefault(j, []).append((by, bi, inv))
    return out


def decode(opcode, u):
    """Read every mapped operand out of `u`, each in its own domain."""
    out = {}
    for idx, (domain, bits) in fields(opcode).items():
        v = 0
        for j, by, bi, inv in bits:
            if by < len(u) and (((u[by] >> bi) & 1) ^ inv):
                v |= 1 << j
        out[idx] = v
    return out


def encode(opcode, values, template=None, allow_uncertified=False, trusted=()):
    """Write `values` - {operand_index: value in that operand's domain} - onto a witness.

    Operands not named keep the template's bits, and so does every bit the map does not name.
    """
    r = record(opcode)
    cert = certified(opcode)
    if cert in ("partial", "none") and not allow_uncertified:
        raise ValueError("opcode %d (%s) is certified %r - its field map does not support "
                         "authoring" % (opcode, r.get("name"), cert))
    fm = fields(opcode)
    u = bytearray(template if template is not None else witness(opcode))
    if len(u) < 16:
        u.extend(b"\x00" * (16 - len(u)))
    for idx, v in values.items():
        if idx not in fm:
            raise ValueError("opcode %d has no field map for operand %d" % (opcode, idx))
        domain, bits = fm[idx]
        if idx in gapped_operands(opcode) and not allow_uncertified:
            raise ValueError("opcode %d operand %d has a MISSING value bit - writing it would name "
                             "a different register than asked" % (opcode, idx))
        # A TRUSTED OPERAND is one the caller stated rather than the compiler derived - a
        # condition code from Apple's own census, or a FLAGR index read off a real instruction. The
        # refusal below exists to stop the compiler INVENTING a packed immediate, which is a
        # different thing from writing one that was measured. Gaps are still refused for everyone:
        # a missing bit makes the value wrong rather than uncertain.
        if idx in failing_operands(opcode) and not allow_uncertified and idx not in trusted:
            raise ValueError("opcode %d operand %d failed the table's multi-bit round trip - it "
                             "is a packed immediate and must come from its own table, not written "
                             "as a number" % (opcode, idx))
        width = max(j for j, _, _, _ in bits) + 1
        if v < 0 or v >> width:
            raise ValueError("operand %d value %d does not fit its %d mapped bits"
                             % (idx, v, len(bits)))
        # EVERY mapped bit is written here, including a value bit with two carriers. That is what
        # the register ladder certifies - 988 of 988 register operands read their ladder back - and
        # it is deliberately NOT what put_lifetime does, because writing both carriers of the
        # lifetime's value bit 4 turns a neighbouring register operand into an expression.
        for j, by, bi, inv in bits:
            b = ((v >> j) & 1) ^ inv
            u[by] = (u[by] & ~(1 << bi)) | (b << bi)
        if domain == "slot":
            _slot_at_length(opcode, idx, v, u, bits)
    return bytes(u)


_SLOT_TRUTH = None


def _slot_truth():
    """{(opcode, operand, length): [(weight, byte, bit, inv)]} from isa/g17-slot-truth-by-length.json."""
    global _SLOT_TRUTH
    if _SLOT_TRUTH is None:
        _SLOT_TRUTH = {}
        path = os.path.join(ROOT, "isa", "g17-slot-truth-by-length.json")
        if os.path.exists(path):
            for key, by_len in json.load(open(path))["truth"].items():
                op, spec = key.split()
                for n, rec in by_len.items():
                    _SLOT_TRUTH[(int(op), int(spec.split(":")[0]), int(n))] = [
                        tuple(f) for f in rec["fields"]]
    return _SLOT_TRUTH


def _slot_at_length(opcode, idx, v, u, bits):
    """The slot the instruction CARRIES at its own length, checked against Apple's decoder's truth.

    This map was fitted at one length, and the layout of a slot depends on the encoded length
    (tools/g17slottruth.py). Where the bits just written do not read back as `v` through the
    decoder-measured layout for this opcode's authored length, the operand is rewritten from that
    layout, and if even that cannot hold `v` the write is refused. Before this, op11994/16 wrote
    every register from 16 up as r105, and op3291/4 wrapped r64 up to 425 by writing a bit past
    byte 4. Where the map was already right nothing changes: the rewrite runs only on a mismatch."""
    try:
        n = length(opcode)
    except (KeyError, FileNotFoundError):
        return
    truth = _slot_truth().get((opcode, idx, n))
    if not truth:
        return
    # THE TWO MAPS COUNT IN DIFFERENT UNITS. The truth is in 16-bit halves; this map counts
    # whatever its operand counts - whole registers for the fourteen forms that lack the H bit,
    # where every truth weight is this map's plus one. The offset is read off the positions the
    # two maps share: the offset at least two thirds of them agree on, or nothing is checked. Not
    # unanimity - a bit this map mislabels is exactly what the check is for, and op11994/16's map
    # calls b11.2 weight 9 where five other shared bits agree the truth is this map's own weight.
    import collections
    at = {(by, bi): w for w, by, bi, _ in truth}
    shifts = collections.Counter(at[(by, bi)] - w for w, by, bi, _ in bits if (by, bi) in at)
    if not shifts:
        return
    k, votes = shifts.most_common(1)[0]
    if k < 0 or 3 * votes < 2 * sum(shifts.values()):
        return
    v = v << k
    read = lambda: sum((((u[by] >> bi) & 1) ^ inv) << w for w, by, bi, inv in truth)
    if read() == v:
        return
    top = max(w for w, _, _, _ in truth)
    if v >> (top + 1) or any((v >> w) & 1 for w in range(top + 1)
                            if w not in {t[0] for t in truth}):
        raise ValueError("opcode %d operand %d: slot value %d does not fit the %d-byte layout "
                         "Apple's decoder measures (weights %s)"
                         % (opcode, idx, v, n, sorted(t[0] for t in truth)))
    for w, by, bi, inv in truth:
        u[by] = (u[by] & ~(1 << bi)) | ((((v >> w) & 1) ^ inv) << bi)
    if read() != v:
        raise ValueError("opcode %d operand %d: slot value %d does not read back at %d bytes"
                         % (opcode, idx, v, n))


def map_gaps(opcode):
    """{operand: [missing value bits]} - bits below an operand's top mapped bit that no
    instruction bit carries, plus, for a `slot` operand, bit 1 if it is absent.

    A GAP IS SILENT AND WRONG, which is why this is a refusal and not a warning. Writing register
    2 into an operand whose map starts at value bit 2 leaves bit 1 holding whatever the witness
    had, so the instruction names register 3 while the caller believes it named register 2 - and a
    round trip does not catch it, because a round trip only ever touches the bits the map names.
    Found by authoring a ladder of registers rather than a single value: 12 of 728 register
    operands in a 400-opcode sample read back 1, 3, 3, 5, 9 for the registers 1, 2, 3, 5, 9.
    Slot bit 0 is the 16-bit half selector and is legitimately absent on a 32-bit class, so it is
    not counted; bit 1 is the first register bit and its absence is a defect.
    """
    out = {}
    for idx, (domain, bits) in fields(opcode).items():
        if domain not in ("slot", "index"): continue    # a raw immediate is legitimately sparse:
        vb = sorted(j for j, _, _, _ in bits)           # the modifier words carry bits 5, 6, 24-31,
        if not vb: continue                             # 33, 37, 41 and 47 and nothing between
        lo = 1 if domain == "slot" else 0
        miss = [j for j in range(lo, max(vb)) if j not in vb]
        if miss: out[idx] = miss
    return out


def residue(opcode, length=None):
    """Bits of the instruction that no operand, opcode or length field explains."""
    r = record(opcode)
    named = set()
    for _, bits in fields(opcode).values():
        named.update((by, bi) for _, by, bi, _ in bits)
    for by, bi, *_ in r.get("opcode_bits", []): named.add((by, bi))
    for by, bi, *_ in r.get("length_bits", []): named.add((by, bi))
    for by, bi in r.get("dead_bits", []): named.add((by, bi))
    n = length if length is not None else 16
    return [(by, bi) for by in range(n) for bi in range(8) if (by, bi) not in named]


# THE LENGTH TABLE IS CHECKED IN, NOT CACHED IN A HOME DIRECTORY. It used to live at
# ~/.cache/agxforge/g17auth-lengths.json, and a consumer that only wanted to compile something found
# the compiler reaching outside the checkout for a file it could not see, could not review and
# could not reproduce - and, when the file was absent, forking Apple's decoder instead. The scan
# acceptance driver refuses both by construction and was right to: a length this backend depends
# on is a recovered specification, so it belongs in isa/ next to every other one.
#
# It is still MEASURED the same way - every witness decoded once - and `--refresh` regenerates it
# deliberately. What changed is that the measurement is a checked-in artifact rather than a side
# effect of whoever ran the compiler first.
#
#     python3 tools/g17auth.py --refresh-lengths     re-derive and rewrite isa/g17-auth-lengths.json
LENGTHS = os.path.join(ROOT, "isa", "g17-auth-lengths.json")
LENCACHE = LENGTHS          # the old name, kept so nothing that imports it breaks
_LEN = None


def lengths(refresh=False):
    """{opcode: instruction length in bytes}, from the checked-in table.

    Decoded in chunks and retried singly on a short result, because Apple's MCInstPrinter calls
    abort() on some encodings and one poisonous witness takes the whole subprocess with it - the
    peer lost 451 of 640 decodes to exactly that before splitting on short output. That decode
    happens under --refresh; a plain call reads the file.
    """
    global _LEN
    if _LEN is None and not refresh and os.path.exists(LENGTHS):
        try:
            doc = json.load(open(LENGTHS))
            _LEN = {int(k): v for k, v in (doc.get("lengths") or doc).items()}
        except ValueError:
            _LEN = None
    if _LEN is None and not refresh:
        raise FileNotFoundError(
            "%s is missing, and deriving it needs Apple's decoder. Regenerate it deliberately "
            "with `python3 tools/g17auth.py --refresh-lengths` - this path does not fork a "
            "subprocess behind a caller's back, and it does not read outside the checkout."
            # RELATIVE TO THE CHECKOUT, NOT TO THIS FILE'S PARENT. These paths are quoted inside
            # diagnostics that end up in retained reports, so anchoring them on the module's
            # location makes a report's TEXT depend on where the module lives: after the move
            # dirname(HERE) became agxforge/, every such path gained a "../", and a truncated refusal
            # in the half-vector report changed from "isa/g17-auth-on-t" to "../isa/g17-auth-o".
            # ROOT is the same string wherever the implementation sits.
            % os.path.relpath(LENGTHS, ROOT))
    if _LEN is None or refresh:
        idx = load(); ops = sorted(idx)
        out = {}
        def run(batch):
            got = _decode_many([witness(o) for o in batch])
            if sum(g is not None for g in got) == 0 and len(batch) > 1:
                return None
            return got
        i = 0
        while i < len(ops):
            batch = ops[i:i+256]
            got = run(batch)
            if got is None:
                got = []
                for o in batch:
                    g = _decode_many([witness(o)])
                    got.append(g[0])
            for o, g in zip(batch, got):
                if g is not None:
                    try: out[o] = int(g[0])
                    except ValueError: pass
            i += 256
        _LEN = out
        # written through a temporary and renamed: several probes run at once and a half-written
        # file is a JSONDecodeError in an unrelated program, which is exactly what happened
        tmp = LENGTHS + ".%d" % os.getpid()
        with open(tmp, "w") as fh:
            json.dump({"source": "tools/g17auth.py --refresh-lengths, one decode per witness",
                       "opcodes": len(out),
                       "lengths": {str(k): v for k, v in sorted(out.items())}}, fh, indent=0)
        os.replace(tmp, LENGTHS)
    return _LEN


def length(opcode):
    n = lengths().get(opcode)
    if n is None:
        raise KeyError("opcode %d has no decodable witness, so its length is not known" % opcode)
    return n


def slot_step(class_name):
    """How much an operand's SLOT moves between consecutive members of its register class.

    A slot names a 16-bit half, so a 32-bit register is two slots and a class whose members are
    tuples steps by however many hardware registers a member covers:

        GPR16                       1     the halves themselves
        GPR32, IRGPR32              2
        GPR32tup2 (plain)           2     member N is R(N)_R(N+1), so members overlap
        GPR32tup2_alignedrc         4     member N is R(2N)_R(2N+1)
        GPR32tup4_alignedrc         8     member N is R(4N)..

    Equivalently: slot = 2 x the index of the FIRST hardware register the operand names. Getting
    this wrong does not fail loudly - writing 1, 2, 3 into an aligned pair's slot field asks for
    values no member of the class has, and the bits that would carry them do not exist, so the
    field reads back 1, 3, 3 and looks like a missing bit rather than a bad question.
    """
    c = class_name or ""
    if c.startswith("GPR16"): return 1
    for k in (2, 4, 8):
        if "tup%d" % k in c: return 2 * k if "aligned" in c else 2
    return 2


# --- THE LADDER IS A PER-OPERAND FACT, NOT A PER-CLASS ONE ---------------------------------
# slot_step() above answers "how far apart are the members of this class", and that is not the
# same question as "what does THIS operand's field value mean". op1934 (ffma, named by isolation)
# has three GPR32 sources and the table calls all three `slot`; measured through Apple's decoder,
# operand 2 steps one register every TWO units - a slot - while operands 4 and 6 step one register
# every unit. Writing 2*r into those two names register 2r, so an authored ffma read two registers
# nobody had written and returned the product of whatever they held.
#
# It fails silently and it defeats a round trip, because a round trip only asks whether the value
# comes back - and it does, faithfully naming the wrong register. It also defeats the affine ladder
# check: fitting printed = base + slope*value with slope free is exactly what hides a wrong slope.
#
# So the slope is measured, once per operand, through the decoder, and cached.
LADDERCACHE = os.path.expanduser("~/.cache/agxforge/g17-ladder.json")
_LADDER = None

def _ladder_cache():
    global _LADDER
    if _LADDER is None:
        try: _LADDER = json.load(open(LADDERCACHE))
        except Exception: _LADDER = {}
    return _LADDER

def _printed_reg(tok, i):
    """The MCRegister the decoder printed for operand i, or None if it did not print a register."""
    if tok is None or len(tok) < 3 + i: return None
    t = tok[2 + i]
    return int(t[4:]) if t.startswith("reg:") else None

def ladders(opcode):
    """{operand: MCRegisters per unit of field value}, measured. None where it cannot be read."""
    c = _ladder_cache()
    key = str(opcode)
    if key not in c:
        d, s = register_operands(opcode)
        ops = d + s
        blobs, tags = [], []
        for i in ops:
            for v in (0, 2, 4):
                try: b = encode(opcode, {i: v}, allow_uncertified=True)
                except Exception: b = None
                blobs.append(b if b is not None else witness(opcode)); tags.append((i, v, b is not None))
        toks = _decode_many(blobs)
        got = {}
        for (i, v, ok), t in zip(tags, toks):
            if ok: got.setdefault(i, {})[v] = _printed_reg(t, i)
        out = {}
        for i in ops:
            p = got.get(i, {})
            p0, p2, p4 = p.get(0), p.get(2), p.get(4)
            if p0 is None or p2 is None or p4 is None: out[str(i)] = None; continue
            a, b = (p2 - p0) / 2.0, (p4 - p0) / 4.0
            out[str(i)] = a if a == b and a > 0 else None
        c[key] = out
        os.makedirs(os.path.dirname(LADDERCACHE), exist_ok=True)
        tmp = LADDERCACHE + ".%d" % os.getpid()
        with open(tmp, "w") as fh: json.dump(c, fh)
        os.replace(tmp, LADDERCACHE)
    return {int(k): v for k, v in c[key].items()}


def imm_field(opcode, operand, printed):
    """The BITS to write so that the decoder prints `printed` for a raw operand.

    An immediate operand need not read back as the bits written: bits outside the operand's own map
    contribute, so the field is affine with a base that is not zero on 466 of 1,014 measured
    operands (the peer's `imm_base`, isa/g17-authoring.jsonl). An encoder that writes v into one of
    those produces base + v and believes it produced v - and it round-trips, because a round trip
    compares bits with bits.

    encode() deliberately keeps the BIT convention: decode() returns bits and encode() writes them,
    so a witness read and written back is unchanged. This is the other convention, for a caller who
    means the number the decoder prints - a condition code taken from a census, a float immediate.
    """
    base = int((record(opcode).get("imm_base") or {}).get(str(operand), 0))
    v = printed - base
    if v < 0:
        raise ValueError("opcode %d operand %d has base %d; %d is below it"
                         % (opcode, operand, base, printed))
    return v


def field_value(opcode, operand, register):
    """The value to write into `operand` so that it names hardware register `register`.

    v = register / slope, and nothing else: the printed index advances by one per hardware register
    in every class this backend allocates, so a slope of 0.5 is the familiar slot (write 2*r) and a
    slope of 1 is a plain register number. Both of the conventions this project has EXECUTED come
    out unchanged - op999's GPR32 source and op767's GPR16 source both measure 0.5 and both keep
    their 2*r - and the operands that measure 1.0 are the ones that were being written wrong.

    TUPLE CLASSES ARE LEFT ALONE. A member of an aligned pair covers two hardware registers, so
    "one printed index per register" is not the right unit for them, and this backend cannot
    allocate a tuple anyway. They keep the table's domain until something executes one.
    """
    r = record(opcode)
    cls = r["operands"][operand] or ""
    # THE TABLE CARRIES THE MEASUREMENT NOW. The peer swept every record - id(v) = base +
    # (v // units) * id_step - and 149 of 149 operand ladders on a random 60-opcode sample agree
    # with the ones measured here through agx3dis, which are two measurements with no shared code.
    # Preferring the table's is what makes this cheap: the local measurement costs a subprocess per
    # opcode, and it stays as the fallback and the cross-check.
    _sl = (r.get("operand_slope") or {}).get(str(operand))
    slope = None
    if _sl and _sl.get("verdict") in ("exact", "interleaved") and _sl.get("units"):
        slope = float(_sl.get("id_step") or 1) / float(_sl["units"])
    if slope is None:
        slope = ladders(opcode).get(operand)
    if slope and "tup" not in cls:
        v = register / slope
        if v != int(v):
            raise ValueError("opcode %d operand %d steps %g printed registers per unit; register "
                             "%d is not reachable" % (opcode, operand, slope, register))
        return int(v)
    domain = fields(opcode)[operand][0]
    return register * 2 if domain == "slot" else register


def register_operands(opcode):
    """([destination operand indices], [source operand indices]) in the table's own order.

    `ndefs` says how many of the instruction's operands are definitions, and the register-like
    operands are the ones whose domain is slot or index - a raw operand is an immediate or a
    modifier word and is not something the register allocator can fill.
    """
    r = record(opcode)
    regs = [i for i, (d, _) in sorted(fields(opcode).items()) if d in ("slot", "index")]
    nd = r["ndefs"]
    return [i for i in regs if i < nd], [i for i in regs if i >= nd]


# THE SOURCE LIFETIME IS A MODIFIER OPERAND, and the convention is the same on every form tested.
# Apple's decoder prints a raw operand immediately after each source register, and differencing
# kernels that differ only in whether a value is read again gives its meaning:
#
#     f[200] = rint(p);                     3770 ... imm:16          10-byte unary
#     f[200] = rint(p); f[201] = floor(p);  3770 ... imm:32          p read again
#     f[200] = p + q;                        998 ... imm:16 imm:16   12-byte two-source
#     f[200] = p + q; f[201] = p * q;        998 ... imm:32 imm:32   both read again
#
#     value bit 5 set = KEEP, value bit 4 set = RELEASE.
#
# Only those two bits are written, never the whole operand: one corpus instance carries 18 rather
# than 16, so the field holds more than the lifetime and overwriting it would discard whatever the
# rest of it says.
# The peer's independent census settles the rest of the field: over 12,000 corpus instances of five
# opcodes, this operand takes EXACTLY the values {0, 2, 16, 18, 20, 32, 34, 36} and nothing else -
# bit 1 negate, bit 2 absolute value, bit 4 release, bit 5 keep, composing freely (18 is
# negate+release, 34 is negate+keep). A destination's modifier takes only 0 or 32, which is the
# consistency check: a definition is always a keep. isa/g17-source-modifier-operand.toml
#
# ALL FOUR BITS ARE WRITTEN, not just the lifetime, and that is not tidiness. A witness taken from a
# mutation walk rather than from Apple's code can carry 0 - neither keep nor release - or 52, which
# is keep AND release AND absolute value at once. Inheriting either is how four opcodes in the
# naming sweep returned the same value for every input.
LIFETIME_KEEP, LIFETIME_RELEASE = 5, 4
MOD_NEG, MOD_ABS = 1, 2


def lifetime_operand(opcode, src):
    """The operand carrying `src`'s lifetime, or None if this form does not express one."""
    m = fields(opcode).get(src + 1)
    if not m or m[0] != "raw":
        return None
    have = {j for j, _, _, _ in m[1]}
    return src + 1 if {LIFETIME_KEEP, LIFETIME_RELEASE} <= have else None


def put_modifier(opcode, u, src, keep, negate=False, absolute=False, choice=None):
    """Write one source's modifier in place; False if the form does not express one.

    `choice` names which carrier to use per value bit. It comes from lifetime_certified, which
    found it by trying each and decoding the result - see carriers().
    """
    idx = lifetime_operand(opcode, src)
    if idx is None:
        return False
    car = carriers(opcode, idx)
    u = bytearray(u)
    want = [(LIFETIME_KEEP, 1 if keep else 0), (LIFETIME_RELEASE, 0 if keep else 1),
            (MOD_NEG, 1 if negate else 0), (MOD_ABS, 1 if absolute else 0)]
    for j, v in want:
        if j not in car:
            if j in (LIFETIME_KEEP, LIFETIME_RELEASE): return False
            continue                       # a form without a negate or abs bit simply lacks one
        by, bi, inv = (choice or {}).get(j) or car[j][0]
        u[by] = (u[by] & ~(1 << bi)) | ((v ^ inv) << bi)
    return bytes(u)


put_lifetime = put_modifier                # the older name, kept for probes already written


# CHECKED IN FOR THE SAME REASON THE LENGTHS ARE. This is a CERTIFICATION - every opcode that
# expresses a source lifetime, authored both ways on its own witness and read back through Apple's
# decoder - so it is recovered evidence, not a memo. Living in ~/.cache made the compiler reach
# outside the checkout for a fact nobody could review, and made its absence a silent fork.
#
#     python3 tools/g17auth.py --refresh-lifetimes   re-certify and rewrite the checked-in table
LIFETIMES = os.path.join(ROOT, "isa", "g17-auth-lifetimes.json")
LIFECACHE = LIFETIMES           # the old name, kept so nothing that imports it breaks
_LIFE = None


def lifetime_certified(opcode, refresh=False):
    """Has writing this opcode's source lifetimes been CHECKED through Apple's decoder?

    Writing a field the map names is not the same as writing it correctly, and this form's lifetime
    is the case that proved it: the write succeeded, the encoding decoded, and one of the register
    operands quietly became an expression. So every opcode that expresses a lifetime is authored
    both ways on its own witness and read back, and the compiler writes one only where both
    polarities leave the opcode and every register operand intact and the modifier reading 32
    and 16 respectively.
    """
    global _LIFE
    if _LIFE is None and not refresh and os.path.exists(LIFETIMES):
        try:
            doc = json.load(open(LIFETIMES))
            _LIFE = {int(k): v for k, v in (doc.get("lifetimes") or doc).items()}
        except ValueError:
            _LIFE = None
    if _LIFE is None and not refresh:
        raise FileNotFoundError(
            "%s is missing, and re-certifying it needs Apple's decoder. Regenerate it deliberately "
            "with `python3 tools/g17auth.py --refresh-lifetimes` - this path does not fork a "
            "subprocess behind a caller's back, and it does not read outside the checkout."
            % os.path.relpath(LIFETIMES, ROOT))
    if _LIFE is None or refresh:
        _LIFE = _certify_lifetimes()
        tmp = LIFETIMES + ".%d" % os.getpid()
        with open(tmp, "w") as fh:
            json.dump({"source": "tools/g17auth.py --refresh-lifetimes, both polarities on each "
                                 "opcode's own witness, read back through Apple's decoder",
                       "opcodes": len(_LIFE),
                       "certified": sum(1 for v in _LIFE.values() if v.get("ok")),
                       "lifetimes": {str(k): v for k, v in sorted(_LIFE.items())}}, fh, indent=0)
        os.replace(tmp, LIFETIMES)
    return bool(_LIFE.get(opcode, {}).get("ok"))


def lifetime_choice(opcode, src):
    r = (_LIFE or {}).get(opcode) or {}
    c = (r.get("choice") or {}).get(str(src))
    return {int(j): tuple(v) for j, v in c.items()} if c else None


# CERTIFICATIONS AGAINST A SUPPLIED TEMPLATE, CHECKED IN. certify_on authors both polarities of a
# source lifetime on the caller's own template and reads them back through Apple's decoder - real
# evidence, and exactly the kind a compiler must not be re-deriving by forking a subprocess every
# time it emits an instruction. The answers are a function of (opcode, template bytes, source), all
# three of which are fixed by the backend's own form registry, so the whole table is finite and
# belongs in the checkout.
#
#     python3 tools/g17authtables.py     compile the known programs and rewrite the table
#
# A MISS RAISES rather than falling back to the decoder. Falling back is what made this a hidden
# dependency; raising names the one command that closes it and keeps the compile path honest.
ON_TABLE = os.path.join(ROOT, "isa", "g17-auth-on-templates.json")
_ONCACHE = {}
_ON_CANON = {}
_ON_LOADED = [False]
_ON_NEW = {}


def canon_template(opcode, template):
    """The template with every OPERAND bit cleared - what is left is the form.

    certify_on's question is about the FORM: can this opcode's lifetime carriers be written here
    without disturbing the opcode or any operand's kind. It is not about which register happens to
    be in the destination field, and keying the table on raw bytes made it about exactly that - the
    first harvest produced ten keys for one form, one per register allocation, and any program the
    harvest had not seen missed.

    MEASURED, NOT ASSUMED: those ten templates differed only in operand bits and returned ONE
    answer per source, both sources, every time. So the operand bits are cleared and the key is the
    form. A template differing in a bit no operand covers still gets its own key, which is the
    distinction certify_on exists to make - op17229's fourteen-byte witness against Apple's own
    eight-byte store.
    """
    u = bytearray(template)
    for _idx, (_kind, bits) in (fields(opcode) or {}).items():
        for _w, by, bi, *_ in bits:
            if by < len(u):
                u[by] &= ~(1 << bi) & 0xFF
    return bytes(u)


def _on_key(opcode, template, src):
    return "%d|%s|%d" % (opcode, canon_template(opcode, template).hex(), src)


def _on_load():
    if _ON_LOADED[0]:
        return
    _ON_LOADED[0] = True
    if not os.path.exists(ON_TABLE):
        return
    try:
        doc = json.load(open(ON_TABLE))
    except ValueError:
        return
    for k, v in (doc.get("certified") or {}).items():
        op, hexs, src = k.split("|")
        _ON_CANON[(int(op), hexs, int(src))] = (
            {int(j): tuple(t) for j, t in v.items()} if v else None)


def on_refresh_allowed():
    """Only tools/g17authtables.py sets this. A compile never does."""
    return os.environ.get("G17_AUTH_REFRESH") == "1"


def on_table_dump():
    """Write everything certified this process, merged over what was already checked in."""
    _on_load()
    out = {k: ({str(j): list(t) for j, t in v.items()} if v else None)
           for k, v in ((("%d|%s|%d" % (op, h, src)), v) for (op, h, src), v in _ON_CANON.items())}
    disagree = []
    for (op, tpl, src), v in sorted(_ONCACHE.items(), key=lambda kv: _on_key(*kv[0])):
        k = _on_key(op, tpl, src)
        val = {str(j): list(t) for j, t in v.items()} if v else None
        if k in out and out[k] != val:
            disagree.append(k)
        out[k] = val
    if disagree:
        raise AssertionError(
            "two templates with the same FORM certified differently, so clearing the operand bits "
            "is losing something real: %s" % disagree[:4])
    tmp = ON_TABLE + ".%d" % os.getpid()
    with open(tmp, "w") as fh:
        json.dump({"source": "tools/g17authtables.py - both lifetime polarities authored on each "
                             "template the backend actually emits, read back through Apple's "
                             "decoder",
                   "entries": len(out),
                   "certified_count": sum(1 for v in out.values() if v),
                   "certified": out}, fh, indent=0)
    os.replace(tmp, ON_TABLE)
    return len(out)


def _same_token(a, b):
    if a.startswith("expr:") and b.startswith("expr:"): return True
    return a == b


def certify_on(opcode, template, src):
    """Can this opcode's source lifetime be written on THIS template? Cached per (opcode, bytes).

    The table's verdict is about the table's witness, and a witness is one encoding of an opcode
    rather than the only one - op17229's is fourteen bytes where Apple's own indexed store is eight,
    and a field's carriers need not behave the same in both. So a template the caller supplies is
    certified against itself: author both polarities, decode, and require the opcode unchanged,
    every register operand still a register, and the modifier reading 32 and 16.

    Returns the carrier choice to use, or None if the lifetime cannot be written here.
    """
    import itertools
    key = (opcode, bytes(template), src)
    _on_load()
    if key in _ONCACHE: return _ONCACHE[key]
    ck = (opcode, canon_template(opcode, template).hex(), src)
    if ck in _ON_CANON:
        _ONCACHE[key] = _ON_CANON[ck]
        return _ONCACHE[key]
    if not on_refresh_allowed():
        raise KeyError(
            "op%d's source %d has not been certified on this template in %s, and certifying it "
            "needs Apple's decoder. Run `python3 tools/g17authtables.py` to re-derive the table - "
            "a compile does not fork a subprocess to answer this."
            % (opcode, src, os.path.relpath(ON_TABLE, ROOT)))
    idx = lifetime_operand(opcode, src)
    out = None
    if idx is not None:
        car = carriers(opcode, idx)
        regs = register_operands(opcode)[0] + register_operands(opcode)[1]
        # WHAT THE OTHER OPERANDS MUST DO IS NOT MOVE. Requiring each to still PRINT as a register
        # refused Apple's own eight-byte indexed store the moment its buffer-address operand was
        # exposed as a register operand: the address is an expression there and always was, so the
        # test was asking the template to be something it never is. The property that matters is
        # that writing the lifetime disturbs nothing else, and that is a comparison against the
        # template's own decode.
        base = _decode_many([template])[0]
        for pick in itertools.product(car.get(LIFETIME_KEEP, []), car.get(LIFETIME_RELEASE, [])):
            choice = {LIFETIME_KEEP: pick[0], LIFETIME_RELEASE: pick[1]}
            blobs = [put_modifier(opcode, template, src, k, choice=choice) for k in (True, False)]
            if any(b is False for b in blobs): continue
            got = _decode_many(blobs)
            ok = True
            for g, want in zip(got, (32, 16)):
                if (g is None or g[1] != str(opcode) or len(g) <= 2 + idx
                        or g[2 + idx] != "imm:%d" % want):
                    ok = False; break
                if base is None:
                    if not all(len(g) > 2 + r and g[2 + r].startswith("reg:") for r in regs):
                        ok = False; break
                # AN EXPRESSION PRINTS AN UNSTABLE NUMBER. agx3dis renders an expression operand
                # as expr:<address>, and the address is the decoder's own temporary - it differs
                # between two runs of the same bytes. So an expression is compared by KIND and a
                # register or immediate by value.
                elif any(len(base) > 2 + r and len(g) > 2 + r and not _same_token(g[2 + r], base[2 + r])
                         for r in regs):
                    ok = False; break
            if ok: out = choice; break
    _ONCACHE[key] = out
    return out


def _certify_lifetimes():
    """{opcode: {"ok": bool, "choice": {src: {value_bit: [byte, bit, inverted]}}}}.

    Every combination of carriers is authored in both polarities and read back. An opcode passes
    only if, for every source it expresses a lifetime for, some combination leaves the opcode
    unchanged, every register operand still a register, and the modifier reading 32 for keep and
    16 for release.
    """
    import itertools
    out = {}
    ops = [o for o in sorted(load())
           if any(lifetime_operand(o, s) is not None for s in register_operands(o)[1])]
    for i in range(0, len(ops), 96):
        jobs, blobs = [], []
        for o in ops[i:i+96]:
            w = witness(o)
            for s in register_operands(o)[1]:
                idx = lifetime_operand(o, s)
                if idx is None: continue
                car = carriers(o, idx)
                for pick in itertools.product(car[LIFETIME_KEEP], car[LIFETIME_RELEASE]):
                    choice = {LIFETIME_KEEP: pick[0], LIFETIME_RELEASE: pick[1]}
                    for keep in (True, False):
                        b = put_modifier(o, w, s, keep, choice=choice)
                        if b is False: continue
                        jobs.append((o, s, idx, choice, keep)); blobs.append(b)
        got = _decode_many(blobs)
        seen = {}
        for (o, s, idx, choice, keep), g in zip(jobs, got):
            want = 32 if keep else 16
            regs = register_operands(o)[0] + register_operands(o)[1]
            ok = (g is not None and g[1] == str(o) and len(g) > 2 + idx
                  and all(len(g) > 2 + r and g[2 + r].startswith("reg:") for r in regs)
                  and g[2 + idx] == "imm:%d" % want)
            key = (o, s, tuple(sorted(choice.items())))
            seen[key] = seen.get(key, True) and ok
        for (o, s, ch), ok in seen.items():
            r = out.setdefault(o, {"ok": {}, "choice": {}})
            if ok and s not in r["choice"]:
                r["choice"][s] = {j: list(v) for j, v in ch}
            r["ok"][s] = r["ok"].get(s, False) or ok
    return {o: {"ok": all(r["ok"].values()), "choice": {str(k): v for k, v in r["choice"].items()}}
            for o, r in out.items()}


def _decode_many(blobs, verbose=False):
    """One agx3dis call for many 16-byte candidates. Returns token lists, None where refused.

    THE STRIDE IS 32, NOT 16, and the padding is zeros. A candidate is sixteen bytes but nothing
    says the decoder will stop there: laid end to end at a 16-byte stride, an instruction that
    decodes longer reads its neighbour's bytes and reports a length the sample does not explain.
    Padding to 32 makes the overrun read zeros instead of data, and any decode claiming more than
    the sample's own sixteen bytes is REJECTED rather than believed. --stride keeps a rejection
    from desynchronising the walk.
    """
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
        for b in blobs:
            fh.write(bytes(b).ljust(16, b"\x00")[:16] + b"\x00" * 16)
        path = fh.name
    try:
        o = subprocess.run([DIS, path, "0", str(32 * len(blobs)), "--pc", "0", "--stride", "32"],
                           capture_output=True, text=True).stdout
    finally:
        os.unlink(path)
    out = [None] * len(blobs)
    for line in o.splitlines():
        p = line.split()
        if len(p) < 2: continue
        try: k = int(p[0], 16) // 32
        except ValueError: continue
        if 0 <= k < len(blobs) and p[1] != "bad":
            try: n = int(p[1])
            except ValueError: continue
            if n <= 16: out[k] = p[1:]
    return out


def _gate(n, points=(1, 2, 3, 5, 9)):
    """Author a LADDER of registers into every register operand and demand Apple's decoder reads
    the ladder back.

    Reproducing a witness proves nothing - it succeeds on any bytes. Authoring one different value
    proves less than it looks, because the decoder prints an absolute MCRegister id whose base
    differs per register class, so an isolated number cannot be checked without knowing the base.
    A ladder settles it without the base: write registers 1, 2, 3, 5, 9 into the operand and demand
    the printed ids form the same affine progression - the intercept, whatever the class's base is,
    cancels, and a field map that is wrong in even one bit breaks the spacing.
    """
    import random
    idx = load()
    cand = [o for o in sorted(idx) if certified(o) in ("operand", "register", "narrow")]
    random.Random(11).shuffle(cand)
    cand = cand[:n]
    jobs, blobs, refused = [], [], []
    for opc in cand:
        fm = fields(opc)
        w = witness(opc)
        for i, (domain, bits) in sorted(fm.items()):
            if domain not in ("slot", "index"): continue
            width = max(j for j, _, _, _ in bits) + 1
            step = slot_step(record(opc)["operands"][i]) if domain == "slot" else 1
            if (max(points) * step) >> width: continue        # a narrow field cannot hold the
                                                              # ladder; that is a clamp, not a fault
            k0 = len(blobs)
            try:
                enc = [encode(opc, {i: pt * step}, w) for pt in points]
            except ValueError:
                refused.append((opc, i)); continue
            blobs.extend(enc)
            jobs.append((opc, i, domain, k0))
    got = _decode_many(blobs)
    ok = bad = unscorable = 0
    shown = 0
    for opc, i, domain, k0 in jobs:
        seen = []
        for t, pt in enumerate(points):
            g = got[k0 + t]
            if g is None or g[1] != str(opc) or len(g) <= 2 + i or not g[2+i].startswith("reg:"):
                seen = None; break
            seen.append(int(g[2+i].split(":")[1]))
        if seen is None:
            unscorable += 1
            continue
        slope = (seen[1] - seen[0]) / float(points[1] - points[0])
        good = slope != 0 and all(abs((seen[t] - seen[0]) - slope * (points[t] - points[0])) < 1e-9
                                  for t in range(len(points)))
        if good: ok += 1
        else:
            bad += 1
            if shown < 8:
                print("   op%-6d %-14s operand %d (%s): %s for registers %s"
                      % (opc, idx[opc].get("name"), i, domain, seen, list(points)))
                shown += 1
    print("register operands scored: %d   ladder read back exactly: %d   broken: %d   "
          "not scorable (operand does not print as a register): %d"
          % (ok + bad, ok, bad, unscorable))
    print("opcodes touched: %d of the %d sampled" % (len({j[0] for j in jobs}), len(cand)))
    if refused:
        print("refused before encoding, map gap: %d operands on %d opcodes  %s"
              % (len(refused), len({o for o, _ in refused}),
                 ", ".join("op%d/%d" % r for r in refused[:6])))
    return bad


def _summary():
    idx = load()
    by = {}
    for o in idx: by[certified(o)] = by.get(certified(o), 0) + 1
    print("authoring table: %d opcodes" % len(idx))
    for k in ("operand", "register", "narrow", "bit", "partial", "gap", "none"):
        if k in by: print("   %-9s %5d" % (k, by[k]))
    named = sum(1 for r in idx.values() if r.get("name"))
    print("   named     %5d" % named)


def main(argv=None):
    """The command-line entry, callable. It used to live inline under `if __name__`, which meant
    neither `python3 tools/g17auth.py` nor the library could invoke it after the move - the
    compatibility module imported this file rather than executing it, so the CLI silently did
    nothing and exited zero. `argv` defaults to the real one so both entries behave identically.
    """
    argv = list(sys.argv if argv is None else argv)
    if "--refresh-lifetimes" in argv:
        t = lifetime_certified(0, refresh=True)
        print("wrote %s: %d opcodes, %d certified"
              % (os.path.relpath(LIFETIMES, ROOT), len(_LIFE),
                 sum(1 for v in _LIFE.values() if v.get("ok"))))
    elif "--refresh-lengths" in argv:
        # THE ONE PLACE THE DECODER IS INVOKED FOR THIS TABLE, and it is invoked because a person
        # asked. Every other caller reads the checked-in file.
        t = lengths(refresh=True)
        print("wrote %s: %d opcodes" % (os.path.relpath(LENGTHS, ROOT), len(t)))
    elif "--summary" in argv: _summary()
    elif "--gate" in argv:
        i = argv.index("--gate")
        n = int(argv[i+1]) if len(argv) > i+1 else 400
        sys.exit(1 if _gate(n) else 0)
    else: print(__doc__)


if __name__ == "__main__":
    main()
