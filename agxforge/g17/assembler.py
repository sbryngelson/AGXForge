#!/usr/bin/env python3
"""A standalone assembler for G17: source text in, machine code out, no decoder in the loop.

WHY THIS EXISTS. Every program this project has executed came out of the compiler, so a failed
experiment has four suspects - instruction selection, register allocation, encoding, and the
harness - and no way to hold three of them still. An assembler removes three: you name the
registers, you name the form, and what runs is what you wrote.

WHAT MAKES IT AN ASSEMBLER RATHER THAN A TEMPLATE FILLER. `g17asm.py` encodes by overwriting the
bits it owns in an instruction Apple already wrote, and the bits it does not own are inherited -
which is why it cannot author a form Apple never emitted. Here the non-operand bits come from the
recovered specification: opcode bits, bits mutation proved forced, and mode bits, all placed by
`g17encode.encode`. No template, and nothing in this file decides what a field means.

WHAT IT REFUSES, and each refusal is a bug that has actually happened here:

  An operand map not VERIFIED by `g17opmap.py` against every cached instance of its opcode.
  Fitting a base to a single witness is unfalsifiable, and doing it is what made ten of eleven
  records in the first oracle batch assemble to bytes Apple's own decoder rejected. Register
  fields turn out to advance by one printed register per TWO encoded units, so maps that looked
  linear and integral were wrong by a factor of two.

  A field with a HOLE in it. op11842's operand 1 has no mapped bits below weight 5, so asking for
  42 quietly encodes 32. Every instruction is read back through the same specification right after
  encoding and compared with what was asked for, which catches this without consulting Apple's
  decoder.

  An operand whose bits fall OUTSIDE the instruction. movimm's eight-byte form cannot carry an
  immediate above 255 because that operand's field map belongs to the ten-byte form; keyed by
  opcode alone an assembler writes past the end and the bytes decode as something else. Forms are
  keyed by (opcode, length).

  A register in two operands of one instruction, a branch displacement out of range or odd, an
  immediate too wide for its field, and an operand count that does not match the form.

SYNTAX

    ; comment - semicolon only, because # begins an immediate
    .entry  0x40
    .form   ld  op12682.l14           ; name any (opcode, length) pair
  top:
    and     r105, r107, #15
    mov     r105, r107
    if      p74                       ; push a mask level, count 1
    if.inv  p74, 2                    ; inverted predicate, count 2
    else    p3
    while   p74, 2
    br.none done                      ; forward branch skipping a fully masked region
    br.any  top                       ; backward branch, repeats while lanes are live
    pop     2
  done:
    end
    .byte   0x0e, 0x00                ; raw bytes, for a form the spec cannot yet author

  Operands: rN register, pN predicate register, #N or a bare integer immediate, [rN] an address,
  or a label. `opK=V` sets operand K directly, the escape hatch for a slot the syntax does not
  name. After a `/` come lifetime and hazard controls:

    add  r105, r107, r109 / waitload wait=3 live keepb

  `waitload` is byte0[3], the load-use wait - the bit whose absence made a device load return
  zero for a whole session. `wait=N` is the five-bit mask at byte1[2..6]. `live` is byte4[3] and
  `keepb` byte8[5]. Each is refused on a form whose specification calls that bit opcode or forced.

    python3 tools/g17as.py prog.s -o prog.bin --list
    python3 tools/g17as.py prog.s --check       also ask Apple's decoder whether it agrees
    python3 tools/g17as.py prog.bin --disasm    render a binary back to annotated source
"""
import fractions, json, os, re, sys

# siblings come from the package
import collections
from agxforge.g17 import cf as g17cf, encode as g17encode, modal as g17modal, slice as g17slice

# ANCHORED ON THE CHECKOUT ROOT: two levels up from agxforge/g17/ where one sufficed from
# tools/. Native helpers stay in tools/ where the Makefile builds them.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ISA = os.path.join(ROOT, "isa")
MAPS = os.path.join(ISA, "g17-operand-maps.jsonl")


class AsmError(Exception):
    def __init__(self, line, msg):
        super(AsmError, self).__init__("line %d: %s" % (line, msg))
        self.line, self.msg = line, msg


# ---------------------------------------------------------------- verified operand maps

_MAPS = None


_FORMBITS = None


def form_bits(op, n):
    """What Apple's instances say about this form's fixed bits - see g17opmap.form_bits."""
    global _FORMBITS
    if _FORMBITS is None:
        path = os.path.join(ISA, "g17-form-bits.json")
        try:
            _FORMBITS = json.load(open(path))
        except (IOError, ValueError):
            _FORMBITS = {}
    return _FORMBITS.get("%d,%d" % (op, n)) or {}


def maps():
    """{(opcode, operand, kind): record} from g17opmap.py, which tested each map on every
    cached instance of its opcode rather than the one it was fitted to."""
    global _MAPS
    if _MAPS is None:
        _MAPS = {}
        if os.path.exists(MAPS):
            for line in open(MAPS):
                r = json.loads(line)
                ln = r.get("length")
                if ln:      # defensive: a map written before positions were length-filtered
                    r["positions"] = [p for p in r.get("positions", []) if p[1] < ln]
                    if r.get("order"):
                        r["order"] = [o for o in r["order"] if o[0] < ln]
                _MAPS[(r["opcode"], ln, r["operand"], r.get("kind"))] = r
        _load_overlay(_MAPS)
    return _MAPS


# THE ATOMIC OVERLAY IS GONE, folded into the map on 2026-09-08 when the re-baseline was adopted.
# The loader stays because the mechanism is sound and the file's absence is handled; what follows
# is the history, kept because it says what the mechanism is FOR rather than describing a file that
# no longer exists. Two of its records - the threadgroup atomics' two-bit tables - were strictly
# better than the re-baselined map and were folded in; the other 93 the map already matched.
#
# The original reasoning:
#
# The atomic forms could be round-tripped but not AUTHORED: their operand fields were fitted from
# one instance each, so base and bit-set were determined together and any value but the witness's
# was refused. Re-fitting them needs the wider population (g17opmap --union), and the wider
# population is only offered to fields the corpus left degenerate or refuted - not to fields marked
# `verified` on a single encoded value, which is exactly what these were.
#
# Fixing that in general reclassifies 2,184 records across 436 opcodes, which is a number the class
# table and the ISA peer's slot-set derivation are denominated in. So the general change waits for a
# deliberate re-baseline, and this file carries ONLY the atomic forms, re-fitted with
# --challenge-contradicted. It is additive: every key here is an opcode the compiler has never been
# able to emit, so nothing that already works can move. If a future re-baseline lands, delete it.
OVERLAY = os.path.join(os.path.dirname(MAPS), "g17-operand-maps-atomics.jsonl")
OVERLAY_OPS = {10018, 10019, 10022, 10023, 10090, 10094, 10095,
               11701, 11703, 11705, 11765, 11769}

# THE SECOND OVERLAY, and it is the same mechanism for the same reason. op586's four-byte form -
# `mov`, 4,707 instructions in 2,156 of the 6,594 corpus programs, the single biggest thing the
# backend could not emit - carries its SOURCE LIFETIME in operand 3, and the main map explains
# only 0.7569 of it with one bit. The eight-byte form of the same operand is verified with two,
# byte1[7] and byte4[2], weights 32 and 16; a four-byte instruction has no byte 4, so the fit had
# nowhere to put the second bit and settled for `conditional`. Flipping every bit of one witness
# says byte2[3], and 32*byte1[7] + 16*byte2[3] is exact on all 4,702 register-source instances.
#
# Additive and narrow, like the first: one record, on an operand the compiler could not write at
# all. ledger/g17-the-four-byte-move-and-its-lifetime.toml
OVERLAY2 = os.path.join(os.path.dirname(MAPS), "g17-operand-maps-mov.jsonl")
OVERLAY2_OPS = {586}

# THE THIRD OVERLAY, and it corrects a fit that measured the wrong thing. op17193's fourteen-byte
# store carries a byte displacement in operand 7, and the inherited record modelled it as base
# -1536 plus three `extra` carrier bits. Two of those - byte7[7] and byte8[0] - are the
# instruction's LENGTH bits: flipped alone on Apple's own instance they re-length it to 8 and 10
# bytes, whereupon the decoder prints operand 7 as 32, and the differential fit read that
# difference as a coefficient of 768. The consequence was not academic. 768 is exactly the weight
# of positions 8 and 9 together, so the two models are degenerate, and asking the encoder for a
# displacement of zero let it pay in length bits instead: the store emitted fourteen bytes that
# decode as a ten-byte store followed by four bytes of another instruction.
OVERLAY3 = os.path.join(os.path.dirname(MAPS), "g17-operand-maps-store.jsonl")
OVERLAY3_OPS = {17193}

# op10091's 12-byte operation field has only one observed value, 262659. The
# inherited differential fit assigns byte11[3] a -512 coefficient, although
# that bit is zero in all seven corpus instances and in the five below-Metal
# compare-exchange placements. Admit only this constant; no operation variants
# are inferred from the unobserved bit.
OVERLAY4 = os.path.join(os.path.dirname(MAPS), "g17-operand-maps-cmpxchg.jsonl")
OVERLAY4_OPS = {10091}


def _load_overlay(store):
    _load_one(store, OVERLAY, OVERLAY_OPS)
    _load_one(store, OVERLAY2, OVERLAY2_OPS)
    _load_one(store, OVERLAY3, OVERLAY3_OPS)
    _load_one(store, OVERLAY4, OVERLAY4_OPS)


def _load_one(store, path, ops):
    if not os.path.exists(path):
        return
    for line in open(path):
        r = json.loads(line)
        ln = r.get("length")
        if not ln or r.get("opcode") not in ops:
            continue
        # A TABLE IS A USABLE MODEL TOO. This test asked "does the record carry a linear fit",
        # which is not the same question as "is this record usable" - it silently dropped every
        # table record, and a table is exactly what a field taking {16, 1, 4, 8} needs. The
        # threadgroup atomics could not be authored for that reason: one operand each, refuted as a
        # ramp in the main map, correctable here only as a table.
        if not (r.get("order") and r.get("table")):
            if r.get("base") is None or r.get("step") is None:
                continue
        r["positions"] = [q for q in r.get("positions", []) if q[1] < ln]
        if r.get("order"):
            r["order"] = [o for o in r["order"] if o[0] < ln]
        store[(r["opcode"], ln, r["operand"], r.get("kind"))] = r


def _field(op, idx, kind, length, line):
    r = maps().get((op, length, idx, kind))
    if r is None:
        raise AsmError(line, "op%d operand %d has no %s map - run tools/g17opmap.py"
                       % (op, idx, kind))
    if r["verdict"] not in ("verified", "table"):
        extra = (" (it explains %s of Apple's own instances)" % r["explains"]
                 if r.get("explains") is not None else "")
        raise AsmError(line, "op%d operand %d %s is %s, not authorable%s"
                       % (op, idx, kind, r["verdict"], extra))
    return r


_ENCCACHE = {}


def field_encode(op, idx, kind, value, length, line):
    """The bits a field must hold for `value`, or an explanation of why it cannot hold it.

    Cached, because the choice of coefficient bits is a subset search and a corpus asks for the
    same (field, value) thousands of times.
    """
    ck = (op, idx, kind, value, length)
    hit = _ENCCACHE.get(ck)
    if hit is not None:
        if isinstance(hit, str):
            raise AsmError(line, hit)
        return hit[0], list(hit[1])
    try:
        got = _field_encode(op, idx, kind, value, length, line)
    except AsmError as e:
        _ENCCACHE[ck] = e.args[0].split(": ", 1)[-1] if e.args else "cannot encode"
        raise
    _ENCCACHE[ck] = (got[0], list(got[1]))
    return got


def variant(r, raw=None, key=None):
    """The map a mode-selected record uses for these bytes, or the record itself.

    Some fields are not one map. op12675's operand 1 has two, and b10.6 says which - the
    coefficients of the other bits are measurably different on each side of it. The mode is part
    of the map, so it is read on the way out and written on the way in.
    """
    ms = r.get("modes")
    if not ms:
        return r
    if key is None:
        key = "".join(str((raw[b] >> i) & 1) if b < len(raw) else "0"
                      for b, i in r["mode_bits"])
    v = ms.get(key)
    if v is None:
        return None
    out = dict(r)
    out.update(v)
    return out


def _modal_score(op, length, bits):
    mb = g17modal.modal().get((op, length)) or {}
    return sum(d[1] for (b, i, want) in bits
               for d in [mb.get("%d.%d" % (b, i))] if d and d[0] == want)


_POLY_CACHE = {}
# THE IDENTITY THE IN-MEMORY ENTRIES BELONG TO, not the identity of the tree right now.
#
# Without this, loading entries under stamp A, editing a source, and saving republished those same
# entries under stamp B - so results solved against one assembler were relabelled as valid for a
# different one. The stamp check on load cannot catch that: by then the entries are already in
# memory and carry no record of where they came from. Root reproduced it and added the control.
_POLY_IDENTITY = [None]
_POLY_STATS = [0, 0]        # [calls, solver runs]
# OUTSIDE THE REPO, beside the build cache this project already keeps there. A derived cache in
# the working tree is a file that gets committed by accident, wiped by a clean, or fought over by
# two sessions sharing the checkout - and it is not source, it is a rebuildable artefact.
_POLY_DIR = os.environ.get("G17_CACHE_DIR") or os.path.expanduser("~/.cache/agxforge/g17")
_POLY_DISK = os.path.join(_POLY_DIR, "poly-solve.json")
_POLY_LOADED = [False]

# EVERY FILE THAT CAN CHANGE WHAT A SOLVE MEANS. A cache keyed only on its inputs is a score for a
# model that no longer exists the moment the code that produced it changes: the peer session lost a
# real measurement tonight because its stamp was the corpus size alone and editing a key-defining
# function left every stale entry looking fresh. The map RECORD is already in the per-entry key, so
# a re-fit invalidates one field; this stamp covers the code, which invalidates all of them.
# THE IMPLEMENTATIONS THE SOLVER'S ANSWERS DEPEND ON, by path, not by basename.
#
# This named ("g17as.py", "g17encode.py") and joined them against a directory. Before the migration
# that directory was tools/ and the names resolved; after it the anchor became the checkout root and
# BOTH resolved to files that do not exist, so each was stamped "missing" and the cache stopped
# noticing that its inputs had changed. Naming the compatibility shims would be the wrong repair -
# they hold no solver logic - so these are the package implementations.
_HERE = os.path.dirname(os.path.abspath(__file__))
_POLY_SOURCES = (os.path.join(_HERE, "assembler.py"), os.path.join(_HERE, "encode.py"))


def _poly_stamp():
    """A content identity for everything a cached solve depends on, or None if it cannot be taken.

    CONTENT, NOT stat. The previous version hashed st_mtime_ns and st_size, which cannot see an
    edit that preserves both - and a checkout, a restore from the evidence archive, or a same-size
    change all do exactly that. sha256 of the bytes cannot.

    NONE, NOT A USABLE KEY, WHEN A SOURCE IS MISSING. Stamping the string "missing" produced a
    perfectly good cache identity for a tree whose sources could not be read, so every stale entry
    validated against it. Callers treat None as "no cache".
    """
    import hashlib
    h = hashlib.sha256()
    for path in _POLY_SOURCES:
        try:
            h.update(hashlib.sha256(open(path, "rb").read()).hexdigest().encode())
            h.update(b";")
        except OSError:
            return None
    try:
        h.update(hashlib.sha256(open(MAPS, "rb").read()).hexdigest().encode())
    except OSError:
        return None
    return h.hexdigest()[:16]


def _poly_key(r, op, idx, value, length):
    """Keyed on the MAP RECORD's content, not just its coordinates, so editing one map invalidates
    the solves for it even when the file stamp happens to match."""
    import hashlib
    h = hashlib.sha256(json.dumps({k: r.get(k) for k in ("poly", "positions", "extra", "step",
                                                         "base", "order", "table", "prefer")},
                                  sort_keys=True, default=str).encode()).hexdigest()[:16]
    return "%d|%s|%d|%s|%s|%s" % (op, length, idx, r.get("kind"), value, h)


def _poly_disk_load():
    if _POLY_LOADED[0]:
        return
    _POLY_LOADED[0] = True
    stamp = _poly_stamp()
    if stamp is None:
        return                           # a source could not be read: use no cache at all
    try:
        blob = json.load(open(_POLY_DISK))
        if blob.get("stamp") != stamp:
            return                       # the code or the maps moved; every entry is suspect
        for k, v in (blob.get("entries") or {}).items():
            _POLY_CACHE[k] = (v[0], [tuple(x) for x in v[1]] if v[1] is not None else None)
    except Exception:
        pass
    finally:
        # Whatever was or was not loaded, the entries now in memory belong to THIS identity, and
        # any solve added later in this process was computed against these same sources.
        _POLY_IDENTITY[0] = stamp


def poly_cache_save():
    """Persist solved bit patterns, atomically, merging rather than overwriting.

    Two of these run at once regularly, so a plain dump loses one run's work and can leave a
    half-written file that the next run reads as garbage. Written to a temp file and moved into
    place under an advisory lock; a failed lock does the work anyway rather than blocking.
    """
    import fcntl, tempfile
    stamp = _poly_stamp()
    if stamp is None:
        return                           # refuse to write a cache nothing can be keyed against
    if _POLY_IDENTITY[0] is not None and _POLY_IDENTITY[0] != stamp:
        # THE SOURCES MOVED UNDER US. These entries were solved or loaded against a different
        # assembler/encoder/maps identity, so writing them now would publish them as valid for the
        # current one. Refuse rather than rebrand; the next process reads the sources as they are
        # and solves what it needs. Discarding here would also be sound, but refusing to persist
        # keeps this run's in-memory answers usable for the run that computed them.
        return
    try:
        os.makedirs(_POLY_DIR, exist_ok=True)
        lock = open(_POLY_DISK + ".lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            pass
        merged = {}
        try:
            blob = json.load(open(_POLY_DISK))
            if blob.get("stamp") == stamp:
                merged.update(blob.get("entries") or {})
        except Exception:
            pass
        for k, v in _POLY_CACHE.items():
            merged[k] = [v[0], v[1]]
        os.makedirs(_POLY_DIR, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=_POLY_DIR, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump({"stamp": stamp, "entries": merged}, fh)
        os.replace(tmp, _POLY_DISK)
    except Exception:
        pass


def _poly_cache_atexit():
    """Any tool that solved something leaves the answers behind for the next run."""
    if _POLY_STATS[1]:
        poly_cache_save()


def _poly_encode(r, op, idx, value, length, line):
    """Memoised wrapper around the solver. THIS WAS 97% OF THE COST OF SCORING THE CORPUS.

    Z3_optimize_check ran 292 times for 3,479 instructions at 288ms a call - 84 of 86.5 seconds -
    because render solves the quadratic afresh for every instruction. The solution depends only on
    (opcode, length, operand, kind, value): the equation comes from the map and the soft
    constraints from the same map's preferences. Values repeat heavily across a corpus of 184,349
    instructions, so nearly every solve is one that has already been done.

    `line` is only carried into error messages and is deliberately not part of the key.
    """
    _poly_disk_load()
    key = _poly_key(r, op, idx, value, length)
    _POLY_STATS[0] += 1
    hit = _POLY_CACHE.get(key)
    if hit is None:
        _POLY_STATS[1] += 1
        hit = _poly_encode_solve(r, op, idx, value, length, line)
        _POLY_CACHE[key] = hit
    enc, bits = hit
    return enc, (list(bits) if bits is not None else None)


def _poly_encode_solve(r, op, idx, value, length, line):
    """Bits that make a SECOND-ORDER field read back as `value`, chosen with a solver.

    const + SUM(c_i x_i) + SUM(c_ij x_i x_j) = value is a quadratic pseudo-Boolean equation, and
    there is no subset search that is both exact and affordable at 26 bits. Z3 solves it directly
    and, given the same equation, also picks WHICH solution: Apple's own instances of this form
    prefer particular values in these bits, so those are soft constraints and the equation is the
    hard one. Without them a satisfying assignment is as arbitrary as the solver's search order.
    """
    import z3
    poly = r["poly"]
    bits = [c for c in g17encode.poly_bits(poly) if c[0] < length]
    if not bits:
        raise AsmError(line, "op%d operand %d has no bits inside this %d-byte form" % (op, idx, length))
    known = set(bits)
    var = {c: z3.Bool("b%d_%d" % c) for c in bits}

    def term(c):
        return z3.If(var[c], 1, 0)

    total = poly["const"]
    for b, i, cf in poly["lin"]:
        if (b, i) in known:
            total = total + cf * term((b, i))
    for b1, i1, b2, i2, cf in poly["quad"]:
        if (b1, i1) in known and (b2, i2) in known:
            total = total + cf * term((b1, i1)) * term((b2, i2))
    # THE THIRD-ORDER TERMS ARE PART OF THE EQUATION. Leaving them out solved a different
    # equation than the map describes, and the solver duly satisfied it: byte-exact fell from
    # 93.5% to 77.8% on maps that read every instance correctly.
    for b1, i1, b2, i2, b3, i3, cf in poly.get("cube", ()):
        if (b1, i1) in known and (b2, i2) in known and (b3, i3) in known:
            total = total + cf * term((b1, i1)) * term((b2, i2)) * term((b3, i3))
    opt = z3.Optimize()
    opt.add(total == value)
    mb = g17modal.modal().get((op, length)) or {}
    for c in bits:
        d = mb.get("%d.%d" % c)
        if d:
            opt.add_soft(var[c] == bool(d[0]), weight=max(1, int(d[1] * 1000)))
    if opt.check() != z3.sat:
        raise AsmError(line, "op%d operand %d cannot represent %s: no assignment of its %d bits "
                             "satisfies the map" % (op, idx, value, len(bits)))
    m = opt.model()
    return None, [(b, i, 1 if z3.is_true(m.eval(var[(b, i)], model_completion=True)) else 0)
                  for (b, i) in bits]


def _field_encode(op, idx, kind, value, length, line):
    r = _field(op, idx, kind, length, line)
    if r.get("poly"):
        return _poly_encode(r, op, idx, value, length, line)
    if r.get("modes"):
        best = None
        for key in sorted(r["modes"]):
            sub = variant(r, key=key)
            try:
                e, p = _encode_with(sub, op, idx, value, length, line)
            except AsmError:
                continue
            bits = list(p) + _scatter(sub, e, length) + [
                (b, i, int(key[j])) for j, (b, i) in enumerate(r["mode_bits"]) if b < length]
            rank = _modal_score(op, length, bits)
            if best is None or rank > best[0]:
                best = (rank, bits)
        if best is None:
            raise AsmError(line, "op%d operand %d cannot represent %s in any of its %d modes"
                           % (op, idx, value, len(r["modes"])))
        return None, best[1]
    return _encode_with(r, op, idx, value, length, line)


def _encode_with(r, op, idx, value, length, line):
    if r["verdict"] == "table":
        # A CODE TABLE, not an integer. op10279's operand 4 is five bits whose all-zeros pattern
        # means 1024, the largest value it takes - so writing the value as a number produces a
        # different instruction, every time, and that one field was 12,096 near misses.
        # The fitter records which spelling Apple writes most often for each value; see the
        # `prefer` block in g17opmap. Falling back to the first is what the encoder always did.
        hit = (r.get("prefer") or {}).get(str(value))
        if hit is None:
            hit = next((k for k, v in r["table"].items() if v == value), None)
        if hit is None:
            raise AsmError(line, "op%d operand %d has no encoding for %s; it takes %s"
                           % (op, idx, value, sorted(set(r["table"].values()))[:8]))
        return 0, [(b, i, int(hit[j])) for j, (b, i) in enumerate(r["order"])]
    pos = [tuple(p) for p in r["positions"]]
    # A FIELD WITH NO WEIGHTS AT ALL is legitimate: its coefficients are not powers of two of a
    # common step, so every one of them is a coefficient bit and nothing is left to scale. It
    # crashed the encoder rather than encoding, which cost op17244 55 of its instances.
    weights = sorted(w for w, _, _, _ in pos)
    lo, hi = (weights[0], weights[-1]) if weights else (0, -1)
    outside = sorted({b for _, b, _, _ in pos if b >= length})
    if outside:
        raise AsmError(line, "op%d operand %d has bits in bytes %s, outside this %d-byte form"
                       % (op, idx, outside, length))
    step = fractions.Fraction(r["step"])
    if step == 0:
        raise AsmError(line, "op%d operand %d has a zero step; it encodes nothing" % (op, idx))
    # COEFFICIENT BITS, chosen EXACTLY rather than greedily. Some fields carry bits that are not
    # binary weights of the field - op10864's fourth operand has one worth 144, a register-bank
    # selector - and spending the largest that fits first is not correct: which ones you spend
    # decides whether the remainder lands on the field's grid. There are few of them, so try
    # every subset and keep one that works. Greedy refused register 117 in op554's four-byte
    # movimm, which Apple encodes 169 times in the first 800 programs alone.
    extras = [(b, i, c) for b, i, c in (r.get("extra") or [])]
    for b, i, _c in extras:
        if b >= length:
            raise AsmError(line, "op%d operand %d needs byte %d, outside this %d-byte form"
                           % (op, idx, b, length))

    def usable(enc):
        if enc.denominator != 1 or enc < 0:
            return None
        e = int(enc)
        if not weights:
            return e if e == 0 else None
        if any(w not in weights and (e >> w) & 1 for w in range(hi + 1)):
            return None
        if e % (1 << lo) or e >> (hi + 1):
            return None
        return e

    # WHICH ENCODING, when more than one produces the same value. A field can carry two bits of
    # the same weight - op10282's b4.6 is one, classed `forced` and named as moving nothing - and
    # then the value does not determine the bytes. Taking the fewest coefficient bits is a
    # choice, and it is not Apple's: it re-encoded 44 of op10282's instances to bytes Apple never
    # wrote. So among the encodings that give the right value, take the one that agrees most with
    # what Apple's own instances of this form put in those bits.
    mb = g17modal.modal().get((op, length)) or {}

    def agreement(take, e):
        want = {(b, i): 1 for b, i, _c in take}
        for b, i, _c in extras:
            want.setdefault((b, i), 0)
        for w, b, i, inv in pos:
            if b < length:
                want[(b, i)] = ((int(e) >> w) & 1) ^ inv
        score = 0.0
        for (b, i), bit in want.items():
            d = mb.get("%d.%d" % (b, i))
            if d and d[0] == bit:
                score += d[1]
        return score

    def reads_back(take, enc):
        """What the DECODER would return for the bytes this choice writes.

        The arithmetic that picks `enc` assumes the positions and the coefficient bits are
        disjoint, and in op595's operand 4 they are not: b1.7 is a weight-5 position AND an extra
        worth 16, so it is counted twice on the way out. Reading stays self-consistent - which is
        why the map scored `verified` on all 107 instances - and writing does not: asked for 32
        the subset search chose both 16s, and the bytes it produced read back as 48.
        A map is certified in both directions; an ENCODER should hold itself to the same rule
        rather than trusting that its own model has no overlap.
        """
        raw = bytearray(length)
        for w, b, i, inv in pos:
            if b < length:
                raw[b] = (raw[b] & ~(1 << i)) | (((((int(enc) >> w) & 1) ^ inv) << i))
        for b, i, _c in extras:
            if b < length:
                on = 1 if (b, i, _c) in take else 0
                raw[b] = (raw[b] & ~(1 << i)) | (on << i)
        e = 0
        for w, b, i, inv in pos:
            if b < length:
                e |= (((raw[b] >> i) & 1) ^ inv) << w
        add = sum(c for b, i, c in extras if b < length and (raw[b] >> i) & 1)
        return r["base"] + e * step + add

    chosen, enc, best = None, None, None
    for mask in range(1 << min(len(extras), 14)):
        take = [extras[j] for j in range(min(len(extras), 14)) if (mask >> j) & 1]
        cand = usable(fractions.Fraction(value - r["base"] - sum(c for _, _, c in take)) / step)
        if cand is None:
            continue
        if reads_back(take, cand) != value:
            continue
        rank = (agreement(take, cand), -len(take))
        if best is None or rank > best:
            chosen, enc, best = take, cand, rank
    if enc is None:
        raise AsmError(line, "op%d operand %d cannot represent %s in its %d-byte form (base %s, "
                       "step %s, weights %s)" % (op, idx, value, length, r["base"], r["step"],
                                                 weights))
    picked = [(b, i, 1) for b, i, _c in chosen]
    for b, i, _c in extras:
        if (b, i, 1) not in picked:
            picked.append((b, i, 0))
    return enc, picked


EXPR_SUBS = ("base", "const", "scale")


def expr_records(op, n, idx, fields=None):
    """The three sub-field maps of an address-expression operand, or None if it has none."""
    if fields is None:
        m = maps()
        rs = {sub: m.get((op, n, idx, "expr." + sub)) for sub in EXPR_SUBS}
    else:
        rs = {sub: fields.get((idx, "expr." + sub)) for sub in EXPR_SUBS}
    return rs if all(rs.values()) else None


def expr_ok(rs):
    """Authorable when every sub-field is either mapped or PINNED.

    A sub-field that carries one value across the whole corpus is not a map and this project does
    not call it one. It is still authorable at that value - the encoder refuses any other - and
    refusing the operand outright would give up the address expression on every load and store
    Apple emits, because the scale of a 32-bit load is 8 in all 3,776 of them.
    """
    for r in rs.values():
        if r["verdict"] in ("verified", "table"):
            if not _usable(r):
                return False
        elif r["verdict"] == "pinned":
            continue          # determined by the form, and more certainly than a modal value
        elif not (r["verdict"] == "degenerate" and r.get("modal") is not None):
            return False
    return True


def _scatter(r, enc, length):
    """The bit writes that put `enc` into the field this record describes.

    A table record carries its own writes and gets none from here.
    """
    if not r or r["verdict"] == "table" or enc is None:
        return []
    return [(b, i, ((int(enc) >> w) & 1) ^ inv)
            for w, b, i, inv in (tuple(p) for p in r["positions"]) if b < length]


def expr_encode(op, idx, value, length, line):
    """Bit writes for one address expression: base operand, constant, scale.

    Written as explicit bits rather than through g17encode.encode, which takes ONE value per
    printed operand and an expression carries three.
    """
    rs = expr_records(op, length, idx)
    if rs is None:
        raise AsmError(line, "op%d operand %d is not an address expression here" % (op, idx))
    if len(value) != 3:
        raise AsmError(line, "op%d operand %d needs [opN+K*S]" % (op, idx))
    banks = []
    for sub, v in zip(EXPR_SUBS, value):
        r = rs[sub]
        if r["verdict"] not in ("verified", "table"):
            pin = r["value"] if r["verdict"] == "pinned" else r.get("modal")
            if v != pin:
                raise AsmError(line, "op%d operand %d: %s is %s in every instance Apple emits, "
                                     "so this map cannot place %s" % (op, idx, sub, pin, v))
            continue
        enc, extra = field_encode(op, idx, "expr." + sub, v, length, line)
        banks += extra
        if r["verdict"] != "table":
            for w, b, i, inv in (tuple(p) for p in r["positions"]):
                if b < length:
                    banks.append((b, i, ((int(enc) >> w) & 1) ^ inv))
    return banks


def expr_decode(op, idx, raw):
    """(base, const, scale) read back out of finished bytes."""
    rs = expr_records(op, len(raw), idx)
    if rs is None:
        return None
    out = []
    for sub in EXPR_SUBS:
        r = rs[sub]
        if r["verdict"] in ("verified", "table"):
            v = field_decode(op, idx, "expr." + sub, raw)
        elif r["verdict"] == "pinned":
            v = r["value"]
        else:
            v = r.get("modal")
        if v is None:
            return None
        out.append(int(v))
    return tuple(out)


def kind_here(op, n, idx, kind, raw):
    """Whether this instruction reads operand `idx` as `kind`, when the slot has two readings.

    The selector is a measured bit (`kind_bit`), not a guess; a slot whose readings no single bit
    separates has none, and the caller falls back to whichever reading the corpus uses more.
    """
    r = maps().get((op, n, idx, kind if kind != "expr" else "expr.base"))
    kb = r.get("kind_bit") if r else None
    if not kb:
        return None
    b, i, want = kb
    return b < len(raw) and ((raw[b] >> i) & 1) == want


def field_decode(op, idx, kind, raw):
    """Keyed by the encoded LENGTH, because an opcode's field map differs between its forms."""
    """Read an operand back out of finished bytes through the same map that wrote it."""
    if kind == "expr":
        return expr_decode(op, idx, raw)
    r = maps().get((op, len(raw), idx, kind))
    if r is None or r["verdict"] not in ("verified", "table"):
        return None
    r = variant(r, raw)
    if r is None:
        return None
    if r.get("poly"):
        return g17encode.poly_value(r["poly"], raw)
    if r["verdict"] == "table":
        code = "".join(str((raw[b] >> i) & 1) if b < len(raw) else "0" for b, i in r["order"])
        return r["table"].get(code)
    v = 0
    for w, b, i, inv in (tuple(p) for p in r["positions"]):
        if b < len(raw):
            v |= (((raw[b] >> i) & 1) ^ inv) << w
    out = r["base"] + v * fractions.Fraction(r["step"])
    for b, i, c in (r.get("extra") or []):
        if b < len(raw) and (raw[b] >> i) & 1:
            out += c
    return out


# ---------------------------------------------------------------- forms

# LIFETIME AND HAZARD CONTROLS, written after a "/" so they cannot be confused with a label:
#     add  r105, r107, r109 / waitload wait=3 live keepb
# Each is a bit this project located and recorded separately, and each is applied only when the
# specification for that (opcode, length) does not call the bit opcode or forced - so asking for
# one on a form where it means something else is refused rather than silently written.
CONTROLS = {
    # ledger/g17-alu-load-use-wait.toml - named by differential, executed with a control
    "waitload": (0, 3),
    # ledger/g17-byte4-liveness-metadata.toml - corpus-correlated, inert under execution
    "live":     (4, 3),
    # the operand-b keep bit, recovered while authoring
    "keepb":    (8, 5),
    # the store's wait bit
    "waitstore": (9, 5),
}
# ledger/g17-the-wait-tag-is-a-mask.toml - five bits at byte1[2..6], a MASK over dependency
# slots rather than an index. Corpus-measured, not yet executed.
WAIT_BITS = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]

# THE ATOMIC OPERATION IS A FIELD NO FORM NAMED, and leaving it unnamed made render refuse 25 of
# 52 op10090 encodings as "near miss": it read the operands correctly, re-encoded with the
# operation bits filled MODALLY, and got a different instruction back. Every one of those 25
# differs from its re-encode in byte5[3], byte6[3] or byte4[5] - which is the three-bit operation
# selector, add 0, and 1, sub 3, max 4, min 5, or 6, xor 7 (float add is also 3; see
# ledger/g17-the-atomic-operation-is-a-field-and-a-length.toml).
#
# Carried as an OPCODE-SCOPED control rather than in CONTROLS, which is global: these bit
# positions mean other things in other opcodes, and a global entry would start naming them
# wherever they happen to be free. Written LSB-first exactly like wait=.
ATOMIC_OPS = {10018, 10019, 10022, 10023, 10090, 10094, 10095,
              11701, 11703, 11705, 11765, 11769}
OPCODE_BITFIELDS = {op: {"aop": [(4, 5), (5, 3), (6, 3)]} for op in ATOMIC_OPS}

# THE SPEC CALLS THREE OF THOSE BITS FIXED AND APPLE VARIES THEM. g17encode.spec_for classes a bit
# `opcode` or `forced` per (opcode, length), and encode_item refuses to write one - correctly, since
# writing an opcode bit produces an instruction the decoder reads as something else, which is how
# malformed bytes get dispatched. But for these seven triples the classification is refuted by
# Apple's own instances: BOTH values occur, decoding to the SAME opcode at the SAME length, so the
# bit is an operand there and the class is a gap in the spec rather than a hazard.
#
# Measured over distinct cached encodings; the count is how many carry that (opcode, length).
#   op10018 l10  b4.5 opcode, b6.3 forced      27 encodings, both values
#   op10018 l12  b6.3 forced                   27
#   op10090 l10  b5.3 opcode                   25
#   op10090 l12  b6.3 forced                   27
#   op10094 l10  b5.3 opcode                   48
#   op10094 l12  b6.3 forced                    9
#
# Scoped to exactly these triples. Anything not listed still goes through the spec's refusal.
SPEC_MEASURED_OPERAND = {
    (10018, 10, 4, 5), (10018, 10, 6, 3), (10018, 12, 6, 3),
    (10090, 10, 5, 3), (10090, 12, 6, 3),
    (10094, 10, 5, 3), (10094, 12, 6, 3),
    # THE THREADGROUP ATOMIC'S OPERATION, added 2026-09-08 on the same standard as the four above:
    # both values of each bit decode as op11765 at length 12 in APPLE'S OWN compilations, which
    # are tga-add..tga-xor in the probe cache. b5.3 is 0 in add and 1 in sub; b6.3 is 0 in add and
    # 1 in min. Flipping b5.3 ALONE in the add witness does decode as op11766 - which is why the
    # form derivation calls it opcode - but the seven shipped instances take both values at this
    # opcode, so it is a bit shared between the opcode and operand 2's table key rather than an
    # opcode bit. tools/g17regress.py re-checks that against the witnesses.
    (11765, 12, 5, 3), (11765, 12, 6, 3),
}

CF = {"if": ("if", False), "if.inv": ("if", True), "else": ("else", False),
      "else.inv": ("else", True), "while": ("while", False), "while.inv": ("while", True),
      "pop": ("pop", False)}
BRANCH = {"br.none": "pop", "br.any": "else", "br.while": "while"}

# EXPLICIT FORMS. A mnemonic names one (opcode, length, operand signature); a slot's role cannot
# be read off the register class, since op423 looks like three registers and is `dest, src, imm`.
# `mods` are modifier slots the syntax does not name; they take the value Apple's own instances
# carry most often, which is recorded rather than invented, and the listing says so.
# Restricted to opcodes this project executed or matched exactly on solver-chosen inputs.
CURATED = {
    "and":  {"op": 423,   "slots": [(0, "reg"), (2, "reg"), (4, "imm")], "mods": [1, 3]},
    "or":   {"op": 13574, "slots": [(0, "reg"), (2, "reg"), (4, "imm")], "mods": [1, 3]},
    "xor":  {"op": 17770, "slots": [(0, "reg"), (2, "reg"), (4, "imm")], "mods": [1, 3]},
    "add":  {"op": 10295, "slots": [(0, "reg"), (2, "reg"), (4, "reg")], "mods": [1, 3, 5]},
    "sub":  {"op": 11680, "slots": [(0, "reg"), (2, "reg"), (4, "reg")], "mods": [1, 3, 5]},
    "mul":  {"op": 10864, "slots": [(0, "reg"), (2, "reg"), (4, "reg")], "mods": [1, 3, 5]},
    "nand": {"op": 13473, "slots": [(0, "reg"), (2, "reg"), (4, "reg")], "mods": [1, 3, 5]},
    "xnor": {"op": 17757, "slots": [(0, "reg"), (2, "reg"), (4, "reg")], "mods": [1, 3, 5]},
    "andn": {"op": 410,   "slots": [(0, "reg"), (2, "reg"), (4, "reg")], "mods": [1, 3, 5]},
    "mov":  {"op": 586,   "slots": [(0, "reg"), (2, "reg")],             "mods": [1, 3]},
}

def _usable(r):
    """A field is offerable if it has any bits inside this form. Whether a PARTICULAR value fits
    is decided per value in field_encode, because a sparse field still reaches some values."""
    if r.get("verdict") == "table":
        return bool(r.get("order"))
    if r.get("poly"):
        # A SECOND-ORDER MAP HAS NO `positions`; its bits are the terms of the polynomial.
        # Testing the wrong field made every form carrying one unofferable, which cost 4,429
        # instructions in one run.
        return bool(r["poly"].get("lin") or r["poly"].get("quad"))
    if r.get("modes"):
        return any(v.get("positions") or v.get("order") for v in r["modes"].values())
    """A field with a hole in its weights cannot hold an arbitrary value, so it cannot be a
    named operand - it would render to a mnemonic and then refuse to assemble."""
    return bool(r.get("positions"))


def derive_form(op, n, fields=None):
    """The operand signature of one (opcode, length), derived rather than hand-written.

    A slot's role cannot be read off its register class: op423 looks like three registers and is
    `dest, src, imm`. So a slot whose class names a register file is a register; a slot with NO
    class is an immediate when Apple's instances give it many different values and a modifier when
    they nearly always give it one.

    THE SLOTS WITH NO CLASS ARE THE POINT. `.form` used to keep only the slots that HAVE a class,
    which silently dropped every immediate - op423 came out as two registers and refused
    `and r105, r107, #15`, while the same opcode and length reached through its curated mnemonic
    took three operands and assembled. Two routes to one form disagreeing about its shape is worse
    than either being wrong, because only one of them tells you.
    """
    if fields is None:
        fields = {}
        for (o, length, idx, kind), r in maps().items():
            if o == op and length == n:
                fields[(idx, kind)] = r
    try:
        classes = g17encode.auth()[op].get("operands") or []
    except Exception:
        return None
    slots, mods = [], []
    for i, c in enumerate(classes):
        # A NAMED CLASS THAT IS NOT A GPR IS STILL A REGISTER. op14060's operand 2 is `SIR32`, a
        # special register, and it fell through both branches - not a slot because the class does
        # not say GPR, not a modifier because the class is not absent - so nothing ever wrote it
        # from its own map and the modal fill guessed its bits. That guess is not legal in every
        # instruction: it is 21 of the 30 encodings that came out as not an instruction at all.
        # Only claim it when it has a map to be written from; otherwise leave today's behaviour.
        if c and "GPR" not in c:
            r = fields.get((i, "reg"))
            if r and r["verdict"] in ("verified", "table") and _usable(r):
                slots.append((i, "reg"))
                continue
        if c and "GPR" in c:
            r = fields.get((i, "reg"))
            rs = expr_records(op, n, i, fields)
            # A slot that is a register in some instances and an ADDRESS EXPRESSION in others is
            # one slot with two readings, not a broken register. Take whichever reading Apple
            # uses more often here: op12682's operand 3 is an expression in 3,776 of its
            # eight-byte instances and a register in 22, and calling it a register loses the
            # 3,776 rather than the 22.
            # A PINNED OPERAND CANNOT BE ENCODED WRONGLY because it cannot be encoded at all:
            # no bit reaches it and every instance of the form carries the same value. It is not
            # a slot and it must not block the form.
            if r and r.get("verdict") == "pinned":
                continue
            reg_ok = bool(r and r["verdict"] in ("verified", "table") and _usable(r))
            exp_ok = bool(rs and expr_ok(rs))
            if reg_ok and exp_ok:
                # BOTH READINGS, when a measured bit says which. op586's operand 2 is a register
                # in 795 of its four-byte instances and an address expression in 990; picking the
                # commoner one threw away the other 795. The slot takes either, the syntax says
                # which, and the selector bit is written to match.
                kb = (r.get("kind_bit"), rs["base"].get("kind_bit"))
                slots.append((i, "reg|expr" if all(kb) else
                              ("expr" if rs["base"]["n"] > r["n"] else "reg")))
            elif reg_ok:
                slots.append((i, "reg"))
            elif exp_ok:
                slots.append((i, "expr"))
            else:
                return None
        elif c is None:
            r = fields.get((i, "imm"))
            # AN UNCLASSED OPERAND CAN STILL BE AN ADDRESS EXPRESSION. This branch only ever
            # looked for an `imm` map, so op3322's operand 0 - whose three expr sub-maps are all
            # verified - fell through to `mods`, and a modifier is filled from an imm or reg map
            # that does not exist. Nothing wrote it. Its const sub-field carries six bits that
            # vary across its instances, so the bits stayed at the skeleton's value and the form
            # re-encoded to a different instruction in 14 of its 16.
            rs = expr_records(op, n, i, fields)
            if not (r and _usable(r)) and rs and expr_ok(rs):
                slots.append((i, "expr"))
                continue
            # A table field is a real operand however few values it takes - the values ARE the
            # field - so it does not have to clear the diversity bar a linear field does.
            if r and _usable(r) and (
                    r["verdict"] == "table"
                    or (r["verdict"] == "verified" and len(r.get("seen") or []) >= 4)):
                slots.append((i, "imm"))
            else:
                mods.append(i)
    if not slots:
        return None
    # AN OPERAND PAST THE END OF THE CLASS LIST is still an operand. Apple's own record for
    # op12674 names nine, and its sixteen-byte form prints ten - so operand 9 was neither a slot
    # nor a modifier and its four bits went wherever the modal fill put them, in every one of its
    # 137 instances. A slot needs a class to know what it is; a modifier only needs a map.
    known = {i for i, _c in enumerate(classes)}
    for i in sorted({i for i, _k in fields} - known):
        if any((fields.get((i, k)) or {}).get("verdict") in ("verified", "table")
               for k in ("imm", "reg")):
            mods.append(i)
    return {"op": op, "len": n, "slots": slots, "mods": mods}


_FORMS = None


def forms_table():
    """Every (opcode, length) whose real operands are all verified, with a derived signature.

    Hand-writing a signature per opcode does not scale and is guesswork: a slot's role cannot be
    read off the register class, since op423 looks like three registers and is `dest, src, imm`.
    Derive it instead. A slot whose class names a register file is a register operand. A slot with
    no class is an immediate if Apple's instances give it many different values and a MODIFIER if
    they nearly always give it one - which is the same distinction the modal-bit work rests on.

    An opcode is only offered when every one of its real operands is `verified`, so a mnemonic
    that renders is a mnemonic that assembles.
    """
    global _FORMS
    if _FORMS is not None:
        return _FORMS
    _FORMS = dict(CURATED)
    for m, f in CURATED.items():
        f.setdefault("len", g17encode.length(f["op"]))
    byop = collections.defaultdict(dict)
    for (op, length, idx, kind), r in maps().items():
        byop[(op, length)][(idx, kind)] = r
    # A CURATED ENTRY NO LONGER SUPPRESSES THE DERIVED ONE. The curated shapes were written when
    # the maps were poor and are now the worse description: `mov` is hand-written as two
    # registers, while op586's four-byte form derives as four operands whose third reads either
    # as a register or as an address expression. Suppressing derivation left render holding only
    # the shape that cannot read those instances - op586 blocked 3,808 instructions with just 68
    # near misses, which is a readability failure wearing an encoding failure's clothes.
    cand = {}
    for (op, n), fields in byop.items():
        if n is None:
            continue
        f = derive_form(op, n, fields)
        if f is None:
            continue
        cand[(op, n)] = f
    seen = collections.Counter(g17slice.KNOWN_OPS.get(o, "op%d" % o) for o, _ in cand)
    for (op, n), f in sorted(cand.items()):
        nm = g17slice.KNOWN_OPS.get(op, "op%d" % op)
        if len([1 for o, _ in cand if o == op]) > 1:
            nm = "%s.l%d" % (nm, n)
        if seen[nm] > 1 or nm in _FORMS:
            nm = "%s@%d" % (nm, op)
        while nm in _FORMS:
            nm += "'"
        _FORMS[nm] = f
    return _FORMS


FORMS = {}


def form_length(f):
    return f.get("len") or g17encode.length(f["op"])


# ---------------------------------------------------------------- parsing

SPLIT = re.compile(r"\s*,\s*")


def parse_operand(tok, line):
    tok = tok.strip()
    idx = None
    m = re.fullmatch(r"op(\d+)=(.+)", tok)
    if m:
        idx, tok = int(m.group(1)), m.group(2).strip()
    mem = re.fullmatch(r"\[\s*([rR]\d+)\s*\]", tok)
    if mem:
        return idx, "addr", int(mem.group(1)[1:])
    # An ADDRESS EXPRESSION, written without commas so it survives operand splitting:
    #     [op0+0*8]      base operand 0, constant 0, scale 8
    ex = re.fullmatch(r"\[\s*op(\d+)\s*([+-]\s*\d+)\s*\*\s*(\d+)\s*\]", tok)
    if ex:
        return idx, "expr", (int(ex.group(1)), int(ex.group(2).replace(" ", "")),
                             int(ex.group(3)))
    if re.fullmatch(r"[rR]\d+", tok):
        return idx, "reg", int(tok[1:])
    if re.fullmatch(r"[pP]\d+", tok):
        return idx, "pred", int(tok[1:])
    if tok.startswith("#") or re.fullmatch(r"[-+]?(0[xXbo])?[0-9A-Fa-f]+", tok):
        try:
            return idx, "imm", int(tok[1:] if tok.startswith("#") else tok, 0)
        except ValueError:
            raise AsmError(line, "malformed immediate %r" % tok)
    if re.fullmatch(r"[A-Za-z_.$][\w.$]*", tok):
        return idx, "label", tok
    raise AsmError(line, "cannot parse operand %r" % tok)


class Inst(object):
    def __init__(self, text, line, mnem, ops, raw=None):
        self.text, self.line, self.mnem, self.ops, self.raw = text, line, mnem, ops, raw
        self.addr, self.bytes, self.mods, self.controls = None, b"", [], []


class Program(object):
    def __init__(self, text, symbols, insts, entry):
        self.text, self.symbols, self.insts, self.entry = text, symbols, insts, entry

    def listing(self):
        out = ["  addr  bytes                         source"]
        for i in self.insts:
            out.append("  %04x  %-29s %-34s%s"
                       % (i.addr, i.bytes.hex(), i.text, ("  ; " + ", ".join(i.mods))
                          if i.mods else ""))
        return "\n".join(out)


def parse(source):
    insts, pending, syms, forms, entry = [], [], {}, dict(forms_table()), None
    for n, src in enumerate(source.splitlines(), 1):
        s = re.sub(r";.*$", "", src).strip()
        while s:
            m = re.match(r"([A-Za-z_.$][\w.$]*)\s*:", s)
            if not m:
                break
            pending.append((m.group(1), n))
            s = s[m.end():].strip()
        if not s:
            continue
        ctl = ""
        if "/" in s:
            s, _, ctl = s.partition("/")
            s = s.strip()
        parts = s.split(None, 1)
        head, rest = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
        if head == ".entry":
            entry = int(rest, 0)
            continue
        if head == ".form":
            a = rest.split()
            m = re.fullmatch(r"op(\d+)\.l(\d+)", a[1]) if len(a) > 1 else None
            if not m:
                raise AsmError(n, ".form NAME opNNN.lLL")
            op, ln = int(m.group(1)), int(m.group(2))
            # THE SAME DERIVATION THE MNEMONIC TABLE USES. Naming a form explicitly used to build
            # its signature from the operand classes alone, keeping only the slots that have one -
            # which drops every immediate, because an immediate's class is None.
            curated = next((dict(f) for f in CURATED.values()
                            if f["op"] == op and f.get("len") == ln), None)
            f = curated or derive_form(op, ln)
            if f is None:
                raise AsmError(n, "op%d has no authorable form at %d bytes" % (op, ln))
            forms[a[0].lower()] = f
            continue
        if head in (".kernel", ".section", ".text"):
            continue
        if head == ".byte":
            raw = bytes(int(x, 0) & 0xFF for x in SPLIT.split(rest) if x.strip())
            it = Inst(s, n, ".byte", [], raw)
        else:
            it = Inst(s, n, head, [parse_operand(t, n) for t in SPLIT.split(rest) if t.strip()])
        it.controls = []
        for tok in ctl.split():
            tok = tok.strip().lower()
            if tok in CONTROLS:
                it.controls.append((tok, 1))
            elif tok.startswith("aux="):
                it.controls.append(("aux", int(tok[4:], 0)))
            elif tok.startswith("aop="):
                try:
                    v = int(tok[4:], 0)
                except ValueError:
                    raise AsmError(n, "malformed %r" % tok)
                if not 0 <= v < 8:
                    raise AsmError(n, "atomic operation code %d does not fit three bits" % v)
                it.controls.append(("aop", v))
            elif tok.startswith("wait="):
                try:
                    v = int(tok[5:], 0)
                except ValueError:
                    raise AsmError(n, "malformed %r" % tok)
                if not 0 <= v < 32:
                    raise AsmError(n, "wait mask %d does not fit five bits" % v)
                it.controls.append(("wait", v))
            else:
                raise AsmError(n, "unknown control %r; known are %s, wait=N and aux=N"
                               % (tok, ", ".join(sorted(CONTROLS))))
        for name, ln in pending:
            if name in syms:
                raise AsmError(ln, "duplicate label %r" % name)
            syms[name] = it
        pending = []
        insts.append(it)
    if pending:
        raise AsmError(pending[0][1], "label %r has no instruction after it" % pending[0][0])
    return insts, syms, forms, entry


def size_of(it, forms):
    if it.mnem == ".byte":
        return len(it.raw)
    if it.mnem == "end" or it.mnem in CF:
        return 4
    if it.mnem in BRANCH:
        return 10
    if it.mnem in forms:
        return form_length(forms[it.mnem])
    raise AsmError(it.line, "unknown mnemonic %r" % it.mnem)


def encode_item(it, syms, forms):
    if it.mnem == ".byte":
        return it.raw
    if it.mnem == "end":
        if it.ops:
            raise AsmError(it.line, "end takes no operands")
        return g17cf.encode_end()
    if it.mnem in CF:
        kind, inv = CF[it.mnem]
        pred, count = 74, 1
        aux = dict(it.controls).get("aux")
        for _, k, v in it.ops:
            if k == "pred":
                pred = v
            elif k == "imm":
                count = v
            else:
                raise AsmError(it.line, "%s takes a predicate register and a count" % it.mnem)
        it.mods = ["count=%d" % count] + (["inverted"] if inv else [])
        if kind != "pop":
            it.mods.append("pred=p%d" % pred)
        try:
            return g17cf.encode_exec(kind, count, pred, inv, aux)
        except ValueError as e:
            raise AsmError(it.line, str(e))
    if it.mnem in BRANCH:
        if len(it.ops) != 1:
            raise AsmError(it.line, "%s takes one target" % it.mnem)
        _, k, v = it.ops[0]
        if k == "label":
            if v not in syms:
                raise AsmError(it.line, "undefined label %r" % v)
            disp = syms[v].addr - it.addr
        elif k == "imm":
            disp = v
        else:
            raise AsmError(it.line, "%s needs a label or a displacement" % it.mnem)
        it.mods = ["-> %+d" % disp]
        try:
            return g17cf.encode_branch(BRANCH[it.mnem], disp,
                                       dict(it.controls).get("aux", 0))
        except ValueError as e:
            raise AsmError(it.line, str(e))

    form = forms[it.mnem]
    op, n = form["op"], form_length(form)
    vals, kinds = {}, {}
    for idx, k, v in it.ops:
        if idx is not None:
            vals[idx], kinds[idx] = v, ({"imm": "imm", "expr": "expr"}.get(k, "reg"))
    positional = [o for o in it.ops if o[0] is None]
    free = [sl for sl in form["slots"] if sl[0] not in vals]
    if len(positional) != len(free):
        raise AsmError(it.line, "%s takes %d operands (%s), got %d"
                       % (it.mnem, len(form["slots"]),
                          ", ".join(k for _, k in form["slots"]), len(positional) + len(vals)))
    for (idx, want), (_, k, v) in zip(free, positional):
        got = k if k in ("imm", "expr") else "reg"
        if want == "reg|expr" and got in ("reg", "expr"):
            want = got
        if got != want:
            raise AsmError(it.line, "%s operand %d must be %s, got %s" % (it.mnem, idx, want, k))
        vals[idx], kinds[idx] = v, want
    # ALIASING IS LEGAL, and I had this backwards. Refusing a register that appears twice threw
    # out 123 of the first 400 real programs: Apple writes `and r105, r105, #15` constantly,
    # because `x = x & 15` is ordinary. There is no evidence of a tied-operand rule here, so the
    # check is opt-in with --no-alias rather than a default I made up.
    if "--no-alias" in sys.argv:
        seen = set()
        for idx, v in sorted(vals.items()):
            if kinds[idx] == "reg":
                if v in seen:
                    raise AsmError(it.line, "r%d appears in two operands" % v)
                seen.add(v)
    # A TABLE OPERAND IS WRITTEN HERE, NOT BY THE LINEAR ENCODER. g17encode.encode places an
    # operand through the opcode-wide field map, which is not length-keyed and for op554's
    # operand 1 includes b0.3 - an opcode bit of the four-byte form. Handing it a value of zero
    # made it clear that bit and destroy the opcode. Table fields carry their own bit writes.
    # THE MAP THAT WAS VERIFIED IS THE MAP THAT WRITES. Handing the encoded value to
    # g17encode.encode scatters it through a DIFFERENT field map - the one the classifier derived
    # for the opcode, with its own steps - and the two disagree: op12682's operand 1 was asked for
    # 25,165,824, encoded correctly against the map that had just been validated on 3,798
    # instances, and read back as 1,048,576. So place every operand from its own record, and let
    # the encoder supply only the skeleton the specification fixes.
    enc, banks = {}, []
    for i, v in vals.items():
        if kinds[i] == "expr":
            banks += expr_encode(op, i, v, n, it.line)
            kb = maps().get((op, n, i, "expr.base"), {}).get("kind_bit")
            if kb:
                banks.append((kb[0], kb[1], kb[2]))
            continue
        e, p = field_encode(op, i, kinds[i], v, n, it.line)
        banks += p
        r = maps().get((op, n, i, kinds[i]))
        banks += _scatter(r, e, n)
        kb = (r or {}).get("kind_bit")
        if kb:
            banks.append((kb[0], kb[1], kb[2]))
    for idx in form.get("mods", []):
        if idx in vals:
            continue                  # the source named it; never overwrite that with a default
        r = maps().get((op, n, idx, "imm")) or maps().get((op, n, idx, "reg"))
        if r and r["verdict"] == "verified" and r.get("modal") is not None:
            # A DEFAULT is best-effort: if the field cannot express Apple's modal value, leave
            # the bits the specification itself places rather than refusing the program. An
            # operand the source names explicitly is never treated this way.
            try:
                e, p = field_encode(op, idx, r["kind"], r["modal"], n, it.line)
                banks += p
                banks += _scatter(r, e, n)
                it.mods.append("op%d=%s (Apple's modal value)" % (idx, r["modal"]))
            except AsmError:
                it.mods.append("op%d left to the spec" % idx)
    try:
        out = g17encode.encode(op, enc, length_hint=n)
    except (KeyError, ValueError) as e:
        raise AsmError(it.line, "op%d: %s" % (op, e))
    if len(out) != n:
        raise AsmError(it.line, "encoder produced %d bytes, form is %d" % (len(out), n))
    if banks:
        buf = bytearray(out)
        for b, i, bit in banks:
            if b < n:
                buf[b] = (buf[b] & ~(1 << i)) | (bit << i)
        out = bytes(buf)
        banks = []
    # BITS NOBODY CLAIMED GET APPLE'S VALUE, not zero. The specification requires nothing of
    # them and zero is as legal as one, but an all-zero op423 is an instruction Apple never emits:
    # b0.5, b6.5 and b6.7 are set in 99.95%, 99.95% and 100% of its instances. These are mostly
    # bits of a modifier operand the source does not name, so they are classed `operand` rather
    # than `invisible` and the free-bit census never reached them. Bits an encoded operand owns
    # are excluded, and the readback below still has to pass.
    owned = set()
    for i2, v in vals.items():
        if kinds[i2] == "expr":
            for sub in EXPR_SUBS:
                r = maps().get((op, n, i2, "expr." + sub)) or {}
                if r.get("verdict") == "table":
                    owned |= {(b, bit) for b, bit in r["order"]}
                elif r.get("verdict") == "verified":
                    owned |= {(b, bit) for _, b, bit, _ in (tuple(x) for x in r["positions"])}
                    owned |= {(b, bit) for b, bit, _ in (r.get("extra") or [])}
                kb = r.get("kind_bit")
                if kb:
                    owned.add((kb[0], kb[1]))
            continue
        r = maps().get((op, n, i2, kinds[i2]))
        if r:
            # THE READING SELECTOR IS OWNED BY WHICHEVER READING IS WRITTEN. It was added to the
            # owned set in the expression branch and not in the register one, so on a register
            # instance the modal fill overwrote it with the majority - which for op12675 is the
            # EXPRESSION reading, 1,603 of its 1,765 instances. The encoder wrote the right bit
            # and a later pass put it back. 64 near misses in 200 programs, every one differing
            # at exactly that bit and nothing else.
            if r.get("kind_bit"):
                owned.add((r["kind_bit"][0], r["kind_bit"][1]))
            if r["verdict"] == "table":
                owned |= {(b, bit) for b, bit in r["order"]}
                continue
            for v in list((r.get("modes") or {}).values()) + [r]:
                owned |= {(b, bit) for _, b, bit, _ in (tuple(p) for p in v["positions"])}
                owned |= {(b, bit) for b, bit, _ in (v.get("extra") or [])}
                if v.get("poly"):
                    owned |= set(g17encode.poly_bits(v["poly"]))
            owned |= {(b, bit) for b, bit in (r.get("mode_bits") or [])}
    spec = g17encode.spec_for(op, n) or {}
    fb = form_bits(op, n)
    # A BIT THE SPECIFICATION DESCRIBES WRONGLY. Four bits in the whole corpus are declared with
    # one value and carried with the other in every instance Apple emits; the specification is
    # the thing that is wrong there, and the measurement says which.
    if fb.get("constant"):
        buf = bytearray(out)
        for k, v in fb["constant"].items():
            b, i2 = (int(x) for x in k.split("."))
            if b < n:
                buf[b] = (buf[b] & ~(1 << i2)) | (v << i2)
        out = bytes(buf)
    mb = g17modal.modal().get((op, n))
    if mb and "--zero-unclaimed" not in sys.argv:
        buf, filled = bytearray(out), 0
        for k, d in mb.items():
            b, i2 = (int(x) for x in k.split("."))
            if b >= n or (b, i2) in owned or d[1] < 0.75:
                continue
            cls = (spec.get(k, {}).get("class") if isinstance(spec.get(k), dict)
                   else spec.get(k))
            if cls in ("opcode", "forced") and [b, i2] not in fb.get("unforced", []):
                continue
            if ((buf[b] >> i2) & 1) != d[0]:
                buf[b] = (buf[b] & ~(1 << i2)) | (d[0] << i2)
                filled += 1
        if filled:
            it.mods.append("%d unclaimed bits set to Apple's modal value" % filled)
        out = bytes(buf)
    if banks:
        buf = bytearray(out)
        for b, i, bit in banks:
            cls = (spec.get("%d.%d" % (b, i)) or {}).get("class")
            if cls == "opcode":
                raise AsmError(it.line, "op%d needs bit b%d.%d, but the specification calls it "
                               "an opcode bit" % (op, b, i))
            buf[b] = (buf[b] & ~(1 << i)) | (bit << i)
        out = bytes(buf)
    if it.controls:
        buf = bytearray(out)
        for name, v in it.controls:
            if name == "wait":
                places = WAIT_BITS
            elif name == "aop":
                bf = OPCODE_BITFIELDS.get(op, {}).get("aop")
                if bf is None:
                    raise AsmError(it.line, "op%d has no atomic operation field" % op)
                places = bf
            else:
                places = [CONTROLS[name]]
            for j, (b, i2) in enumerate(places):
                if b >= n:
                    raise AsmError(it.line, "%s needs byte %d, outside this %d-byte form"
                                   % (name, b, n))
                k = "%d.%d" % (b, i2)
                cls = (spec.get(k, {}).get("class") if isinstance(spec.get(k), dict)
                       else spec.get(k))
                if cls in ("opcode", "forced") and (op, n, b, i2) not in SPEC_MEASURED_OPERAND:
                    raise AsmError(it.line, "%s wants b%d.%d, which op%d calls %s"
                                   % (name, b, i2, op, cls))
                bit = (v >> j) & 1 if name in ("wait", "aop") else v
                buf[b] = (buf[b] & ~(1 << i2)) | (bit << i2)
            it.mods.append("%s=%d" % (name, v) if name in ("wait", "aop") else name)
        out = bytes(buf)
    # THE BITS NOBODY ASKED TO CHANGE. Writing modal and bank bits after the encoder runs is the
    # same act that made a peer's template probe silently test nothing: bits get overwritten by a
    # later pass and the caller never learns. So assert the identity bits survived - every bit the
    # specification calls opcode or forced must still hold its specified value.
    # A FORCED BIT APPLE VARIES IS NOT FORCED. The classification is measured against the
    # instances of the form in g17opmap, and a bit an operand map owns and Apple's own encodings
    # disagree on is recorded as `unforced` - asserting it refused 192 of op17257's 265 instances.
    unforced = {(b, i2) for b, i2 in fb.get("unforced", [])}
    for (o, l, _i, _k), r in maps().items():
        if o == op and l == n:
            unforced |= {(b, i2) for b, i2 in (r.get("unforced") or [])}
    for k, d in spec.items():
        if not isinstance(d, dict) or d.get("class") not in ("opcode", "forced"):
            continue
        b, i2 = (int(x) for x in k.split("."))
        if (b, i2) in unforced and d.get("class") == "forced":
            continue
        # The same seven measured triples the controls pass admits. `unforced` covers a FORCED bit
        # the maps disagree with; three of these are classed `opcode`, where the check is otherwise
        # absolute and rightly so. They are listed one at a time, with both values observed decoding
        # to this opcode at this length in Apple's own instances - see SPEC_MEASURED_OPERAND.
        if (op, n, b, i2) in SPEC_MEASURED_OPERAND:
            continue
        want = fb.get("constant", {}).get(k, d["value"])
        if b < n and ((out[b] >> i2) & 1) != want:
            raise AsmError(it.line, "op%d: b%d.%d is %s and a later pass changed it to %d"
                           % (op, b, i2, d["class"], (out[b] >> i2) & 1))
    # READ BACK THROUGH THE SAME SPECIFICATION. A field with a hole in it encodes a plausible
    # different value; nothing else catches that, and Apple's decoder is not consulted here.
    for idx, v in vals.items():
        back = field_decode(op, idx, kinds[idx], out)
        if back is not None and back != v:
            raise AsmError(it.line, "op%d operand %d was asked for %s and reads back as %s"
                           % (op, idx, v, back))
    return out


def assemble(source, entry=None):
    """Source text to a Program: .text bytes, .symbols {name: address}, .insts, .entry."""
    insts, syms, forms, declared = parse(source)
    addr = 0
    for it in insts:
        it.addr = addr
        addr += size_of(it, forms)
    for it in insts:
        it.bytes = encode_item(it, syms, forms)
    text = b"".join(it.bytes for it in insts)
    ent = entry if entry is not None else (declared if declared is not None else 0)
    return Program(text, {k: v.addr for k, v in syms.items()}, insts, ent)


def _read_operands(op, ln, form, raw):
    """The printed operands of this instruction under this form, or None if it cannot read it.

    A slot that takes either reading is read the way THIS instruction encodes it, decided by the
    measured selector bit. A slot that takes only the other reading means the form is the wrong
    description for these bytes - which is a reason to try another form, not to give up.
    """
    args = []
    for idx, kind in form["slots"]:
        if kind == "reg|expr":
            kind = ("reg" if kind_here(op, ln, idx, "reg", raw) else
                    "expr" if kind_here(op, ln, idx, "expr", raw) else None)
            if kind is None:
                return None
        elif kind_here(op, ln, idx, kind, raw) is False:
            return None
        v = field_decode(op, idx, kind, raw)
        if v is None:
            return None
        args.append(("[op%d%+d*%d]" % v) if kind == "expr"
                    else (("r%d" % v) if kind == "reg" else ("#%d" % v)))
    return args


def render(code, spans, notes=None):
    """Bytes plus [(offset, length, opcode)] back to source, so a program can be round-tripped.

    `notes` collects why an instruction fell back to raw bytes, keyed by offset, so a caller can
    still tell "no form for this opcode" from "a form that re-encodes to something else".
    """
    tbl = forms_table()
    by_op = collections.defaultdict(list)
    for m, f in tbl.items():
        by_op[(f["op"], f["len"])].append(m)
    out = [".entry 0x%x" % (spans[0][0] if spans else 0)]
    for off, ln, op in spans:
        raw = code[off:off + ln]
        name = {"exec": None}.get(op)
        if op == 684:
            out.append("        end")
            continue
        if op in (573, 574, 575, 576, 577, 578, 579, 582, 583):
            kind, count, pred, inv, aux = g17cf._decode(raw)
            mn = kind + (".inv" if inv and kind != "pop" else "")
            args = "%d" % count if kind == "pop" else "p%d, %d" % (pred, count)
            tail = "" if aux == g17cf.AUX_DEFAULT[kind] else " / aux=%d" % aux
            out.append("        %-7s %-20s%s" % (mn, args, tail))
            continue
        if op in (450, 458, 462):
            k = {462: "br.none", 458: "br.any", 450: "br.while"}[op]
            aux = g17cf.aux_of(raw)
            out.append("        %-7s %-20s%s" % (k, g17cf._disp(raw),
                                                 "" if aux == 0 else " / aux=%d" % aux))
            continue
        # MORE THAN ONE FORM CAN DESCRIBE ONE (opcode, length), and only some can read a given
        # instruction. Taking whichever was registered first made render refuse a third of
        # op586's instances with the only shape that cannot read them.
        # THE FORM THAT EXPLAINS THE MOST, not the one registered first. Two forms can both read
        # an instruction while covering different amounts of it: (op10295, 12) has `add` with
        # three register slots and `add@10295` with five, and operand 5 differs from its modal
        # value in 121 of 200 instances - so the three-slot form reads, renders, and re-encodes
        # to something else. Taking the wider one first also moves in the endpoint's direction,
        # because a slot the form does not name is a bit inherited from Apple's witness.
        mn, args = None, None
        for cand_mn in sorted(by_op.get((op, ln), []),
                              key=lambda m: -len(tbl[m].get("slots") or ())):
            got = _read_operands(op, ln, tbl[cand_mn], raw)
            if got is not None:
                mn, args = cand_mn, got
                break
        if mn is None:
            if notes is not None:
                notes[off] = "no form" if not by_op.get((op, ln)) else "operand not readable"
            out.append("        .byte   " + ", ".join("0x%02x" % b for b in raw))
            continue
        # CARRY THE MODIFIER OPERANDS TOO. A modifier slot is an ordinary operand with a verified
        # map; the only reason the syntax does not name it is that Apple nearly always gives it
        # the same value. Nearly is not always - op423's operands 1 and 3 differ from their modal
        # value in 189 of its instances - and writing the modal there reproduces a different
        # instruction. Named explicitly, and only when it differs, so the listing stays readable.
        for idx in tbl[mn].get("mods", []):
            r = maps().get((op, ln, idx, "imm")) or maps().get((op, ln, idx, "reg"))
            if not r or r["verdict"] not in ("verified", "table"):
                continue
            v = field_decode(op, idx, r["kind"], raw)
            if v is None or v == r.get("modal"):
                continue
            args.append("op%d=%s" % (idx, ("r%d" % v) if r["kind"] == "reg" else "#%d" % v))
        # Carry the lifetime and hazard bits across the round trip. Without this a rendered
        # program silently loses whichever of them Apple set, and the difference shows up as an
        # instruction that reads back with the right operands and the wrong bytes.
        # Only name a control the assembler would accept: these bits carry the meaning this
        # project measured on the ALU family, and on another opcode the same position is part of
        # the opcode. Emitting it there produces source that will not assemble.
        sp = g17encode.spec_for(op, ln) or {}

        def free(b, i):
            d = sp.get("%d.%d" % (b, i))
            return (d.get("class") if isinstance(d, dict) else d) not in ("opcode", "forced")

        ctl = []
        for name, (b, i) in sorted(CONTROLS.items()):
            if b < len(raw) and (raw[b] >> i) & 1 and free(b, i):
                ctl.append(name)
        w = 0
        for j, (b, i) in enumerate(WAIT_BITS):
            if b < len(raw) and (raw[b] >> i) & 1 and free(b, i):
                w |= 1 << j
        if w:
            ctl.append("wait=%d" % w)
        bf = OPCODE_BITFIELDS.get(op, {}).get("aop")
        if bf and all(b < ln for b, _ in bf):
            ctl.append("aop=%d" % sum(((raw[b] >> i) & 1) << j for j, (b, i) in enumerate(bf)))
        text = "        %-7s %-30s%s" % (mn, ", ".join(args),
                                          (" / " + " ".join(ctl)) if ctl else "")
        # RENDER MUST NOT PRODUCE SOURCE THAT WILL NOT REPRODUCE. A form can be verified on the
        # values Apple used and still be unable to hold one of them - a field whose bits belong
        # to a longer form of the same opcode, a step that only fits its own witnesses. Assembling
        # is not enough: a mnemonic that assembles to DIFFERENT bytes is worse than raw bytes,
        # because it silently changes the instruction and takes a whole program down with it.
        # So run the line and require the bytes back, or emit the bytes instead.
        try:
            probe = Inst(text.strip(), 1, mn, [parse_operand(t, 1) for t in
                                               SPLIT.split(", ".join(args)) if t.strip()])
            for tok in ctl:
                if tok.startswith("wait="):
                    probe.controls.append(("wait", int(tok[5:], 0)))
                elif tok.startswith("aop="):
                    probe.controls.append(("aop", int(tok[4:], 0)))
                elif tok.startswith("aux="):
                    probe.controls.append(("aux", int(tok[4:], 0)))
                else:
                    probe.controls.append((tok, 1))
            probe.addr = 0
            if encode_item(probe, {}, tbl) != raw:
                raise ValueError("re-encodes differently")
        except Exception as e:
            if notes is not None:
                notes[off] = ("near miss" if isinstance(e, ValueError)
                              and str(e) == "re-encodes differently" else "will not assemble")
            out.append("        .byte   " + ", ".join("0x%02x" % b for b in raw))
            continue
        out.append(text)
    return "\n".join(out) + "\n"


def disasm(path, entry=0):
    """Render a binary back to source, annotated with addresses, bytes and every modifier.

    Getting instruction BOUNDARIES out of a flat binary needs the decoder - an instruction's
    length is part of its encoding. That is a disassembler's job and it is not encoding, so the
    separation this file maintains is intact: nothing here feeds back into how bytes are made.
    """
    # PACKAGE MODULES, NOT BARE NAMES. These two survived the migration's import rewrite because
    # they sit on a line that also imports stdlib, and the rewrite matched whole import statements
    # of siblings. Nothing caught it: importing this module never executes a deferred import, so
    # every import-only check passed while disasm() was broken in any process without tools/ on
    # sys.path. The linker hit the identical shape in facts.text_of and reported this one.
    import subprocess, tempfile
    from agxforge.g17 import metal as g17metal, slice as g17slice
    code = open(path, "rb").read()
    with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
        fh.write(code); fh.flush()
        r = subprocess.run([g17metal.DIS, fh.name, str(entry), str(len(code) - entry),
                            "--pc", str(entry), "--expr"], capture_output=True, text=True)
    spans, seen = [], []
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) < 3 or p[1] == "bad":
            continue
        off, ln, op = int(p[0], 16), int(p[1]), int(p[2])
        spans.append((off, ln, op))
        seen.append(" ".join(p[3:]))
    src = render(code, spans)
    body = [l for l in src.splitlines() if not l.startswith(".entry")]
    print("  addr  bytes                         source                         decoded")
    for (off, ln, op), text, dec in zip(spans, body, seen):
        print("  %04x  %-29s %-30s %s %s" % (off, code[off:off + ln].hex(), text.strip(),
                                             g17slice.KNOWN_OPS.get(op, "op%d" % op), dec[:34]))
    auth = sum(1 for l in body if ".byte" not in l)
    print("\n  %d instructions, %d rendered as mnemonics, %d as raw bytes"
          % (len(spans), auth, len(spans) - auth))
    return 0


def check(prog):
    """Assemble, then ask Apple's decoder whether it agrees. A test, never part of encoding."""
    import subprocess, tempfile
    from agxforge.g17 import metal as g17metal
    with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
        fh.write(prog.text); fh.flush()
        r = subprocess.run([g17metal.DIS, fh.name, "0", str(len(prog.text)), "--pc", "0",
                            "--expr"], capture_output=True, text=True)
    got = {}
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) >= 3 and p[1] != "bad":
            got[int(p[0], 16)] = (int(p[1]), int(p[2]), p[3:])
    ok = bad = 0
    for it in prog.insts:
        if it.mnem == ".byte":
            continue
        d = got.get(it.addr)
        if d is None or d[0] != len(it.bytes):
            print("  %04x  %-10s DOES NOT DECODE   %s" % (it.addr, it.mnem, it.bytes.hex()))
            bad += 1
        else:
            ok += 1
            print("  %04x  %-10s op%-6d %s" % (it.addr, it.mnem, d[1], " ".join(d[2])[:52]))
    print("\n  %d decode, %d do not" % (ok, bad))
    return bad == 0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        print(__doc__)
        return 2
    if "--disasm" in sys.argv:
        ent = int(sys.argv[sys.argv.index("--entry") + 1], 0) if "--entry" in sys.argv else 0
        return disasm(args[0], ent)
    src = open(args[0]).read()
    try:
        prog = assemble(src)
    except AsmError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    if "--list" in sys.argv:
        print(prog.listing())
        print("\n  %d instructions, %d bytes, entry 0x%x, symbols %s"
              % (len(prog.insts), len(prog.text), prog.entry, sorted(prog.symbols)))
    if "-o" in sys.argv:
        path = sys.argv[sys.argv.index("-o") + 1]
        open(path, "wb").write(prog.text)
        print("wrote %d bytes to %s" % (len(prog.text), path))
    if "--check" in sys.argv:
        return 0 if check(prog) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


import atexit as _atexit
_atexit.register(_poly_cache_atexit)
