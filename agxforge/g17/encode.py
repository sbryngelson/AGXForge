#!/usr/bin/env python3
"""Encode a G17 instruction from the specification, with no template.

`g17asm` encodes by taking an instruction Apple wrote and overwriting the bits it owns. Every bit
it does not own is inherited, which is why it cannot author a form Apple never emitted: nobody
knows what to put in the bits nobody examined.

This encoder starts from nothing. Each bit gets its value from one of four places, and every one
of them was established by MUTATING THAT BIT and asking the decoder what happened:

    opcode      flipping it decodes as a different opcode, so its value is the identity of the
                instruction and comes from isa/g17-bit-spec.jsonl
    forced      flipping it makes the encoding undecodable, so the recorded value is the ONLY
                legal one - carrying it is a proof, not an inheritance
    operand     flipping it moves a printed operand, so the value comes from the caller
    invisible   flipping it changes nothing the decoder prints. Emitted as zero, and flagged,
                because the decoder not showing a difference is not the same as there being none

An encoding this produces is checked by decoding it back before it is returned. A mismatch is
raised rather than papered over - the point of the exercise is that a form we cannot build from
the specification is one we do not understand, and silently falling back to a template would hide
exactly that.

    python3 tools/g17encode.py --reconstruct [--limit N]
        For every opcode with an encoding Apple wrote, rebuild that instruction from the
        specification alone and compare. This is the coverage number the mission asks for:
        canonical, template-free encoding coverage.
"""
import collections
import os, json, os, subprocess, sys, tempfile
# siblings come from the package
from agxforge.g17 import metal as g17metal, slice as g17slice

HERE = os.path.dirname(os.path.abspath(__file__))
# ANCHORED ON THE CHECKOUT ROOT: two levels up from agxforge/g17/. One level short pointed every
# table in this module at agxforge/isa/, which does not exist - eight paths from one anchor.
ROOT = os.path.dirname(os.path.dirname(HERE))
ISA = os.path.join(ROOT, "isa")
SPEC = os.path.join(ISA, "g17-bit-spec.jsonl")
FREE = os.path.join(ISA, "g17-free-bits.jsonl")
TABLES = os.path.join(ISA, "g17-code-tables.jsonl")
FORMSPEC = os.path.join(ISA, "g17-form-spec.jsonl")
_FORMS = None
_FREE = None
_TABLES = None
AUTH = os.path.join(ISA, "g17-authoring.jsonl")
_SPEC = None
_AUTH = None


def spec():
    global _SPEC
    if _SPEC is None:
        _SPEC = {}
        if os.path.exists(SPEC):
            for line in open(SPEC):
                d = json.loads(line)
                _SPEC[d["opcode"]] = d.get("bits") or {}
    return _SPEC


def freebits():
    """What each invisible bit turned out to be, from agreement across Apple's own instances.

    An invisible bit is one whose flip changes nothing the decoder prints, so decoding cannot say
    whether Apple's value is required. The corpus can: a bit every instance agrees on is a
    CANONICAL constant of the instruction and the encoder emits it - recorded because many
    independent instances agree, not because one witness had it. A bit instances disagree on is
    genuinely FREE and the encoder may emit anything.
    """
    global _FREE
    if _FREE is None:
        _FREE = {}
        if os.path.exists(FREE):
            for line in open(FREE):
                d = json.loads(line)
                # keyed by FORM: the same bit position means different things at different
                # lengths, so pooling them compares bits that are not the same bit
                _FREE[(d["opcode"], d.get("length"))] = d.get("bits") or {}
    return _FREE


def forms():
    """{(opcode, length): {"witness": hex, "bits": {...}}} - the specification, per FORM.

    An opcode is not one instruction. 190 of the 717 Apple emits appear at more than one encoded
    length, and a bit specification derived from one witness describes only that form - re-encoding
    a twelve-byte fadd from a six-byte spec produces six bytes, which is a wrong form rather than a
    wrong value. The unit of specification is (opcode, length).
    """
    global _FORMS
    if _FORMS is None:
        _FORMS = {}
        if os.path.exists(FORMSPEC):
            for line in open(FORMSPEC):
                d = json.loads(line)
                _FORMS[(d["opcode"], d["length"])] = d
    return _FORMS


def spec_for(op, n=None):
    """The bit specification for one FORM, falling back to the opcode-wide one.

    Keyed by (opcode, length) where a per-form specification exists, because 190 opcodes have more
    than one encoded length and a spec derived from the six-byte form says nothing correct about
    the twelve-byte one.
    """
    if n is not None:
        d = forms().get((op, n))
        if d:
            return d.get("bits") or {}
    return spec().get(op) or {}


def code_tables():
    """{opcode: {"j|sub": {bits, table}}} for operand fields that index a table, not a number.

    A field whose bit deltas are multiples of the operand's step but not powers of two is an
    enumeration - the base of an address expression is the clearest case, where one bit takes it
    from op0 to op5. The table was derived by holding the rest of Apple's encoding fixed, walking
    the field's bits through every combination and decoding each, so it is measured rather than
    modelled. A value the table does not contain is left alone rather than approximated.
    """
    global _TABLES
    if _TABLES is None:
        _TABLES = {}
        if os.path.exists(TABLES):
            for line in open(TABLES):
                d = json.loads(line)
                _TABLES[d["opcode"]] = d.get("tables") or {}
    return _TABLES


def semantic_operands(op, real):
    """Operand values read from the DECODE of a real instruction, not back through the field map.

    Reading them through the map would make reconstruction circular: a map wrong in the same way
    twice would pass. These are used only for the fields the map cannot express - the code-table
    ones - so the rest of the test stays exactly as it was.
    """
    from agxforge.g17 import canon as g17canon
    d = decode([real])
    if not d.get(0):
        return {}
    out = {}
    for j, tok in enumerate(d[0][1]):
        for key, v in g17canon.token_values(tok).items():
            out[(j, key)] = v
    return out


_CTX = None


def context_bits():
    """{(opcode, length): {(byte, bit): row}} - fixed-class bits Apple varies, from
    isa/g17-context-bits.jsonl (tools/g17contextbits.py). Empty if the table is absent."""
    global _CTX
    if _CTX is None:
        _CTX = {}
        path = os.path.join(ISA, "g17-context-bits.jsonl")
        if os.path.exists(path):
            for line in open(path):
                d = json.loads(line)
                _CTX[(d["opcode"], d["length"])] = {tuple(map(int, k.split("."))): v
                                                   for k, v in d["bits"].items()}
    return _CTX


def auth():
    global _AUTH
    if _AUTH is None:
        _AUTH = {r["opcode"]: r for r in (json.loads(l) for l in open(AUTH))}
    return _AUTH


def fields_of(op):
    """{operand index: [(value bit, byte, bit, inverted)]} from the authoring field map."""
    out = {}
    for k, v in (auth()[op].get("fields") or {}).items():
        out[int(k.split(":")[0])] = [tuple(e) for e in v if len(e) >= 4]
    return out


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


_EXT_CACHE = {}


def extended_fields(op, length_hint=None, own_form_only=False):
    """Memoised wrapper. Pure over (opcode, length): it reads two loaded specs and derives from
    them, and it was being recomputed once per INSTRUCTION - 2,157 calls for a handful of distinct
    keys, driving 1.2 million dict lookups per twenty-five programs.

    The result is deep-copied on the way out because callers build on it, so a cached entry can
    never be mutated by one caller and served altered to the next."""
    key = (op, length_hint, own_form_only)
    hit = _EXT_CACHE.get(key)
    if hit is None:
        hit = _extended_fields(op, length_hint, own_form_only)
        _EXT_CACHE[key] = hit
    # FOUR VALUES, NOT THREE. Unpacking this as a triple raised ValueError on every call, every
    # caller caught it, and every instruction came back "no form" - 169,640 byte-exact to 14,195.
    # It survived a digest check because the digest was taken over runs where nearly everything
    # threw and appended b"", so both sides were identically broken. Verify with a number that
    # can tell the outcomes apart.
    fields, steps, unresolved, bases = hit
    return ({k: list(v) for k, v in fields.items()}, dict(steps), list(unresolved), dict(bases))


_BASES = None
_BASES_MISSES = []


def bases_misses():
    """The (opcode, length) pairs whose bases were derived by forking the decoder this process."""
    return list(_BASES_MISSES)


def _table_bases(op, length_hint):
    """The checked-in bases for this form, or None to derive them. Refresh bypasses the table."""
    global _BASES
    if os.environ.get("G17_BASES_REFRESH") == "1":
        return None
    if _BASES is None:
        try:
            from agxforge.g17 import bases as g17bases
            _BASES = {k: g17bases.unpack(v) for k, v in g17bases.load().items()}
        except Exception:
            _BASES = {}
    return _BASES.get("%d,%s" % (op, "" if length_hint is None else length_hint))


def _extended_fields(op, length_hint=None, own_form_only=False):
    """The field map plus every operand bit the classifier could place, with its weight.

    The prober's map is incomplete - 6,290 opcodes have operand bits it never assigned - and an
    encoder that leaves those zero writes an instruction with the right opcode and the wrong
    registers. That is exactly what 468 of the 717 reconstructions were doing.

    The classifier mutated each bit and recorded how the PRINTED operand moved. That delta is the
    bit's weight times the operand's step: a register field whose printed value is an MCRegister
    id steps by two or more, so the smallest delta across the operand's bits is the step and every
    other delta must be a power-of-two multiple of it. A bit whose delta is not such a multiple is
    left out rather than guessed, and reported.

    Returns ({operand: [(weight, byte, bit, inverted)]}, step per operand, unresolved positions).
    """
    # THE AUTHORING MAP BELONGS TO ONE LENGTH. It was probed on the prober's witness, and for
    # 1,301 of 1,401 per-form specs that witness is a different length from the form. Laid over a
    # per-form spec it placed bits the form does not have (op13575's 16-byte map reaches byte 9 of
    # a 4-byte form) and, worse, its positions went into `known` below, so the per-form operand bits
    # they collided with were skipped: op13575/4's register bit 0.5 was read as immediate weight 33.
    # Where the form has its own spec and the map was probed at another length, the spec alone
    # places the operands.
    #
    # OPT-IN, because the closed-form encoder and the per-form bases read this too, and dropping the
    # map there changes compiled bytes that were validated on hardware. encode_operands asks the
    # decoder about every setting, so it takes the corrected fields; the compiler path is unchanged
    # until a GPU batch says the corrected bytes execute.
    base = fields_of(op)
    if own_form_only and length_hint is not None and (op, length_hint) in forms():
        probed = len(bytes.fromhex(auth()[op].get("witness") or ""))
        if probed != length_hint:
            base = {}
    out = {k: list(v) for k, v in base.items()}
    known = {(b, i) for v in base.values() for _, b, i, _ in v}
    # length_hint selects the per-FORM specification where one exists; without it the
    # opcode-wide spec is used, which describes only the form it was derived from.
    bits = spec_for(op, length_hint) if length_hint is not None else (spec().get(op) or {})
    cand = collections.defaultdict(list)
    for key, d in bits.items():
        if not isinstance(d, dict) or d.get("class") != "operand":
            continue
        b, i = (int(x) for x in key.split("."))
        if (b, i) in known:
            continue
        for entry in (d.get("moved") or []):
            # the schema carries a SUB-FIELD name now: an address expression has a base, a
            # constant and a scale, each with its own bits
            if len(entry) == 4:
                j, key, before, after = entry
            else:
                j, before, after = entry
                key = ""
            if before is None or after is None or before == after:
                continue
            cand[(j, key)].append(((b, i), d.get("value", 0), after - before))
            break
    steps, unresolved = {}, []
    for j, lst in cand.items():
        mags = [abs(delta) for _, _, delta in lst if delta]
        if not mags:
            unresolved += [p for p, _, _ in lst]
            continue
        step = 0
        for m in mags:
            step = _gcd(step, m)
        steps[j] = step
        for (b, i), held, delta in lst:
            q = abs(delta) // step
            if q == 0 or (q & (q - 1)):
                unresolved.append((b, i))
                continue
            w = q.bit_length() - 1
            # a bit contributes positively when clearing it lowers the value
            inv = 0 if ((held == 0 and delta > 0) or (held == 1 and delta < 0)) else 1
            out.setdefault(j, []).append((w, b, i, inv))
    # THE BASE OFFSET. A field holds (value - base) / unit. Reconstruction never noticed the base
    # was missing because it read the value back through the same map - self-consistent and wrong.
    # Asking the encoder for register 105 produced register 157 the moment the value came from
    # somewhere else. Derive it from this form's own witness: the base is whatever the field's
    # bits do not account for.
    # THE BASES ARE READ FROM THE REPOSITORY, not asked of the vendor at compile time. Deriving
    # them needs a DECODE of the form's witness, and doing that here forked Apple's disassembler
    # once per form - so the emitted bytes depended on that tool's behaviour DURING the build.
    # (tools/agx3dis is gitignored but built from tracked tools/agx3dis.c, so it is pinned by its
    # source; the defect is the compile-time dependence, not the tracking. It links the local
    # SDK's MCDisassembler, so the same source can decode differently on another machine.)
    # Compiling the FP16 scan forked it for op1004, op1016 and op17193. The derivation is
    # unchanged and lives below; isa/g17-form-bases.json holds its answers, harvested by
    # tools/g17bases.py --refresh and re-checked by --check. A form the table does not cover still
    # falls through to the decoder rather than failing the build, and every such miss is counted
    # so that "no oracle was consulted" is a measurement rather than a hope.
    bases = _table_bases(op, length_hint)
    if bases is not None:
        return out, steps, unresolved, bases
    _BASES_MISSES.append((op, length_hint))
    bases = {}
    try:
        wit = None
        if length_hint is not None:
            wit = (forms().get((op, length_hint)) or {}).get("witness")
        wit = wit or auth()[op].get("apple_witness") or auth()[op]["witness"]
        real = bytes.fromhex(wit)
        semv = semantic_operands(op, real)
        for key, positions in out.items():
            k = key if isinstance(key, tuple) else (key, "")
            if not semv or k not in semv:
                continue
            enc = 0
            for w, b, i, inv in positions:
                if b < len(real):
                    enc |= (((real[b] >> i) & 1) ^ inv) << w
            bases[key] = semv[k] - enc * steps.get(key, 1)
    except Exception:
        bases = {}
    return out, steps, unresolved, bases


def length(op):
    r = auth()[op]
    return len(bytes.fromhex(r.get("apple_witness") or r["witness"]))


def encode(op, operands=None, invisible=0, modes=None, tabled=None, hints=None, length_hint=None,
           semantic=False, _fields=None):
    """Bytes for `op` with the given {operand index: value}. Raises if the spec is incomplete.

    TWO OPERAND CONVENTIONS, and the difference is not cosmetic. By default a value is
    `field * step` - what `reconstruct` reads back out of Apple's own bytes, and base-free.
    With `semantic=True` a value is the thing the disassembler PRINTS: register 425, not the
    index 0 that a field counting from 425 holds. Ten of eleven records in the first oracle
    batch were undecodable because they were written the semantic way into a base-free encoder.
    """
    bits = spec_for(op, length_hint)
    if not bits:
        raise KeyError("op%d has no bit specification - run tools/g17canon.py --all-bits" % op)
    # the FORM decides the size: an opcode with a six-byte and a twelve-byte form must produce
    # whichever was asked for, and sizing from the witness silently produces the other one.
    n = length_hint or length(op)
    out = bytearray(n)
    unresolved = []
    for key, d in bits.items():
        b, i = (int(x) for x in key.split("."))
        if b >= n:
            continue
        cls = d["class"] if isinstance(d, dict) else d
        if cls in ("opcode", "forced"):
            if d["value"]:
                out[b] |= 1 << i
        elif cls == "invisible":
            fb = (freebits().get((op, n)) or freebits().get((op, None)) or {}).get(key)
            if fb and fb.get("verdict") == "constant":
                if fb["value"]:
                    out[b] |= 1 << i
            elif hints is not None and (b, i) in hints:
                # A FREE BIT IS AN ENCODER INPUT, not a specification gap. Apple's own instances
                # of the same opcode disagree on it, so no value is required and none can be
                # derived - byte7 bit1 alone varies across 388 instances of op774. Exposing it
                # lets a caller reproduce a specific instruction; leaving it zero is equally
                # legal.
                if hints[(b, i)]:
                    out[b] |= 1 << i
            elif invisible:
                out[b] |= 1 << i
        elif cls == "mode":
            # THE OPERAND'S KIND. `modes` says which kind each operand should take; absent an
            # instruction, the recorded value is the one Apple's own encodings use, and it is
            # recorded as a mode rather than inherited blindly because the mutation showed
            # exactly which operands it switches.
            want = None
            if modes:
                for j, frm, to in (d.get("switches") or []):
                    if j in modes:
                        want = d["value"] if modes[j] == frm else 1 - d["value"]
                        break
            if (d["value"] if want is None else want):
                out[b] |= 1 << i
        elif cls == "operand":
            pass                      # written below, from the caller's values
        else:
            unresolved.append((b, i))
    # THE FIELD MAP OF THIS FORM, not of the opcode. Pooled across lengths it places weights in
    # bytes a shorter form does not have and, worse, on bits the shorter form calls opcode:
    # op554's operand 1 includes b0.3, which is an opcode bit of the four-byte movimm, and writing
    # a zero there destroyed the instruction. An operand never overwrites a bit this same
    # specification calls opcode or forced - if a weight lands on one, the value cannot be placed
    # and the caller's readback says so, which is a refusal rather than a corrupted opcode.
    ext, steps, _, bases = (_fields if _fields is not None
                            else extended_fields(op, length_hint=n))
    for k, positions in ext.items():
        if operands is None or k not in operands:
            continue
        raw = operands[k] - bases.get(k, 0) if semantic else operands[k]
        v = raw // steps.get(k, 1) if k in steps else raw
        for vb, b, i, inv in positions:
            if b >= n:
                continue
            d = bits.get("%d.%d" % (b, i))
            if (d.get("class") if isinstance(d, dict) else d) in ("opcode", "forced"):
                continue
            want = ((v >> vb) & 1) ^ inv
            out[b] = (out[b] & ~(1 << i)) | (want << i)
    # CODE-TABLE FIELDS, applied last and only when a value is supplied. A field this cannot
    # place keeps whatever the linear pass wrote, so adding tables can improve the encoding and
    # never worsen it.
    for tk, t in (code_tables().get(op) or {}).items():
        j, _, sub = tk.partition("|")
        want = (tabled or {}).get((int(j), sub))
        if want is None:
            continue
        hit = next((int(m) for m, v in t["table"].items() if v == want), None)
        if hit is None:
            continue
        for k, (b, i) in enumerate(t["bits"]):
            if b < n:
                out[b] = (out[b] & ~(1 << i)) | (((hit >> k) & 1) << i)
    if unresolved:
        raise ValueError("op%d has %d bits with no class: %s" % (op, len(unresolved), unresolved[:6]))
    return bytes(out)


FORM_AUTHORING = os.path.join(ISA, "g17-form-authoring.jsonl")
_FORM_AUTH = None


def _unkey(k):
    j, _, sub = k.partition("|")
    return (int(j), sub) if _ else int(j)


def form_authoring():
    """{(opcode, length): row} - forms the closed-form encoder authors at their OWN width.

    Written by tools/g17formauthor.py, which lists a form only after this encoder, reading only
    the row, reproduced Apple's witness and every sampled held-out instance byte-exact from their
    printed operands. Empty if the table is absent."""
    global _FORM_AUTH
    if _FORM_AUTH is None:
        _FORM_AUTH = {}
        if os.path.exists(FORM_AUTHORING):
            for line in open(FORM_AUTHORING):
                d = json.loads(line)
                _FORM_AUTH[(d["opcode"], d["length"])] = d
    return _FORM_AUTH


def form_fields(row):
    """(fields, steps, unresolved, bases) in extended_fields' shape, from a table row."""
    ext = {_unkey(k): [tuple(p) for p in v] for k, v in row["fields"].items()}
    steps = {_unkey(k): v for k, v in row["steps"].items()}
    bases = {_unkey(k): v for k, v in row["bases"].items()}
    return ext, steps, [], bases


def encode_form(op, length, operands, modes=None, tabled=None, hints=None):
    """Bytes for (op, length) from PRINTED operand values, with no decoder and no opcode-wide map.

    `operands` is {operand index or (index, sub-field): the value the disassembler prints};
    `modes` is {operand index: kind}. Raises KeyError for a form the table does not list - the
    compiler's own-width authoring is exactly the verified set, never a fallback."""
    row = form_authoring().get((op, length))
    if row is None:
        raise KeyError("op%d/%d is not in isa/g17-form-authoring.jsonl" % (op, length))
    # A CODE-TABLE FIELD IS KEYED BY THE SAME PRINTED VALUE, so it comes from the request too -
    # never from decoding anything.
    if tabled is None:
        tabled = {(k if isinstance(k, tuple) else (k, "")): v for k, v in operands.items()}
    for j, kind in (modes or {}).items():
        allowed = row.get("kinds", {}).get(str(j))
        if allowed is not None and kind not in allowed:
            raise ValueError("op%d/%d operand %d cannot be a %s at this width (it can be %s)"
                             % (op, length, j, kind, "/".join(allowed)))
    out = encode(op, operands, modes=modes, tabled=tabled, hints=hints, length_hint=length,
                 semantic=True, _fields=form_fields(row))
    _read_back(row, op, length, operands, out)
    return out


def form_encoder(op, length, operands, dsts, srcs):
    """An encoder for cc's `machine` hook that emits (op, length) through encode_form.

    `operands` gives every NON-register operand as printed ({index or (index, sub): value}, plus
    the operand kinds under "kinds"); `dsts` and `srcs` are the operand indices the allocator fills.
    A register operand's printed value is its file's base plus the allocated register, where the base
    is the row's own base for that operand - so an operand whose file this row does not start at
    register zero, or has no field for, is refused by encode_form's read-back, never guessed."""
    row = form_authoring().get((op, length))
    if row is None:
        raise KeyError("op%d/%d is not in isa/g17-form-authoring.jsonl" % (op, length))
    kinds = dict(operands.get("kinds") or {})
    fixed_ops = {k: v for k, v in operands.items() if k != "kinds"}

    def enc(template, defs, uses):
        ops = dict(fixed_ops)
        for j, r in list(zip(dsts, defs)) + list(zip(srcs, uses)):
            base = row["bases"].get("%d|" % j)
            if base is None:
                tie = (row.get("ties") or {}).get("%d|" % j)
                base = row["bases"].get(tie) if tie else None
            if base is None:
                raise ValueError("op%d/%d has no register field for operand %d" % (op, length, j))
            ops[j] = ops[(j, "")] = base + r
            kinds[j] = "reg"
        return encode_form(op, length, ops, modes=kinds)
    return enc


def _read_back(row, op, length, operands, out):
    """Refuse bytes that do not carry every requested operand - read through the row, no decoder.

    Apple's instances cannot show that a form lacks a field for an operand they never vary, so a
    row admitted on them can still be unable to place a value. op9749/8's operand 3 has no field at
    eight bytes; asked for register 441 it wrote 425. A field can also be too narrow, or run into
    a bit this form calls opcode, where encode() leaves the bit alone. Each of those is a silent
    wrong operand unless the result is read back, so it is, and a mismatch raises."""
    ext, steps, _, bases = form_fields(row)
    fixed = {_unkey(k): v for k, v in (row.get("fixed") or {}).items()}
    ties = {_unkey(k): _unkey(m) for k, m in (row.get("ties") or {}).items()}
    asked = {(k if isinstance(k, tuple) else (k, "")): v for k, v in operands.items()}
    tables = code_tables().get(op) or {}
    for k, want in operands.items():
        key = k if isinstance(k, tuple) else (k, "")
        # A CODE TABLE IS NOT A PLACEMENT. encode() applies one only when the value is in it, and
        # the tables are opcode-level, measured at another width. Skipping every tabled operand let
        # op1015/4 take operand 1 = 2147483680 and emit 32. So a tabled operand is accepted only if
        # the table's own bits, inside this width, read back as the value - or the linear field does.
        t = tables.get("%d|%s" % key)
        if t is not None and all(b < length for b, _ in t["bits"]):
            code = sum(((out[b] >> i) & 1) << n for n, (b, i) in enumerate(t["bits"]))
            if t["table"].get(str(code)) == want:
                continue
        field = ext.get(key, ext.get(key[0]) if key[1] == "" else None)
        if field is None:
            if key in ties:
                if asked.get(ties[key], want) != want:
                    raise ValueError("op%d/%d operand %s has no field at this width and always "
                                     "equals operand %s; %s and %s were asked for"
                                     % (op, length, key, ties[key], want, asked[ties[key]]))
            elif key in fixed and fixed[key] != want:
                raise ValueError("op%d/%d has no field for operand %s at this width; it is %s, "
                                 "and %s was asked for"
                                 % (op, length, key, "not placeable" if fixed[key] is None
                                    else "always %s" % fixed[key], want))
            continue
        fk = key if key in ext else key[0]
        raw = 0
        for w, b, i, inv in field:
            if b < length and w < 64:
                raw |= (((out[b] >> i) & 1) ^ inv) << w
        got = bases.get(fk, 0) + raw * steps.get(fk, 1)
        if got != want:
            raise ValueError("op%d/%d operand %s: asked for %s, the bytes carry %s - the field "
                             "cannot hold it" % (op, length, key, want, got))


def poly_value(poly, raw):
    """The value a SECOND-ORDER operand map reads out of these bytes.

    Some fields are not a sum of their bits. See tools/g17diff.py quadratic(): the map is
    const + SUM(c_i x_i) + SUM(c_ij x_i x_j), measured by second differences against the decoder.
    """
    v = poly["const"]
    for b, i, c in poly["lin"]:
        if b < len(raw) and (raw[b] >> i) & 1:
            v += c
    for b1, i1, b2, i2, c in poly["quad"]:
        if b1 < len(raw) and b2 < len(raw) and (raw[b1] >> i1) & 1 and (raw[b2] >> i2) & 1:
            v += c
    for b1, i1, b2, i2, b3, i3, c in poly.get("cube", ()):
        if (b1 < len(raw) and b2 < len(raw) and b3 < len(raw)
                and (raw[b1] >> i1) & 1 and (raw[b2] >> i2) & 1 and (raw[b3] >> i3) & 1):
            v += c
    return v


def poly_bits(poly):
    """Every (byte, bit) a second-order map decides."""
    out = {(b, i) for b, i, _c in poly["lin"]}
    for b1, i1, b2, i2, _c in poly["quad"]:
        out.add((b1, i1))
        out.add((b2, i2))
    for b1, i1, b2, i2, b3, i3, _c in poly.get("cube", ()):
        out.update({(b1, i1), (b2, i2), (b3, i3)})
    return sorted(out)


_DECODE_CACHE = {}
_DECODE_STATS = [0, 0]      # [calls, subprocesses]


def decode(blob, stride=32):
    """Decode a list of instruction blobs, MEMOISED on the bytes.

    THIS WAS THE WHOLE COST OF SCORING THE CORPUS. It forks the disassembler once per call and the
    encode path calls it once per instruction, so a full-corpus byte-exactness run spent its time
    in fork/poll: 9 instructions per second, 358 minutes for 184,349 instructions. Decoding is a
    pure function of the bytes, and Apple's corpus repeats instructions heavily, so the same blob
    is decoded thousands of times.

    Cached on the exact blob tuple and the stride. Nothing here mutates the result - callers read
    it - and the cache is per-process, so a map edit between runs cannot be served a stale answer:
    the bytes are the key, and different maps produce different bytes.
    """
    key = (tuple(bytes(b) for b in blob), stride)
    _DECODE_STATS[0] += 1
    hit = _DECODE_CACHE.get(key)
    if hit is not None:
        return hit
    _DECODE_STATS[1] += 1
    out = _decode_uncached(blob, stride)
    if len(_DECODE_CACHE) < 400000:
        _DECODE_CACHE[key] = out
    return out


def _decode_uncached(blob, stride=32):
    with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
        fh.write(b"".join(v.ljust(stride, b"\x06") for v in blob))
        fh.flush()
        r = subprocess.run([g17metal.DIS, fh.name, "0", str(stride * len(blob)), "--pc", "0",
                            "--stride", str(stride), "--expr"], capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) < 2:
            continue
        try:
            off = int(p[0].rstrip(":"), 16)
        except ValueError:
            continue
        if off % stride == 0:
            out[off // stride] = None if p[1] == "bad" else (int(p[2]), tuple(p[3:]))
    return out


def reconstruct(ops):
    """Rebuild each opcode's real instruction from the spec and compare with Apple's own."""
    A = auth()
    built, want, keys, hinted = [], [], [], []
    for op in ops:
        r = A[op]
        w = r.get("apple_witness")
        if not w:
            continue
        real = bytes.fromhex(w)
        # the operand values Apple's own instruction carries, read through the field map
        ext, steps, _, bases = extended_fields(op)
        vals = {}
        for k, positions in ext.items():
            v = 0
            for vb, b, i, inv in positions:
                if b < len(real):
                    v |= (((real[b] >> i) & 1) ^ inv) << vb
            vals[k] = v * steps.get(k, 1)
        fb = freebits().get((op, len(real))) or {}
        hints = {}
        for key, v in fb.items():
            if v.get("verdict") in ("varies", "free"):   # free: settled by execution
                b, i = (int(x) for x in key.split("."))
                if b < len(real):
                    hints[(b, i)] = (real[b] >> i) & 1
        try:
            got = encode(op, vals, tabled=semantic_operands(op, real))
            withhints = encode(op, vals, tabled=semantic_operands(op, real), hints=hints)
        except (KeyError, ValueError):
            continue
        hinted.append(withhints)
        built.append(got)
        want.append(real)
        keys.append(op)
    d = decode(built)
    ref = decode(want)
    modulo = sum(1 for j in range(len(keys)) if hinted[j] == want[j])
    exact = same = wrong = bad = 0
    detail = []
    for j, op in enumerate(keys):
        g, r = d.get(j), ref.get(j)
        if built[j] == want[j]:
            exact += 1
            same += 1
        elif g is None:
            bad += 1
            detail.append((op, "does not decode"))
        elif r is not None and g == r:
            same += 1
        else:
            wrong += 1
            detail.append((op, "decodes as op%s not op%d" % (g[0] if g else "?", op)))
    return keys, exact, same, wrong, bad, detail, modulo


def diagnose(ops):
    """Why each reconstruction failed, at the level a fix can act on.

    Ranked by HOW MANY INSTRUCTIONS A MISSING RULE BLOCKS rather than by how many bits it covers:
    one register-value rule shared across a hundred opcodes is worth more than ten isolated bits,
    and counting bits hides that completely.
    """
    A = auth()
    built, want, keys, hinted = [], [], [], []
    for op in ops:
        r = A[op]
        w = r.get("apple_witness")
        if not w:
            continue
        real = bytes.fromhex(w)
        ext, steps, unres, _bases = extended_fields(op)
        vals = {}
        for k, positions in ext.items():
            v = 0
            for vb, b, i, inv in positions:
                if b < len(real):
                    v |= (((real[b] >> i) & 1) ^ inv) << vb
            vals[k] = v * steps.get(k, 1)
        fb = freebits().get((op, len(real))) or {}
        hints = {}
        for key, v in fb.items():
            if v.get("verdict") in ("varies", "free"):   # free: settled by execution
                b, i = (int(x) for x in key.split("."))
                if b < len(real):
                    hints[(b, i)] = (real[b] >> i) & 1
        try:
            got = encode(op, vals, tabled=semantic_operands(op, real))
            withhints = encode(op, vals, tabled=semantic_operands(op, real), hints=hints)
        except (KeyError, ValueError):
            continue
        hinted.append(withhints)
        built.append(got); want.append(real); keys.append((op, unres, ext))
    d = decode(built); ref = decode(want)
    blockers = collections.Counter()
    opblock = collections.Counter()
    rows = []
    for j, (op, unres, ext) in enumerate(keys):
        g, r = d.get(j), ref.get(j)
        if built[j] == want[j]:
            continue
        diff = [(b, i) for b in range(len(want[j]))
                for i in range(8) if ((built[j][b] ^ want[j][b]) >> i) & 1]
        if g is not None and r is not None and g == r:
            kind = "same instruction, different bytes"
        elif g is None:
            kind = "does not decode"
        elif r is None:
            kind = "reference does not decode"
        elif g[0] != r[0]:
            kind = "wrong opcode"
        else:
            bad = [k for k, (x, y) in enumerate(zip(r[1], g[1])) if x != y]
            kind = "operands differ: %s" % ",".join(str(k) for k in bad)
            for k in bad:
                blockers["operand %d has an unmapped bit" % k] += 1
        # a differing bit that the classifier could not place is the actionable blocker
        for pos in diff:
            if pos in set(unres):
                blockers["unplaced operand bit"] += 1
                opblock[op] += 1
                break
        rows.append((op, kind, len(diff), sorted(set(unres))[:4]))
    return rows, blockers, opblock


def main():
    named = g17slice.KNOWN_OPS
    if "--diagnose" in sys.argv:
        ops = sorted(o for o, r in auth().items() if r.get("apple_witness") and o in spec())
        rows, blockers, opblock = diagnose(ops)
        kinds = collections.Counter(k for _, k, _, _ in rows)
        print("RECONSTRUCTION FAILURES, %d of %d witnessed opcodes" % (len(rows), len(ops)))
        for k, v in kinds.most_common(10):
            print("   %-46s %d" % (k, v))
        print("")
        print("BLOCKERS, ranked by instructions blocked:")
        for k, v in blockers.most_common(10):
            print("   %-46s %d" % (k, v))
        print("")
        print("failures with NO unplaced bit (residue is a rule, not a weight): %d"
              % sum(1 for op, _, _, u in rows if not u))
        print("")
        print("first twelve failures:")
        for op, kind, nd, u in rows[:12]:
            print("   op%-6d %-24s %-40s %d bits differ, unplaced %s"
                  % (op, named.get(op, "--"), kind, nd, u))
        return
    if "--reconstruct" in sys.argv:
        lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
        ops = sorted(o for o, r in auth().items() if r.get("apple_witness") and o in spec())
        if lim:
            ops = ops[:lim]
        keys, exact, same, wrong, bad, detail, modulo = reconstruct(ops)
        print("CANONICAL RECONSTRUCTION - rebuild Apple's own instruction from the spec alone")
        print("  opcodes with an Apple encoding and a bit spec: %d" % len(keys))
        print("  byte-for-byte identical                       %d  (%.1f%%)"
              % (exact, 100 * exact / max(1, len(keys))))
        print("  decodes to the same instruction               %d  (%.1f%%)"
              % (same, 100 * same / max(1, len(keys))))
        print("  byte-exact given the same free-bit hints      %d  (%.1f%%)"
              % (modulo, 100 * modulo / max(1, len(keys))))
        print("  decodes to something else                     %d" % wrong)
        print("  does not decode                               %d" % bad)
        for op, why in detail[:12]:
            print("     op%-6d %-24s %s" % (op, named.get(op, "--"), why))
        return
    print(__doc__)


if __name__ == "__main__":
    main()




def _forms_main():
    """Reconstruct every (opcode, length) form through the full encoder."""
    F = forms()
    built, want, keys = [], [], []
    for (op, ln), d in sorted(F.items()):
        real = bytes.fromhex(d["witness"])
        ext, steps, _, bases = extended_fields(op)
        vals = {}
        for k, positions in ext.items():
            v = 0
            for vb, b, i, inv in positions:
                if b < len(real):
                    v |= (((real[b] >> i) & 1) ^ inv) << vb
            vals[k] = v * steps.get(k, 1)
        try:
            got = encode(op, vals, tabled=semantic_operands(op, real), length_hint=ln)
        except (KeyError, ValueError):
            got = None
        built.append(got if got is not None else b"")
        want.append(real)
        keys.append((op, ln))
    g = decode([b for b in built])
    r = decode(want)
    exact = sum(1 for j in range(len(keys)) if built[j] == want[j])
    same = sum(1 for j in range(len(keys)) if g.get(j) is not None and g.get(j) == r.get(j))
    print("PER-FORM RECONSTRUCTION - every (opcode, length) with a real encoding")
    print("  forms                                        %d" % len(keys))
    print("  byte-for-byte identical                      %d  (%.1f%%)"
          % (exact, 100 * exact / max(1, len(keys))))
    print("  decodes to the same instruction              %d  (%.1f%%)"
          % (same, 100 * same / max(1, len(keys))))


if __name__ == "__main__" and "--forms" in sys.argv:
    _forms_main()


def encode_operands(op, length, want_tokens, hints=None):
    """Build an instruction of this form whose printed operands are `want_tokens`.

    The closed-form field model - weight, unit, base - is right often enough to reconstruct
    Apple's own instructions and NOT right enough to place a value that came from elsewhere: it
    reproduced register 157 when asked for 105, because a field map with a missing bit gives a
    base that absorbs the error and only agrees with itself.

    So use the decoder as the oracle instead of trusting the model. Each operand field is solved
    independently - a field moves one operand, so the search is a sum over fields rather than a
    product - by walking the field's values and keeping the one whose printed token matches. What
    comes back is exact by construction, and where it agrees with the closed-form model that is
    evidence the model is right.

    Returns (bytes, unmatched operand indices).
    """
    # START FROM THE CANONICAL BASE, NOT THE WITNESS. Starting from Apple's own encoding made the
    # search trivially succeed whenever the operands asked for were the ones already there - it
    # "authored" 984 of 984 forms byte-identically by changing nothing, which measures the witness
    # and not the specification. The base is the template-free encoding: opcode and forced bits at
    # their proven values, canonical invisible bits, everything else zero.
    try:
        base = bytearray(encode(op, {}, length_hint=length))
    except (KeyError, ValueError):
        return None, list(range(len(want_tokens)))
    if len(base) < length:
        return None, list(range(len(want_tokens)))
    base = bytes(base[:length])
    from agxforge.g17 import canon as g17canon
    ext, steps, _, _ = extended_fields(op, length, own_form_only=True)
    # SEARCH THE BITS THE MODEL LEFT OUT. extended_fields drops an operand bit whose decoder delta
    # is not a power-of-two multiple of the step - a register-class bit takes op2191/8's operand 6
    # from 427 to 294 - because a closed-form weight for it would be a guess. The search below
    # needs no weight: it enumerates settings and asks the decoder. So a bit the spec says moved
    # operand j joins j's search, and stays out of every weight-based path.
    placed = {(b, i) for v in ext.values() for _, b, i, _ in v}
    for bk, d in sorted((spec_for(op, length) or {}).items()):
        if not isinstance(d, dict) or d.get("class") != "operand" or not d.get("moved"):
            continue
        b, i = (int(x) for x in bk.split("."))
        if (b, i) in placed or b >= length:
            continue
        entry = d["moved"][0]
        j, sub = entry[0], (entry[1] if len(entry) == 4 else "")
        key = (j, sub) if (j, sub) in ext or sub or j not in ext else j
        ext.setdefault(key, []).append((64 + len(ext.get(key, [])), b, i, 0))
        placed.add((b, i))
    cur = bytearray(base)
    # THE OPERAND KINDS ARE PART OF THE REQUEST. A `mode` bit switches one operand between kinds -
    # register, immediate, expression - and its `switches` record says which kind each value gives.
    # The canonical base carries the witness's value, so an instance whose operand is the other
    # kind could never be matched: no operand field can turn a register into an expression. The
    # requested token names its kind, so the bit is set from it, as encode() does with `modes`.
    kind = lambda t: t.split(":")[0] if ":" in t else t.split("(")[0]
    for bk, d in (spec_for(op, length) or {}).items():
        if not isinstance(d, dict) or d.get("class") != "mode":
            continue
        b, i = (int(x) for x in bk.split("."))
        if b >= length:
            continue
        for j, frm, to in (d.get("switches") or []):
            if j < len(want_tokens) and kind(want_tokens[j]) in (frm, to) and frm != to:
                v = d.get("value", 0) if kind(want_tokens[j]) == frm else 1 - d.get("value", 0)
                cur[b] = (cur[b] & ~(1 << i)) | (v << i)
                break
    # HINTS FIRST. A hinted bit is one the caller has chosen - a free bit's value - and op11487/10's
    # free bits 4.3 and 4.4 move immediate 7. Applied only after the search, they shifted operands the
    # search had already matched; applied before, the search solves in the context it will ship in.
    # They are applied again at the end, so a search that touched one cannot override the caller.
    for (b, i), v in (hints or {}).items():
        if b < length:
            cur[b] = (cur[b] & ~(1 << i)) | ((1 if v else 0) << i)
    # SWEEP UNTIL IT SETTLES. Fields are solved one at a time, but a field can move another
    # operand - op11487/10's bit 4.1 is recorded against operand 8 and also shifts immediate 7 -
    # so a single pass in a fixed order leaves the earlier operand wrong. Repeating the pass
    # from the current bytes lets each field re-solve in its neighbours' final context. It
    # stops when every operand matches or a pass changes nothing; three passes is the bound.
    def settle(cur):
        for _sweep in range(3):
            before = bytes(cur)
            unmatched = []
            for key, positions in sorted(ext.items(), key=lambda kv: str(kv[0])):
                j = key[0] if isinstance(key, tuple) else key
                if j >= len(want_tokens):
                    continue
                target = want_tokens[j]
                # A SUB-FIELD IS SCORED ON ITS SUB-VALUE. An address expression's base, constant and scale
                # are separate fields; scoring each against the whole printed expression meant no single
                # field could ever match, since the other two were still at their canonical values.
                sub = key[1] if isinstance(key, tuple) else ""
                want_sub = g17canon.token_values(target).get(sub) if sub else None
                def hits(r):
                    if not r or j >= len(r[1]):
                        return False
                    if want_sub is None:
                        return r[1][j] == target
                    return g17canon.token_values(r[1][j]).get(sub) == want_sub
                d0 = decode([bytes(cur)])
                if d0.get(0) and hits(d0[0]):
                    continue
                # ENUMERATE THE FIELD'S BITS, NOT ITS VALUE RANGE. A modifier field can have weights 24
                # to 27 - sparse, high, and perfectly ordinary - and taking 2**(max weight + 1) asked for
                # 2**48 trials and gave up. The number of settings is 2**(number of bits), which is small
                # for every field in the ISA.
                if len(positions) > 16:
                    unmatched.append(j)
                    continue
                order = sorted(positions, key=lambda t: t[0])
                trials, vals = [], []
                for v in range(1 << len(order)):
                    m = bytearray(cur)
                    for k, (w, b, i, inv) in enumerate(order):
                        if b < length:
                            m[b] = (m[b] & ~(1 << i)) | ((((v >> k) & 1) ^ inv) << i)
                    trials.append(bytes(m)); vals.append(v)
                got = decode(trials)
                hit = None
                for k in range(len(trials)):
                    r = got.get(k)
                    if r and r[0] == op and hits(r):
                        hit = k
                        break
                if hit is None:
                    unmatched.append(j)
                else:
                    cur = bytearray(trials[hit])
            if bytes(cur) == before:
                break
            now = decode([bytes(cur)]).get(0)
            if now and now[0] == op and list(now[1][:len(want_tokens)]) == list(want_tokens):
                break
        return cur, unmatched

    def score(c):
        r = decode([bytes(c)]).get(0)
        if not r or r[0] != op:
            return -1
        return sum(1 for j, t in enumerate(want_tokens) if j < len(r[1]) and r[1][j] == t)

    cur, unmatched = settle(cur)
    # BITS THE SPEC FIXED FROM ONE WITNESS AND APPLE VARIES. isa/g17-context-bits.jsonl lists, per
    # form, the opcode- and forced-class bits that take both values in Apple's own instances of that
    # (opcode, length) - the SPEC_MEASURED_OPERAND standard, computed from the corpus. The search
    # never writes them, so an instance carrying the other value was unreachable. When the sweep
    # leaves operands unmatched, flip one such bit at a time, re-settle, and keep a flip only if more
    # operands match; the whole-instruction check below still refuses a different opcode.
    ctx = [p for p in sorted(context_bits().get((op, length), {}))
           if p[0] < length and p not in (hints or {})]
    best = score(cur)
    for _round in range(len(ctx)):
        if best == len(want_tokens):
            break
        improved = False
        for b, i in ctx:
            m = bytearray(cur)
            m[b] ^= 1 << i
            m, un = settle(m)
            sc = score(m)
            if sc > best:
                cur, unmatched, best, improved = m, un, sc, True
                if best == len(want_tokens):
                    break
        if not improved:
            break
    if hints:
        for (b, i), v in hints.items():
            if b < length:
                cur[b] = (cur[b] & ~(1 << i)) | ((1 if v else 0) << i)
    # CHECK THE FINISHED INSTRUCTION. Solving each field in turn and reporting the per-field
    # matches is not the same as the assembled instruction being what was asked for: a later field
    # can move an earlier operand, and a field can change the OPCODE while its own operand still
    # reads correctly. Both were reproduced - asked for op774, got op766, no unmatched operands
    # reported. So decode the result and compare the whole thing.
    #
    # The reviewer's other point stands too and is not fixed here: 826 bit/form entries move more
    # than one operand sub-field, so per-field independence is an approximation. This check turns
    # that approximation from a silent wrong answer into a reported failure.
    fin = decode([bytes(cur)])
    r = fin.get(0)
    if r is None:
        return bytes(cur), sorted(set(unmatched) | {-1})
    if r[0] != op:
        return bytes(cur), sorted(set(unmatched) | {-2})
    for j, tok in enumerate(want_tokens):
        if j >= len(r[1]) or r[1][j] != tok:
            unmatched.append(j)
    return bytes(cur), sorted(set(unmatched))


def authorable():
    """How many forms can be built to a REQUESTED operand list, not just reconstructed.

    Reconstruction asks whether the closed-form model reproduces Apple's bytes. This asks the
    question a compiler asks: given a form and the operands I want, produce that instruction. The
    search solves each field against the decoder, so a form counts only if every operand the
    witness carries can be placed - which is exactly the condition for the selector to be allowed
    to choose it.
    """
    F = forms()
    ok = exact = partial = 0
    rows = []
    for (op, ln), d in sorted(F.items()):
        real = bytes.fromhex(d["witness"])
        got = decode([real])
        if not got.get(0):
            continue
        want = list(got[0][1])
        enc, un = encode_operands(op, ln, want)
        if enc is None:
            continue
        if not un:
            ok += 1
            exact += (enc == real)
        else:
            partial += 1
        rows.append((op, ln, un, enc == real))
    print("AUTHORABLE FORMS - every operand placed as requested")
    print("  forms tried                    %d" % len(rows))
    print("  every operand placed           %d  (%.1f%%)" % (ok, 100 * ok / max(1, len(rows))))
    print("  and byte-identical to Apple's  %d  (%.1f%%)" % (exact, 100 * exact / max(1, len(rows))))
    print("  some operand not placeable     %d" % partial)
    import collections as _c
    hist = _c.Counter(len(u) for _, _, u, _ in rows)
    print("  unplaceable-operand histogram: %s" % dict(sorted(hist.items())[:8]))
    return rows


if __name__ == "__main__" and "--authorable" in sys.argv:
    authorable()
