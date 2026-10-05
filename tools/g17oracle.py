#!/usr/bin/env python3
"""AN EXECUTION ORACLE FOR AUTHORED ENCODINGS - one instruction, one image, one dispatch.

Everything the ISA side can execute today goes through SUBSTITUTION: an opcode dropped into a slot
Apple compiled, with every operand bit left as Apple wrote it. That reaches only forms whose
surrounding instruction already exists. The cases where a wrong specification survives forever are
the other ones - an opcode Apple never emitted, encoded from the spec, where decoding cannot say
whether the model is right.

The image side can answer those now: since the whole archive is generated from payload sizes and
Metal accepts a metallib this project writes, a program of any length can be wrapped and run with
no host archive to splice into. This file is that entry point.

WHAT IT TAKES - a JSON list of records, each one instruction:

    {"id":     "op998.l12",              a label carried through to the result
     "op":     998,                      the opcode, for the safety gate
     "length": 12,                       the form; an opcode is not one instruction
     "bytes":  "092dac0c1011",           the whole instruction, assembled by the caller
     "dest":   {"map": [[0,4],[0,5]], "slope": 1, "base": 0},
     "srcs":  [{"map": [[1,1],[1,2]], "slope": 1, "base": 0},
                {"map": [[3,1],[3,2]], "slope": 2, "base": 425}],
     "cases": [[3235663872, 1073741824]],   u32 values for the sources, one list per case
     "expect": [1082130432],                optional, compared as raw u32
     "witness_required": false,            waives only the "no Apple witness" refusal
     "author": "caller"}                   or "fieldmap" - see below

TWO WAYS TO AUTHOR THE SAME INSTRUCTION, and running both is how a disagreement gets attributed.
With `author: "caller"` the record's own bytes and field maps place the operands. With
`author: "fieldmap"` the record supplies only the opcode and this project's own recovered field map
places them, through the same generic form the compiler uses for every other instruction. If the
second returns the expected value and the first returns zero, the harness is delivering operands
correctly and the caller's map is what is wrong - which is a different finding from "the opcode is
misnamed", and neither side can tell them apart from one number.

`map` is the field's (byte, bit) positions, least significant first, and the value written is

    field = base + register * slope + half

BASE AND SLOPE ARE BOTH PER-OPERAND FACTS and neither has a safe default. A field does not hold a
register number: for op10295 a field value of zero names register 425. Writing raw register numbers
into a field that counts from 425 produced eleven zeros on the first batch and looked like eleven
wrong opcodes. So `base` is REQUIRED on every operand a record describes - a record that omits it is
refused rather than assumed to start at zero, because the failure it causes is silent.

WHAT IT RETURNS, per record: the value and the command-buffer status SEPARATELY, because
"the encoding is wrong" and "the encoding is illegal" are different findings and only the second
one shows up as a non-zero status.

WHAT IT CHECKS BEFORE DISPATCHING. A caller's field solver can return a different opcode while
reporting success - asked for op774, produced op766 - so the bytes are decoded AFTER the registers
are written into them and the record is refused if the decoded opcode is not the declared one. The
decode is reported either way, so when the two sides disagree about what an instruction is, that is
visible in the results file rather than inferred from a wrong number.

WHAT IT REFUSES. tools/g17safe.py gates every record on Apple's own two flag words before anything
is authored: memory, atomic and texture opcodes, and anything that may load, store, branch, call,
return, terminate or have unmodelled side effects. Two substitutions hung this GPU and killed
WindowServer once (memory agx-mutation-gpu-hang-hazard), so that gate is not waivable from the
input file - a record's `allow` list only takes effect when the RUNNER is invoked with
--allow-unsafe, which is a decision for whoever is sitting at the machine, not for whoever wrote
the batch.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "spike", "accel", "re"))

import g17auth, g17cc, g17ir as ir, g17program, g17ref, g17safe, g17forms
import g17imgconst_scalar as K

ENTRY = 0x40
FILLER = bytes.fromhex("0600")
SLOT0 = 8                     # the first output word; every case writes SLOT0 + 2*i

# THE CANARY IS WRITTEN LAST, so its absence says the program DID NOT FINISH rather than that the
# instruction computed something. The output buffer is filled with 0xDEADBEEF before the dispatch,
# and a run that faults part-way leaves the case slots holding whatever they held - which reads as
# a value. Three runs of one unchanged program once gave the right answer, a failed command buffer,
# and a WRONG answer with the status reporting success (memory g17-metrics-lie-by-default), so the
# status alone is not enough to know a program ran to the end.
CANARY_SLOT = 6
PER_LANE_BASE = 256           # per_lane records store case i, lane L at word 256 + 32*i + L
CANARY = 0x5A17C0DE


def field_encoder(template, dest, srcs):
    """An encoder for `ir.machine` that writes the allocator's registers into a caller's bytes.

    The template is the caller's assembled instruction; only the operand fields move. Anything the
    caller did not describe keeps the bits they wrote, so what the dispatch measures is their
    encoding and not this file's idea of one.
    """
    def put(u, spec, reg):
        if not spec:
            return u
        v = int(spec["base"]) + reg * int(spec.get("slope", 1)) + int(spec.get("half", 0))
        u = bytearray(u)
        for i, (by, bi) in enumerate(spec["map"]):
            u[by] = (u[by] & ~(1 << bi)) | (((v >> i) & 1) << bi)
        return bytes(u)

    def enc(_t, defs, uses):
        u = bytes(template)
        if dest and defs:
            u = put(u, dest, defs[0])
        for i, spec in enumerate(srcs):
            if i < len(uses):
                u = put(u, spec, uses[i])
        return u
    return enc


# THE LOW HALF OF REGISTER r IS DECODER ID 425 + r. MEASURED, and it is not what this comment
# first said. The first version wrote "425 + 2r" - a GPR16 operand names a slot, 2 x register +
# half (g17auth.slot_step) - and was wrong by a factor of two. The hardware-verified path says
# so directly: the 6-byte op1004 authored by the field map, whose h2f of the low half is
# measured on silicon, loads the case into r16 (id 121) and READS id 441 = 425 + 16. Loads say
# the same: op12646@14 with destination id 432 wrote the low half of the register holding the
# identity that op12682's destination id 112 (r7) overwrote (isa/g17-execution-unsafe2-results.json).
# Under "425 + 2r" every 12-byte op1004 record read the low half of r32 or r52, registers nothing
# wrote, and returned 0 on every input three batches running. The check that let it through
# compared this encoding with Apple's decoder reading of the SAME encoding, which cannot fail;
# the test now compares with the field map's id for the same register instead. How the HIGH
# half is named is not established here, so an id this module cannot place is refused.
HALF0 = 425


def _is_half(op, idx):
    ops = g17auth.record(op).get("operands") or []
    return idx < len(ops) and str(ops[idx] or "").startswith("GPR16")


def decoder_id(op, idx, reg):
    """The decoder id that names allocator register `reg` (its low half, for a GPR16 operand)."""
    return HALF0 + reg if _is_half(op, idx) else reg + g17auth.REG0


def allocator_reg(op, idx, ident):
    """The inverse of decoder_id, or None where the id is outside the range it produces."""
    if _is_half(op, idx):
        r = ident - HALF0
        return r if 0 <= r < 126 else None
    return ident - g17auth.REG0


def assembler_encoder(op, width, template, wrote):
    """An encoder that writes the allocator's registers into an Apple template at ITS OWN WIDTH.

    `g17auth.fields` is keyed to one width per opcode, so a fieldmap record cannot author any
    other width - its bit positions belong to the other form (see refusal()). The assembler's maps
    ARE keyed (op, length, operand, kind), and `g17as.field_encode` returns the field value for a
    register at a stated length. This writes only the register fields; every other bit, the
    lifetime operands included, keeps the value Apple wrote. `wrote` collects what each call
    asked for, IN THE ALLOCATOR'S NUMBERING, so _check can compare it with what Apple's decoder
    reads back.

    TWO REGISTER NUMBERINGS, AND THE ASSEMBLER USES THE DECODER'S. `g17auth.REG0` is 105: "the
    decoder prints register n as MCRegister id 105 + n". The assembler's maps are built from the
    decoder, so a field whose `base` is 105 holds allocator r0, not r105. The first version of this
    encoder passed allocator numbers straight to `field_encode`, so an instruction asked to use
    r105/r107/r108 was written to use allocator r0/r2/r3 - and every case of all five dispatched
    short forms returned 0. It is translated here, once.
    """
    import g17as as _as
    maps = _as.maps()
    dsts, srcs = g17auth.register_operands(op)

    def put(u, idx, reg):
        m = maps.get((op, width, idx, "reg"))
        if not m or not m.get("positions"):
            raise ValueError("op%d@%d operand %d has no plain register field at this width"
                             % (op, width, idx))
        v, extra = _as.field_encode(op, idx, "reg", decoder_id(op, idx, reg), width, 0)
        if not isinstance(v, int):
            raise ValueError("op%d@%d operand %d cannot hold r%d: %s" % (op, width, idx, reg, v))
        u = bytearray(u)
        for bi, by, bit, inv in (tuple(x) for x in m["positions"]):
            x = (v >> bi) & 1
            if inv:
                x ^= 1
            u[by] = (u[by] & ~(1 << bit)) | (x << bit)
        for by, bit, val in (extra or []):
            u[by] = (u[by] & ~(1 << bit)) | ((val & 1) << bit)
        return bytes(u)

    def enc(_t, defs, uses):
        u = bytes(template)
        asked = []
        if dsts and defs:
            u = put(u, dsts[0], defs[0]); asked.append(defs[0])
        for idx, reg in zip(srcs, uses):
            u = put(u, idx, reg); asked.append(reg)
        wrote.append(asked)
        return u
    return enc


def decode(b):
    """(opcode, length) the reference decoder reads out of these bytes, or (None, None)."""
    try:
        r = list(g17ref.walk(bytes(b), 0))
    except Exception:
        return None, None
    if len(r) != 1 or r[0][1] != len(b):
        return None, None
    return r[0][2], r[0][1]


def ladder_program(name):
    """Compile a named ladder program and append the canary as the LAST thing it writes.

    The oracle's own records are one instruction wrapped in a scaffold, which cannot express an
    instruction with no destination: a store computes nothing the next instruction can hold, so
    `r = machine(...); store(r)` has nothing to put in r. The compiler's own programs can - the
    effect IS the observable - and dispatching them measures the thing the mission actually asks
    about, which is the backend rather than one encoding.

    The canary goes in the LAST block, immediately before its terminator, so its absence still means
    the program did not reach the end. A program whose final block is not the one that runs last
    would break that, which is why the two programs written for this take a single path.
    """
    import g17ladder
    # PROBES LIVE APART FROM THE LADDER, because g17scorecard counts every public g17ladder function
    # as the compiler's corpus. A name the ladder lacks is looked up among the store probes.
    if hasattr(g17ladder, name):
        f = getattr(g17ladder, name)()
    else:
        import g17storeprobes
        f = getattr(g17storeprobes, name)()
    out_buf = next((bf for bf in f.buffers if bf.slot == 2), f.buffers[-1])
    blk = f.blocks[-1]
    term = blk.ops.pop() if blk.ops and blk.ops[-1].kind in ir.TERMS else None
    b = ir.Builder(f, blk)
    b.store(out_buf, ir.Imm(CANARY_SLOT), b.const(CANARY, name="canary"))
    if term is not None:
        blk.ops.append(term)
    return g17cc.compile_function(f)


def _program_as_written(name):
    """The named program compiled WITHOUT the canary - what a program record's `op` is counted in."""
    import g17ladder
    if hasattr(g17ladder, name):
        f = getattr(g17ladder, name)()
    else:
        import g17storeprobes
        f = getattr(g17storeprobes, name)()
    return g17cc.compile_function(f).code


def program_opcodes(name):
    """Every opcode a compiled ladder program contains - what the safety gate has to be asked
    about. A record naming one opcode says nothing about the other eight instructions around it,
    and "refuse loops and the memory opcodes across the WHOLE emitted program" is the rule."""
    return sorted({op for _at, _ln, op in g17ref.walk(ladder_program(name).code, 0)})


def highest_output_word_read(rec):
    """The largest word index of the output buffer this record's readback touches.

    ONE DEFINITION, BECAUSE THE DISPATCH SCOPE IS ENFORCED IN TWO PLACES. The peer lane's
    retraction (2026-09-18) measured the mechanism exactly: a dispatch whose `dim` is too small
    for the output extent returns everything past dim*dim*esz bytes as fill or stale data with NO
    error - dim=22 failing and dim=23 passing for a value at byte offset 1024. A record reading
    past its dispatch's scope therefore presents as a dead instruction, indistinguishable from the
    constant-output population this census already holds hundreds of records of.

    This mirrors what the runner's `_child` ACTUALLY reads, not a conservative envelope: the case
    slots or the named `read_slots` (never both - `read_slots` replaces them), the preload identity
    words, and the canary. The audit that used to compute this inline covered only the first of
    those three, so a preload at a high slot or a moved canary was outside the question it asked;
    it happened not to matter, because the worst record in the corpus reads word 67 of a
    guaranteed 1024 - a wide margin that nothing had checked.
    """
    if rec.get("read_slots"):
        hi = max(int(k) for k in rec["read_slots"])
    elif rec.get("cases"):
        hi = SLOT0 + 2 * (len(rec["cases"]) - 1)
    else:
        # NO CASE SLOT IS READ AT ALL, and `SLOT0 + 2 * max(0, n - 1)` reported one anyway - the
        # floor at zero makes an empty case list look like a single case. Conservative and
        # therefore harmless, but this value decides a REFUSAL, and a bound that over-reports
        # refuses records for words they never touch.
        hi = CANARY_SLOT
    pre = rec.get("preload")
    if pre:
        hi = max(hi, int(pre["slot"]) + int(pre["n"]) - 1)
    return max(hi, CANARY_SLOT)


def refusal(rec, allow_unsafe=False):
    """Why this record must not be dispatched, or None."""
    # A PLAN THAT NO LONGER BUILDS WHAT IT MEASURED. 42 retained records rebuild today with an
    # operand VALUE their result never ran - a source lifetime the liveness pass now writes, or a
    # destination modifier (isa/g17-execution-rebuild-drift.json). Re-dispatching one would file a
    # new instruction's behaviour under the old record's id. The retained bytes are the evidence;
    # a record that needs re-running is re-authored with its own `bytes`, under a new id.
    import g17rebuilddrift
    drifted = g17rebuilddrift.not_reproducible().get(g17rebuilddrift.plan_digest(rec))
    if drifted:
        return ("not reproducible: %s/%s now rebuilds %s where the retained result ran %s (%s); "
                "re-author it under a new id" % (drifted["batch"], drifted["id"], drifted["rebuilt"],
                                                 drifted["retained"], drifted["cls"]))
    if rec.get("author", "caller") != "fieldmap":
        for name, spec in [("dest", rec.get("dest"))] + [
                ("src%d" % i, s) for i, s in enumerate(rec.get("srcs") or [])]:
            if spec is not None and "base" not in spec:
                return ("%s has no `base`; a register field holds base + register*slope and "
                        "assuming zero fails silently" % name)
    allow = tuple(rec.get("allow", ())) if allow_unsafe else ()
    req = bool(rec.get("witness_required", True))
    if rec.get("program"):
        # EVERY opcode in the program, not the one the record happens to name. A whole-program
        # record has no single opcode under test, and the two hangs this project has caused both
        # came from an instruction nobody had filtered rather than from the one being studied.
        try:
            ops = program_opcodes(rec["program"])
        except Exception as e:
            return "cannot compile %r: %s: %s" % (rec["program"], type(e).__name__, e)
        bad = []
        for o in ops:
            w = g17safe.why_unsafe(o, allow=allow, require_apple_witness=req)
            if w:
                bad.append("op%d %s" % (o, "/".join(w)))
        return "; ".join(bad) or None
    why = g17safe.why_unsafe(int(rec["op"]), allow=allow, require_apple_witness=req)
    if rec.get("per_lane"):
        n = len(rec.get("cases") or [])
        want = [PER_LANE_BASE + 32 * i + L for i in range(n) for L in range(32)]
        if int(rec.get("threads") or 1) != 32:
            why.append("per_lane stores one word per lane of a 32-thread group; `threads` must be 32")
        if not rec.get("lane_varying"):
            why.append("per_lane without lane_varying reads back 32 copies of one uniform value")
        if [int(k) for k in (rec.get("read_slots") or [])] != want:
            why.append("per_lane needs read_slots to be exactly words %d..%d in case/lane order"
                       % (want[0], want[-1]) if want else "per_lane with no cases")
    # A FIELDMAP RECORD'S TEMPLATE MUST BE THE WIDTH ITS FIELD MAP DESCRIBES. `g17auth.fields` is
    # keyed to ONE width per opcode, and with `author: fieldmap` that map writes the allocator's
    # registers into whatever bytes the record supplies. Supply an Apple instance at any other
    # width and the bits land in positions that belong to the other form. Measured on op998: an
    # Apple 4-byte template (`reg:105 imm:32 reg:105 imm:16 reg:106 imm:16`) came back as
    # `reg:105 imm:0 reg:105 imm:0 reg:138 imm:0` - decoder ids, so allocator r0, r0 and r33
    # (REG0 is 105), where the allocator had put the sources in r16 and r17: the instruction reads
    # registers nobody loaded. Every lifetime is 0, which Apple never emits (32 keeps, 16 releases).
    # [Corrected: this comment first called reg:138 "outside the R0-R125 file". It is a decoder id,
    # allocator r33, inside the file - the defect is the WRONG registers, not an out-of-file one.]
    # The opcode and the
    # width both survived, so `_check` reported ok and the record would have dispatched malformed
    # bytes. The patch path below already refuses this ("the field map's bit positions belong to
    # the other form"); this route to the same defect was not guarded. For all eleven evidence-gap
    # forms that are encodable at their own width, the authoring table describes a different one.
    # The repair is a width-aware author - the assembler's maps are keyed (op, length, operand) -
    # not a looser check here.
    if rec.get("author") == "fieldmap" and rec.get("bytes"):
        width = len(bytes.fromhex(rec["bytes"]))
        described = g17auth.length(int(rec["op"]))
        if width != described:
            why.append("the template is %d bytes and op%d's authoring field map describes the "
                       "%d-byte form, so its bit positions belong to the other form; a "
                       "fieldmap record cannot author a width its map does not describe"
                       % (width, int(rec["op"]), described))
    lane = rec.get("lane_varying")
    if lane:
        # A LANE-VARYING SOURCE AT ONE THREAD IS THE UNIFORM PROBE WITH A NEW FIELD NAME, and
        # that is the exact failure this feature exists to end. Ten determinations were retracted
        # on 2026-09-18 because every lane held the same constant, so `simd.fmax.f16` and
        # `simd.fmin.f16` returned byte-identical vectors and both "fitted" a truncation. A record
        # that asks for lane variation and dispatches one thread would reproduce that silently and
        # look like a fresh measurement, so it is refused rather than run.
        if int(rec.get("threads") or 1) < 2:
            why.append("lane_varying needs `threads` > 1; at one thread every lane holds the "
                       "same constant, which is the uniform-lane probe whose results were "
                       "retracted")
        if not lane.get("sources"):
            why.append("lane_varying names no sources, so nothing would vary")
        unknown = [k for k in lane if k not in ("sources", "builtin", "axis", "step")]
        if unknown:
            why.append("lane_varying has unknown keys %s; a misspelled key would silently leave "
                       "the probe uniform" % sorted(unknown))
    return ", ".join(why) or None


# THE OPCODES THAT NEED A THREADGROUP BINDING DECLARED. A threadgroup allocation attaches to a
# BINDING, and a signature that does not declare one produces an image where
# setThreadgroupMemoryLength has nothing to attach to - the round trip reads zero and the zero means
# nothing. That cost a day. g17mdgen now REFUSES an unmatched threadgroup signature rather than
# quietly handing back the class for the same signature without one, so the failure mode here is a
# build error naming the missing class instead of a null.
TG_OPCODES = {12364, 13288,
              # THE THREADGROUP ATOMICS ARE THREADGROUP ACCESSES TOO. Without them a
              # kernel whose only threadgroup use is an atomic gets no binding and no
              # allocation, and writes nothing - which presents as a dead kernel rather
              # than as a missing declaration.
              11701, 11703, 11705, 11765, 11769}


def _wrap(code, rec, ops=()):
    """Put emitted code into a dispatchable image at ENTRY, refusing anything unwalkable."""
    text = bytes.fromhex("0e000000") + FILLER * ((ENTRY - 4) // 2) + code
    if len(text) % 16:
        text += FILLER * ((16 - len(text) % 16) // 2)
    list(g17ref.walk(text, ENTRY))
    tg = any(o in TG_OPCODES for o in ops)
    return g17program.G17Program(
        text=text, entry=ENTRY,
        buffers=[1, 2, 0] if tg else [1, 2],
        binding_kinds=(["device_buffer", "device_buffer", "threadgroup"] if tg else None),
        stats_md=K.STATS_MD)


def program(rec):
    """The whole program for one record: the constants, the instruction under test, the stores.

    Returns (G17Program, decode) where `decode` says what the reference decoder reads out of the
    emitted code - the check that the bytes that will RUN are still the opcode the record declared.
    """
    if rec.get("program"):
        p = ladder_program(rec["program"])
        code = p.code
        patched = []
        if rec.get("patch"):
            # WRITING AN OPERAND INSIDE A WHOLE PROGRAM, which is what a confound on a barrier or a
            # threadgroup load needs. Those opcodes' functions were proven by a PROGRAM - the
            # round trip, the guarded store - so the only way to ask whether the answer rests on an
            # inherited operand is to write that operand in the program that proved it and run it
            # again. A fieldmap record cannot: a barrier returns no value.
            #
            # The patch names (opcode, operand, value) and is applied to EVERY instance of that
            # opcode in the emitted code, through the same field map the authoring path uses. The
            # emitted length must be the one the table describes, or the bit positions belong to a
            # different instruction - which is the mismatch g17confound now reports separately.
            code = bytearray(code)
            for spec in rec["patch"]:
                op, val = int(spec["op"]), int(spec["value"])
                if "bit" in spec:
                    # A RAW BIT, for a position no operand in the table covers. The operand form
                    # above is the one to use wherever it applies - it goes through the field map
                    # and so cannot mean something different in a different encoding. This form
                    # exists for the bits that have no operand: op12364 has six that its map does
                    # not reach, and asking what they do needs a way to write them.
                    by, bi = int(spec["byte"]), int(spec["bit"])
                    hit = 0
                    for at, ln, o in g17ref.walk(bytes(code), 0):
                        if o != op or at + by >= len(code):
                            continue
                        code[at + by] = (code[at + by] & ~(1 << bi)) | ((val & 1) << bi)
                        hit += 1
                    if not hit:
                        raise ValueError("%s: no op%d in program %r to patch"
                                         % (rec.get("id"), op, rec["program"]))
                    patched.append([op, "b%d[%d]" % (by, bi), val, hit])
                    continue
                idx = int(spec["operand"])
                fm = g17auth.fields(op)
                if idx not in fm:
                    raise ValueError("%s: op%d has no operand %d to patch" % (rec.get("id"), op, idx))
                hit = 0
                for at, ln, o in g17ref.walk(bytes(code), 0):
                    if o != op:
                        continue
                    if ln != g17auth.length(op):
                        raise ValueError(
                            "%s: op%d is %d bytes here and the authoring table describes %d - the "
                            "field map's bit positions belong to the other form"
                            % (rec.get("id"), op, ln, g17auth.length(op)))
                    for j, by, bi, inv in fm[idx][1]:
                        b = (val >> j) & 1
                        if inv:
                            b ^= 1
                        code[at + by] = (code[at + by] & ~(1 << bi)) | (b << bi)
                    hit += 1
                if not hit:
                    raise ValueError("%s: no op%d in program %r to patch"
                                     % (rec.get("id"), op, rec["program"]))
                patched.append([op, idx, val, hit])
            code = bytes(code)
            # THE PATCH MUST NOT HAVE CHANGED WHAT THE INSTRUCTION IS. Writing an operand that
            # turns the opcode into a different one is the failure this catches before dispatch.
            after = sorted({o for _a, _l, o in g17ref.walk(code, 0)})
            if after != sorted({o for _a, _l, o in g17ref.walk(p.code, 0)}):
                raise ValueError("%s: the patch changed the program's opcodes" % rec.get("id"))
        ops = sorted({op for _a, _l, op in g17ref.walk(code, 0)})
        # A PROGRAM RECORD THAT NAMES AN OPCODE IS EVIDENCE FOR THAT FORM, so it must contain it.
        # The runner copies `op` and `length` from the plan into the result, and the compiler
        # inventory's isolated index counts any finished ok record by exactly those two fields -
        # so without this, a record labelled op17235/8 would count for 17235@8 whatever its
        # program held. Exactly ONE instance, at the declared length: two would make the landing
        # word ambiguous, and another length is a different form.
        want, enc = None, []
        if rec.get("op") is not None:
            want = int(rec["op"])
            # COUNTED IN THE PROGRAM AS WRITTEN, without the canary this wrapper appends: the canary
            # is itself a default store (op17244 at 8 bytes), so counting the wrapped code would
            # refuse every probe of op17244/8 for an instance the scaffold, not the probe, put there.
            written = [(at, ln) for at, ln, op in g17ref.walk(_program_as_written(rec["program"]), 0)
                       if op == want]
            # `instances` (default 1) says how many copies the program writes. More than one is honest
            # only where the observable needs EVERY copy right - a half-vector store's members each
            # arrive through their own op590 packing move, so none reaches its word unless its move did.
            n_want = int(rec.get("instances", 1))
            if (len(written) != n_want or rec.get("length") is None
                    or any(ln != int(rec["length"]) for _a, ln in written)):
                raise ValueError("%s: names op%d at length %s, and program %r contains it as %s"
                                 % (rec.get("id"), want, rec.get("length"), rec["program"],
                                    [ln for _a, ln in written] or "no instance"))
            # the probe's instance precedes the canary, which is the last thing before the return
            mine = [(at, ln) for at, ln, op in g17ref.walk(code, 0)
                    if op == want and ln == written[0][1]]
            enc = [code[mine[0][0]:mine[0][0] + mine[0][1]].hex()]
        return _wrap(code, rec, ops), {"ok": True, "want": want, "found": len(p.layout),
                                    "cases": len(rec.get("read_slots") or []),
                                    "encoded": enc, "from_template": False, "opcodes": ops,
                                    "patched": patched}
    tmpl = bytes.fromhex(rec["bytes"]) if rec.get("bytes") else b""
    if rec.get("bytes") and rec.get("length") and len(tmpl) != int(rec["length"]):
        raise ValueError("%s: %d bytes given for a length-%d form"
                         % (rec.get("id"), len(tmpl), int(rec["length"])))
    assembler = rec.get("author") == "assembler"
    caller = assembler or (rec.get("author", "caller") != "fieldmap"
                           and (rec.get("dest") or rec.get("srcs")))
    rec["_asm_wrote"] = []
    if assembler:
        if not tmpl:
            raise ValueError("%s: author=assembler needs `bytes`, an Apple instance at the width "
                             "under test" % rec.get("id"))
        enc = assembler_encoder(int(rec["op"]), len(tmpl), tmpl, rec["_asm_wrote"])
    else:
        enc = field_encoder(tmpl, rec.get("dest"), rec.get("srcs") or []) if caller else None
    f = ir.Function("oracle", [ir.Buffer("A", 0), ir.Buffer("B", 1), ir.Buffer("C", 2)])
    b = ir.Builder(f, f.block("e"))
    # THE IDENTITY PRELOAD, for records that ask which REGISTER an instruction read rather than
    # what it computed. Register k is given the value 1<<k, so a result is a bitmask naming its
    # inputs directly instead of a number someone has to attribute. Two properties make it work
    # and both come from the allocator rather than from hope:
    #   - a range store PRE-COLOURS its values to consecutive registers, and
    #   - a pre-coloured value is never released (Alloc.run: `v not in pre`), so the identities
    #     survive every instruction between here and the read-back.
    # The form encodes n as 1..4, so N identities become ceil(N/4) range stores, each taking its
    # own consecutive run. Which run each group gets is NOT assumed - it is read back.
    ids = []
    pre = rec.get("preload")
    if pre:
        ids = [b.const(1 << k, name="id%d" % k) for k in range(int(pre["n"]))]
    # ONE FIELD VALUE PER CASE, so a single dispatch can read many registers instead of one. The
    # encoder is rebuilt per case with that case's literal; without this every case in a program
    # reads the SAME register, which is one reading per dispatch and the reason a full sweep looked
    # like thousands of them.
    percase = rec.get("per_case_field")
    flips = rec.get("per_case_flip")
    for i, case in enumerate(rec["cases"]):
        # A SOURCE THAT DIFFERS BETWEEN LANES, which every cross-lane opcode needs and no record
        # could ask for. `b.const` materialises one constant, identical in all 32 lanes, so a
        # reduction over the simdgroup returns its own input and a maximum cannot be told from a
        # minimum - which is exactly how ten cross-lane determinations came to be retracted.
        #
        # The lane index comes from `thread_position_in_threadgroup`, read through the builder's
        # existing builtin. That is deliberate: the compiler already reads it for spill indexing,
        # and reading it in a kernel that does NOT DECLARE it returns zero in every lane, silently,
        # which is the same uniform-lane symptom one level down. Going through the builder means
        # the declaration is produced by the path that already gets it right, and no new encoder
        # is grown here.
        lane = rec.get("lane_varying")
        wanted = set(lane.get("sources") or ()) if lane else set()
        srcs = []
        for j, v in enumerate(case):
            value = b.const(int(v) & 0xFFFFFFFF, name="c%d_%d" % (i, j))
            if j in wanted:
                tid = b.builtin(lane.get("builtin", "thread_position_in_threadgroup"),
                                axis=lane.get("axis", "x"), name="tid%d_%d" % (i, j))
                step = int(lane.get("step", 1))
                if step != 1:
                    tid = b.mul(tid, b.const(step, name="step%d_%d" % (i, j)),
                                name="scaled%d_%d" % (i, j))
                value = b.add(value, tid, name="lane%d_%d" % (i, j))
            srcs.append(value)
        e = enc
        if caller and (percase is not None or flips is not None):
            base_i = int(percase[i]) if percase is not None else None
            spec = ([dict(x, base=base_i) for x in (rec.get("srcs") or [])]
                    if base_i is not None else (rec.get("srcs") or []))
            t_i = tmpl
            if flips is not None and flips[i] is not None:
                # ONE BIT OF THE INSTRUCTION FLIPPED, PER CASE. This is what turns the probe from
                # "which register does field value V select" into "does this BIT select the
                # register at all" - the question the decoder cannot answer for a bit two operand
                # maps both claim, and the reason 901 such pairs exist.
                u = bytearray(t_i)
                by, bi = flips[i]
                u[by] ^= (1 << bi)
                t_i = bytes(u)
            e = field_encoder(t_i, rec.get("dest"), spec)
        # A FIELDMAP RECORD MAY CHOOSE ITS WITNESS. Authoring places the operands; every other bit
        # comes from a template, and which template that is has been fixed at the authoring table's
        # own witness. Supplying one is what lets a record ask "does this bit carry anything" -
        # clear it, author the same operands onto it, and require the same answer. That is the move
        # that settled the store's byte1 bit2, done here without editing a shared constants file.
        # A RECORD MAY STATE A NON-REGISTER OPERAND. Everything the IR does not fill keeps the
        # witness's bits, which is fine until one of those bits is the answer: op621's operand 5 is
        # 32 in the witness this project authored onto AND 32 is where its shift was measured to
        # saturate, so "saturates at 32" and "saturates at whatever operand 5 says" fit the same
        # data. Writing the operand is the only way to tell them apart.
        _imms = {int(k): int(v) for k, v in (rec.get("imms") or {}).items()}
        # AN OPERAND THE COMPILER OWNS CANNOT BE STATED BY A RECORD, AND ASKING SILENTLY GAVE
        # BACK A DIFFERENT PROGRAM. A source's LIFETIME is an operand (32 keeps, 16 releases), and
        # the liveness pass writes it into the finished bytes with put_modifier after g17auth.encode
        # has already honoured whatever the record said - by design, so an authored program cannot
        # emit a stale lifetime (cc.py: "an unchecked lifetime is exactly how a value that is read
        # again gets released"). The record's value is therefore overwritten without a word.
        #
        # This is not hypothetical. op612's operand 3 is the lifetime carrier for its register
        # source at operand 2, and a peer lane reads that operand as a register-source marker whose
        # zero forces "the result is always 0 regardless of src". A record asking for `imms: {"3":
        # 0}` came back byte-identical to the record that asked for nothing, returned the ordinary
        # answer, and its two mismatches read exactly like hardware refuting their sentence. A
        # decode of the emitted bytes is what caught it. Refusing here is what makes the decode
        # unnecessary: a record that cannot be built must not run instead.
        # SCOPED TO THE REGISTER SOURCES, because a lifetime belongs to one. lifetime_operand
        # answers for any operand index it is handed, so asking it about all six of op612's
        # operands claims 3, 4 and 5 are all owned - and 4 and 5 are its lo and width, which the
        # batch above provably wrote and read back. A guard that refuses a case known to work is
        # measuring itself; register_operands is the population that has lifetimes at all.
        _rsrc = g17auth.register_operands(int(rec["op"]))[1]
        _owned = {c: s_ for s_ in _rsrc
                  for c in (g17auth.lifetime_operand(int(rec["op"]), s_),) if c is not None}
        _clash = sorted(set(_imms) & set(_owned))
        if _clash:
            raise SystemExit(
                "%s: imms names operand(s) %s of op%d, which the liveness pass owns - each is the "
                "lifetime carrier for source operand %s (32 keeps, 16 releases) and is rewritten "
                "into the bytes after encoding, so the value asked for here would be silently "
                "discarded and the record would measure the default program under another name. "
                "To vary a lifetime, drive it from liveness; to author these bits directly, give "
                "the record explicit `bytes` and operand specs so it builds through the caller's "
                "own encoder."
                % (rec.get("id"), _clash, int(rec["op"]), [_owned[c] for c in _clash]))
        r = (b.machine(int(rec["op"]), *srcs, name="r%d" % i, encoder=e, template=tmpl)
             if caller else b.machine(int(rec["op"]), *srcs, name="r%d" % i,
                                      template=(tmpl or None), imms=_imms))
        if rec.get("per_lane"):
            # EVERY LANE'S RESULT, NOT LANE 0'S. A cross-lane opcode read back from one constant
            # slot tells an identity from a reduction only on the lanes where they differ, and the
            # family laws were fitted on that one word. Each lane stores its own result at
            # PER_LANE_BASE + 32*i + lane through the register-indexed store, and the record's
            # read_slots names exactly those words (refusal() checks both).
            tid = b.builtin("thread_position_in_threadgroup", axis="x", name="plane%d" % i)
            at = b.add(b.const(PER_LANE_BASE + 32 * i, name="pbase%d" % i), tid, name="pidx%d" % i)
            b.store_at(f.buffers[2], at, r, width="word")
        else:
            b.store(f.buffers[2], ir.Imm(SLOT0 + 2 * i), r)
    # AFTER the instruction under test, so the read-back proves the identities were still live
    # while it ran rather than only before it - the release-bit hazard this project has been bitten
    # by four times (memory g17-modifier-operand-lifetimes) shows up here as a changed identity.
    for j in range(0, len(ids), 4):
        b.store_range(f.buffers[2], ir.Imm(int(pre["slot"]) + j), ids[j:j + 4])
    b.store(f.buffers[2], ir.Imm(CANARY_SLOT), b.const(CANARY, name="canary"))
    b.ret(); ir.verify(f)
    # THE POOL IS A HARNESS CHOICE, NOT A HARDWARE LIMIT, and a record that preloads two dozen
    # identity registers needs more of the file than one that does not. It stays at 0..39 unless a
    # record asks, so every previously dispatched record allocates exactly as it did - c1 still
    # builds byte-identical to the delivery result that was proven on silicon.
    # [MEASURED 2026-09-22 FALSE for 537 of the 1,688 ok records (isa/g17-execution-rebuild-drift.json,
    # tools/g17rebuilddrift.py): 495 rebuild with different REGISTER NUMBERS only, and 42 with an
    # operand VALUE changed - 16 of them a source lifetime the liveness pass now writes (16, release)
    # where the dispatched bytes carry 0 or 2, 24 an operand 1, 2 an address expression. None changes
    # opcode, length, operand kind or instance count. The pool is unchanged; the compiler around it
    # is not. The functions those records measured stand, but re-running one of the 42 today
    # dispatches a different instruction from the one measured.]
    pool =int(rec.get("pool") or 40)
    code, _ = g17cc.emit(g17cc.Alloc(regs=range(0, pool)).run(g17cc.select(f)))
    check = _check(rec, code)
    text = bytes.fromhex("0e000000") + FILLER * ((ENTRY - 4) // 2) + code
    if len(text) % 16:
        text += FILLER * ((16 - len(text) % 16) // 2)
    list(g17ref.walk(text, ENTRY))       # refuses rather than dispatching an unwalkable program
    # THE BINDINGS COME FROM THE FUNCTION THAT WAS COMPILED. This read `buffers=[1, 2]`, hardcoded,
    # while the scaffold above declares THREE - A at 0, B at 1, C at 2 - so every image built here
    # omitted the binding at rank 0 that its own code was compiled against. tools/g17endtoend.py
    # states the consequence exactly: "building the image for one list while the code was compiled
    # against the other is a store at a rank the image does not declare, which returns the fill
    # value at status 0 and looks exactly like a broken opcode."
    #
    # THAT IS WHY THIS RUNNER STOPPED REPRODUCING ITS OWN CONTROL. Every record went through this
    # path: pipeline built, ac_run_ps returned 0, and not one word of any bound buffer changed.
    # With the list derived, CONTROL.op10279 reproduces its retained values 4100, 4101, 4356, 8196
    # and the canary 0x5A17C0DE, in the same process and with the same eight-byte store form -
    # so the store form was never the fault, and neither was the image class, the harness call or
    # the metadata.
    #
    # `_wrap` above still passes [1, 2] and is correct by coincidence rather than by derivation:
    # every ladder program it wraps declares exactly those two slots, which is why whole-program
    # records kept working while scaffold records went silent. Deriving it there needs the compiled
    # function's buffer list, which that path does not currently carry.
    P = g17program.G17Program(text=text, entry=ENTRY,
                              buffers=sorted({buf.slot for buf in f.buffers}),
                              stats_md=K.STATS_MD)
    return P, check


def _apple_registers(raw):
    """The register operands agx3dis - Apple's decoder - reads out of one instruction, in order."""
    import re as _re, subprocess as _sp, tempfile as _tf
    with _tf.NamedTemporaryFile(suffix=".bin") as fh:
        fh.write(bytes(raw)); fh.flush()
        out = _sp.run([g17ref.binary(), fh.name, "0", str(len(raw)), "--pc", "0"],
                      capture_output=True, text=True).stdout
    line = out.splitlines()[0] if out.strip() else ""
    if not line or " bad" in line:
        return None
    return [int(x) for x in _re.findall(r"reg:(\d+)", line)]


def _check(rec, code):
    """The opcodes actually present in the emitted program, and whether the declared one survived.

    The registers are written into the caller's template after they send it, and a field map with a
    missing bit can turn one opcode into another; a caller's own solver has done exactly that. So
    the finished bytes are decoded, not the ones that arrived.
    """
    want, ncases = int(rec["op"]), len(rec["cases"])
    tlen = len(bytes.fromhex(rec["bytes"])) if rec.get("bytes") else None
    seen = [(at, ln, op) for at, ln, op in g17ref.walk(code, 0)]
    mine = [(at, ln, op) for at, ln, op in seen if op == want]
    ok = len(mine) == ncases and (tlen is None or all(ln == tlen for _a, ln, _o in mine))
    enc = [code[at:at + ln].hex() for at, ln, _o in mine]
    # DID THE RECORD'S OWN BYTES REACH THE PROGRAM? Every other guard here can pass while the
    # answer is no. program() selects caller mode with `author != "fieldmap" and (dest or srcs)`,
    # and an EMPTY srcs list is falsy - so a record that supplies bytes and no operand specs is
    # silently authored from the table instead, at whatever length the table likes. A four-byte
    # op13588 template came back as a ten-byte instruction that way and a whole sweep measured
    # instructions it had not written, with the safety gate, the decode check, the canary and
    # three-run agreement all passing.
    #
    # A record that supplies `bytes` gets told when the emitted instruction is not built on them.
    # THE FIELDMAP EXEMPTION IS GONE, because the reason for it is. It was here because a fieldmap
    # record IGNORED its own bytes and was authored from the table instead; now it authors onto them,
    # so the same length check has to cover it or this guard would quietly stop guarding the case it
    # was written for.
    if rec.get("bytes"):
        want_len = len(bytes.fromhex(rec["bytes"]))
        if any(len(bytes.fromhex(e)) != want_len for e in enc):
            ok = False
    # A RECORD MAY NAME A `length` IT DOES NOT GET, AND NOTHING CHECKED IT. The length check above
    # only runs when the record supplies its own BYTES, so a fieldmap record asking for length 6
    # sailed through: `g17auth.length(3290)` is 14, the encoder authored 14 bytes, and the result
    # file went out saying `length: 6`. The census was not fooled - it reads the width off the
    # decoded bytes - but a results file carrying a length its own bytes contradict is exactly the
    # kind of field that later gets quoted.
    #
    # A measured function belongs to the FORM dispatched. If a record cannot have the form it asks
    # for, that is a capability gap to name, not a number to publish.
    if rec.get("length") and enc:
        asked = int(rec["length"])
        if any(len(bytes.fromhex(e)) != asked for e in enc):
            ok = False
    # THE OPERANDS, READ BACK BY APPLE'S OWN DECODER. Every check above can pass while the
    # registers are wrong: the opcode and the width survive a field map applied at the wrong
    # width, and op998 on a 4-byte template came back as reg:138 with every lifetime 0 while this
    # function reported ok. For the assembler author the registers the encoder was asked to write
    # are known, so they are compared with what agx3dis - Apple's decoder, not this project's -
    # reads out of the finished bytes. A mismatch, or any register at or above R126 (squashed as a
    # whole instruction, measured), refuses the record rather than dispatching it.
    operand_check = None
    if rec.get("author") == "assembler":
        wrote = rec.get("_asm_wrote") or []
        # IN ONE NUMBERING. The encoder records allocator registers and agx3dis prints decoder ids
        # (allocator + REG0). The first version compared them directly - and passed, because the
        # allocator had been pinned to r105..r108 and the instruction had been written to decoder
        # 105..108: the same four numbers meaning two different sets of registers. A check that
        # compares values from two namespaces cannot fail for the reason it exists.
        # Converted PER OPERAND: a GPR16 operand is 425-based and counts halves (decoder_id).
        order = [i for i in (g17auth.register_operands(want)[0][:1] + g17auth.register_operands(want)[1])]
        read = []
        for r in (_apple_registers(bytes.fromhex(e)) for e in enc):
            if r is None:
                read.append(None)
                continue
            got = [allocator_reg(want, idx, x) for idx, x in zip(order, r)] + \
                  [x - g17auth.REG0 for x in r[len(order):]]
            read.append(got)
        operand_check = dict(asked=wrote, apple_reads=read)
        if len(read) != len(wrote) or any(r is None for r in read):
            ok = False
            operand_check["why"] = "Apple's decoder did not read every instance"
        else:
            for asked, got in zip(wrote, read):
                if got[:len(asked)] != asked:
                    ok = False
                    operand_check["why"] = "asked %s, Apple reads %s" % (asked, got)
                    break
                if any(r is None or r >= 126 or r < 0 for r in got[:len(asked)]):
                    ok = False
                    operand_check["why"] = ("a register outside allocator r0..r125 (R126+ is "
                                            "squashed as a whole instruction, measured)")
                    break
    return dict(ok=ok, want=want, found=len(mine), cases=ncases, encoded=enc,
                operands=operand_check,
                from_template=bool(rec.get("bytes")) and bool(enc)
                and all(len(bytes.fromhex(e)) == len(bytes.fromhex(rec["bytes"])) for e in enc),
                opcodes=sorted(set(op for _a, _l, op in seen)))
