#!/usr/bin/env python3
"""The opcode-centric model: everything Apple's tables say about a G17 instruction, in one place.

tools/g17ir.py is the authoring IR, the compiler's input language. This is the other direction:
what a decoded instruction IS, keyed by Apple's opcode id rather than by a hand-assigned class
name. It is the layer the framing tools, the encoder checks and any later dataflow work should
read from, so that knowledge lives in one table instead of in each script's assumptions.

What is mechanical, straight out of Apple's metadata and not inferred:

    opcode id           stable, 17796 of them, 216 seen in the corpus
    operand count       MCInstrDesc.NumOperands, matches the decoder exactly
    defs vs uses        MCInstrDesc.NumDefs - the first NumDefs operands are definitions
    register class      per operand, named: GPR32, GPR32tup2, FLAGR, IRGPR32, SIR32, ...
    operand type        0 for a plain immediate, non-zero for a typed one (4 is PC-relative)
    scheduling class    44 of them in the corpus, and they cluster functionally
    tsflags             target-specific flag word, not yet decomposed
    implicit uses/defs  present on 402 and 7 opcodes, on NONE of the corpus 216
    register names      MCRegister id to name, GPR block starts at 105 so R_n = 105+n

What is NOT here, deliberately:

    mnemonics           they do not exist, see docs/agx3-oracle.md section 6
    lengths             not a descriptor property; only the decoder knows, and the same opcode
                        appears at several widths (ledger/g17-length-is-base-plus-extension-bit)
    semantics           no claim about what an opcode DOES is made mechanically. A reading of
                        the scheduling classes is recorded in the ledger as inference, and
                        belongs in isa/g17-scalar-isa.toml only once something causal backs it.

    python3 tools/g17model.py <object>        decode with names, defs and uses resolved
    python3 tools/g17model.py --classes       scheduling-class clustering over the corpus
"""
import collections, functools, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# ANCHORED ON THE CHECKOUT ROOT, and the sys.path inserts are gone: agxdis and machobj are package
# modules since 4a, so they are imported rather than found by path. agx3meta is compiled on demand
# and stays in tools/ beside its tracked source.
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS = os.path.join(ROOT, "tools")

META = os.path.join(TOOLS, "agx3meta")
GPR_BASE = 105          # MCRegister id of R0; verified against 300 registers over 60 shaders


META_SNAPSHOTS = {m: os.path.join(ROOT, "isa", "g17-agx3meta-%s.txt" % m) for m in ("regs", "classes", "instrs")}


def _meta(mode):
    # THE COMMITTED SNAPSHOT FIRST (isa/g17-agx3meta-<mode>.txt): the register, class and instruction tables are
    # Apple's and fixed for one OS build. Reading them from a committed file keeps a compile free of external
    # processes AND of uncommitted inputs - cc's release guard reads registers() on every compile (MM 25.144.6),
    # and the delivery paths audit both. test_g17ccguards holds each snapshot equal to the binary's output, so an
    # OS update that changes Apple's tables reds instead of drifting; without a snapshot this runs the binary.
    snap = META_SNAPSHOTS.get(mode, "")
    if os.path.exists(snap):
        with open(snap) as f:
            return [l for l in f.read().splitlines() if not l.startswith("#")]
    src = META + ".c"
    if not os.path.exists(META) or os.path.getmtime(src) > os.path.getmtime(META):
        subprocess.run(["clang", "-O2", "-o", META, src], check=True)
    out = subprocess.run([META, mode], capture_output=True, text=True, check=True).stdout
    return [l for l in out.splitlines() if not l.startswith("#")]


@functools.lru_cache(maxsize=1)
def registers():
    """MCRegister id -> name. Index 0 is unused, as in every LLVM target."""
    names = {}
    for line in _meta("regs"):
        p = line.split(maxsplit=1)
        names[int(p[0])] = p[1].strip() if len(p) > 1 else ""
    return names


@functools.lru_cache(maxsize=1)
def reg_classes():
    """register class id -> (name, bits, count)."""
    out = {}
    for line in _meta("classes"):
        i, name, bits, n = line.split()
        out[int(i)] = (name, int(bits), int(n))
    return out


class Opcode:
    """One row of MCInstrDesc, with the operand descriptors resolved to class names."""

    __slots__ = ("id", "nops", "ndefs", "sched", "tsflags", "operands", "implicit_uses",
                 "implicit_defs")

    def __init__(self, id, nops, ndefs, sched, tsflags, operands):
        self.id, self.nops, self.ndefs = id, nops, ndefs
        self.sched, self.tsflags, self.operands = sched, tsflags, operands
        self.implicit_uses, self.implicit_defs = (), ()

    def signature(self):
        """One token per operand: the register class name, or imm / imm.tN."""
        cls = reg_classes()
        out = []
        for rc, ty, fl in self.operands:
            if rc >= 0:
                out.append(cls.get(rc, ("class%d" % rc, 0, 0))[0])
            else:
                out.append("imm" if ty == 0 else "imm.t%d" % ty)
        return out

    def is_def(self, index):
        """MCInstrDesc puts definitions first, so the first NumDefs operands are written."""
        return index < self.ndefs

    def __repr__(self):
        return "op%d(%s)" % (self.id, " ".join(self.signature()))


@functools.lru_cache(maxsize=1)
def opcodes():
    """opcode id -> Opcode, for all 17796."""
    out = {}
    for line in _meta("instrs"):
        p = line.split()
        # agx3meta's current format has two fixed fields between tsflags and
        # the operand descriptors: flags8, then uses=/defs=.  The old parser
        # started at p[5], so it tried to parse the flags word (or ``uses=-``)
        # as an operand and made every multi-operand opcode unusable.  Slice
        # by NumOperands rather than consuming the remainder; this also makes
        # a malformed metadata line fail loudly instead of silently shifting
        # operands into the next field.
        nops = int(p[1])
        operand_start = 8
        operand_end = operand_start + nops
        if len(p) < operand_end or not p[6].startswith("uses=") or not p[7].startswith("defs="):
            raise ValueError("malformed agx3meta instrs row: %s" % line)
        ops = [tuple(int(x) for x in f.split(":")) for f in p[operand_start:operand_end]]
        if any(len(op) != 3 for op in ops):
            raise ValueError("malformed operand descriptor in agx3meta row: %s" % line)
        out[int(p[0])] = Opcode(int(p[0]), nops, int(p[2]), int(p[3]), p[4], ops)
    return out


class Inst:
    """A decoded instruction: the decoder's framing plus what the tables say about it.

    Deliberately carries no semantic interpretation. defs and uses are the mechanical split by
    NumDefs, register numbers are resolved to Apple's names, and nothing else is claimed.
    """

    __slots__ = ("offset", "size", "raw", "opcode", "values")

    def __init__(self, offset, size, raw, opcode, values):
        self.offset, self.size, self.raw = offset, size, raw
        self.opcode, self.values = opcode, values

    def _operands(self):
        names = registers()
        for i, (kind, value) in enumerate(self.values):
            if kind == "reg":
                yield i, names.get(value, "reg%d" % value)
            elif kind == "expr":
                yield i, "expr:0x%x" % value
            else:
                yield i, "%d" % value

    def defs(self):
        return [t for i, t in self._operands() if self.opcode and self.opcode.is_def(i)]

    def uses(self):
        return [t for i, t in self._operands() if not (self.opcode and self.opcode.is_def(i))]

    def __str__(self):
        d, u = self.defs(), self.uses()
        return "%08x %2d op%-6d sched=%-4d %s%s" % (
            self.offset, self.size, self.opcode.id if self.opcode else -1,
            self.opcode.sched if self.opcode else -1,
            (", ".join(d) + " <- ") if d else "", ", ".join(u))


@functools.lru_cache(maxsize=4096)
def _decode_stdout(t, start):
    """Return decoder text for an exact byte stream and start offset.

    Encoding validation calls the decoder for the same instruction witnesses repeatedly.  Cache
    only the native decoder's textual result within this process: parsing still happens for each
    caller, and the key includes the complete bytes and offset.  This is an execution-local
    optimization; it is not a persistent result cache and cannot reuse output across code or
    interpreter changes.
    """
    from agxforge.g17 import ref as g17ref
    binary = g17ref.binary()
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        f.write(t)
        f.flush()
        r = subprocess.run([binary, f.name, str(start), str(len(t) - start), "--pc", str(start)],
                           capture_output=True, text=True)
    return r.stdout


def operand_value(v):
    """One `key:VALUE` token from the disassembler, as an int.

    EXTRACTED SO IT CAN BE TESTED. This logic lived inline in `decode`, which shells out to a
    binary, so the only way to exercise it was to find a witness that happened to produce the
    token shape you wanted - and the shape that broke it, `unknown:0x00`, appears for just 72 of
    the 6,001 repair-walk witnesses and none of the 717 Apple ones.

    The disassembler emits some values in hex. `int(v)` is base 10, so `0x00` raised ValueError
    and the whole instruction's decode was lost with it, costing those 72 opcodes their width.

    STRICTLY ADDITIVE: a decimal token takes the same branch and the same value as before, and
    the 0x form previously raised. Anything that is neither still raises, loudly, rather than
    being skipped - a token silently dropped would leave the operand list short and the caller
    reading a value that belongs to another field.
    """
    if v[:2].lower() in ("0x", "-0") and "x" in v.lower():
        return int(v, 16)
    return int(v)


class StructuralVariant(Inst):
    """An instruction Apple's compiler emitted and Apple's decoder refuses: a NAMED REFUSAL.

    T11 (docs/g17-tensorops-machine-model.md section 25.71). Apple-compiled int8 widening-MMA
    streams hold `a700a518220ea182c00a`: an op10384 whose bit 71 is set. Clearing bit 71 alone
    gives an ordinary 10-byte op10384; with it set, agx3dis rejects the bytes and, before this
    class, `decode` yielded nothing from that offset on - three 84 KB kernels read as 4,364
    bytes. The meaning of bit 71 is NOT known, so this is deliberately not an op10384: `opcode`
    is None, `values` is empty (reading the cleared form's operands would claim B = R84_R85, and
    that is exactly what is unsettled), and the record carries what IS known - the family it
    derives from, the bit, and the ten-byte framing, which rests on the cleared form's length and
    on the suffix decoding continuously to EOF in all three retained streams (section 24 L1).
    """

    __slots__ = ("variant_of", "structural_bit")

    def __init__(self, offset, raw, variant_of, structural_bit):
        super().__init__(offset, len(raw), raw, None, [])
        self.variant_of, self.structural_bit = variant_of, structural_bit

    def __str__(self):
        return "%08x %2d refused: op%d with structural bit %d set (unsupported variant, meaning unknown)" % (
            self.offset, self.size, self.variant_of, self.structural_bit)


# The only variant admitted: bit 71 of the int8 widening MMA family, with and without C. Nothing
# else is named - a refusal elsewhere still ends the walk, as Apple's decoder does.
STRUCTURAL_VARIANTS = {71: (10384, 10385)}
VARIANT_LENGTH = 10


def structural_variant(raw):
    """(opcode, bit) if `raw` is a known structural-bit variant, else None.

    Decided by Apple's decoder, not by a pattern: the bit must be set, and clearing it alone
    must decode as a single VARIANT_LENGTH-byte instruction of the named family."""
    raw = bytes(raw[:VARIANT_LENGTH])
    if len(raw) < VARIANT_LENGTH:
        return None
    for bit, family in STRUCTURAL_VARIANTS.items():
        if not raw[bit // 8] >> (bit % 8) & 1:
            continue
        cleared = bytearray(raw)
        cleared[bit // 8] &= ~(1 << (bit % 8)) & 0xFF
        first = [line.split() for line in _decode_stdout(bytes(cleared), 0).splitlines()[:1]]
        if first and len(first[0]) >= 3 and first[0][1] != "bad":
            if int(first[0][0], 16) == 0 and int(first[0][1]) == VARIANT_LENGTH and int(first[0][2]) in family:
                return int(first[0][2]), bit
    return None


class DecoderUnavailable(RuntimeError):
    """tools/libagx3dis.dylib is not built (make native-tools) or could not reach Apple's decoder."""


_LIB = [None]


def _decode_text_nofork(t, start):
    """Apple's decoder IN THIS PROCESS (tools/libagx3dis.dylib, built from tools/agx3dislib.c): the same text
    tools/agx3dis prints, with no subprocess. A compile must not start an external process (MM 25.144.6), and
    this is how cc's release guard reads the reference decode anyway. Never builds the library: a missing one
    raises DecoderUnavailable, because building it is itself an external process."""
    import ctypes
    if _LIB[0] is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "tools", "libagx3dis.dylib")
        if not os.path.exists(path):
            raise DecoderUnavailable("%s is not built (make native-tools)" % path)
        lib = ctypes.CDLL(path)
        lib.agx3dis_lib_init.restype = ctypes.c_int
        lib.agx3dis_lib_decode.restype = ctypes.c_long
        lib.agx3dis_lib_decode.argtypes = [ctypes.c_char_p, ctypes.c_long, ctypes.c_long, ctypes.c_long,
                                           ctypes.c_long, ctypes.c_char_p, ctypes.c_long]
        rc = lib.agx3dis_lib_init()
        if rc:
            raise DecoderUnavailable("agx3dis_lib_init failed (%d)" % rc)
        _LIB[0] = lib
    lib = _LIB[0]
    cap = max(4096, 64 * len(t))
    while True:
        out = ctypes.create_string_buffer(cap)
        k = lib.agx3dis_lib_decode(t, len(t), start, len(t) - start, start, out, cap)
        if k == -1:
            raise DecoderUnavailable("agx3dis_lib_decode refused the range")
        if k >= 0:
            return out.raw[:k].decode()
        cap = -k + 1


def decode_nofork(t, start=0):
    """decode() through the in-process decoder: the same Inst records, no external process. A known structural
    variant or a refusal ends the walk (decode's variant recovery forks, so it is not attempted here)."""
    t = bytes(t)
    table = opcodes()
    for line in _decode_text_nofork(t, start).splitlines():
        p = line.split()
        if len(p) >= 2 and p[1] == "bad":
            return
        if len(p) < 3:
            continue
        off, size, op = int(p[0], 16), int(p[1]), int(p[2])
        vals = []
        for tok in p[3:]:
            if ":" not in tok:
                continue
            k, v = tok.split(":", 1)
            vals.append((k, operand_value(v)))
        yield Inst(off, size, t[off:off + size], table.get(op), vals)


def decode(t, start=0):
    """Decode from `start`, yielding Inst. Framing comes from tools/g17ref.py.

    Where Apple's decoder stops on a known structural variant (STRUCTURAL_VARIANTS), a
    StructuralVariant is yielded in its place and decoding resumes after it. Any other refusal
    ends the walk, as before."""
    t = bytes(t)
    table = opcodes()
    pos = start
    while True:
        bad = None
        for line in _decode_stdout(t, pos).splitlines():
            p = line.split()
            if len(p) >= 2 and p[1] == "bad":
                bad = int(p[0], 16)
                continue
            if len(p) < 3:
                continue
            off, size, op = int(p[0], 16), int(p[1]), int(p[2])
            vals = []
            for tok in p[3:]:
                if ":" not in tok:
                    continue
                k, v = tok.split(":", 1)
                vals.append((k, operand_value(v)))
            yield Inst(off, size, t[off:off + size], table.get(op), vals)
        if bad is None:
            return
        named = structural_variant(t[bad:bad + VARIANT_LENGTH])
        if named is None:
            return
        yield StructuralVariant(bad, t[bad:bad + VARIANT_LENGTH], named[0], named[1])
        pos = bad + VARIANT_LENGTH
        if pos >= len(t):
            return


def _classes_report():
    import glob, json, tempfile
    from agxforge.g17 import machobj, agxdis, ref as g17ref
    table = opcodes()
    sc = collections.defaultdict(lambda: dict(n=0, ops=set(), sig=collections.Counter(), ln=set()))
    for d in sorted(glob.glob(os.path.expanduser("~/.cache/agxforge/agx/*"))):
        arc, obj = d + "/s.arc.metallib", d + "/out/object/0-0"
        if not (os.path.exists(arc) and os.path.exists(obj)):
            continue
        try:
            loc = machobj.locate(arc, obj)
            f, sz = agxdis.sections(loc["obj"])
            t = loc["obj"][f:f + sz]
            walk = list(g17ref.walk(t, loc["syms"]["_agc.main"]))
        except Exception:
            continue
        for off, ln, op in walk:
            e = table.get(op)
            if not e:
                continue
            s = sc[e.sched]
            s["n"] += 1
            s["ops"].add(op)
            s["ln"].add(ln)
            s["sig"][tuple(e.signature())] += 1
    print("%-6s %-8s %-5s %-16s %s" % ("sched", "instrs", "ops", "lengths", "dominant shape"))
    for k, v in sorted(sc.items(), key=lambda x: -x[1]["n"]):
        sig = v["sig"].most_common(1)[0][0]
        print("%-6d %-8d %-5d %-16s %s" % (k, v["n"], len(v["ops"]),
              ",".join(str(x) for x in sorted(v["ln"])), " ".join(sig)[:64]))


def main(argv=None):
    """The command-line entry, callable. It was inline under `if __name__`, so after the move
    neither the shim nor the library could invoke it - the compatibility module imports this file
    rather than executing it. Same defect the auth CLI had at 14a13a34.
    """
    argv = list(sys.argv if argv is None else argv)
    if "--classes" in argv:
        _classes_report()
    else:
        from agxforge.g17 import machobj, agxdis
        blob = open(argv[1], "rb").read()
        f, sz = agxdis.sections(blob)
        t = blob[f:f + sz]
        for inst in decode(t, 0):
            print(inst)


if __name__ == "__main__":
    main()
