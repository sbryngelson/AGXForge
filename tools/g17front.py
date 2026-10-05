#!/usr/bin/env python3
"""A SOURCE FRONT END: Metal to this backend's IR, through AIR.

Everything this compiler has ever compiled was a Python program that built g17ir by hand. That is
fine for probes and useless as a compiler: a backend nobody can hand a .metal file to is a backend
whose reach nobody can measure, and "what does it refuse?" has had no answer except "whatever the
person writing the IR did not try".

WHY AIR AND NOT MSL. `xcrun metal -S` emits AIR as LLVM IR text and needs no LLVM installation.
It has already done the parsing, the type checking, the constant folding and the addressing
arithmetic, and - crucially - it is the level at which Apple's own backend receives the program,
so a construct that reaches here is one their compiler had to lower too. Parsing MSL instead would
mean re-deriving semantics this file gets for free and getting them subtly wrong.

WHAT IT DOES NOT DO, said plainly rather than discovered later. This reads STRAIGHT-LINE kernels:
buffers, the thread-position builtins, integer and float arithmetic, loads and stores. Control
flow, vectors wider than the position builtins, atomics, textures, threadgroup memory and calls to
the Metal library all REFUSE BY NAME. That is not a design position, it is where the work stopped,
and tools/g17frontcensus.py counts what the refusals cost over Apple's own corpus rather than
guessing.

    python3 tools/g17front.py k.metal          the IR it builds, or the refusal
    python3 tools/g17front.py --air k.ll       from AIR text directly
"""
import os
import re
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Unsupported(Exception):
    """A construct this front end does not read. Named, never guessed at."""


# --- AIR text -------------------------------------------------------------------------------

def air_of(metal_path):
    """AIR text for a .metal file. One `xcrun metal -S`, no LLVM installation."""
    with tempfile.NamedTemporaryFile(suffix=".ll", delete=False) as fh:
        out = fh.name
    try:
        r = subprocess.run(["xcrun", "metal", "-S", "-o", out, metal_path],
                           capture_output=True, text=True)
        if r.returncode:
            # The Apple front end can reject a Metal argument before producing AIR. Preserve the
            # source construct in our refusal rather than reducing a useful named failure to
            # "3 errors generated." This is especially important for unsupported builtins: our
            # own surface census needs to distinguish a named front-end gap from an opaque SDK
            # diagnostic, and the source is the only instrument available at this phase.
            source = open(metal_path, encoding="utf-8").read()
            attrs = sorted(set(re.findall(r"\[\[\s*([A-Za-z_]\w*)", source)))
            hint = ("; source attributes: " + ", ".join(attrs)) if attrs else ""
            detail = (r.stderr.strip().splitlines() or ["?"])[-1][:160]
            raise Unsupported("xcrun metal refused the source%s: %s" % (hint, detail))
        return open(out).read()
    finally:
        if os.path.exists(out):
            os.unlink(out)


_DEFINE = re.compile(r"^define\s+\w+\s+@(\w+)\((.*?)\)\s*(?:local_unnamed_addr\s*)?#\d+\s*\{",
                     re.M | re.S)
_MDLINE = re.compile(r"^!(\d+) = (?:distinct )?!\{(.*)\}\s*$", re.M)
_KERNELMD = re.compile(r"^!air\.kernel = !\{!(\d+)\}", re.M)


def _metadata(air):
    return {int(n): body for n, body in _MDLINE.findall(air)}


def _argument_nodes(air, md):
    """The per-argument metadata node numbers, in the kernel's parameter order.

    The same walk `_arguments` does - !air.kernel names an argument list, whose items are the
    argument nodes - factored out so a second reader (the atomic field type) can index by POSITION
    without re-deriving the order. A non-node item yields None so positions never shift.
    """
    m = _KERNELMD.search(air)
    if not m:
        return []
    knode = _md_list(md, int(m.group(1)))
    argsnode = None
    for item in knode:
        if item.startswith("!") and item[1:].isdigit():
            argsnode = int(item[1:])
    if argsnode is None:
        return []
    out = []
    for ref in _md_list(md, argsnode):
        ref = ref.strip()
        out.append(int(ref[1:]) if ref.startswith("!") and ref[1:].isdigit() else None)
    return out


def _md_list(md, n):
    """The !N metadata node split into its comma-separated items, quotes kept."""
    out, depth, cur = [], 0, ""
    for ch in md[n]:
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip()); cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


# ARGUMENT KINDS WHOSE SPECIAL REGISTER IS MEASURED, not derived. Each value is the name this
# project's SR table gives the register Apple's compiler reads for a source using ONLY that builtin;
# the backend already emits every one of them, so this widens the FRONT END and lowers nothing new.
#
#     thread_index_in_simdgroup        reg:45   SR_SIMD_ELEM
#     simdgroup_index_in_threadgroup   reg:46   SR_SIMD_GRP
#     threads_per_threadgroup .x/.y/.z reg:55/57/59   SR_TG_X/Y/Z_SIZE
#
# THE HELD-OUT PREDICTION WAS WRONG AND SO WAS THE CONCLUSION IT PRESCRIBED, which is worth more
# than either. Preregistered: y and z would decode to reg:56 and reg:57, one and two above x's 55,
# because the SR table numbers them 24/25/26 consecutively - and if they did not, "the table's
# numbering is not the register numbering and the front-end mapping cannot be written from the
# table", so only x would be mapped.
#
# They decode to 57 and 59. The operands step by TWO, so the first half of that is confirmed: the
# decoded operand is not linear in the table's number. But the conclusion does not follow. The axis
# mechanism applies to the TABLE number, and `threads_per_threadgroup` .x, .y and .z through this
# front end emit exactly reg:55, reg:57 and reg:59 - Apple's own registers on all three authored
# sources. So the table IS the right basis for the mapping; what is not linear is the decoder's
# printed operand, which the mapping never used.
#
# Recording it this way rather than quietly mapping all three: a preregistered consequence can be
# wrong the same way a preregistered prediction can, and the difference between noticing that and
# rationalising it is whether the measurement that overturns it is shown. All three components are
# measured against Apple's compiler, one builtin per source.
#
# WHAT IS NOT MAPPED: nothing about the component selector is inferred - AIR names the argument kind
# once and the component comes from the body, so there is no separate y or z kind to map. Every
# other argument kind refuses by name.
MEASURED_BUILTINS = {
    "thread_index_in_simdgroup": "SR_SIMD_ELEM",
    "simdgroup_index_in_threadgroup": "SR_SIMD_GRP",
    "threads_per_threadgroup": "SR_TG_X_SIZE",
}


def arguments(air):
    """[(kind, detail)] for each kernel argument, in order, from !air.kernel's metadata.

    kind is "buffer" with (location index, declared element type name), or "builtin" with the
    air.* name. Anything else - a texture, a sampler, a threadgroup binding - is returned as
    ("other", the raw node) so the caller refuses it by name rather than mis-reading it as a buffer.

    THE ELEMENT TYPE IS PART OF THE DECLARATION AND DROPPING IT IS NOT FREE. This returned the
    location index alone, so `to_ir` built every ir.Buffer at the default i32 and a `device float *`
    at public index 7 reached the contract as `uint`. The bytes were right - element width is four
    either way - and the image refused with "contract element uint, allocation float", which is the
    contract doing its job about a fact the front end had thrown away. Apple records it as
    `air.arg_type_name`, in exactly the spellings ir.ELEM_NAMES uses.
    """
    md = _metadata(air)
    m = _KERNELMD.search(air)
    if not m:
        raise Unsupported("no !air.kernel metadata - this is not a compute kernel")
    knode = _md_list(md, int(m.group(1)))
    argsnode = None
    for item in knode:
        if item.startswith("!") and item[1:].isdigit():
            argsnode = int(item[1:])
    if argsnode is None:
        raise Unsupported("!air.kernel names no argument list")
    out = []
    for ref in _md_list(md, argsnode):
        if not (ref.startswith("!") and ref[1:].isdigit()):
            continue
        items = _md_list(md, int(ref[1:]))
        text = ", ".join(items)
        if '!"air.buffer"' in text:
            loc, elem = None, None
            for i, it in enumerate(items):
                if it == '!"air.location_index"':
                    loc = int(items[i + 1].split()[-1])
                elif it == '!"air.arg_type_name"':
                    elem = items[i + 1].strip().lstrip("!").strip('"')
            out.append(("buffer", (loc, elem)))
        elif '!"air.thread_position_in_grid"' in text:
            out.append(("builtin", "thread_position_in_grid"))
        elif '!"air.thread_position_in_threadgroup"' in text:
            out.append(("builtin", "thread_position_in_threadgroup"))
        elif '!"air.threadgroup_position_in_grid"' in text:
            out.append(("builtin", "threadgroup_position_in_grid"))
        elif '!"air.thread_index_in_threadgroup"' in text:
            # NOT A REGISTER READ. Unlike the three coordinate builtins above, this one is
            # COMPUTED from the local coordinates and the launch shape - see _linear_local_index
            # for why, and for why the suggestive SR_LIN_ID is not what it lowers to.
            out.append(("builtin", _LOCAL_INDEX_AIR_NAME))
        elif any('!"air.%s"' % k in text for k in MEASURED_BUILTINS):
            # FOUR MORE ARGUMENT KINDS, and the register for each was MEASURED from an authored
            # single-builtin source rather than taken from the SR table. The measurement matters:
            # Apple's own corpus kernels that use `thread_index_in_simdgroup` read reg:38, and a
            # kernel using ONLY that builtin reads reg:45. Those kernels use two builtins and read
            # one register, so attributing that register to either was a guess - and reg:38 would
            # have gone into this table as a plausible number. results/g17-front-builtins-v1 has
            # the five authored sources, one builtin each, with their objects.
            kind = next(k for k in MEASURED_BUILTINS if '!"air.%s"' % k in text)
            out.append(("builtin", MEASURED_BUILTINS[kind]))
        else:
            kind = next((it for it in items if it.startswith('!"air.') and "arg_" not in it),
                        items[1] if len(items) > 1 else "?")
            out.append(("other", kind.strip('!"')))
    return out


_FN = re.compile(r"^define\s+(?:\w+\s+)*?(\S+)\s+@([\w.$]+)\((.*?)\)[^{]*\{\n(.*?)\n\}", re.M | re.S)
_CALL = re.compile(r"^%([\w.]+) = (?:tail |notail |musttail )?call (?:[\w]+ )*(\S+) @([\w.$]+)\((.*)\)")


def _split_args(text):
    """Top-level comma split of a call's argument list, each reduced to its value token."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "([{<": depth += 1
        if ch in ")]}>": depth -= 1
        if ch == "," and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [a.strip().split()[-1] for a in out]


def inline_calls(air):
    """Every call to a function DEFINED IN THIS MODULE, replaced by the callee's body; the callee
    definitions then removed, so the kernel is the only `define` left. -> the rewritten AIR.

    A NOINLINE CALL IS A CALL TO KNOWN CODE, and Apple keeps it as op450 because it chose to; the
    source's meaning is the same inlined. This backend has no call/return lowering yet, so inlining
    is how a program with calls compiles at all. Only the shape whose substitution is exact is
    taken: a callee of ONE basic block ending in `ret`, called with plain values. A callee with
    branches, a void result used as a value, or recursion is refused by name.

    THIS ALSO REPAIRS KERNEL SELECTION. `_DEFINE` matched the FIRST `define` in the module, which
    with a callee present is the callee - the kernel's buffers then read as undefined values."""
    fns = {m.group(2): m for m in _FN.finditer(air)}
    called = {c.group(3) for c in (_CALL.match(l.strip()) for l in air.splitlines()) if c}
    local = called & set(fns)
    if not local:
        return air
    counter = [0]

    def expand(name, actuals, depth):
        if depth > 8:
            raise Unsupported("call nesting deeper than 8 at @%s: recursion is not inlined" % name)
        m = fns[name]
        params = [p.strip().split()[-1] for p in m.group(3).split(",") if p.strip()]
        if len(params) != len(actuals):
            raise Unsupported("@%s takes %d arguments and is called with %d" % (name, len(params), len(actuals)))
        lines = [l.split(", !")[0].rstrip() for l in m.group(4).splitlines()]
        lines = [l for l in lines if l.strip() and not l.strip().startswith(";")]
        if any(re.match(r"^\s*\d+:", l) or l.strip().startswith("br ") for l in lines):
            raise Unsupported("@%s has more than one basic block; only single-block callees are "
                              "inlined, and this backend has no call/return lowering" % name)
        counter[0] += 1
        tag = "inl%d_" % counter[0]
        sub = dict(zip(params, actuals))
        def rn(line):
            def rep(mm):
                tok = "%" + mm.group(1)
                return sub.get(tok, "%" + tag + mm.group(1))
            return re.sub(r"%([\w.]+)", rep, line)
        out, ret = [], None
        for l in lines:
            t = l.strip()
            if t.startswith("ret "):
                parts = t.split()
                ret = None if parts[1] == "void" else rn(parts[-1])
                continue
            c = _CALL.match(t)
            if c and c.group(3) in local:
                val, sub_lines = expand(c.group(3), [rn(a) for a in _split_args(c.group(4))], depth + 1)
                out += sub_lines
                sub["%" + c.group(1)] = val
                continue
            out.append("  " + rn(t))
        return ret, out

    kernel = [n for n in fns if n not in local]
    if len(kernel) != 1:
        raise Unsupported("after inlining %s, %d functions remain; one kernel is expected" % (sorted(local), len(kernel)))
    km = fns[kernel[0]]
    klines, subst = [], {}
    for l in km.group(4).splitlines():
        t = l.strip()
        c = _CALL.match(t)
        if c and c.group(3) in local:
            val, body_lines = expand(c.group(3), [subst.get(a, a) for a in _split_args(c.group(4))], 0)
            if val is None:
                raise Unsupported("@%s returns void and its result is used" % c.group(3))
            klines += body_lines
            subst["%" + c.group(1)] = val
            continue
        if subst:
            l = re.sub(r"%([\w.]+)", lambda mm: subst.get("%" + mm.group(1), "%" + mm.group(1)), l)
        klines.append(l)
    new_kernel = air[km.start():km.start(4)] + "\n".join(klines) + "\n}"
    out = air[:km.start()] + new_kernel + air[km.end():]
    for n in sorted(local, key=lambda n: -fns[n].start()):
        mm = [x for x in _FN.finditer(out) if x.group(2) == n][0]
        out = out[:mm.start()] + out[mm.end():]
    return out


def body(air):
    """The kernel function's instruction lines, without metadata attachments."""
    m = _DEFINE.search(air)
    if not m:
        raise Unsupported("no kernel definition in this AIR")
    rest = air[m.end():]
    end = rest.index("\n}")
    lines = []
    for line in rest[:end].splitlines():
        line = line.split(", !")[0].strip()
        if line and not line.startswith(";"):
            lines.append(line)
    return m.group(1), lines


# --- AIR to g17ir ---------------------------------------------------------------------------

_ASSIGN = re.compile(r"^%(\S+) = (.*)$")
_INT = re.compile(r"^-?\d+$")
# ORDINARY FP32 LITERALS, IN THE THREE FORMS APPLE'S AIR ACTUALLY USES. Measured over root's
# retained AIR for all sixteen literal candidates (results/g17-source-admission-v2/
# construct-ranking.json): 195 exponent tokens, 170 LLVM hex tokens, 16 plain decimals.
#
#   decimal / exponent   1.000000e+00, 5.000000e-01, 1.000000e+03   -> float(), then binary32 RNE
#   LLVM hex             0x3FF003AFC0000000                          -> DOUBLE bits, then binary32
#
# THE HEX FORM IS A DOUBLE BIT PATTERN AND NOT A FLOAT ONE. LLVM prints a `float` constant in hex as
# the sixty-four-bit double whose value the float equals exactly, so reading those bits as a float32
# would be the "numeric conversion versus bit reinterpretation" confusion this campaign is warned
# about - 0x3FF003AFC0000000 is 1.0009000301361084, not whatever its low 32 bits spell. Measured:
# all 150 distinct hex constants in the candidates are EXACTLY representable in binary32, 150 of
# 150, so exactness is asserted rather than assumed. A hex constant that is not exactly a float is
# not a float constant, and it refuses.
_FLOAT_DEC = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")
_FLOAT_HEX = re.compile(r"^0x[0-9A-Fa-f]{16}$")
# LLVM's other hex literal widths, refused by name rather than reinterpreted: 0xH half, 0xK x86_fp80,
# 0xL fp128, 0xM ppc_fp128. Their bit layouts are not this one and a silent read would be wrong.
_FLOAT_HEX_OTHER = re.compile(r"^0x[HKLM][0-9A-Fa-f]+$")


# Set to read AIR as this front end read it before FP32 literals were parsed, so the byte-identity
# of every previously-compiled source stays measurable in one process rather than asserted. It
# restores BOTH halves of the change: the parser below, and the two character classes that were
# missing `+`.
_NO_FP32_LITERALS = False


# THE BINARY32 RANGE, and both edges of it. Root's review found `1.0e+40` and
# `0x7FEFFFFFFFFFFFFF` raising OverflowError out of `struct.pack` - a crash, not a refusal - and the
# same check turns up a third case root did not name: `1.0e-60` narrowed SILENTLY to 0x00000000.
# The overflow is loud and wrong; the underflow is quiet and worse, because a nonzero constant
# becoming exactly zero is the shape that cost a whole hardware campaign when 1.0 was emitted as
# 1.4e-45. Both are refused by name, and so is a value that lands in the subnormal range: what this
# backend does with a denormal is opcode-dependent and unmeasured (op3850 flushes its input to zero
# where op3978 does not), so a literal that can only be represented as one is not converted here.
#
# THE GUARD ACCEPTS GROUND TRUTH. Every literal in root's retained population is around 1.0, 0.5 or
# 1000.0 - no overflow, no underflow, no subnormal - so these three refusals are a boundary rather
# than a wall, and the 63 admissions are unaffected.
_F32_MAX = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]          # 3.4028234663852886e+38
_F32_MIN_NORMAL = struct.unpack("<f", struct.pack("<I", 0x00800000))[0]   # 1.1754943508222875e-38


def _f32_bits_in_range(value, tok):
    """-> the binary32 bits of a finite `value`, or Unsupported naming which edge it crossed."""
    if abs(value) > _F32_MAX:
        raise Unsupported("floating-point literal %r is %r, outside binary32's finite range "
                          "(+-%r); the IEEE result would be an infinity and this front end refuses "
                          "non-finite values" % (tok, value, _F32_MAX))
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    narrowed = struct.unpack("<f", struct.pack("<I", bits))[0]
    if value != 0.0 and narrowed == 0.0:
        raise Unsupported("floating-point literal %r is %r and narrows to exactly zero in binary32; "
                          "a nonzero constant silently becoming zero is the defect that made "
                          "g17ir.const refuse a Python float, so it refuses here too" % (tok, value))
    if narrowed != 0.0 and abs(narrowed) < _F32_MIN_NORMAL:
        raise Unsupported("floating-point literal %r is %r, a binary32 SUBNORMAL; what this backend "
                          "does with a denormal is opcode-dependent and unmeasured (op3850 flushes "
                          "its input to zero where op3978 does not), so it is refused rather than "
                          "converted" % (tok, value))
    return bits


def f32_literal_bits(tok):
    """-> the binary32 BIT PATTERN of an AIR float literal, or None if `tok` is not one.

    Raises Unsupported for a literal this front end will not convert: a non-finite value, a hex
    constant that is not exactly a float, or one of LLVM's other floating-point widths. Returning
    bits rather than a number is deliberate - g17ir.const takes the bit pattern and REFUSES a Python
    float at a float type, because passing 1.0 once emitted the immediate 1, which as an IEEE single
    is 1.4e-45, and a whole hardware campaign recorded the resulting zeros as a dependency failure.
    """
    if _NO_FP32_LITERALS:
        return None
    if _FLOAT_HEX_OTHER.match(tok):
        raise Unsupported("floating-point literal %r: LLVM's %s width is not binary32 and its bits "
                          "are laid out differently; this front end converts f32 only"
                          % (tok, {"H": "half", "K": "x86_fp80", "L": "fp128",
                                   "M": "ppc_fp128"}[tok[2]]))
    return _f32_from_decimal_or_hex(tok)


_FLOAT_HEX_HALF = re.compile(r"^0xH[0-9A-Fa-f]{4}$")
_NO_FP16_LITERALS = False


def f16_literal_bits(tok):
    """-> the binary16 BITS of an LLVM half literal, or None if the token is not one.

    NO ROUNDING QUESTION EXISTS HERE, and that is the whole difference from the f32 parser above.
    LLVM prints a half constant as `0xH` followed by the four hex digits of its binary16 encoding,
    so the token IS the bit pattern: there is no decimal to convert, nothing to narrow, and no
    edge of the range to cross. The f32 parser needs three refusals (overflow, narrows-to-zero,
    subnormal) because it converts a value; this one reads sixteen bits that are already binary16.
    A denormal or a non-finite pattern is still refused, because what this backend DOES with one
    is unmeasured either way - and `0xH7C00` is what an infinity would arrive as.
    """
    if _NO_FP16_LITERALS or not _FLOAT_HEX_HALF.match(tok):
        return None
    bits = int(tok[3:], 16)
    if (bits >> 10) & 0x1F == 0x1F:
        raise Unsupported("half literal %r is an infinity or a NaN; nothing here has measured what "
                          "this backend does with a non-finite half" % tok)
    if (bits >> 10) & 0x1F == 0 and bits & 0x3FF:
        raise Unsupported("half literal %r is a binary16 SUBNORMAL; what this backend does with a "
                          "denormal is opcode-dependent and unmeasured" % tok)
    return bits


def _f32_from_decimal_or_hex(tok):
    """The binary32 half of the parser, split out so the half literal can precede it."""
    if _FLOAT_HEX.match(tok):
        as_double = struct.unpack("<d", struct.pack("<Q", int(tok, 16)))[0]
        if as_double != as_double or as_double in (float("inf"), float("-inf")):
            raise Unsupported("floating-point literal %r is not finite; nothing here has measured "
                              "what this backend does with an infinity or a NaN" % tok)
        if abs(as_double) > _F32_MAX:
            raise Unsupported("floating-point literal %r is %r, outside binary32's finite range "
                              "(+-%r)" % (tok, as_double, _F32_MAX))
        narrowed = struct.unpack("<f", struct.pack("<f", as_double))[0]
        if narrowed != as_double:
            raise Unsupported("floating-point literal %r is %r as a double and %r as binary32: "
                              "LLVM prints a float constant in hex only when the two agree, so "
                              "this is not an f32 constant and reinterpreting it would be a guess"
                              % (tok, as_double, narrowed))
        return _f32_bits_in_range(as_double, tok)
    if _FLOAT_DEC.match(tok) and not _INT.match(tok):
        value = float(tok)
        if value != value or value in (float("inf"), float("-inf")):
            raise Unsupported("floating-point literal %r is not finite; nothing here has measured "
                              "what this backend does with an infinity or a NaN" % tok)
        # RNE to binary32 is what a decimal float constant MEANS; struct does it once. Signed zero
        # survives it - -0.0 packs to 0x80000000 - and that is a distinct bit pattern this returns
        # rather than collapsing, because the sign of a zero is observable in a stored word.
        return _f32_bits_in_range(value, tok)
    return None

# LLVM binary operators this backend has an IR op for. The name on the right is the Builder method.
BINOPS = {"add": "add", "mul": "mul", "sub": "sub", "and": "and", "or": "or", "xor": "xor",
          "shl": "shl", "lshr": "shr", "fadd": "fadd", "fmul": "fmul", "fsub": "fsub"}


def _binop(b, head, x, y, name):
    """One BINOPS operation. fsub has no instruction of its own: it is the float add with the
    NEGATE SOURCE MODIFIER on the subtrahend (ledger g17-float-source-modifiers: "there is no fsub"),
    which cc folds into the add's modifier word - one op998, as Apple emits it. IEEE-exact: x - y IS
    x + (-y) for every binary32 pair, signed zeros included (x - (+0) = x + (-0)), because negation
    is exact. A constant subtrahend has its sign bit flipped here instead, the same exact operation,
    so no modifier is asked to ride on an immediate."""
    if BINOPS[head] != "fsub":
        return getattr(b, BINOPS[head])(x, y, name=name)
    ir = _ir()
    if isinstance(y, ir.Imm):
        flipped = (y.v ^ 0x80000000) & 0xFFFFFFFF
        return b.fadd(x, b.const(flipped, name="k%x" % flipped), name=name)
    return b.fadd(x, b.fneg(y, type=ir.I32, name=name + "_n"), name=name)

# SIXTY-FOUR-BIT INTEGER DATA AS TWO EXPLICIT THIRTY-TWO-BIT WORDS.
#
# The six sources root assessed all STORE i64 results - none of their wide values is a getelementptr
# index - so they need real wide data arithmetic. What they do NOT need is a new instruction:
# Apple's paired-word store is its CHOICE, and a `ulong` element's two halves are reachable with
# the word store this backend already emits (ir.load_word_component / store_word_component).
#
# THE CARRY AND BORROW ARE BITWISE, not a flag this backend has. For unsigned 32-bit words:
#
#     lo     = a0 + b0                                     carry = 1 exactly when a0 + b0 >= 2^32
#     carry  = ((a0 & b0) | ((a0 | b0) & ~lo)) >> 31        hi = a1 + b1 + carry
#     lo     = a0 - b0                                      borrow = 1 exactly when a0 < b0
#     borrow = ((~a0 & b0) | ((~a0 | b0) & lo)) >> 31       hi = a1 - b1 - borrow
#
# Both are the standard full-adder/full-subtractor sign-bit extractions and both are exact over all
# 2^64 operand pairs, which the reference test checks on edges and random cases rather than
# asserting. `~x` is `x ^ 0xFFFFFFFF` because the dedicated not form op11190 is refused by the
# checker; xor with an all-ones immediate is an operation this front end already emits.
#
# WHAT ESTABLISHES A PAIR is an allowlist, deliberately: an i64 value is usable as data only when
# BOTH of its words are known, and only three shapes establish that - a component load from a wide
# declaration, a zext from 32 bits (whose high word is zero BY DEFINITION), and the result of one
# of these compositions. Anything else refuses by name rather than being given a guessed high word,
# which is the failure mode that would silently compute on garbage.
_NO_WIDE_ARITHMETIC = False
_WIDE_INT_TYPES = ("i64",)
_WORD_MASK = 0xFFFFFFFF
_WIDE_SIGN_SHIFT = 31


def _wide_not(b, x, name):
    """~x as `x ^ 0xFFFFFFFF`; the dedicated not form is refused by the checker."""
    return getattr(b, "xor")(x, b.const(_WORD_MASK, name="ones"), name=name)


def _wide_add_words(b, a, c):
    """(lo, hi) of a + c over word pairs, with the carry taken from the sign bit."""
    lo = b.add(a[0], c[0], name="wlo")
    both = getattr(b, "and")(a[0], c[0], name="wboth")
    either = getattr(b, "or")(a[0], c[0], name="weither")
    spill = getattr(b, "and")(either, _wide_not(b, lo, "wnlo"), name="wspill")
    carry = b.shr(getattr(b, "or")(both, spill, name="wcbits"),
                  ir_imm(_WIDE_SIGN_SHIFT), name="wcarry")
    hi = b.add(b.add(a[1], c[1], name="whi0"), carry, name="whi")
    return lo, hi


def _wide_sub_words(b, a, c):
    """(lo, hi) of a - c over word pairs, with the borrow taken from the sign bit."""
    lo = b.sub(a[0], c[0], name="wlo")
    na = _wide_not(b, a[0], "wna")
    only = getattr(b, "and")(na, c[0], name="wonly")
    spill = getattr(b, "and")(getattr(b, "or")(na, c[0], name="weither"), lo, name="wspill")
    borrow = b.shr(getattr(b, "or")(only, spill, name="wbbits"),
                   ir_imm(_WIDE_SIGN_SHIFT), name="wborrow")
    hi = b.sub(b.sub(a[1], c[1], name="whi0"), borrow, name="whi")
    return lo, hi


def ir_imm(v):
    return _ir().Imm(v)


def _wide_bitwise_words(b, a, c, which):
    """and/or/xor over word pairs: componentwise, and exact with no carry between halves."""
    return (getattr(b, which)(a[0], c[0], name="wb0"),
            getattr(b, which)(a[1], c[1], name="wb1"))


def _wide_shift_words(b, a, amount, direction):
    """A CONSTANT 64-bit shift over a word pair, or a named refusal.

    WHY THIS EXISTS AT ALL. `r-shr64a` compiled to 100 bytes and computed the WRONG ANSWER: its
    `shl i64`, `or i64` and `lshr i64` each fell through to a 32-bit arm, so the program evaluated
    the shift on the low word alone. Root interpreted the delivered code at one thread and I
    reproduced it - for B0 = 1, 7, 8 and 0x10000001 the program stored 0, 0, 1 and 0x02000000
    where the source means 0x20000000, 0xE0000000, 1 and 0x22000000. Three of four wrong.
    My own test had asserted that program's BYTE COUNT was unchanged, which locked the defect in:
    unchanged bad bytes are not a guard, they are a bug with a regression test.

    ONLY A CONSTANT AMOUNT IS LOWERED. A variable shift needs a runtime select between the three
    cases below and that is a different construct; it refuses by name rather than being invented.
    Shifts of 64 or more are undefined in the source language and refuse too, rather than being
    normalised into something the source did not say.
    """
    ir = _ir()
    if not isinstance(amount, ir.Imm):
        raise Unsupported(
            "a 64-bit %s by a VARIABLE amount: the word-pair lowering is defined per case - below "
            "32, exactly 32, and above - and choosing between them at runtime is a different "
            "construct from the constant shift this front end lowers" % direction)
    n = amount.v & 0xFFFFFFFFFFFFFFFF
    if n >= 64:
        raise Unsupported(
            "a 64-bit %s by %d: a shift of 64 or more is undefined in the source language, so this "
            "refuses rather than normalising it into an answer the source does not state"
            % (direction, n))
    zero = b.const(0, name="wz")
    if n == 0:
        return a
    if direction == "shr":
        if n == 32:
            return (a[1], zero)
        if n > 32:
            return (b.shr(a[1], ir.Imm(n - 32), name="wsh"), zero)
        return (getattr(b, "or")(b.shr(a[0], ir.Imm(n), name="wlo0"),
                                 b.shl(a[1], ir.Imm(32 - n), name="wlo1"), name="wlo"),
                b.shr(a[1], ir.Imm(n), name="whi"))
    if n == 32:
        return (zero, a[0])
    if n > 32:
        return (zero, b.shl(a[0], ir.Imm(n - 32), name="wsh"))
    return (b.shl(a[0], ir.Imm(n), name="wlo"),
            getattr(b, "or")(b.shl(a[1], ir.Imm(n), name="whi0"),
                             b.shr(a[0], ir.Imm(32 - n), name="whi1"), name="whi"))


def _wide_low_word(held):
    """The low 32-bit word of an established pair, whether a promise or a materialised pair."""
    if isinstance(held, tuple) and len(held) == 2 and held[0] == "zext32":
        return held[1]
    return held[0]


def _wide_target_declares_wide(rhs, val, argmap):
    """Whether an i64 load/store's buffer DECLARES a wide scalar.

    WHY THIS GUARD DEFERS INSTEAD OF REFUSING. A retained control compiles a `ulong` source with
    the declaration deliberately relabelled to `uint`, reproducing byte for byte what this front
    end emitted before wide declarations existed - a four-byte load on an eight-byte element. Those
    bytes are the null arm of an earlier batch: they are how that behaviour is KNOWN to be wrong
    rather than merely different, and refusing them here would delete the control. So when the AIR
    says i64 and the declaration says otherwise, the declaration still wins exactly as before, and
    the component path engages only where the declaration is genuinely wide.
    """
    ir = _ir()
    m = re.search(r"i64 addrspace\(\d+\)\*\s+%([\w.]+)", rhs)
    if m is None:
        m = re.search(r"addrspace\(\d+\)\*\s+%([\w.]+)", rhs)
    if m is None:
        return False
    key = m.group(1)
    held = val.get(key)
    buf = None
    if isinstance(held, tuple) and held and held[0] == "gep":
        buf = held[1]
    elif key in argmap and argmap[key][0] == "buffer":
        buf = argmap[key][1]
    return buf is not None and buf.elem in ir.WORD_COMPONENTS


def _wide_int_data(head, rhs, val, argmap, words, wide_results):
    """Whether this line is a 64-bit integer DATA operation this arm owns.

    DELIBERATELY NARROW, because a wider guard regresses a source that already compiles.
    `r-shr64a` is admitted today and contains `shl i64`, `or i64` and `lshr i64`; its wide values
    never become established word pairs, so intercepting every i64-typed line would refuse a
    program that works. This owns four shapes - a wide load, a wide store, a wide add or sub, and
    a truncation of a pair - and otherwise claims nothing.

    THE SECOND CLAUSE IS THE REFUSAL SIDE OF THE SAME RULE: once a value's two words ARE
    established here, any operation this arm does not lower must refuse by name rather than reach
    a 32-bit arm that would silently compute on the low word alone. That is scoped to values this
    arm created, which is why it cannot affect `r-shr64a`.
    """
    if _NO_WIDE_ARITHMETIC or _NARROW_ELEMENTS_AS_WORD:
        # DECLINING IS A BETTER NULL ARM THAN A MESSAGE OF MY OWN. With the capability off this
        # arm claims nothing, so each source refuses with exactly the refusal it had before the
        # batch - naming the operation, the declared spelling and its 8 bytes - which is the
        # measurement the gain is scored against. A generic "switched off" string would have
        # replaced that historical message with a new one and quietly lost it.
        # _NARROW_ELEMENTS_AS_WORD is here for the same reason it is everywhere else: it exists to
        # reproduce four retained WRONG programs byte for byte.
        return False
    if head == "load":
        return (bool(re.match(r"load i64\b", rhs))
                and _wide_target_declares_wide(rhs, val, argmap))
    if head == "store":
        return (bool(re.match(r"store i64\b", rhs))
                and _wide_target_declares_wide(rhs, val, argmap))
    if head in ("add", "sub", "or", "and", "xor", "shl", "lshr"):
        if not re.match(r"(?:add|sub|or|and|xor|shl|lshr) (?:\w+ )*i64\b", rhs):
            return False
        # OWN IT WHEN THE WORDS ARE THERE, REFUSE WHEN SOME ARE, DEFER WHEN NONE ARE.
        #
        # Both established: lower it as a word pair. SOME established: refuse, because completing
        # it would need a guessed high word and letting it reach a 32-bit arm would silently
        # compute on low words. NONE established: this is not a wide computation this arm created
        # at all - the retained control relabels a `ulong` declaration as `uint`, so its operands
        # are ordinary words and its `sub i64` must keep emitting the four-byte arithmetic those
        # bytes document as wrong. Deferring is what preserves that null arm.
        operands = [t.lstrip("%") for t in re.findall(r"%[\w.]+", rhs)]
        return any(t in words for t in operands)
    # NO `trunc` BRANCH HERE ON PURPOSE: the zext/trunc arm runs earlier in the dispatch chain and
    # takes the low word there. A branch here would be unreachable, and an unreachable arm is a
    # claim nobody can check.
    if not wide_results:
        return False
    return any(tok.lstrip("%") in wide_results for tok in re.findall(r"%[\w.]+", rhs))


def _wide_declaration(buf, what):
    """The buffer must DECLARE a wide scalar; its declaration is preserved either way."""
    ir = _ir()
    if buf.elem not in ir.WORD_COMPONENTS:
        raise Unsupported(
            "a 64-bit %s on buffer %s, whose declared element is %s: the word-component access is "
            "for a declared wide scalar (%s), and this declaration is not one - relabelling it "
            "would lose the element the source states"
            % (what, buf.name, buf.elem, "/".join(sorted(ir.WORD_COMPONENTS))))

# AIR library calls this backend has an IR operation for: (Builder method, argument count).
# Everything else refuses BY NAME, which is what makes the census's top rows readable as work.
# THE CALLS THIS FRONT END READS, by AIR name. Everything else refuses by name, which is why the
# conversion family dominates the refusal histogram: until now this table had one entry.
#
# air.convert.f.f32.u.i32 is unsigned 32-bit integer to FP32, the largest ordinary-compute group in
# that histogram (14 first refusals). The INVERSE (u.i32.f.f32, 8) and the HALF-RESULT conversions
# (f.f16.u.i32, 6) stay absent deliberately: they are different instructions and nothing here has
# measured them. So does everything Apple compiles to op11185, whose operand 0 is degenerate and
# which has no corpus instance at all. CORRECTED 2026-09-23: isa/g17-vendor-corpus-forms.json counts
# 3,338 op11185/10 (10 in the 6,594-program corpus, 3,328 in the vendor set), so that absence was the
# older spans index's, not Apple's; its operand 0 is still unestablished.
# Set to measure this front end as it was before the constant-address scalar store was routed
# through store_at; see the store branch below.
_NO_CONSTANT_STORE_AT = False

# THE ELEMENT WIDTH OF AN ACCESS IS A FORM, NOT A TYPE ANNOTATION. Only `half` is mapped, and
# every other 16-bit-looking element refuses by name: whether op17193 is a WIDTH form that also
# carries `short` or a HALF form is not measured here, and guessing would put an unmeasured claim
# in the one layer every kernel passes through. `float`/`i32`/`uint` and the other 32-bit elements
# are the word form, which is what this front end has always emitted.
_HALF_ELEMENTS = ("half",)
# SIXTEEN-BIT INTEGER ELEMENTS REFUSE, AND THE REASON IS MEASURED RATHER THAN ASSUMED.
#
# WHAT WAS WRONG. `load i16` and `store i16` went through the WORD form - op12682/14 and
# op17229/8 - so a program touched FOUR bytes where its source touches two, at byte offset 4i
# where the source means 2i. Four of the frozen 63 did exactly that (mn-s16_s32.add-4 and
# mn-u16_u32.and-4 load a short; mn-s32_s16.shl-2 and mn-u32_u16.xor-2 store one) and their
# hashes are in baseline-native-code.json, so the wrong width was frozen. The same programs also
# lost their `sext`/`zext`/`trunc`, which this front end passed through as no-ops.
#
# WHAT APPLE ACTUALLY SELECTS, measured through the vendor's own compiler (Metal -> Apple's AIR ->
# Apple's native code -> this project's decoder, four one-variable kernels plus word and half
# controls):
#
#     short  s[i] -> int      op12646/14 into reg:425, then op10284/12 (reg:105 <- reg:425,
#                             imm:24) - the SIGN EXTENSION - then the word store
#     ushort s[i] -> uint     op12646/14 into reg:426, then op555/4, then the word store
#     (short)x   -> s[i]      op17193/10
#     (ushort)x  -> s[i]      op17193/10
#     int    s[i] -> int      op12682/14  (the word control)
#     half   s[i] -> float    op12646/14  (the half control - the SAME load form as short)
#
# So op12646 and op17193 are ELEMENT-WIDTH forms, shared by half and by 16-bit integers: that
# question is now closed, and it closes in the direction that says those four programs' bytes must
# move. What is NOT closed is the extension: this backend has no integer 16<->32 widening or
# narrowing at all (only f16_to_f32 / f32_to_f16_rte), and op10284/12's source-width and
# sign-versus-zero semantics are unmeasured here. A 16-bit load whose value is then used as a
# 32-bit operand cannot be expressed correctly, so it is REFUSED rather than approximated.
#
# THE DEFAULT IS THE REFUSAL. `True` restores the old, wrong behaviour and exists only as an
# explicit negative-control arm: it reproduces the four retained programs byte for byte, which is
# what keeps them usable as negative evidence.
_NARROW_ELEMENTS_AS_WORD = False
# THE TWO CAPABILITIES THIS BATCH ADDS, EACH SWITCHABLE AT THE POINT OF MEASUREMENT. A census taken
# before a change is only comparable to one taken after if the change can be turned off in the same
# process, which is the reason _NO_FP32_LITERALS and _NO_CONSTANT_STORE_AT exist above. The first
# time these two were measured the off arm was a flag nobody read, so both arms returned 66 of 197
# and the "control" could not have failed. Two arms agreeing exactly is the tell.
_NO_HALF_CONVERSIONS = False      # fptrunc/fpext -> op1016/op1004
_NO_HALF_ELEMENT_WIDTH = False    # `half` load/store -> op12646/op17193 rather than the word form
_NARROW_ELEMENTS = ("short", "ushort", "i16", "bfloat", "char", "uchar", "i8")


def _fast_math(rhs, what):
    """The call's fast-math flags, refusing when the composition needs them and they are absent."""
    flags = set(re.findall(r"\b(fast|nnan|ninf|nsz|arcp|contract|afn|reassoc)\b",
                           rhs.split("@")[0]))
    if not ({"fast"} <= flags or {"nnan", "ninf", "nsz"} <= flags):
        raise Unsupported(
            "%s without the fast-math flags its lowering needs (has %s): op9700's value on NaN is "
            "unmeasured and so is the sign of a zero result, so `nnan` and `nsz` are required, and "
            "the executed domain is finite, so `ninf` is too. `fast` carries all three. The strict "
            "contract needs a measurement of op9700 on NaN and on signed zero"
            % (what, ", ".join(sorted(flags)) or "no flags"))
    return flags


def _ir():
    import g17ir as ir
    return ir


def ir_accessible():
    return _ir().ACCESSIBLE_ELEMS


def _element_width(rhs, what):
    """"word" or "half" for an access, from the type the AIR instruction itself names."""
    m = re.match(r"(?:load|store) ([\w.]+)[ ,]", rhs)
    if not m:
        raise Unsupported("%s %r" % (what, rhs))
    ty = m.group(1)
    if ty in _HALF_ELEMENTS:
        return "word" if _NO_HALF_ELEMENT_WIDTH else "half"
    if ty in _NARROW_ELEMENTS and not _NARROW_ELEMENTS_AS_WORD:
        # THE TWO DIRECTIONS ARE NO LONGER THE SAME QUESTION, and this refusal used to treat them
        # as one. It said this backend has no integer 16-bit widening OR narrowing; the narrowing
        # now exists (ir.low16 -> op590/4, resting on 425+n being the low half of word n, witnessed
        # by the imageblock prologue and measured for the read view by root's op10283 evidence), so
        # a STORE of a sixteen-bit element can be carried: compute in a word, take its low half.
        #
        # A LOAD NOW CAN, and this is where that changed. Both routes to a widening were once
        # unmeasured - op10284/12's source width and sign-versus-zero behaviour, or op10283's
        # zero-first-operand endpoint - and root measured the second one twice: the zero first
        # operand (results/g17-integer16-zero-first-v1) and then the half-load readiness that
        # feeding it a loaded halfword requires (results/g17-integer16-half-load-v1, whose
        # non-waiting control is falsified). So the widening exists, the load keeps its element
        # width, and _NO_U16_TO_U32 below is what switches the capability off - it preserves the
        # refusal this used to raise unconditionally rather than asserting the facts are missing.
        if what == "load" and not _NO_U16_TO_U32:
            # THE LOAD KEEPS ITS ELEMENT WIDTH and the widening happens at the USE, through
            # op10283's measured zero-first form. Admitting the load is only correct because that
            # widening exists: without it the loaded value could not reach a 32-bit operand at all,
            # which is what this refusal used to say for both directions at once.
            return "half"
        if what != "store" or _NO_NARROW_ELEMENT_STORE:
            raise Unsupported(
                "a %s-element %s: Apple selects op12646/14 and op17193/10 for it - the same "
                "element-width forms as `half` - and a sixteen-bit element STORE is carried as the "
                "low half of a word (ir.low16). A LOAD needs a widening to reach a 32-bit operand, "
                "and this backend HAS one - op10283's measured zero-first form with the load wait "
                "(results/g17-integer16-zero-first-v1 and results/g17-integer16-half-load-v1) - "
                "but %s, so the load cannot be admitted here. op10284/12's source width and its "
                "sign-versus-zero extension remain unmeasured and are not used"
                % (ty, what, "_NO_U16_TO_U32 switches that widening off"
                   if _NO_U16_TO_U32 else "_NO_NARROW_ELEMENT_STORE switches the element store off"))
        # AN ADMITTED SIXTEEN-BIT STORE TAKES THE ELEMENT-WIDTH FORM, which is the whole point:
        # falling through to "word" here is what wrote four bytes where the source writes two.
        return "half"
    return "word"


# VECTOR LANES: a <N x T> access is N accesses of T, and every lane is an ordinary SSA value.
#
# The back end already has all of this: the emitted address strides by the ACCESS width (a word
# access carries scale 4, a half access 2) rather than by the buffer's declared element, and the
# declared element reaches only the ABI. So a lane access needs no new form, no new opcode and no
# new address arithmetic - it needs the front end to stop treating <N x T> as one indivisible
# thing. ir.LANE_ELEM names which declarations have an accessible lane; the gate there admits a
# load or store of exactly the lane's width and refuses everything else, so a whole-element
# access still refuses by name.
#
# WHAT IS NOT CLAIMED: that N scalar operations equal one vector operation numerically. Nothing
# here composes a new arithmetic; each lane runs the SAME measured scalar operation the scalar
# sources already run, and a lane whose scalar counterpart is not measured refuses by name rather
# than being approximated. Lane ORDER is explicit - lane k of element i is index i*lanes + k - and
# a permutation is applied to the lane list, never to the addresses.
_NO_VECTOR_LANES = False
_LANE_WIDTH = {"i32": "word", "float": "word", "half": "half", "i16": "half"}
_LANE_COUNTS = (2, 4)           # the widths this front end indexes; 3-lane memory refuses by name


class _Poison:
    """An undef/poison lane. Consuming one refuses; carrying one is how AIR spells a splat."""
    __slots__ = ()
    def __repr__(self):
        return "poison"


POISON = _Poison()


_IR_FLAGS = ("fast", "nnan", "ninf", "nsz", "arcp", "contract", "afn", "reassoc",
             "exact", "nsw", "nuw", "inbounds", "volatile")


def _vector_type(text):
    """(lanes, lane type) for a leading <N x T>, past any flag words, else None.

    `fadd fast <4 x float> %a, %b` puts `fast` between the head and the type, and reading only the
    first token after the head saw no vector there: float4 and half4 fell through to the scalar
    parser and refused with a parse message instead of being expanded or refused by name.
    """
    m = re.match(r"\s*((?:(?:%s)\s+)*)<(\d+) x (\w+)>" % "|".join(_IR_FLAGS), text)
    return (int(m.group(2)), m.group(3)) if m else None


def _lanes_of(v):
    """The lane list of a vector value, or None if this is not one."""
    return list(v[1]) if isinstance(v, tuple) and len(v) == 2 and v[0] == "lanes" else None


def _lane_value(lanes):
    return ("lanes", list(lanes))


def _vector_access(rhs, what):
    """(lanes, lane type, access width) for a vector load/store, refusing by name what it cannot do."""
    vt = _vector_type(rhs[len(what):])
    if vt is None:
        return None
    lanes, lty = vt
    if _NO_VECTOR_LANES:
        raise Unsupported("a %d-lane %s of %s: the vector-lane capability is switched off here, "
                          "which is the state the gain is measured against" % (lanes, what, lty))
    width = _LANE_WIDTH.get(lty)
    if width is None:
        raise Unsupported("a %d-lane %s of %s: this front end indexes lanes of %s only"
                          % (lanes, what, lty, ", ".join(sorted(_LANE_WIDTH))))
    if lanes not in _LANE_COUNTS:
        raise Unsupported("a %d-lane %s: lane indexing is written for %s lanes, and a count "
                          "outside that is refused rather than indexed by a guess"
                          % (lanes, what, " and ".join(str(n) for n in _LANE_COUNTS)))
    return lanes, lty, width


def _lane_index(b, base, lanes, k):
    """Index of lane k of element `base` - i*lanes + k, in LANE units. Lane order is this line."""
    if isinstance(base, _ir().Imm):
        flat = base.v * lanes + k
        return b.const(flat, name="k%d" % flat)
    shift = {2: 1, 4: 2}[lanes]
    scaled = b.shl(base, _ir().Imm(shift), name="lane%d" % lanes)
    return scaled if k == 0 else b.add(scaled, _ir().Imm(k), name="l%d" % k)


def _lane_declaration(buf, lanes, width, what):
    """The declared element must BE a vector of this many lanes at this width."""
    declared = getattr(buf, "elem", None)
    phys = _ir().DECL_PHYSICAL.get(declared, {})
    lane_elem = getattr(_ir(), "LANE_ELEM", {}).get(declared)
    if lane_elem is None:
        raise Unsupported("a %d-lane %s on buffer %s, which the source declares as %s: that "
                          "declaration has no lane this backend accesses, so the access refuses "
                          "rather than reinterpreting the declaration"
                          % (lanes, what, getattr(buf, "name", "?"), declared))
    if phys.get("lanes") != lanes:
        raise Unsupported("a %d-lane %s on buffer %s, which the source declares as %s with %s "
                          "lanes: the access and the declaration disagree about the lane count "
                          "and this front end does not choose between them"
                          % (lanes, what, getattr(buf, "name", "?"), declared, phys.get("lanes")))
    if _ir().lane_width_name(lane_elem) != width:
        raise Unsupported("a %s-wide lane %s on buffer %s, which the source declares as %s whose "
                          "lane is %s: the width and the declaration disagree"
                          % (width, what, getattr(buf, "name", "?"), declared, lane_elem))


# THREADGROUP MEMORY: an addrspace(3) array global, a GEP into it, and a barrier.
#
# WHAT THE BACKEND ALREADY HAS, and what it rests on. store_tg/load_tg are a scratchpad indexed by
# a register, measured at 32 lanes - every lane writes its own slot and any lane can read another's
# after a barrier (ledger/g17-threadgroup-exchange-at-32-lanes.toml) - and the barrier's byte1 is
# its SCOPE, with the device scope measured from a kernel that asks for one. So this front end adds
# a mapping and no new instruction.
#
# THE BARRIER FLAGS ARE READ, NOT GUESSED. Apple's own compiler fixes their meaning: compiling
# `threadgroup_barrier(mem_flags::mem_device)` yields `air.wg.barrier(i32 1, i32 1)` and
# `mem_threadgroup` yields `(i32 2, i32 1)`, while `simdgroup_barrier(mem_flags::mem_threadgroup)`
# yields `air.simdgroup.barrier(i32 2, i32 4)` - twelve instances across the retained 197 agree,
# and the source text of each is retained beside it. So the first operand is the MEMORY flag and
# the second the execution scope, and only the two combinations whose backend scope is measured are
# mapped. Everything else - a simdgroup execution scope, an unlisted flag pair - refuses by name,
# because BARRIER_SCOPE has no simdgroup entry and inventing one is inventing a scope.
_NO_THREADGROUP_MEMORY = False
# THE ATOMIC OPERATIONS THIS FRONT END LOWERS, and the flag triple it accepts.
#
# Apple spells a device read-modify-write `air.atomic.global.<op>.<signedness>.i32` and passes
# three trailing flags. The two operations here are the ones MEASURED on the per-lane device form:
# compiling `atomic_fetch_add_explicit` and `atomic_fetch_and_explicit` gives op10090/10 in both
# cases with only the operation field moving, 0 for add and 1 for `and`
# (results/g17-atomic-assessment-v1). Every other member of the fetch family is left out rather
# than assumed from the field table: the field is measured, but which operations Apple actually
# emits at this form is a separate question and only these two were asked.
#
# THE FLAGS ARE NOT INTERPRETED, THEY ARE MATCHED. (0, 2, true) is what Apple emits for
# `memory_order_relaxed` at device scope - verified by compiling that source and reading its own
# AIR, where the triple is identical to the one both retained sources carry. No other ordering or
# scope has been measured, so any other triple refuses by name rather than being treated as
# equivalent. Guessing a memory scope is the one thing an atomic must never do.
# THE STRUCT'S STORAGE gives the WIDTH and the kind; the METADATA gives the signedness. Both are
# read and they must agree.
#
# Root caught this: `atm-i-and-dev-used-1-r1` and `probe-aq-0-before` both have `type { i32 }`
# storage, and their `air.struct_type_info` nodes say `!"int"` and `!"uint"` respectively. Mapping
# from the storage alone collapsed a signed atomic into the unsigned spelling - losing a fact the
# source states explicitly, which is the whole failure mode the declaration-carrying design exists
# to prevent. So the metadata name decides the spelling, the structural pointee is checked against
# it, and a contradiction refuses rather than one silently winning.
_ATOMIC_STORAGE = {"i32": {"int", "uint"}, "float": {"float"}}
_ATOMIC_SPELLINGS = {"int": "atomic_int", "uint": "atomic_uint", "float": "atomic_float"}


def _atomic_declarations(air):
    """{argument position: the atomic spelling} for each `metal::_atomic` parameter.

    THE INNER TYPE IS NOT IN `air.arg_type_name`, which spells every atomic the same. Two sources
    of truth are combined:

      * the STRUCT the signature points at - `%"struct.metal::_atomic" = type { i32 }` - which
        gives the storage, hence the width;
      * that argument's `air.struct_type_info` node - `!{i32 0, i32 4, i32 0, !"int", !"__s"}` -
        whose field name gives the declared type INCLUDING signedness.

    AND IT IS RESOLVED PER ARGUMENT, not per program. Three sources in the frozen population
    declare TWO different atomic structs at once (`%"struct.metal::_atomic"` and
    `%"struct.metal::_atomic.0"`), so a single per-program lookup would label one of them with the
    other's field. The signature's parameter order is the metadata's argument order.
    """
    fields = dict(re.findall(r'%("struct\.metal::_atomic[^"]*") = type \{ (\w+) \}', air))
    m = re.search(r"define [^@]*@[\w.$]+\((.*?)\)\s*(?:local_unnamed_addr|#|\{)", air, re.S)
    if not m:
        return {}
    storage = {}
    for i, arg in enumerate(re.split(r",(?![^<]*>)", m.group(1))):
        hit = re.search(r'%("struct\.metal::_atomic[^"]*")\s+addrspace\(\d+\)\*', arg)
        if hit is not None:
            storage[i] = fields.get(hit.group(1))
    if not storage:
        return {}
    md = _metadata(air)
    stated = {}
    for i, node in enumerate(_argument_nodes(air, md)):
        items = _md_list(md, node) if node is not None else []
        for k, it in enumerate(items):
            if it == '!"air.struct_type_info"':
                ref = items[k + 1].strip()
                if ref.startswith("!") and ref[1:].isdigit():
                    info = _md_list(md, int(ref[1:]))
                    names = [x.strip().lstrip("!").strip('"') for x in info
                             if x.strip().startswith('!"')]
                    # !{i32 0, i32 4, i32 0, !"int", !"__s"} - the second integer is the field's
                    # SIZE, and it is read and checked rather than trusted to agree: the node
                    # states the width independently of the struct's storage, so the two are two
                    # facts about the same field and a disagreement means one reader is wrong.
                    sizes = [int(x.split()[-1]) for x in info
                             if re.match(r"^\s*i32 -?\d+\s*$", x)]
                    if names:
                        stated[i] = (names[0], sizes[1] if len(sizes) > 1 else None)
    out = {}
    for i, store in storage.items():
        entry = stated.get(i)
        if entry is None:
            raise Unsupported(
                "the atomic at parameter %d declares no air.struct_type_info, so its field's "
                "declared type - and its signedness - cannot be preserved" % i)
        name, width = entry
        allowed = _ATOMIC_STORAGE.get(store)
        if allowed is None or name not in allowed:
            raise Unsupported(
                "the atomic at parameter %d has struct storage %r and metadata field type %r, "
                "which contradict each other; neither is preferred silently" % (i, store, name))
        spelling = _ATOMIC_SPELLINGS[name]
        declared = _ir().ELEM_BYTES[spelling]
        if width is not None and width != declared:
            raise Unsupported(
                "the atomic at parameter %d states field size %d in its air.struct_type_info and "
                "%d in the element this front end would carry (%s); the width is stated twice and "
                "the two do not agree" % (i, width, declared, spelling))
        out[i] = spelling
    return out


# THE ESTABLISHED ATOMIC ADDRESS FORM, as an ALLOWLIST rather than a property test.
#
# This was `_lane_varying(index)`, and root broke it twice over. That predicate asks whether the
# index DEPENDS on a per-lane builtin, which is provenance, not the property that matters: insert
# `%u = and i32 %tid, 0` or `%u = sub i32 %tid, %tid` and the index still depends on the builtin
# while being identically zero for every lane. Both mutations compiled. It is the same mistake as
# the bounds check that asked whether a builtin was PRESENT instead of what the index could be.
#
# Proving non-cancellation in general needs a symbolic solver, and root's instruction is explicit
# that this is not the place for one. So the address is required to MATCH THE ESTABLISHED FORM
# instead: the measured probe indexes directly by a per-lane position builtin, that is the only
# address form any measurement here covers, and anything computed - however it is computed -
# refuses by name. An allowlist cannot be defeated by an expression nobody anticipated, which is
# exactly the property the two previous versions of this check lacked.
_ATOMIC_ADDRESS_BUILTINS = ("thread_position_in_grid", "thread_position_in_threadgroup")


def _is_established_atomic_index(index):
    """True only for an index that IS a per-lane position builtin, with nothing applied to it."""
    ir_mod = _ir()
    if not isinstance(index, ir_mod.Value) or index.op is None:
        return False
    if index.op.kind != "builtin":
        return False
    return index.op.attrs.get("which") in _ATOMIC_ADDRESS_BUILTINS


def _describe_atomic_index(index):
    """Why an index is not the established form, in the terms the refusal needs."""
    ir_mod = _ir()
    if isinstance(index, ir_mod.Imm):
        return "a constant index"
    if not isinstance(index, ir_mod.Value) or index.op is None:
        return "an index this front end cannot attribute"
    if index.op.kind != "builtin":
        return "a COMPUTED index (%s), which the measurement does not cover however it is computed"\
               % index.op.kind
    return "the builtin %r, which is not one that varies per lane" % index.op.attrs.get("which")


_ATOMIC_OPERATIONS = {"add", "and"}
_ATOMIC_FLAGS = (0, 2, "true")
# THE CAPABILITY-OFF ARM FOR THIS BATCH. True restores the state before it: `metal::_atomic`
# refused at the DECLARATION, which is how all three of these sources refused - including the one
# that contains no atomic operation at all. Gating the declaration rather than only the call is
# what makes the off arm reproduce that state for all three rather than for two of them.
_NO_ATOMIC = False

_BARRIER_FLAGS = {("air.wg.barrier", 1, 1): "device",
                  ("air.wg.barrier", 2, 1): "threadgroup"}
# The lane width of a threadgroup array element. The scratchpad is declared in 32-bit WORDS, so a
# narrower element would need a packing rule this front end has not established - it refuses.
_TG_WORD_ELEMENTS = {"i32": 4, "float": 4}
# The domain the threadgroup exchange is MEASURED on: every lane writes its own slot and any lane
# reads another's after a barrier, at 32 lanes (ledger/g17-threadgroup-exchange-at-32-lanes.toml).
# This is a statement about evidence, not about any source.
_TG_MEASURED_LAUNCH = (32, 1, 1)
_TG_THREAD_BUILTINS = ("thread_position_in_threadgroup", "thread_position_in_grid",
                       "thread_index_in_threadgroup")


_NO_LOCAL_INDEX = False
_LOCAL_INDEX = "thread_index_in_threadgroup"
# THE SAME NAME, FROZEN SEPARATELY, and the separation is the point. The guard further down
# refuses if this argument kind ever reaches the generic single-register builtin read. Keyed on
# _LOCAL_INDEX it would compare against the very variable that decides the dispatch, so any edit
# that made the dispatcher miss would make the guard miss too - a check that cannot fail. Keyed on
# the AIR name it fires whenever the kind arrives, whatever the dispatcher thinks.
_LOCAL_INDEX_AIR_NAME = "thread_index_in_threadgroup"
_LOCAL_POSITION = "thread_position_in_threadgroup"
# The three local coordinate reads, MEASURED against Apple's compiler on 2026-09-15 rather than
# read off the SR table. Three one-variable sources - `lp.x`, `lp.y`, `lp.z` of a `uint3
# [[thread_position_in_threadgroup]]`, everything else held fixed - compiled through xcrun metal
# (compile-only; no command buffer, nothing dispatched) and decoded:
#
#     Apple      lp.x -> reg:26      lp.y -> reg:27      lp.z -> reg:28
#     this front end, same three coordinates, decoded from its own bytes: reg:26, reg:27, reg:28
#
# So all three components are Apple's own registers, which is what makes the multidimensional
# formula below a composition of established reads rather than an extrapolation. Worth naming
# because the analogous prediction for `threads_per_threadgroup` was WRONG - its decoded operands
# step by two (55/57/59), not one - so consecutiveness here is a measurement, not a pattern.
# reg:27 is additionally a register isa/g17-special-registers.toml lists as UNREAD in the corpus
# (SR_LOCAL_Y); the corpus not containing a read is not the register being unreadable, and an
# authored one-variable source is what separates those.
_LOCAL_COORD_REGISTERS = {"x": 26, "y": 27, "z": 28}


def _linear_local_index(b, launch, ir):
    """thread_index_in_threadgroup, COMPUTED from the local coordinates and the EXACT launch.

    Metal defines this builtin as the linear position of the thread within its threadgroup:

        index = x + size_x * (y + size_y * z)

    THIS IS NOT LOWERED AS A REGISTER READ, and the reason is worth stating because the ISA offers
    a tempting one. `isa/g17-special-registers.toml` has a register literally named SR_LIN_ID, and
    nothing measured connects it to this builtin - no authored source has read it and no corpus
    kernel attributes it. Mapping a builtin onto a register because their NAMES agree is the shape
    of error this project keeps retracting, so the name is not used and the index is built from the
    three coordinate reads that ARE measured (see _LOCAL_COORD_REGISTERS above).

    THE SHAPE HAS TO BE EXACT, because the formula is a function OF it. `size_x` is a multiplier in
    the emitted code, so a program compiled for one shape computes a different function at another
    - which is why an unresolved shape REFUSES rather than defaulting. The ABI's required_size is
    already a size the runtime must launch with exactly, so the contract that decides this is the
    launch contract, not the array extent.

    WHAT IS DROPPED AND WHY IT IS NOT AN ASSUMPTION. On a shape whose size_z is 1, every thread has
    z == 0, so the z term is identically zero and is not emitted; likewise y on a 1-D shape. That
    is a consequence of the resolved shape, not a guess about the launch - and on a shape where
    they are NOT 1 the terms are emitted, which is what the 2-D and 3-D controls check. A 1-D shape
    reduces to the x read alone with no multiply and no add.
    """
    if launch is None:
        raise Unsupported(
            "thread_index_in_threadgroup with no resolved threadgroup shape: the linear index is a "
            "function OF the shape (size_x multiplies the y coordinate), so a program compiled "
            "without one would compute a different value at any other launch. Pass the exact "
            "launch contract rather than leaving it to be derived")
    if len(launch) != 3 or any(not isinstance(n, int) or n < 1 for n in launch):
        raise Unsupported(
            "thread_index_in_threadgroup on the threadgroup shape %r: each dimension must be a "
            "resolved positive thread count for the linear formula to be exact" % (launch,))
    size_x, size_y, size_z = launch
    x = b.builtin(_LOCAL_POSITION, name="lx", axis="x")
    if size_y == 1 and size_z == 1:
        # ONE DIMENSION: the linear index IS the x coordinate, with nothing to add.
        return x
    inner = b.builtin(_LOCAL_POSITION, name="ly", axis="y")
    if size_z != 1:
        z = b.builtin(_LOCAL_POSITION, name="lz", axis="z")
        inner = b.add(inner, b.mul(z, b.const(size_y, name="lsy"), name="lzy"), name="lyz")
    return b.add(x, b.mul(inner, b.const(size_x, name="lsx"), name="lsxi"), name="lidx")


def _tg_indexed_per_thread(text):
    """Does the kernel index threadgroup memory with a per-thread builtin? Then the launch is bounded."""
    return any(("air." + name) in text for name in _TG_THREAD_BUILTINS)


def _threadgroup_arrays(text):
    """{global name: (extent, element, alignment)} for every addrspace(3) array the module declares."""
    found = {}
    for line in text.splitlines():
        m = re.match(r"(@[\w.$]+) = [^=]*addrspace\(3\) global \[(\d+) x (\w+)\] \w+, align (\d+)",
                     line.strip())
        if m:
            found[m.group(1)] = (int(m.group(2)), m.group(3), int(m.group(4)))
    return found


def _threadgroup_words(extent, element, name):
    """The scratchpad's size in 32-bit words, refusing an element whose packing is unestablished."""
    width = _TG_WORD_ELEMENTS.get(element)
    if width is None:
        raise Unsupported("threadgroup array %s of %s: the scratchpad is declared in 32-bit words "
                          "and this front end has established no packing for a %s element, so its "
                          "size refuses rather than being rounded" % (name, element, element))
    return extent


# INDEX BOUNDS ARE PROVED, NOT INFERRED FROM THE PRESENCE OF A BUILTIN.
#
# My first threadgroup batch checked "is the index derived from a per-thread builtin, and is the
# thread count within the array" - and root broke it in one line: take the retained tgm-u64, whose
# index is `and i32 %14, 63`, and write `add i32 %14, 64` instead. The thread count is still 32 and
# the extent still 64, so that check passed, and the program compiled and read past the array. The
# check was about the builtin's PRESENCE; the index is an EXPRESSION.
#
# So the bound is now computed over the expression, conservatively, and an access whose bound
# cannot be proved below the extent refuses. A mask is what makes the retained sources provable -
# `and x, 255` is at most 255 whatever x is - which is why every real source passes and the
# mutation does not. An index derived from memory or from a kernel argument has no bound at all.
_BOUND_DEPTH = 16
# The index is an unsigned 32-bit value. Reading AIR's literals as Python's signed integers
# made `-1` a small number instead of 0xFFFFFFFF, which is what let root's `add %x, -1`
# mutation prove a bound of 62 for an index that is UINT_MAX at lane 0.
_U32 = 0xFFFFFFFF


def _index_bounds(lines):
    """{AIR name: rhs} for bound computation - the definitions, in one pass."""
    defs = {}
    for line in lines:
        m = _ASSIGN.match(line)
        if m:
            defs[m.group(1)] = m.group(2)
    return defs


def _typed_binary(text):
    """(head, width, (left, right)) for a binary integer instruction, or None.

    METADATA IS NOT AN OPERAND. This read used `re.findall` over the whole right-hand side, so
    `and i32 %x, 63, !annotation !0` handed the bound rule a third "operand" of 0 taken from the
    metadata ID - and because `and` takes the MINIMUM, the bound became 0. Root found it by calling
    the function directly. A spuriously SMALL bound is unsound in the admitting direction: with
    `!tbaa !24` attached, `and %x, 4095` would report 24 and pass in a 64-element array. So the two
    typed operands are parsed exactly and anything after them is discarded, and an instruction that
    does not match this shape yields no bound at all rather than a guess.
    """
    body = re.split(r",\s*!|\s+!", text)[0]
    m = re.match(r"^(\w+)\s+i(\d+)\s+(\S+),\s*(\S+)\s*$", body.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2)), (m.group(3), m.group(4))


def _bound_of(name, defs, threads, kinds=None, depth=0):
    """A conservative upper bound for an AIR value, or None when nothing bounds it.

    `threads` bounds a per-threadgroup position because the launch domain is stated; a grid
    position is NOT bounded by it, so it comes back None and only a mask can save such an index.

    EVERY VALUE HERE IS AN UNSIGNED 32-BIT QUANTITY, and reading the literals as Python's signed
    integers was unsound in three separate places. Root's second counterexample was `add i32 %x,
    -1` over a masked index: signed arithmetic made the bound 63 + (-1) = 62 and admitted it,
    while the actual lane 0 index is 0 + 0xFFFFFFFF = UINT_MAX. The same slip made `and %x, -1` -
    which is the IDENTITY, not a mask - report a bound of -1 that compared below every extent, and
    `urem %x, -1` report -2. So a literal is read as its unsigned 32-bit value, and any
    composition whose exact result leaves the 32-bit range refuses: unsigned wrap means the
    computed value is not bounded by the operands at all, and a wrapped index is precisely the
    out-of-bounds read this proof exists to stop.
    """
    if depth > _BOUND_DEPTH:
        return None
    kinds = kinds or {}
    text = defs.get(name)
    if text is None:
        return None
    if re.match(r"^\d+$", text.strip()):
        return int(text.strip())
    parts = text.split()
    head = parts[0]
    def operand_bound(token):
        token = token.rstrip(",")
        if re.match(r"^-?\d+$", token):
            return int(token) & _U32                  # -1 is 0xFFFFFFFF, not a small number
        if token.startswith("%"):
            return _bound_of(token[1:], defs, threads, kinds, depth + 1)
        return None
    if head == "extractelement":
        # WHICH builtin this reads is in the kernel's SIGNATURE, not in this line: the line says
        # `extractelement <3 x i32> %4, i64 0` and %4's attribute is what names it. Reading the
        # text alone matched nothing and made every position index unbounded, so the argument map
        # is consulted. Only a position WITHIN THE THREADGROUP is bounded by the stated launch; a
        # grid position is not bounded by it at all, and only a mask can prove such an index.
        source = re.search(r"%([\w.]+)", text)
        kind = kinds.get(source.group(1)) if source else None
        if kind in ("thread_position_in_threadgroup", "thread_index_in_threadgroup"):
            return max(threads - 1, 0) if threads else None
        return None
    if head in ("zext", "trunc", "bitcast"):
        m = re.search(r"%([\w.]+)", text)
        if not m:
            return None
        inner = _bound_of(m.group(1), defs, threads, kinds, depth + 1)
        if inner is None:
            return None
        if head == "trunc":
            # truncation keeps the low bits, so the result is at most the narrower width's max
            w = re.search(r"to i(\d+)", text)
            if w:
                return min(inner, (1 << int(w.group(1))) - 1)
        return inner
    if head not in ("and", "urem", "add", "or", "mul", "shl"):
        return None
    parsed = _typed_binary(text)
    if parsed is None:
        return None
    _, width, (left_token, right_token) = parsed
    if width > 32:
        return None                                   # wider than the index space this proves
    ceiling = (1 << width) - 1
    left, right = operand_bound(left_token), operand_bound(right_token)
    if head == "and":
        masks = [c for c in (left, right) if c is not None]
        return min(masks) if masks else None          # x & m <= m for unsigned m
    if head == "urem":
        # the remainder is bounded by the DIVISOR alone, so an unbounded left operand is fine here
        if not right:                                 # a zero divisor is undefined, not bounded
            return None
        return right - 1
    if left is None or right is None:
        return None
    if head == "shl":
        if right >= width:                            # an over-wide shift is undefined
            return None
        exact = left << right
    elif head == "mul":
        exact = left * right
    else:
        exact = left + right                          # x | y <= x + y for unsigned operands
    return None if exact > ceiling else exact         # wrap leaves the operands proving nothing


def _cc():
    """The backend module, for facts the front end must not duplicate - e.g. which compare
    relations have been RECOVERED. Importing it here keeps one statement of that set."""
    from agxforge.g17 import cc as _module
    return _module


def _tg_index(text, global_name, extent, b, operand, defs, threads, kinds):
    """The lane index of a GEP into the threadgroup array, bounds-checked where it is constant.

    Both spellings appear: a separate instruction `getelementptr inbounds [N x T], [N x T]
    addrspace(3)* @g, i64 0, i64 %i` and the inline constant form inside a load or store operand,
    `... addrspace(3)* getelementptr inbounds ([N x T], [N x T] addrspace(3)* @g, i64 0, i64 7)`.
    The first index must be zero - it selects the array itself - and a non-zero one would address
    past the object, so it refuses.
    """
    m = re.search(re.escape(global_name) + r",\s*i\d+ (\S+?),\s*i\d+ (%?[\w.]+)", text)
    if not m:
        raise Unsupported("a threadgroup getelementptr this front end cannot read: %r" % text[:90])
    first, second = m.group(1).rstrip(","), m.group(2).rstrip(")")
    if first not in ("0", "i64 0"):
        raise Unsupported("a threadgroup getelementptr whose first index is %s rather than 0: that "
                          "addresses past the array object" % first)
    value = operand(second)
    if isinstance(value, _ir().Imm):
        if not 0 <= value.v < extent:
            raise Unsupported("a threadgroup access at constant index %d of a %d-element array: "
                              "out of bounds" % (value.v, extent))
        return b.const(value.v, name="tg%d" % value.v)
    bound = _bound_of(second.lstrip("%"), defs, threads, kinds)
    if bound is None:
        raise Unsupported(
            "a threadgroup access at %s, whose value nothing bounds: the index is an expression "
            "and this front end proves a bound over it rather than trusting that it came from a "
            "thread builtin. A mask (`and x, %d`) proves one; a value from memory, from a kernel "
            "argument or from unbounded arithmetic does not, and an unproved index refuses"
            % (second, extent - 1))
    if bound >= extent:
        raise Unsupported(
            "a threadgroup access at %s, whose proved upper bound is %d in a %d-element array: "
            "that reads or writes past the array, so it refuses" % (second, bound, extent))
    return value


# FORWARD CONTROL FLOW: the single-level if, and nothing beyond what is measured.
#
# WHAT THE BACKEND HAS. A conditional region is a mask push (op582), the guarded instructions, and
# a restore (op577), with the compare's immediate form op10369 - and a forward branch plus a phi
# compiles today (I measured that before writing this). What is MEASURED is narrower than what
# compiles: ledger/g17-exec-mask-reconvergence.toml says in terms "Only the single-level,
# single-conditional case was measured. Nested conditionals and loops containing conditionals were
# not tried", and ir.MAX_PRED_DEPTH = 4 is the deepest nesting APPLE emits rather than a depth this
# tree has run.
#
# SO THIS ADMITS ONE SHAPE: a single two-way branch whose arms rejoin at one block that ends the
# kernel. Anything else refuses BY NAME - a back edge (a loop, which this assignment excludes), a
# phi, a second conditional, a join that is not the exit. That is deliberately less than "generic
# acyclic": the sources in scope are one such if, and shipping untested lowerings for shapes no
# source exercises would be the opposite of measuring.
_NO_FORWARD_CONTROL_FLOW = False
_LABEL = re.compile(r"^(\d+):(?:\s*;\s*preds\s*=\s*(.*))?$")


def _split_blocks(lines):
    """[(label, preds, [lines])] in textual order; the entry block's label is None."""
    blocks, label, preds, current = [], None, [], []
    for line in lines:
        m = _LABEL.match(line)
        if m:
            blocks.append((label, preds, current))
            label = m.group(1)
            preds = [p.strip().lstrip("%") for p in (m.group(2) or "").split(",") if p.strip()]
            current = []
        else:
            current.append(line)
    blocks.append((label, preds, current))
    return blocks


# GENERIC FINITE CONSTANT-TRIP LOOPS, PROVED THEN UNROLLED.
#
# Two retained sources are counted loops over a threadgroup array: an induction phi from 0 stepping
# by 1, an accumulator phi threading a sequential chain, and `icmp eq %step, 16` on the back edge.
# Root's feasibility experiment showed that their HAND-unrolled AIR compiles through the existing
# path - so what is missing is not a lowering but the PROOF plus the expansion, and that is what
# this adds. The expansion is AIR to AIR, so every other construct in the program keeps the exact
# path it already had, and nothing about the emitted code for the straight-line parts is new.
#
# WHAT IS PROVED BEFORE ANYTHING IS UNROLLED, each refusing BY NAME when it does not hold:
#   * exactly one self-looping block, entered from exactly one outside edge;
#   * a terminator that is a two-way branch on an `icmp eq`/`ne` against a CONSTANT;
#   * an induction phi whose entry value is a constant and whose latch value is that compare's
#     operand, defined as `add <induction>, K` for a constant non-zero K;
#   * a trip count that is exact: (N - start) divisible by K, positive, and within the cap below;
#   * every phi's incoming edges being exactly the entry edge and the latch, read explicitly
#     rather than by position, so a malformed or three-way phi refuses instead of being guessed.
#
# THE ORDERED COMPARES ARE REFUSED, not approximated. `icmp ult %i, N` is a perfectly ordinary
# counted loop and its trip formula differs; no retained source uses one, so implementing it would
# be an unmeasured path and it refuses by name instead.
#
# THE CAP IS A REFUSAL, NOT A CLAMP. A loop past it refuses rather than being partly unrolled,
# because a partial unroll is a different program and this construct has no back edge to leave.
_NO_COUNTED_LOOPS = False
_MAX_UNROLL_TRIPS = 64
_PHI = re.compile(r"^\s*%([\w.]+) = phi (\S+) \[ ([^,]+), %([\w.]+) \], \[ ([^,]+), %([\w.]+) \]\s*$")


def _rename_values(text, mapping):
    """Substitute whole SSA tokens in one pass, so no replacement feeds another."""
    if not mapping:
        return text
    pattern = re.compile(r"%(" + "|".join(re.escape(k) for k in sorted(mapping, key=len,
                                                                      reverse=True)) + r")\b")
    return pattern.sub(lambda m: mapping[m.group(1)], text)


def _defined_name(line):
    m = _ASSIGN.match(line)
    return m.group(1) if m else None


def _counted_loop_plan(lines):
    """A proof for the one self-looping block, or None when the function has no back edge.

    Returns (entry_lines, body_lines_to_repeat, exit_lines, trips, induction, step_name,
    step_expr_name, accumulators, literal_of) - everything the expansion needs and nothing it
    should have to re-derive.
    """
    blocks = _split_blocks(lines)
    looping = [i for i, (label, preds, _) in enumerate(blocks) if label and label in preds]
    if not looping:
        return None
    if _NO_COUNTED_LOOPS:
        raise Unsupported(
            "a kernel with a back edge: the counted-loop capability is switched off here, which "
            "is the state the gain is measured against")
    if len(looping) > 1:
        raise Unsupported(
            "%d self-looping blocks: this construct proves and unrolls ONE counted loop, and "
            "nested or sibling loops refuse rather than being unrolled one at a time with the "
            "others left behind" % len(looping))
    index = looping[0]
    label, preds, body = blocks[index]
    outside = [p for p in preds if p != label]
    if len(outside) != 1:
        raise Unsupported(
            "a loop block entered from %d edges besides its own latch: the induction's start value "
            "is read from THE entry edge, and with more than one there is no single start"
            % len(outside))
    entry_label = outside[0]
    term = body[-1].strip() if body else ""
    m = re.match(r"br i1 %([\w.]+), label %([\w.]+), label %([\w.]+)", term)
    if not m:
        raise Unsupported("a loop whose terminator this front end cannot read: %r" % term)
    cond, if_true, if_false = m.group(1), m.group(2), m.group(3)
    if label not in (if_true, if_false):
        raise Unsupported("a loop whose back edge is not one of its terminator's targets")
    exit_label = if_false if if_true == label else if_true
    exits_when_true = if_true == exit_label
    compare = next((l for l in body if _defined_name(l) == cond), None)
    if compare is None:
        raise Unsupported("a loop whose exit condition %%%s is not defined in the loop" % cond)
    cm = re.match(r"\s*%[\w.]+ = icmp (\w+) (\S+) %([\w.]+), (-?\d+)\s*$", compare)
    if cm is None:
        raise Unsupported(
            "a loop exit condition this front end cannot prove a trip count from (%s): it reads "
            "an `icmp eq`/`ne` of one value against a CONSTANT" % compare.strip())
    relation, stepped, bound = cm.group(1), cm.group(3), int(cm.group(4))
    compare_type = cm.group(2)
    if not ((relation == "eq" and exits_when_true) or (relation == "ne" and not exits_when_true)):
        raise Unsupported(
            "a loop that exits on %r with the branch %s: the proved shape leaves when the stepped "
            "induction EQUALS its bound, and an ordered or inverted compare has a different trip "
            "formula that no retained source uses" % (relation, "true" if exits_when_true else "false"))
    phis, induction, accumulators = [], None, []
    induction_type = step_type = None
    for line in body:
        if " = phi " not in line:
            continue
        pm = _PHI.match(line)
        if pm is None:
            raise Unsupported(
                "a phi this front end will not read (%s): the proved shape is exactly two incoming "
                "edges, the entry and the latch, and a malformed or wider phi refuses rather than "
                "having its edges guessed by position" % line.strip())
        name, phi_type, v0, p0, v1, p1 = (pm.group(1), pm.group(2), pm.group(3).strip(),
                                          pm.group(4), pm.group(5).strip(), pm.group(6))
        if {p0, p1} != {entry_label, label}:
            raise Unsupported(
                "a phi whose incoming blocks are %%%s and %%%s, not the entry %%%s and the latch "
                "%%%s" % (p0, p1, entry_label, label))
        enter, latch = (v0, v1) if p0 == entry_label else (v1, v0)
        phis.append((name, enter, latch))
        if latch == "%" + stepped:
            induction = (name, enter, latch)
            induction_type = phi_type
        else:
            accumulators.append((name, enter, latch))
    if induction is None:
        raise Unsupported(
            "no induction phi whose latch value is the compared %%%s: without it the start value "
            "and the step are not established" % stepped)
    step_line = next((l for l in body if _defined_name(l) == stepped), None)
    sm = re.match(r"\s*%[\w.]+ = add(?: nuw| nsw)* (\S+) %([\w.]+), (-?\d+)\s*$",
                  step_line or "")
    if sm is None or sm.group(2) != induction[0]:
        raise Unsupported(
            "a step this front end cannot prove (%s): the proved shape is `add <induction>, K` "
            "with a constant K" % (step_line or "missing").strip())
    step, step_type = int(sm.group(3)), sm.group(1)
    # THE WIDTH IS PART OF THE PROOF, because the trip count is computed in Python integers and
    # the machine's is not. Root's point: an i8 induction from 0 by 1 toward 300 has no i8 value
    # equal to 300 at all - the real sequence WRAPS at 256 and exits early - while the linear
    # arithmetic below would happily report 300 trips and unroll a program the source never
    # describes. So the declared width is read from the phi, the step and the compare (which must
    # AGREE), and the endpoints must be representable in it; between two representable endpoints a
    # monotone sequence cannot wrap, which is what makes the linear computation exact.
    widths = {induction_type, step_type, compare_type}
    if len(widths) != 1 or not re.match(r"^i\d+$", induction_type):
        raise Unsupported(
            "a loop whose induction, step and bound are not one integer width (%s): the trip count "
            "is proved in that width, and disagreeing widths would prove it in none of them"
            % ", ".join(sorted(widths)))
    bits = int(induction_type[1:])
    if not re.match(r"^-?\d+$", induction[1]):
        raise Unsupported(
            "an induction whose start value is %r rather than a constant: a trip count cannot be "
            "proved from a value this front end does not know" % induction[1])
    start = int(induction[1])
    if step == 0:
        raise Unsupported("an induction with step 0: the loop does not terminate")
    span = bound - start
    if span == 0 or (span % step) != 0 or (span // step) <= 0:
        raise Unsupported(
            "a loop stepping from %d by %d toward %d: the stepped induction never equals the bound "
            "exactly, so the trip count is not finite and proved" % (start, step, bound))
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    for name, value in (("start", start), ("bound", bound)):
        if not low <= value <= high:
            raise Unsupported(
                "a loop whose %s is %d, outside the %s it is declared in (%d..%d): the sequence "
                "would wrap, and a trip count computed as ordinary arithmetic would describe a "
                "program the source does not" % (name, value, induction_type, low, high))
    trips = span // step
    if trips > _MAX_UNROLL_TRIPS:
        raise Unsupported(
            "a proved trip count of %d, past the %d this construct unrolls: a partial unroll would "
            "be a different program and this lowering has no back edge to leave, so it refuses "
            "rather than approximating" % (trips, _MAX_UNROLL_TRIPS))
    return dict(index=index, label=label, entry_label=entry_label, exit_label=exit_label,
                bits=bits, induction_type=induction_type,
                body=body, blocks=blocks, trips=trips, start=start, step=step, bound=bound,
                induction=induction, accumulators=accumulators, stepped=stepped, cond=cond)


def _unroll_counted_loop(lines):
    """The proved loop expanded into straight-line AIR, or `lines` unchanged when there is none.

    THE EXPANSION IS ORDER-PRESERVING BY CONSTRUCTION. Iteration t's lines are emitted in their
    original order before iteration t+1's, so every store, load and accumulator step keeps the
    sequence the source wrote - which is the whole of what a sequential accumulator chain needs.
    The induction becomes a LITERAL per iteration and each accumulator phi becomes the previous
    iteration's value, so no phi survives and nothing has to be rewritten later.

    The exit block's references to loop values are remapped to the LAST iteration, which is what
    the phi's exit value meant; the induction and its stepped form are remapped to the literals
    they hold on leaving.
    """
    plan = _counted_loop_plan(lines)
    if plan is None:
        return lines
    blocks, index = plan["blocks"], plan["index"]
    label, entry_label, exit_label = plan["label"], plan["entry_label"], plan["exit_label"]
    body, trips, start, step = plan["body"], plan["trips"], plan["start"], plan["step"]
    induction, accumulators = plan["induction"], plan["accumulators"]
    stepped, cond = plan["stepped"], plan["cond"]

    by_label = {lab: (lab, preds, ls) for lab, preds, ls in blocks}
    # THE ENTRY BLOCK CARRIES NO LABEL LINE. LLVM numbers it - the loop's pred list says `%5` -
    # but there is no `5:` in the text, so _split_blocks records it as None. Resolving it by
    # position rather than by name is safe precisely because it is the only unlabelled block.
    if entry_label not in by_label and None in by_label:
        by_label[entry_label] = by_label[None]
    if entry_label not in by_label or exit_label not in by_label:
        raise Unsupported("a loop whose entry %%%s or exit %%%s is not a block in this function"
                          % (entry_label, exit_label))
    if len(blocks) != 3:
        raise Unsupported(
            "a function with %d blocks around the loop: the proved shape is entry, loop and exit, "
            "and anything else refuses rather than being flattened by guess" % len(blocks))

    # WHICH LOOP-CONTROL VALUES CAN BE DROPPED, decided by counting real uses rather than assumed.
    exit_lines = by_label[exit_label][2]
    def used_by(name, exclude_defs):
        hits = 0
        for line in list(body) + list(exit_lines):
            if _defined_name(line) in exclude_defs:
                continue
            if " = phi " in line and _defined_name(line) == induction[0]:
                continue                        # the induction phi's own latch reference
            if re.search(r"%" + re.escape(name) + r"\b", line):
                hits += 1
        return hits
    drop = set()
    if used_by(cond, {cond}) == 1:               # only its own terminator
        drop.add(cond)
    if used_by(stepped, {stepped, cond}) == 0:
        drop.add(stepped)

    defined = [_defined_name(l) for l in body]
    defined = [d for d in defined if d]
    carried = {induction[0]} | {a[0] for a in accumulators}
    repeated = [l for l in body
                if l is not body[-1] and " = phi " not in l and _defined_name(l) not in drop]

    out = list(by_label[entry_label][2])
    entry_term = out[-1].strip() if out else ""
    if not re.match(r"br label %" + re.escape(label) + r"\s*$", entry_term):
        raise Unsupported("an entry block whose terminator is not the branch into the loop: %r"
                          % entry_term)
    out = out[:-1]

    previous = {a[0]: a[1] for a in accumulators}
    last, folded, phi_in_last = {}, {}, {}
    for t in range(trips):
        folded = {}
        value = start + t * step
        mapping = {induction[0]: str(value)}
        for name, enter, _latch in accumulators:
            mapping[name] = previous[name]
        for name in defined:
            if name in carried or name in drop:
                continue
            mapping[name] = "%%%s_u%d" % (name, t)
        if stepped in drop:
            mapping[stepped] = str(start + (t + 1) * step)
        for line in repeated:
            renamed = _rename_values(line, mapping)
            own = _defined_name(line)
            # A WIDTH CONVERSION OF A CONSTANT IS THE CONSTANT IN ITS SOURCE WIDTH.
            #
            # Substituting the induction turns `%31 = zext i32 %29 to i64` into
            # `zext i32 0 to i64`, and the conversion arm reads its operand as `(%\S+)` - a VALUE -
            # so a literal source refuses with a message about zext that says nothing about loops.
            # Root hit that too and folded it by hand in the feasibility probe.
            #
            # AND THE FOLD MUST MASK TO THE SOURCE WIDTH FIRST, which is the correction root's
            # second counterexample forced. An i8 induction starting at -1 substitutes the literal
            # `-1`, and `zext i8 -1 to i32` is 255, not -1: zero extension reads the source's BIT
            # PATTERN. I had passed the Python integer through unchanged, so a source computing
            # 10 + 255 = 265 expanded to 10 + (-1) = 9. My comment had said "non-negative" while
            # the code accepted negatives - a scope stated in prose and not enforced, for the
            # third time in this work. Masking by the source width handles both signs exactly.
            #
            # `sext` and `bitcast` are still NOT folded: sign extension of a negative literal and a
            # bitcast's reinterpretation are different questions, and they refuse downstream rather
            # than being folded on a guess.
            fold = re.match(r"(zext|trunc) i(\d+) (-?\d+) to i(\d+)\s*$",
                            renamed.split(" = ", 1)[1] if " = " in renamed else "")
            if own and fold is not None:
                literal = int(fold.group(3)) & ((1 << int(fold.group(2))) - 1)
                if fold.group(1) == "trunc":
                    literal &= (1 << int(fold.group(4))) - 1
                mapping[own] = str(literal)
                folded[own] = str(literal)
                continue
            if own and own not in carried and own not in drop:
                renamed = re.sub(r"^(\s*)%" + re.escape(own) + r"\b",
                                 lambda m: "%s%%%s_u%d" % (m.group(1), own, t), renamed)
            out.append(renamed)
        last = dict(mapping)
        # WHAT THE PHI ITSELF DENOTES IN THIS ITERATION, recorded before `previous` advances.
        # Root's counterexample: a loop whose EXIT STORES THE PHI rather than the latch. The phi's
        # value on the last iteration is the accumulator entering that iteration, and the latch's
        # is the result of it - one apart. I had mapped the phi to the final latch, so a source
        # storing the phi (11) got the latch (12). They are different values and the exit chooses
        # between them, so both are carried and the exit uses whichever name it actually names.
        for name, _enter, _latch in accumulators:
            phi_in_last[name] = mapping[name]
        for name, enter, latch in accumulators:
            previous[name] = last.get(latch.lstrip("%"), latch)

    tail_map = {}
    for name in defined:
        if name in carried or name in drop:
            continue
        tail_map[name] = folded.get(name, "%%%s_u%d" % (name, trips - 1))
    for name, enter, latch in accumulators:
        # the PHI's own name resolves to what it held in the last iteration; the latch's name is
        # left to the ordinary per-iteration rename above, so an exit that names the latch gets it
        tail_map[name] = phi_in_last[name]
    tail_map[induction[0]] = str(start + (trips - 1) * step)
    if stepped in drop:
        tail_map[stepped] = str(plan["bound"])
    for line in exit_lines:
        out.append(_rename_values(line, tail_map))
    return out


_PURE_HEADS = {"add", "sub", "mul", "and", "or", "xor", "shl", "lshr", "ashr", "fadd", "fsub", "fmul",
               "fneg", "zext", "sext", "trunc", "bitcast", "uitofp", "sitofp", "fptoui", "fptosi",
               "icmp", "select", "extractelement", "insertelement"}


def _if_convert_switch(lines):
    """A `switch` whose cases are PURE single blocks rejoining at one phi, as straight-line AIR.

    Apple lowers a small switch to compare-and-branch with exec-mask updates (the feature census:
    no jump table). When every arm only computes a value, the switch's meaning is exactly: compute
    each arm, then choose - so each phi becomes a chain of `icmp eq %x, K` and `select`, the default
    first. Nothing is speculated that could fault or differ: an arm holding a load, a store, a
    call, a division or anything else outside _PURE_HEADS refuses by name. A function with no
    switch is returned unchanged."""
    at = next((i for i, l in enumerate(lines) if l.startswith("switch ")), None)
    if at is None:
        return lines
    head = re.match(r"switch i32 (%[\w.]+), label %(\w+) \[", lines[at])
    if not head:
        raise Unsupported("switch %r: only an i32 switch is if-converted" % lines[at])
    x, default = head.group(1), head.group(2)
    cases, j = [], at + 1
    while not lines[j].startswith("]"):
        c = re.match(r"i32 (-?\d+), label %(\w+)$", lines[j])
        if not c:
            raise Unsupported("switch case %r" % lines[j])
        cases.append((int(c.group(1)), c.group(2)))
        j += 1
    entry, rest = lines[:at], lines[j + 1:]
    blocks, cur = {}, None
    order = []
    for l in rest:
        m = re.match(r"^(\w+):", l)
        if m:
            cur = m.group(1); blocks[cur] = []; order.append(cur)
            continue
        blocks[cur].append(l)
    arm_labels = [lab for _k, lab in cases] + [default]
    targets = {re.match(r"br label %(\w+)$", blocks[lab][-1]).group(1)
               for lab in arm_labels if lab in blocks and re.match(r"br label %(\w+)$", blocks[lab][-1])}
    if len(targets) != 1:
        raise Unsupported("a switch whose arms do not all branch to one join (%s)" % sorted(targets))
    join = targets.pop()
    arms = [lab for lab in dict.fromkeys(arm_labels) if lab != join]
    body_lines = []
    for lab in arms:
        blk = blocks.get(lab)
        if blk is None or not blk or blk[-1] != "br label %%%s" % join:
            raise Unsupported("switch arm %%%s is not one block ending in a branch to the join" % lab)
        for l in blk[:-1]:
            h = re.match(r"%[\w.]+ = (?:\w+ )*?(\w+)", l)
            op = h.group(1) if h else l.split()[0]
            if op not in _PURE_HEADS:
                raise Unsupported("switch arm %%%s holds %r: an arm is if-converted only when it "
                                  "computes a value and nothing else" % (lab, l))
            body_lines.append(l)
    jb = blocks[join]
    phis = [l for l in jb if " = phi " in l]
    out = entry + body_lines
    entry_label = None
    for n, l in enumerate(phis):
        pm = re.match(r"%([\w.]+) = phi (\S+) (.*)$", l)
        name, ty = pm.group(1), pm.group(2)
        incoming = {lab: v for v, lab in re.findall(r"\[ (\S+), %(\w+) \]", pm.group(3))}
        from_entry = [lab for lab in incoming if lab not in arms]
        if len(from_entry) > 1:
            raise Unsupported("phi %%%s has %d incoming edges from outside the arms" % (name, len(from_entry)))
        entry_label = from_entry[0] if from_entry else None
        def value_for(lab):
            if lab == join:
                if entry_label is None:
                    raise Unsupported("the default reaches the join with no incoming value in %%%s" % name)
                return incoming[entry_label]
            return incoming[lab]
        acc = value_for(default)
        for k, (key, lab) in enumerate(cases):
            if lab == default:
                continue
            out.append("%%%s_sw%dc%d = icmp eq i32 %s, %d" % (name, n, k, x, key))
            nxt = name if k == len(cases) - 1 else "%s_sw%ds%d" % (name, n, k)
            out.append("%%%s = select i1 %%%s_sw%dc%d, %s %s, %s %s" % (nxt, name, n, k, ty, value_for(lab), ty, acc))
            acc = "%" + nxt
        if not cases:
            raise Unsupported("a switch with no cases")
    out += [l for l in jb if " = phi " not in l]
    for lab in order:
        if lab not in arms and lab != join:
            raise Unsupported("a switch followed by more control flow (block %%%s)" % lab)
    return out


def _if_then_shape(blocks):
    """(cond, then label, join label) for the one measured shape, or a refusal naming why not.

    Accepts: entry ends `br i1 %c, label %T, label %J`; block T ends `br label %J`; block J is the
    last block and ends the kernel. Everything else refuses, including the else arm - no source in
    the population has one, so admitting it would be untested.
    """
    if len(blocks) != 3:
        raise Unsupported("a control-flow shape with %d basic blocks: the measured case is one "
                          "two-way branch rejoining at the exit (three blocks), and a deeper or "
                          "wider shape refuses rather than being lowered untested" % len(blocks))
    (_e, _ep, entry), (t_label, t_preds, then_body), (j_label, j_preds, join) = blocks
    for label, preds, _body in blocks[1:]:
        if label in preds:
            raise Unsupported("block %s is its own predecessor: that is a loop, and a back edge is "
                              "outside this front end's forward-only control flow" % label)
    cond = re.match(r"br i1 %([\w.]+), label %(\d+), label %(\d+)$", entry[-1] if entry else "")
    if not cond:
        raise Unsupported("an entry block ending %r: the measured shape ends in a two-way "
                          "conditional branch" % (entry[-1] if entry else "nothing"))
    if cond.group(2) != t_label or cond.group(3) != j_label:
        raise Unsupported("a two-way branch to %%%s and %%%s where the blocks that follow are %s "
                          "and %s: the guarded arm must be the next block and the other arm the "
                          "join" % (cond.group(2), cond.group(3), t_label, j_label))
    if then_body[-1:] != ["br label %%%s" % j_label]:
        raise Unsupported("a guarded block ending %r rather than branching to the join %%%s"
                          % (then_body[-1] if then_body else "nothing", j_label))
    if sorted(j_preds) != sorted({b for b in (t_label, blocks[0][0] or "entry")} | set(j_preds)):
        pass          # the preds comment is informational; the shape above is what decides
    if not any(line.startswith("ret") for line in join):
        raise Unsupported("a join block that does not end the kernel: the measured case rejoins at "
                          "the exit")
    return cond.group(1), t_label, j_label


def _agree(buf, width, what):
    """The access width and the buffer's DECLARED element must agree, or neither is trustworthy."""
    declared = getattr(buf, "elem", None)
    # A DECLARATION-ONLY ELEMENT REFUSES HERE, WHERE THE ACCESS IS DECIDED.
    #
    # Carrying a `ulong` or a `uint2` declaration is what lets an ordinary scalar body lower at all
    # (the program never touches that buffer), so the buffer now EXISTS in the IR and an access to
    # it is no longer stopped by the declaration being unreadable. ir.Block.add still refuses every
    # operation that names one, but that raises IRError, and an IRError reaching the census is
    # recorded as the compiler CRASHING - a refusal is what this is. So the front end refuses it
    # knowingly here, and the IR gate stays the backstop whose firing means a defect in this file
    # rather than an unsupported program.
    if (declared is not None and declared in _ir().BFLOAT_TRANSPORT and what == "load"
            and width == "half" and not _NO_BFLOAT_WIDEN):
        # THE ONE BFLOAT ACCESS THIS FRONT END ADMITS: the two-byte read. bfloat is two bytes on a
        # backend that loads two-byte scalars, so the load is the measured element-width form and
        # nothing new; what differs from `half` is the FORMAT, and the widening below is a shift
        # rather than a conversion instruction. A bfloat STORE still falls through to the refusal,
        # deliberately - narrowing needs a rounding contract whose signed-zero and
        # overflow-to-infinity domains are unadjudicated.
        return
    if declared is not None and declared not in ir_accessible():
        phys = _ir().DECL_PHYSICAL.get(declared, {})
        raise Unsupported("a %s on buffer %s, which the source declares as %s (%s bytes, %s-byte "
                          "alignment, %s): the declaration is carried so the rest of the program "
                          "can lower, but this backend loads and stores 2- and 4-byte scalars "
                          "only, so an access to it refuses rather than picking a width"
                          % (what, getattr(buf, "name", "?"), declared, phys.get("bytes"),
                             phys.get("alignment"), phys.get("signedness")))
    name = {16: "half", 32: "word"}.get(getattr(declared, "bits", None))
    if name is None:
        name = "half" if str(declared) in ("f16", "i16") else "word"
    if name != width:
        if (_NARROW_ELEMENTS_AS_WORD or _NO_HALF_ELEMENT_WIDTH) and width == "word" and name == "half":
            return          # the known defect above: a narrow element on the word form
        raise Unsupported("a %s-element %s on buffer %s, which the source declares as %s: the "
                          "access width and the declaration disagree and this front end does not "
                          "choose between them" % (width, what, getattr(buf, "name", "?"), declared))


CALLS = {"air.fma.f32": ("fma", 3), "air.convert.f.f32.u.i32": ("u32_to_f32", 1),
         # fma(half, half, half): op798, binary16 with one rounding, measured against Apple's own (MM 25.196)
         "air.fma.f16": ("fma16", 3)}

# FLOAT SELECT AND SATURATE, ADMITTED ONLY WITH THE FAST-MATH FLAGS, and the reason is the same one
# the composed half arithmetic already turns on.
#
# op9700 is Apple's own max(float, float) and min(float, float): one instruction, sources (a, b, a,
# b), field 7 for max and 3 for min, and this backend's two templates are cut from Apple's sw_f_max
# and sw_f_min. ir.ARITH states the limit exactly - "what the value on NaN is has not been
# measured; the operation on finite operands is the one Apple selects for max()/min()".
#
# WHAT IS MEASURED. The form is op9700 at FOURTEEN bytes, which is the one this backend emits and
# the one Apple's corpus carries 271 times. It has EXECUTED: the softmax program of
# results/g17-attention-softmax-execution-v1 contains 31 instances of it, that receipt reports
# status passed with gpu_dispatched true, and the bytes carrying them are the ones the receipt's own
# `programs/softmax/program.bin` digest names (789bc7b4e67f) - not another image's execution
# borrowed for this one. Apple selects op9700 at SIX bytes for this particular source; a different
# length is a different form, so nothing here claims to reproduce that choice.
#
# WHAT IS NOT. NaN, and the sign of a zero result. Both AIR calls carry `fast`, which is the
# source's own statement that neither is part of its contract - the same gate `_COMPOSED_HALF` uses.
# A strict `air.fmax.f32` has no such statement and is not in this table, so it still refuses.
_FAST_SELECT_CALLS = {"air.fast_fmax.f32": ("fmax", 2), "air.fast_fmin.f32": ("fmin", 2)}
_NO_FLOAT_SELECT = False          # capability-off arm: restores the refusal these two replace

# SATURATE IS THE CLAMP THE IR ALREADY BUILDS, not a new instruction.
#
# `saturate(x)` is clamp(x, 0, 1), and for finite x that is exactly fmin(fmax(x, 0.0f), 1.0f) - two
# instructions of the form above plus the two constants. The IR's own `erf` expansion is built from
# this shape (fmax(fmin(x*1e30, 1), -1)), so the operand shape is not new either.
#
# APPLE FUSES IT DIFFERENTLY AND THAT IS WORTH STATING: for this source Apple emits op3138 at eight
# bytes, one instruction per `saturate(f[i] * 0.5f)`, so the multiply and the clamp are one
# operation there. op3138 is not in this backend and has no instance in the 6,594-program corpus
# table [CORRECTED 2026-09-23: isa/g17-vendor-corpus-forms.json, decoding that same file rather than its
# spans index, counts 7 op3138/8 - the zero was the index's, not Apple's], so there is nothing here to reproduce; this composition is a different program that
# computes the same function on finite inputs.
#
# The other candidate was op1062, which this backend has as `fsat`. It is NOT used: it has zero
# corpus instances and its evidence is one dispatch over six finite values (2.5, -1, 0.7, -0.3, 3.9
# and 0.5), while the input here is an arbitrary loaded float. Two instructions of an executed form
# beat one instruction of a form measured at six points.
_NO_FLOAT_SATURATE = False
_NO_DOT_SQRT = False            # capability-off arm: air.dot.vNf32 and air.fast_sqrt.f32 refuse again
_SATURATE_CALLS = {"air.fast_saturate.f32"}
# THE CAPABILITY-OFF ARM for the sanitized ctz composition: True restores the refusal
# that named clz(0) as the missing fact, which is the state the gain is measured against.
_NO_CTZ = False
# THE CAPABILITY-OFF ARM for 32->16 truncation: True restores the refusal that said a
# 16-bit width change is a real operation with no lowering here, which it was.
_NO_TRUNC_I32_I16 = False
# THE CAPABILITY-OFF ARM for the sixteen-bit element STORE: True restores the refusal
# that covered both directions at once.
_NO_NARROW_ELEMENT_STORE = False

# THE MASK/RE-VIEW ROUTE IS GONE, SUPERSEDED BY A MEASUREMENT. An earlier assessment of mine
# showed that `a_word & 0xFFFF` would widen using only forms already lowered, needing just the
# permission to read a half-written word. Root measured op10283's zero-first endpoint instead, which
# gives a real half-to-word instruction, so that route is not pursued: it would have needed
# allocator support for a pure re-view AND a permission this evidence does not grant. The algebra
# was sound; the instruction is better.

# SIXTEEN-BIT WIDENING BY op10283 WITH A ZERO FIRST OPERAND - the route root measured.
#
# op10283 computes `a + (b & 0xFFFF)`: a 32-bit first source and a SIXTEEN-BIT view of the second.
# With a = 0 that is exactly zext16(b). Root dispatched the endpoint
# (results/g17-integer16-zero-first-v1) and the observations are independently reproduced here: both
# configurations, six queries, 2208 output words, and for b in
# [0, 1, 65535, 65536, 32768, 2147483648, 4294967295] the results are
# [0, 1, 65535, 0, 32768, 0, 65535] - exactly b & 0xFFFF, with the interleaved echo confirming the
# first operand was zero. root-review.json records the scope: constant-produced zero-first operands
# on the retained forms, and NO half-load readiness, opposite-half or arbitrary-modifier claim.
#
# SO THE ROUTE IS THE RETAINED FORM, NOT A NEW ENCODER. Every one of root's seven sites used the
# baseline tuple - raw 3702045a2900a312a8032100, [reg110, imm32, reg110, imm16, reg432, imm16] -
# with only the materialised constants differing, so this emits op10283 through the same machine
# route with the witness's own template and immediates. Nothing invents a modifier or a wait value.
#
# THE LOAD DEPENDENCY IS MEASURED ON THIS FORM. It used to be borrowed from a neighbouring one:
# while op10283's half-load readiness was unmeasured, this route made the loaded half ready with
# the executed wait-family's op10289/12 copy and widened the copy. Root then dispatched the direct
# pair - results/g17-integer16-half-load-v1, two 440-byte programs differing only in seven byte0[3]
# wait bits, seven op12646 half loads feeding op10283 as 0 + low16 over a 368-element ushort source.
# The waiting arm returns the source halfword on every query and the non-waiting arm returns wrong,
# query-to-query unstable values. So op10283 consumes the load directly with the wait bit set, the
# copy is gone, and the readiness claim now cites a receipt for the instruction that makes it.
#
# SIGNED EXTENSION is then the standard fold on top, in forms already lowered:
#     sext16(x) = (zext16(x) ^ 0x8000) - 0x8000
# 0x8000 exceeds the eight-bit immediate slot, so it is materialised and the xor is the reg-reg
# form op17771; the subtract is op11666. Exact for all 65536 values, tested.
_NO_U16_TO_U32 = False

# A CONVERSION THIS BACKEND HAS NO INSTRUCTION FOR, BUILT FROM TWO IT HAS - and admitted only
# because the composition is PROVABLY the correctly-rounded conversion, not because it looks close.
#
# THE CLAIM. For every u in [0, 2^32), rounding u to binary32 and then to binary16 gives exactly
# the binary16 value that correctly-rounded conversion of u gives. Both steps are round-to-nearest-
# even, which is what op11179 and op1016 do.
#
# THE PROOF, in two exhaustive cases:
#
#   u < 2^24   binary32 has a 24-bit significand, so u is EXACTLY representable: f32(u) = u, and
#              f16(f32(u)) = f16(u) trivially. This case contains every u whose binary16 result is
#              finite, because the largest finite binary16 is 65504 < 2^24.
#   u >= 2^24  the first step may round, and it cannot matter. binary32's ulp at 2^24 is 2, so
#              f32(u) >= 2^24 - 1. binary16 rounds everything at or above 65520 to +infinity, and
#              2^24 - 1 = 16777215 > 65520, so f16(f32(u)) = +infinity. And u >= 2^24 > 65520 gives
#              f16(u) = +infinity directly. Both sides are +infinity.
#
# The two cases cover [0, 2^32), so no input can distinguish the composition from the direct
# conversion. THE DOUBLE ROUNDING IS THE PART THAT LOOKS DANGEROUS AND IS NOT: it is harmless
# because the intermediate format is exact over the whole finite range of the narrow one.
#
# HOW FAR THE PROOF ACTUALLY REACHES - corrected after root's review, because the first version of
# this comment claimed less than the proof gives and then used that to justify the table's shape.
# It said "the same composition for u64 to f16 through f32 would not be safe", and NEITHER CASE OF
# THE PROOF DEPENDS ON THE SOURCE WIDTH: case 1 needs only that the value is below 2^24, case 2
# only that it is at or above 2^24 and that the intermediate conversion is correctly rounded and in
# range. So the argument covers any NONNEGATIVE INTEGER format whose conversion to binary32 has
# those properties - u64 included. What does NOT generalise is a source that can be NEGATIVE or
# already floating: f64 to f16 through f32 is a different question, since binary64 values below
# 2^24 need not be exact in binary32.
#
# THE TABLE STAYS u32-ONLY ANYWAY, and for a different reason: op11179 is the conversion this
# backend HAS. That is a supported-lowering-domain fact, not a mathematical one, and conflating
# the two is what the first version of this comment did.
#
# An exhaustive sweep over a subrange is kept as a test, but it is a control ON the proof and not
# the argument: sampling cannot establish a claim over 2^32 inputs.
COMPOSED_CALLS = {"air.convert.f.f16.u.i32": (("u32_to_f32", "f32_to_f16_rte"), 1)}
_NO_U32_TO_F16 = False
# THE CAPABILITY-OFF ARM FOR THE TWO SCALAR-HALF CONSTRUCTS. True makes both arms DECLINE, so each
# source refuses with exactly the call refusal it had before this batch - the measurement the gain
# is scored against - rather than with a message of my own invention.
_NO_SCALAR_HALF = False
# THE ORDERED FLOAT COMPARE, THE BOOLEAN-TO-FLOAT, AND THE SIXTEEN-BIT REVERSAL.
_NO_FLOAT_ORDER = False
_NO_REVERSE16 = False
_NO_MULHI16 = False
_NO_BFLOAT_WIDEN = False
_F32_SIGN_SHIFT = 31
_F32_MAGNITUDE = 0x7FFFFFFF
_F32_ONE_BITS = 0x3F800000
_WORD_ONES = 0xFFFFFFFF


def _f32_order_key(b, bits, ir, tag):
    """An unsigned key whose ORDER is the float order, for finite values.

    IEEE-754 was designed so that the bit patterns of like-signed finite floats compare as
    integers. The standard total-order key makes that work across signs: a non-negative value gets
    its sign bit set, and a negative value is complemented, so the unsigned order of the keys is
    the numeric order of the floats.

        non-negative   key = bits ^ 0x80000000
        negative       key = bits ^ 0xFFFFFFFF

    SIGNED ZERO IS THE ONE PLACE THIS DISAGREES WITH IEEE, and it is handled before the key rather
    than argued away: -0 and +0 are numerically EQUAL but their patterns differ, so -0 would key
    below +0 and `-0 >= +0` would come out false where IEEE says true. So a value whose magnitude
    is zero has its sign bit cleared first, which maps -0 onto +0 exactly. That is the whole of the
    signed-zero handling and it is measured by a control, not left implicit.

    NaN IS NOT HANDLED AND MUST NOT BE, which is why the caller requires the AIR's `fast` flag: an
    unrestricted ordered compare has to answer false for NaN operands on BOTH sides, and the
    integer order cannot express that - a NaN pattern keys above every finite value. So this is
    admitted only where the source itself declares the finite domain.
    """
    magnitude = getattr(b, "and")(bits, b.const(_F32_MAGNITUDE, name="mag"), name="%smag" % tag)
    is_zero = b.icmp(magnitude, b.const(0, name="zero"), rel="eq", name="%sisz" % tag)
    clear = b.shl(is_zero, ir.Imm(_F32_SIGN_SHIFT), name="%sclr" % tag)
    keep = getattr(b, "xor")(clear, b.const(_WORD_ONES, name="ones"), name="%skeep" % tag)
    canonical = getattr(b, "and")(bits, keep, name="%scan" % tag)
    sign = b.shr(canonical, ir.Imm(_F32_SIGN_SHIFT), name="%ssgn" % tag)
    spread = b.sub(b.const(0, name="z0"), sign, name="%sspr" % tag)      # 0 or 0xFFFFFFFF
    mask = getattr(b, "or")(getattr(b, "and")(spread, b.const(_F32_MAGNITUDE, name="m2"),
                                              name="%smlo" % tag),
                            b.const(1 << _F32_SIGN_SHIFT, name="top"), name="%smask" % tag)
    return getattr(b, "xor")(canonical, mask, name="%skey" % tag)


def _bool_to_f32(b, value, name="b2f"):
    """0 or 1 in a register to the binary32 patterns of 0.0 and 1.0, with no multiply.

    `sub(0, v)` spreads the bit to 0 or 0xFFFFFFFF and the AND selects 1.0's pattern - exact
    because the comparison forms this consumes yield exactly 0 or 1.
    """
    spread = b.sub(b.const(0, name="bz"), value, name="bspr")
    return getattr(b, "and")(spread, b.const(_F32_ONE_BITS, name="one"), name=name)

# HALF ARITHMETIC WITHOUT A HALF ALU: widen both operands, operate in binary32, and NARROW AFTER
# EVERY SOURCE OPERATION. The narrowing per operation is the whole point - substituting the chain
# with binary32 and casting once at the end is a different function, and it is what batch 3 caught
# this front end doing implicitly.
#
# WHY THIS IS THE SAME FUNCTION AS NATIVE BINARY16 ARITHMETIC, per operation:
#
#   THE WIDENING IS EXACT ON EVERY VALUE EXCEPT NEGATIVE ZERO. binary16 is a subset of binary32 -
#   11 significand bits against 24, and the whole binary16 exponent range including its subnormals
#   (down to 2^-24) lies inside binary32's normal range (down to 2^-126) - so no MAGNITUDE is
#   perturbed and the only roundings in the composition are the binary32 operation and the final
#   narrowing. BUT op1004 IS AN fadd OF +0, NOT A CAST, and -0.0 + 0.0 is +0.0: a half -0.0 widens
#   to f32 +0.0 (0x8000 in, 0x00000000 out, where a cast gives 0x80000000). Root's review of
#   c8760654 found this and it is why the gate below requires `nsz`. It is also why the exhaustive
#   check had to be re-run: the first one used numpy's cast and so never exercised the instruction
#   this lowering actually emits.
#
#   SUBNORMALS REMAIN AN OPEN QUESTION rather than a verified one. The add of +0 is exact on a
#   subnormal in software, and root's retained receipts "do not isolate general rounding/NaN
#   semantics", so whether the hardware flushes a subnormal operand is unmeasured. `fast` does not
#   license flushing, so this is recorded as a limitation of the evidence and not as permission.
#
#   MULTIPLICATION IS EXACT IN THE INTERMEDIATE. Two 11-bit significands multiply to at most 22
#   bits, which fits binary32's 24, and the product's exponent cannot leave binary32's range
#   (65504^2 is about 4.3e9, and the smallest subnormal squared about 3.6e-15). So there is only
#   ONE rounding - the narrowing - and single rounding is the definition of correct.
#
#   ADDITION AND SUBTRACTION ARE DOUBLE ROUNDED, AND THE DOUBLE ROUNDING IS INNOCUOUS BECAUSE THE
#   INTERMEDIATE HAS EXACTLY ENOUGH BITS. Figueroa's bound: rounding to p2 bits and then to p1 is
#   equivalent to rounding once to p1, for addition, when p2 >= 2*p1 + 2. Here p1 = 11 and
#   2*p1 + 2 = 24, which is exactly binary32's significand width. The bound is MET, not exceeded -
#   so this argument does not survive a change of either format, and the entry is per pair.
#   Subtraction would need no separate NUMERICAL argument - negation is exact in both formats, so
#   a - b is the addition case with an operand from the same finite set, and the exhaustive check
#   below covers it because the finite binary16 set is closed under negation. It is still refused
#   here: fp32 fsub now lowers as the add with the negate source modifier (_binop, MM 25.180,
#   hardware-matched against Apple), but the HALF composition of it has not been run on hardware,
#   and a provable numerical argument is not an execution.
#
# AND IT IS CHECKED EXHAUSTIVELY, over the whole domain rather than a sample: all 63,488 finite
# binary16 values against all 63,488, both operand orders included, comparing the composition
# against the correctly-rounded sum computed in binary64 - which is exact for these inputs, since
# the exact sum of two binary16 values needs at most about 40 significand bits and binary64 has 53.
# 4,030,726,144 pairs, zero differing. That is a complete verification of the admitted domain, and
# it is stated alongside the structural reason rather than instead of it.
#
# WHAT IS NOT ADMITTED. Half fsub, for the reason above. fdiv: the exact quotient of two binary16 values is not generally
# representable in any finite precision, so the 2p+2 bound does not apply and the double rounding
# is not innocuous. Non-finite INPUTS are outside the verified domain - infinities do propagate
# correctly through the composition by construction, but NaN payload and signalling behaviour is
# not modelled here and no source in the population carries either (every half operation in the
# five sources is `fadd fast half`, and `fast` asserts no NaN or infinity).
_COMPOSED_HALF = {"fadd": "fadd", "fmul": "fmul"}
_NO_COMPOSED_HALF_ARITH = False

# AIR's ten integer relations. The backend measures three condition codes and derives the rest by
# swapping operands or complementing with an xor; the names are passed through unchanged so the
# SIGNEDNESS travels with them - `ult` and `slt` are different codes and agree on every
# non-negative pair, which is why they were separated by running (-5, 3).
# f32 -> u32, BUILT FROM MASKS, SHIFTS AND TWO SELECTS, because no conversion opcode exists for it.
#
# THE CONTRACT FIRST, and the first version of this comment got it wrong in the one direction that
# matters. I derived the contract from C/C++, where a float-to-integer conversion whose integral
# part is unrepresentable is UNDEFINED, and concluded that NaN was undefined here too. IT IS NOT:
# root's review cites MSL 2026-06-04 section 8.6 - float-to-integer conversion of a NaN yields
# ZERO, and fast math does not change conversion accuracy. Metal DEFINES what C leaves open, so
# reading the C rule and stopping was reading the wrong specification.
#
# The defect that followed was real and is kept below as a rejected control: 0x7fc00000, a quiet
# NaN, produced 0x80000000; the signaling NaN 0x7f800001 produced 0x200; 0x7fffffff produced
# 0xfffffe00. Every NaN payload gave a different wrong answer, because the formula was shifting a
# NaN's mantissa as if it were a significand.
#
#     DEFINED    every v with trunc(v) in [0, 2^32) - that is v in (-1, 2^32), and it INCLUDES
#                -0.0 and the whole (-1, 0] interval, which truncate to 0
#     DEFINED    NaN -> 0, for BOTH SIGNS, quiet and signaling, and every payload (MSL 8.6)
#     NOT ESTABLISHED HERE  +-infinity and finite v outside [0, 2^32). The cited section settles
#                NaN; I have not read a Metal rule for the others, so this lowering does not claim
#                one. What it happens to produce for an infinity is 0, because the masked left
#                shift discards the hidden bit - that is an artefact of the encoding, recorded as
#                an observation and not as a guarantee.
#
# THE DECOMPOSITION. For binary32 bits x with exponent field e = (x >> 23) & 0xFF and significand
# m = (x & 0x7FFFFF) | 0x800000, the value is m * 2^(e - 150). So the integer part is a shift of m:
#
#     e < 127           |v| < 1, and e == 0 (zero or subnormal) lands here too  ->  0
#     127 <= e <= 150   the binary point falls inside m                         ->  m >> (150 - e)
#     150 <  e          the binary point falls past m                           ->  m << (e - 150)
#
# Both shift amounts are masked to five bits so neither shift is ever ISSUED out of range: on the
# branch where an amount is used it is already in [0, 31], and masking keeps the discarded one from
# depending on unmeasured shift-by->=32 behaviour. The two selects are op11375, whose `gt` is the
# only relation this needs; e is a masked 8-bit value and the thresholds are 127 and 150, so signed
# and unsigned agree over the whole comparison range and the code's signedness does not matter here.
#
# WHAT THE SIGN DOES, stated rather than implied: the formula reads only e and m, so a NEGATIVE
# input whose magnitude is at least 1 yields that MAGNITUDE. That is one permitted outcome of an
# undefined conversion, not a defined result, and nothing here claims otherwise. Inside the defined
# part of the negative range - (-1, 0] - the first case returns 0, which IS the required answer.
#
# CHECKED against an independent trunc() reference over every exponent boundary and a per-exponent
# sweep, and against MSL's NaN rule over EVERY NaN ENCODING - all 16,777,214 of them, both signs,
# quiet and signaling, every payload.
_NO_F32_TO_U32 = False
# fneg, fast_fabs and ashr: three ordinary operations this backend could always express and the
# front end simply never mapped. Switchable together for the capability-off census.
_NO_SIGN_OPS = False


_NO_F2I_OP9320 = False


def _f32_to_u32(b, x, ir):
    """The decomposition above, as IR. `x` is a 32-bit register holding the binary32 encoding.

    `and` and `or` are set on the Builder by name from the ARITH table, so they are reached with
    getattr rather than as attributes - `b.and(...)` is not spellable in Python.
    """
    # THE MASKS ARE REGISTERS, NOT IMMEDIATES. The bitwise-immediate form's slot is EIGHT BITS -
    # `and immediate 8388607 exceeds the 8-bit slot` - so 0x7FFFFF and 0x800000 are materialised
    # and the register-register form is used. Same bound as the compare immediate; the fix is the
    # same shape, and a shift amount (23) still fits the slot it has.
    b_and, b_or = getattr(b, "and"), getattr(b, "or")
    e = b_and(b.shr(x, ir.Imm(23), name="ef"), b.const(0xFF, name="kff"), name="e")
    mf = b_and(x, b.const(0x7FFFFF, name="kman"), name="mf")
    m = b_or(mf, b.const(0x800000, name="khid"), name="m")
    rsh = b_and(b.sub(b.const(150, name="k150"), e, name="rs"), b.const(31, name="k31"), name="rsh")
    lsh = b_and(b.sub(e, b.const(150, name="k150b"), name="ls"), b.const(31, name="k31b"), name="lsh")
    right = b.shr(m, rsh, name="r")
    left = b.shl(m, lsh, name="l")
    big = b.csel(e, b.const(150, name="k150c"), left, right, rel="gt", name="big")
    small = b.csel(b.const(127, name="k127"), e, b.const(0, name="z"), big, rel="gt", name="u")
    # NaN IS DEFINED AND IT IS ZERO. e == 255 with a NONZERO mantissa is a NaN; e == 255 with a
    # zero mantissa is an infinity, which this arm must not catch, so the mantissa is tested too.
    # The raw mantissa `mf` is reused rather than recomputed. Both signs are covered because
    # neither `e` nor `mf` carries the sign bit.
    nan = getattr(b, "and")(
        b.icmp(e, b.const(0xFF, name="k255"), rel="eq", name="e255"),
        getattr(b, "xor")(b.icmp(mf, b.const(0, name="kz"), rel="eq", name="m0"),
                          ir.Imm(1), name="mnz"),
        name="isnan")
    # `test` is op11375's code 1, a BIT TEST (ledger/g17-csel-code-one-is-a-bit-test.toml). This line
    # used to say `eq`, and it was right by coincidence: `nan` is 0 or 1 and is tested against 1, where
    # (nan & 1) != 0 and nan == 1 agree.
    return b.csel(nan, b.const(1, name="k1"), b.const(0, name="z2"), small, rel="test", name="c")


_ICMP_RELATIONS = ("eq", "ne", "ugt", "uge", "ult", "ule", "sgt", "sge", "slt", "sle")
_NO_INTEGER_COMPARE = False
# THE CAPABILITY-OFF ARM FOR CARRIED DECLARATIONS. True restores the state before this batch: a
# buffer whose declared element this backend cannot ACCESS refuses the whole program, even when the
# program never touches that buffer. It is kept because the measurement it preserves is the one the
# gain is measured against - 137 unused declarations across the frozen population, 105 of them wider
# than any form here can address - and because a switch nothing reads proves nothing (a capability
# flag the front end ignored is how a census once returned 66 on both arms).
_NO_WIDE_DECLARATIONS = False


def to_ir(air, name=None, threadgroup_size=None):
    """g17ir.Function for a straight-line AIR kernel, or Unsupported naming the construct.

    `threadgroup_size` is the launch domain for a program that uses threadgroup memory. No source
    states one - a Metal kernel declares `threadgroup float s[32]` and nothing about how many
    threads will run - and the ABI's required_size is a size the runtime must launch with EXACTLY,
    so deriving it from the array extent would invent a source attribute: `[256 x float]` does not
    mean 256 threads, it means the index must stay under 256. When the caller says nothing the
    default is the domain the EVIDENCE covers, 32 lanes
    (ledger/g17-threadgroup-exchange-at-32-lanes.toml), stated as such rather than derived from the
    array; a caller who knows its own launch passes it and overrides that.
    """
    import g17ir as ir
    air = inline_calls(air)
    fname, lines = body(air)
    # A PROVED COUNTED LOOP BECOMES STRAIGHT-LINE AIR BEFORE ANYTHING ELSE LOOKS AT IT, so every
    # other construct in the program keeps exactly the path it already had. A function with no
    # back edge is returned unchanged, which is why this cannot affect the 117.
    lines = _unroll_counted_loop(lines)
    lines = _if_convert_switch(lines)
    args = arguments(air)
    other = [d for k, d in args if k == "other"]
    if other:
        raise Unsupported("argument kind %r: this front end reads buffers and the position "
                          "builtins only" % other[0])

    # DECLARE WHAT THE SOURCE DECLARES. The first version of this dropped buffers the kernel
    # never touches, because doing so made six of the project's own end-to-end kernels dispatch
    # correctly where declaring all of them returned 0xDEADBEEF on every lane. Six witnesses.
    #
    # The linker measured it over 12,044 corpus kernels, comparing each source's declared buffer
    # set against the binding indices its metadata records:
    #
    #     records EXACTLY what the source declares   11,312   93.9%
    #     records FEWER (eliminated)                    470    3.9%
    #     records MORE (an internal binding added)      256    2.1%
    #     neither a subset nor a superset                 6
    #
    # Metal's own compiler records the DECLARATION, not the use. So the rule that fixed six kernels
    # was fitted against a 93.9% majority, and shipping it here would have put a 3.9% case in the
    # one layer where every future kernel passes through it. It is not shipped.
    #
    # WHICH MEANS THE SIX HAVE A DIFFERENT CAUSE, still open: with every buffer declared their
    # stores carry a rank that returns the fill value at status 0. The rank the IMAGE declares and
    # the index the HOST binds have to agree, and reconciling them is docs/archive/g17-scan-acceptance.md
    # item 4. The 2.1% row is a live candidate for the other direction: 256 kernels record MORE
    # bindings than the source names, so a rank computed from the source list alone is off by one
    # on every one of them.
    # THE DECLARED ELEMENT TYPE IS PRESERVED, and an unrecognised one REFUSES rather than
    # defaulting. Defaulting is what put a `device float *` into the contract as `uint`: the
    # emitted bytes are identical because the width is the same, so nothing upstream notices and
    # the image refuses later about a declaration this layer had already lost. The spellings are
    # Apple's own `air.arg_type_name` values and ir.ELEM_NAMES is the same table, read backwards -
    # so a type this IR has not recovered (a vector element, a struct, a 64-bit scalar) lands in
    # the refusal below with its name in the message instead of silently becoming a uint.
    # THE SIGNED SPELLINGS ARE THE SAME ELEMENT, AND THAT IS A FACT ABOUT THIS IR. ir.ELEM_NAMES
    # carries Metal's unsigned/float names only, so reading it backwards alone would newly refuse
    # `device int *` and `device short *` - which this front end handled before, at the same width,
    # because the IR's element type records WIDTH and int-vs-float and not signedness. Refusing them
    # would be a regression dressed as strictness. MEASURED over the pinned 197-tag population, every
    # declared air.arg_type_name and its count:
    #
    #     uint 245   float 126   half 95   ushort 26        <- ir.ELEM_NAMES, read backwards
    #     int 43     short 23                               <- the same elements, signed spelling
    #     ulong 28   long 17                                <- eight bytes, DECLARATION-ONLY
    #     uchar 1   bfloat 7                                <- one and two bytes, DECLARATION-ONLY
    #     half2/uint2/float2/short2 61   uint4/float4/half4/int4 57   bfloat2/bfloat4 2   <- vectors,
    #                                                          DECLARATION-ONLY
    #     metal::_atomic 16   P 2                           <- not carried, still refused by name
    #
    # THE THREE `DECLARATION-ONLY` ROWS ARE NEW, AND THE COUNT ABOVE THEM WAS TRUE WHEN WRITTEN.
    # It said 535 of 673 expressible with 138 refusing by name, and the reason 138 refused was that
    # a declaration this backend cannot ACCESS refused the program that merely DECLARED it. 137 of
    # the population's declarations are never touched by their own program, so that refusal was
    # about a type nothing reads. Now every spelling the corpus declares and the linker has
    # recovered is CARRIED with its size, alignment and signedness (ir.DECL_PHYSICAL), an ACCESS to
    # one refuses in _agree with those facts in the message, and only `metal::_atomic` and the
    # template parameter `P` - 18 declarations - still refuse at the declaration itself. The five
    # programs root named lower on this; a `long` stays a `long` and a `uint2` stays a `uint2`.
    elem_of = {v: k for k, v in ir.ELEM_NAMES.items()}
    elem_of.update({"int": ir.I32, "short": ir.I16})
    atomic_decls = _atomic_declarations(air)
    buffers, argmap = [], {}
    for i, (kind, detail) in enumerate(args):
        if kind == "buffer":
            slot, type_name = detail
            if type_name is None:
                raise Unsupported("buffer at index %s declares no air.arg_type_name, so its "
                                  "element type cannot be preserved" % slot)
            if type_name == "metal::_atomic":
                if _NO_ATOMIC:
                    raise Unsupported(
                        "buffer element type 'metal::_atomic' at index %s: the atomic capability "
                        "is switched off here, which is the state the gain is measured against"
                        % slot)
                # the metadata spells every atomic the same; the STRUCT says which one it is
                type_name = atomic_decls.get(i)
                if type_name is None:
                    raise Unsupported(
                        "buffer at index %s declares metal::_atomic but the signature's parameter "
                        "%d does not point at an atomic struct this front end can resolve, so its "
                        "field type and width cannot be preserved" % (slot, i))
                if type_name not in elem_of:
                    raise Unsupported(
                        "buffer at index %s declares metal::_atomic over field type %r: the "
                        "fields this IR carries are %s" % (slot, type_name,
                        ", ".join(sorted(_ATOMIC_STRUCT_FIELDS.values()))))
            if type_name not in elem_of:
                raise Unsupported("buffer element type %r at index %s: this IR carries %s"
                                  % (type_name, slot, ", ".join(sorted(elem_of))))
            if _NO_WIDE_DECLARATIONS and elem_of[type_name] not in ir.ACCESSIBLE_ELEMS:
                raise Unsupported("buffer element type %r at index %s is carried as a declaration "
                                  "only, and the capability-off arm refuses it: this is the state "
                                  "the gain is measured against" % (type_name, slot))
            # THE SOURCE SPELLING IS CARRIED, NOT JUST THE LOWERED ONE. `type_name` is the
            # declaration as AIR wrote it - `int` and `short` among them - while
            # `elem_of[type_name]` is what this backend lowers to, and those differ for exactly
            # the two signed aliases. The lowering is unchanged; the spelling is retained beside
            # it so a projected program can state the ORIGINAL declarations.
            b = ir.Buffer("b%d" % slot, slot, elem=elem_of[type_name],
                          declared_element=type_name)
            buffers.append(b)
            argmap["%d" % i] = ("buffer", b)
        else:
            argmap["%d" % i] = ("builtin", detail)

    fn = ir.Function(name or fname, buffers)
    # THE RESOLVED LAUNCH SHAPE, for anything that is a function OF it rather than merely bounded
    # by it. An explicit threadgroup_size IS the exact launch contract and is used as given. With
    # no explicit one, the shape is resolved only for a program that declares threadgroup memory,
    # where the ABI already records required_size from the measured 32-lane domain; a program with
    # neither has NO resolved shape and `None` is carried so the consumer refuses rather than
    # defaulting. Note what this is not: the array extent is never consulted, because `[75 x i32]`
    # bounds the index and does not state a thread count.
    resolved_launch = tuple(threadgroup_size) if threadgroup_size is not None else None
    if resolved_launch is not None and (len(resolved_launch) != 3
                                        or any(not isinstance(n, int) or n < 1
                                               for n in resolved_launch)):
        # A MALFORMED EXPLICIT LAUNCH REFUSES HERE rather than reaching the IR, which used to
        # raise a bare IndexError on a two-element shape - an internal error where a named refusal
        # belongs. No source passes a launch, so this cannot move any of the 197.
        raise Unsupported("threadgroup shape %r: the launch contract is three positive thread "
                          "counts" % (threadgroup_size,))
    if any(kind == "builtin" and detail == _LOCAL_INDEX for kind, detail in args):
        # AN EXPLICIT PYTHON ARGUMENT IS NOT AN ENFORCEABLE LAUNCH CONTRACT, and this refusal is
        # here because a peer review found the hole and root reproduced it: a kernel that uses the
        # linear index and declares NO threadgroup memory compiled at threadgroup_size=(8,4,1) to
        # 54 bytes whose typed contract carried `threadgroup: null` and whose emission recorded no
        # required_size. The emitted code MULTIPLIES BY size_x, so it is a different function at a
        # different launch - and nothing in the delivered artifact would have told a host which
        # launch it was compiled for. `fn.declare_threadgroup` is what writes required_size, and
        # only a program with threadgroup memory reaches it, so that is exactly the condition.
        #
        # Refusing rather than widening the contract schema is root's call, recorded in the queue.
        if not _threadgroup_arrays(air):
            raise Unsupported(
                "thread_index_in_threadgroup in a kernel with no threadgroup memory: the linear "
                "index is compiled FOR a shape (size_x multiplies the y coordinate), and only a "
                "program that declares threadgroup memory records required_size in its typed "
                "contract - so here the shape the code assumes would reach no host, and a launch "
                "the contract does not state is not a contract")
        if _NO_LOCAL_INDEX:
            # THE OFF ARM REPRODUCES THE ORIGINAL REFUSAL, and it has to be raised here rather
            # than left to fall through: the generic builtin path would read a single register
            # named by this kind, which is an UNMEASURED read, so falling through would emit a
            # wrong program instead of declining. Off must decline, not guess.
            raise Unsupported("argument kind 'air.%s': this front end reads buffers and the "
                              "position builtins only" % _LOCAL_INDEX_AIR_NAME)
    # THREADGROUP ARRAYS ARE MODULE GLOBALS, so they are read before the body and declared once.
    tg_arrays = _threadgroup_arrays(air)
    tg_global = tg_extent = tg_element = None
    if tg_arrays:
        if _NO_THREADGROUP_MEMORY:
            raise Unsupported("a threadgroup array global (%s): the threadgroup-memory capability "
                              "is switched off here, which is the state the gain is measured "
                              "against" % ", ".join(sorted(tg_arrays)))
        if len(tg_arrays) > 1:
            raise Unsupported("%d threadgroup array globals: the backend's scratchpad is ONE "
                              "register-indexed region and no offset rule between two arrays is "
                              "established, so this refuses rather than packing them by guess"
                              % len(tg_arrays))
        tg_global, (tg_extent, tg_element, tg_align) = next(iter(tg_arrays.items()))
        words = _threadgroup_words(tg_extent, tg_element, tg_global)
        # THE LAUNCH DOMAIN IS THE MEASURED ONE, NOT THE ARRAY'S SIZE.
        #
        # No source states a threadgroup size - none of these kernels carries
        # max_total_threads_per_threadgroup - and the ABI's required_size is a size the runtime
        # must launch with EXACTLY. Deriving it from the array extent would be the invented
        # attribute: `[256 x float]` does not mean 256 threads, it means the index must stay under
        # 256. So the default is the domain the EVIDENCE covers - the threadgroup exchange is
        # measured at 32 lanes, ledger/g17-threadgroup-exchange-at-32-lanes.toml - and a caller
        # who knows better passes its own.
        launch = tuple(threadgroup_size) if threadgroup_size is not None \
            else _TG_MEASURED_LAUNCH
        resolved_launch = launch
        threads = launch[0] * launch[1] * launch[2]
        # AND THE BOUND IS CHECKED, which is what "unsafe bounds refuse" has to mean here: a
        # per-thread index into an N-element array is in bounds only while the threadgroup has at
        # most N threads. coop4-n190's array is exactly 32 floats, so 32 lanes is its maximum
        # rather than a free choice.
        # NO PRESENCE-BASED LAUNCH CHECK. This used to refuse when the thread count exceeded the
        # array, on the strength of a per-thread builtin appearing anywhere in the kernel - and
        # that check was both unsound (root broke it with one line: the index is an expression, so
        # `add x, 64` passes it and reads past the array) and too strong (a masked index stays in
        # bounds at ANY launch, so refusing it was wrong). The per-access proof in _tg_index
        # replaces it and subsumes it: an unmasked threadgroup-position index carries the bound
        # `threads - 1`, so an oversized launch is caught there, by the bound rather than by the
        # builtin's presence.
        fn.declare_threadgroup(words, size=launch, alignment=tg_align)
        tg_defs, tg_threads = _index_bounds(lines), threads
        tg_kinds = {k: v[1] for k, v in argmap.items() if v[0] == "builtin"}
    b = ir.Builder(fn, fn.block("entry"))
    val = {}          # AIR register name -> ir.Value, ir.Imm, or a ("gep", buffer, index) tuple
    words = {}        # AIR register name -> (low word, high word) for an ESTABLISHED i64 value
    # THE PAIRS WHOSE VALUE IS ONLY MEANINGFUL AS A PAIR: a component load and a wide add/sub
    # result. A zext-derived pair is in `words` too, because a wide operand may read it, but it is
    # NOT here - its scalar low word is equally valid and the existing arms use it that way. That
    # distinction is what keeps `r-shr64a` compiling: it applies `shl i64` to a zext-derived value,
    # and refusing every operation that touches any pair would have refused a program that works.
    wide_results = set()
    literals = {}        # literal TOKEN TEXT -> the materialised f32 constant, one per distinct token
    # AIR values this front end produced with a comparison, so a later `zext i1` can be the
    # identity for exactly those and refuse for anything else.
    booleans = set()

    def operand(tok):
        tok = tok.strip()
        if tok.startswith("%"):
            k = tok[1:]
            if k in val:
                return val[k]
            if k in argmap and argmap[k][1] == _LOCAL_INDEX and not _NO_LOCAL_INDEX:
                # INTERCEPTED BEFORE THE GENERIC PATH BELOW, which would read a single register on
                # axis x. For a 1-D shape that happens to be the right answer, and taking it
                # silently on a multidimensional one would be wrong by exactly the y and z terms.
                if k not in val:
                    val[k] = _linear_local_index(b, resolved_launch, ir)
                return val[k]
            if k in argmap and argmap[k][0] == "builtin":
                if argmap[k][1] == _LOCAL_INDEX_AIR_NAME:
                    # UNREACHABLE BY DESIGN, asserted rather than trusted: the linear index is
                    # computed above and the capability-off arm refuses before the body is read.
                    # If this ever fires, something added a path that would emit an unmeasured
                    # single-register read for a builtin that is not one.
                    raise Unsupported(
                        "internal: %s reached the generic builtin read, which would emit one "
                        "unmeasured register where the linear index is a computed value"
                        % _LOCAL_INDEX_AIR_NAME)
                # A SCALAR DECLARATION HAS NO extractelement. `uint t [[thread_position_in_grid]]`
                # arrives as an i32 argument and is used directly, where `uint3 tp` arrives as a
                # vector and is picked apart - so refusing a builtin used whole refused every
                # kernel written the first way, which is how all 36 of this project's own
                # end-to-end kernels are written.
                if k not in val:
                    val[k] = b.builtin(argmap[k][1], name="t", axis="x")
                return val[k]
            raise Unsupported("AIR value %%%s is used before this front end defines it" % k)
        if _INT.match(tok):
            return ir.Imm(int(tok))
        half = f16_literal_bits(tok)
        if half is not None:
            # A HALF LITERAL IS MATERIALISED AS A SIXTEEN-BIT VALUE, not as an f32 the consumer
            # then narrows. Narrowing it here would introduce a rounding step the source does not
            # have, and the half store and the half ALU both name the 16-bit register file - so the
            # type is I16, which is what those consumers accept.
            if tok not in literals:
                literals[tok] = b.const(half, type=ir.I16, name="h%d" % len(literals))
            return literals[tok]
        bits = f32_literal_bits(tok)
        if bits is not None:
            # MATERIALISED ONCE PER DISTINCT LITERAL. sl32-u148 carries 146 distinct hex constants
            # across 148 fused operations, and a fresh const op per USE would emit an instruction
            # per reference rather than per value. Keyed on the token text, which is how AIR spells
            # the value; two spellings of one value materialise twice, which is honest about what
            # the source said and costs a const the allocator can coalesce.
            if tok not in literals:
                literals[tok] = b.const(bits, type=ir.F32, name="f%d" % len(literals))
            return literals[tok]
        raise Unsupported("operand %r" % tok)

    def _pointee(k, val, argmap, b):
        """The (buffer, index) a pointer names. A kernel that writes `h[0]` has no getelementptr at
        all - AIR hands the argument pointer straight to the load - and reading only geps refused
        sixteen of the first three hundred corpus kernels for a construct that is element zero."""
        g = val.get(k)
        if isinstance(g, tuple) and g[0] == "gep":
            return g
        if k in argmap and argmap[k][0] == "buffer":
            return ("gep", argmap[k][1], ir.Imm(0))
        return g

    # FORWARD CONTROL FLOW, recognised once before the body is walked: the blocks have to exist
    # before any instruction can branch to them, and the shape has to be the measured one before
    # any of it is emitted.
    ir_blocks, cf_cond, cf_join = {}, None, None
    tg_defs = tg_defs if tg_arrays else {}
    tg_threads = tg_threads if tg_arrays else 0
    tg_kinds = tg_kinds if tg_arrays else {}
    if any(_LABEL.match(l) for l in lines):
        if _NO_FORWARD_CONTROL_FLOW:
            raise Unsupported("a kernel with %d basic blocks: the forward-control-flow capability "
                              "is switched off here, which is the state the gain is measured "
                              "against" % (sum(1 for l in lines if _LABEL.match(l)) + 1))
        cf_cond, then_label, cf_join = _if_then_shape(_split_blocks(lines))
        ir_blocks = {then_label: fn.block("guarded"), cf_join: fn.block("join")}
    # A BRANCH PREDICATE IS A DIFFERENT COMPARE FROM A VALUE. icmp that yields a value lowers to
    # op11372, which the IR will not accept as a branch condition ("br_cond predicate must come
    # directly from a cmp"): the branch's compare is the immediate form op10369, whose result is
    # mask state rather than a register. So an icmp whose result reaches a branch is lowered with
    # Builder.cmp, and one whose result is stored or arithmetic keeps op11372. Knowing which needs
    # the USE, so the branch conditions are collected here rather than guessed at per instruction.
    branch_conditions = {m.group(1) for m in
                         (re.match(r"br i1 %([\w.]+),", l) for l in lines) if m}

    for line in lines:
        label = _LABEL.match(line)
        if label:
            b.at(ir_blocks[label.group(1)])
            continue
        m = _ASSIGN.match(line)
        dest, rhs = (m.group(1), m.group(2)) if m else (None, line)
        head = rhs.split()[0]
        if head in ("tail", "call", "musttail"):
            callee = re.search(r"@([\w.$]+)", rhs)
            fname_ = callee.group(1) if callee else None
            # A VECTOR INTRINSIC IS N CALLS OF ITS OWN SCALAR INTRINSIC, and only where that
            # scalar one is already wired. air.fast_fmin.v2f32 becomes two air.fast_fmin.f32 -
            # the same measured op9700 select the scalar sources emit - and the lane count in the
            # name must equal the lane count of every argument, which is checked rather than
            # assumed. A vector intrinsic whose scalar counterpart this front end does not have
            # refuses by name, naming the scalar it would need.
            if fname_ in ("air.wg.barrier", "air.simdgroup.barrier"):
                flags = [int(x) for x in re.findall(r"i32 (-?\d+)", rhs)]
                if len(flags) != 2:
                    raise Unsupported("a call to %s with %d integer flags; the measured form takes "
                                      "two" % (fname_, len(flags)))
                scope = _BARRIER_FLAGS.get((fname_, flags[0], flags[1]))
                if scope is None:
                    raise Unsupported(
                        "a call to %s(%d, %d): the memory flag and execution scope are read from "
                        "Apple's own source-to-AIR mapping (mem_device -> (1, 1), mem_threadgroup "
                        "-> (2, 1)), and this combination has no barrier scope this backend has "
                        "measured - g17asm.BARRIER_SCOPE carries none, threadgroup, imageblock, "
                        "device and texture, and no simdgroup execution scope at all"
                        % (fname_, flags[0], flags[1]))
                b.barrier(scope=scope)
                continue
            atomic = re.match(r"air\.atomic\.(\w+)\.(\w+)\.(\w)\.i(\d+)$", fname_ or "")
            if atomic is not None:
                space, aop, signedness, bits = (atomic.group(1), atomic.group(2),
                                                atomic.group(3), int(atomic.group(4)))
                if _NO_ATOMIC:
                    raise Unsupported(
                        "a call to %s: the atomic capability is switched off here, which is the "
                        "state the gain is measured against" % fname_)
                if space != "global":
                    raise Unsupported(
                        "a call to %s: this front end lowers the DEVICE atomic only. The "
                        "threadgroup family is a different form (op11765 carries no address "
                        "operand at all) and none of it is measured through this path" % fname_)
                if bits != 32:
                    raise Unsupported(
                        "a call to %s: the measured per-lane device atomic is 32-bit, and a "
                        "%d-bit one is a width this form has not been read at" % (fname_, bits))
                if aop not in _ATOMIC_OPERATIONS:
                    raise Unsupported(
                        "a call to %s: the operations measured on the per-lane device form are "
                        "%s. The operation is a field the encoder can place, but WHICH operations "
                        "Apple emits at this form was asked only of those, so the rest refuse "
                        "rather than being assumed from the table"
                        % (fname_, "/".join(sorted(_ATOMIC_OPERATIONS))))
                flags = re.findall(r"i32 (-?\d+), i32 (-?\d+), i1 (\w+)\)", rhs)
                if not flags:
                    raise Unsupported("a call to %s whose trailing flags this front end cannot "
                                      "read: %r" % (fname_, rhs))
                got = (int(flags[-1][0]), int(flags[-1][1]), flags[-1][2])
                if got != _ATOMIC_FLAGS:
                    raise Unsupported(
                        "a call to %s with flags %s: the only triple measured is %s, which is what "
                        "Apple emits for memory_order_relaxed at device scope. Another ordering or "
                        "scope is not the same instruction and is not assumed equivalent"
                        % (fname_, got, _ATOMIC_FLAGS))
                mm = re.search(r"addrspace\(1\)\*\s+(?:nocapture\s+)?%([\w.]+)", rhs)
                g = _pointee(mm.group(1), val, argmap, b) if mm else None
                if not (isinstance(g, tuple) and g[0] == "gep"):
                    raise Unsupported("a call to %s whose address is not a getelementptr of a "
                                      "kernel buffer: %r" % (fname_, rhs))
                buf, index = g[1], g[2]
                # THE STRUCT FIELD INDEX IS CHECKED, NOT DROPPED. The address is
                # `getelementptr %"struct.metal::_atomic", ... %0, i64 %5, i32 0` and the generic
                # getelementptr arm reads the ELEMENT index only, ignoring the trailing field
                # index. For a one-field struct 0 is the only correct value, so a non-zero one
                # would address past the object and must refuse rather than be ignored.
                gep_line = next((l for l in lines
                                 if l.strip().startswith("%" + mm.group(1) + " =")), "")
                field = re.search(r"i64 (?:%[\w.]+|\d+),\s*i32 (\d+)", gep_line)
                if field is not None and int(field.group(1)) != 0:
                    raise Unsupported(
                        "an atomic on struct field %s of %s: `metal::_atomic` has ONE field and "
                        "field 0 is it, so a non-zero index addresses past the object"
                        % (field.group(1), buf.name))
                # THE ADDRESS MUST BE PER-LANE, AND A CONSTANT INDEX IS NOT THE ONLY WAY TO FAIL
                # THAT. This checked `isinstance(index, ir.Imm)`, which catches `&a[0]` and misses
                # every COMPUTED uniform address - root found the case: this population's atomic
                # `and` indexes with `and(threadgroup_position_in_grid.x, 7)`, which is one value
                # for the whole threadgroup, and its thread_position_in_threadgroup argument is
                # declared but never used. So the check is now the established lane-varying
                # question (cc.LANE_VARYING_SR: the grid and threadgroup THREAD positions vary,
                # the threadgroup's own position does not), which subsumes the constant case.
                #
                # The distinction is not cosmetic. op10090 is selected BY the address being
                # per-lane; Apple compiles a uniform address to op10094 wrapped in a lane election
                # (op10372, op582, the atomic, op577, then an op14157 broadcast), and the per-lane
                # form at a uniform address does not serialise - every lane reads the same old
                # value. Emitting it there would be a different program, not a different encoding.
                if not _is_established_atomic_index(index):
                    raise Unsupported(
                        "a call to %s whose address is not the established per-lane form (%s): the "
                        "measured probe indexes DIRECTLY by a per-lane position builtin, and only "
                        "that address form is established. op10090 is selected BY the address "
                        "being per-lane; Apple compiles a uniform one to op10094 wrapped in a lane "
                        "election, and the per-lane form at a uniform address does not serialise, "
                        "so every lane would read the same old value - a different program rather "
                        "than a different encoding"
                        % (fname_, _describe_atomic_index(index)))
                value = operand(re.findall(r"i32 (%[\w.]+|-?\d+),\s*i32 -?\d+, i32 -?\d+, i1",
                                           rhs)[-1])
                if isinstance(value, ir.Imm):
                    value = b.const(value.v & 0xFFFFFFFF, name="k%d" % (value.v & 0xFFFFFFFF))
                val[dest] = b.atomic_rmw(aop, buf, index, value, name="old")
                booleans.discard(dest)
                continue
            dm = re.match(r"air\.dot\.v(\d)f32$", fname_ or "")
            if dm is not None and not _NO_DOT_SQRT:
                # APPLE'S ORDER, READ OFF APPLE'S CODE (MM 25.180): dot(float4 x, float4 y) is one
                # fmul (op3290) of lane 0 and then an fma (op2190) per lane, left to right -
                # fma(x3, y3, fma(x2, y2, fma(x1, y1, x0 * y0))). The same chain here, so the
                # rounding is the one Apple's compiler chose, not a reassociation `fast` would permit.
                _fast_math(rhs, "a call to %s" % fname_)
                tokens = re.findall(r"<\d+ x float> (%[\w.]+)", rhs)
                sides = [_lanes_of(val.get(t.lstrip("%"))) for t in tokens]
                if len(sides) != 2 or any(sd is None or len(sd) != int(dm.group(1)) for sd in sides) \
                        or any(isinstance(v, (_Poison, ir.Imm)) for sd in sides for v in sd):
                    raise Unsupported("a call to %s whose operands are not two held %s-lane float "
                                      "vectors" % (fname_, dm.group(1)))
                x, y = sides
                acc = b.fmul(x[0], y[0], name="dot0")
                for k in range(1, len(x)):
                    acc = b.fma(x[k], y[k], acc, name="dot%d" % k)
                val[dest] = acc
                continue
            vname = re.match(r"(air\.[\w.]+?)\.v(\d+)([a-z]\d+)$", fname_ or "")
            if vname is not None and not _NO_VECTOR_LANES:
                scalar_name = "%s.%s" % (vname.group(1), vname.group(3))
                want_lanes = int(vname.group(2))
                if scalar_name not in _FAST_SELECT_CALLS and scalar_name not in CALLS:
                    raise Unsupported("a call to %s: its per-lane form %s is not one this front "
                                      "end has, so the lanes are refused rather than composed"
                                      % (fname_, scalar_name))
                if want_lanes not in _LANE_COUNTS:
                    raise Unsupported("a call to %s: %d lanes is outside the counts this front "
                                      "end indexes" % (fname_, want_lanes))
                tokens = re.findall(r"<\d+ x \w+> (%[\w.]+)", rhs)
                sides = []
                for token in tokens:
                    held = _lanes_of(val.get(token.lstrip("%")))
                    if held is None:
                        raise Unsupported("a call to %s on %s, which this front end does not hold "
                                          "as lanes" % (fname_, token))
                    if len(held) != want_lanes:
                        raise Unsupported("a call to %s on a %d-lane value"
                                          % (fname_, len(held)))
                    for k, lane in enumerate(held):
                        if isinstance(lane, _Poison):
                            raise Unsupported("a call to %s reads lane %d of %s, which is poison"
                                              % (fname_, k, token))
                    sides.append(held)
                if scalar_name in _FAST_SELECT_CALLS:
                    method, argc = _FAST_SELECT_CALLS[scalar_name]
                    _fast_math(rhs, "a call to %s" % fname_)   # raises, naming what it needs
                else:
                    method, argc = CALLS[scalar_name]
                if len(sides) != argc:
                    raise Unsupported("a call to %s with %d vector arguments; its scalar form "
                                      "takes %d" % (fname_, len(sides), argc))
                val[dest] = _lane_value(getattr(b, method)(*lane_args, name="p%d" % k)
                                        for k, lane_args in enumerate(zip(*sides)))
                continue
            if fname_ == "air.fast_fabs.f32" and not _NO_SIGN_OPS:
                # CLEARING THE SIGN BIT IS WHAT fabs IS. IEEE 754 defines abs as a sign-bit
                # operation, not as a comparison-and-negate, so `and` with 0x7FFFFFFF is exact for
                # every encoding including -0.0 (which becomes +0.0, as required) and a NaN (whose
                # payload is preserved with the sign cleared). The `fast_` prefix licenses more
                # than this needs; the bit operation satisfies the strict definition too.
                args_ = re.findall(r"float\s+(%?[\w.+-]+)", rhs)
                if len(args_) < 1:
                    raise Unsupported("a call to %s with %d arguments" % (fname_, len(args_)))
                v = operand(args_[0])
                if isinstance(v, ir.Imm) or getattr(v, "type", None) not in (None, ir.I32):
                    raise Unsupported("a call to %s on %s: the sign clear reads a 32-bit register"
                                      % (fname_, args_[0]))
                val[dest] = getattr(b, "and")(v, b.const(0x7FFFFFFF, name="abs"), name="fa")
                continue
            # TWO SCALAR-HALF CONSTRUCTS, each composed from routes this backend already has.
            #
            # THE WIDENING IS EXACT, which is what makes both compositions the same function as the
            # half operation rather than an approximation of it: binary16 is a subset of binary32 -
            # 11 significand bits against 24, and its whole exponent range including subnormals
            # lies inside binary32's normal range - so no value is perturbed going up.
            if fname_ == "air.convert.f.f32.u.i1" and not _NO_FLOAT_ORDER:
                args_ = re.findall(r"i1\s+(%?[\w.]+)", rhs)
                if len(args_) < 1:
                    raise Unsupported("a call to %s with %d arguments" % (fname_, len(args_)))
                source = args_[0].lstrip("%")
                if source not in booleans:
                    raise Unsupported(
                        "a call to %s on the i1 %s, which did not come from a comparison this "
                        "front end lowered: an i1 has no register representation here except as a "
                        "comparison's 0-or-1 result, and 0.0/1.0 is selected from that bit"
                        % (fname_, args_[0]))
                val[dest] = _bool_to_f32(b, operand(args_[0]))
                continue
            if fname_ == "air.reverse_bits.i16" and not _NO_REVERSE16:
                # SIXTEEN-BIT REVERSAL THROUGH THE MEASURED WORD REVERSE, with zero kept out of it.
                #
                # reverse16(x) is the top half of reverse32(zext32(x)), because zero-extending puts
                # x in the low half and reversing sends it to the high half.
                #
                # WHY ZERO IS SANITISED, which is the part worth stating rather than copying: the
                # word reverse op14047 was EXECUTED on 4096, 4097, 4352 and 8192 only
                # (isa/g17-execution-sweep-results.json), and zero was not among them. Its result
                # there is unmeasured, so the composition never feeds it: `x | isz` is non-zero for
                # every input, and the `isz << 15` xor removes the artefact that introduces,
                # restoring the exact zero answer. For x = 0 that is reverse32(1) = 0x80000000,
                # whose top half is 0x8000, xored with 0x8000 to give 0.
                args_ = re.findall(r"i16\s+(%?[\w.]+)", rhs)
                if len(args_) < 1:
                    raise Unsupported("a call to %s with %d arguments" % (fname_, len(args_)))
                v = operand(args_[0])
                if isinstance(v, ir.Imm) or getattr(v, "type", None) is not ir.I16:
                    raise Unsupported(
                        "a call to %s on %s: the composition widens a sixteen-bit register, and "
                        "this operand is not one" % (fname_, args_[0]))
                wide = b.u16_to_u32(v, name="rw")
                is_zero = b.icmp(wide, b.const(0, name="rz"), rel="eq", name="risz")
                safe = getattr(b, "or")(wide, is_zero, name="rsafe")
                top = b.shr(b.reverse(safe, name="rrev"), ir.Imm(16), name="rtop")
                fixed = getattr(b, "xor")(top, b.shl(is_zero, ir.Imm(15), name="rfix"),
                                          name="rout")
                val[dest] = b.low16(fixed, name="r16")
                continue
            if fname_ == "air.mul_hi.u.i16" and not _NO_MULHI16:
                # THE UNSIGNED SIXTEEN-BIT HIGH PRODUCT, EXACTLY, through the word multiply.
                #
                # mul_hi.u.i16(a, b) is the upper sixteen bits of the 32-bit product of two
                # unsigned 16-bit values. THE WHOLE ARGUMENT IS THAT NOTHING IS LOST: the largest
                # possible product is 65535 * 65535 = 4294836225, and 2^32 = 4294967296, so every
                # product of two unsigned 16-bit values fits in a 32-bit word with room to spare.
                # So the word multiply's result IS the exact full product - not a truncation of a
                # wider one - and its upper half is the answer by definition rather than by
                # approximation. `4294836225 < 4294967296` is asserted below so that this is a
                # checked premise and not a remark.
                #
                # WHICH STEP IS ACTUALLY LOAD-BEARING, measured rather than assumed - and it is
                # not the one I first wrote down. I claimed the LOGICAL shift was critical, on the
                # grounds that an arithmetic shift replicates bit 31 and bit 31 is set for every
                # product at or above 2^31. Running it says otherwise: over 1024 pairs an
                # arithmetic shift gets ZERO wrong, because the sign extension lands entirely in
                # bits 16 and above and the truncation below discards exactly those. `shr` is
                # still what the operation MEANS and is what is emitted; it is simply not where
                # the correctness lives here.
                #
                # THE ZERO EXTENSION IS. With the operands sign-extended instead, 599 of those
                # same 1024 pairs come out wrong, because a sign-extended operand is a different
                # number and its product is not the one asked for. A bit-15 operand is NECESSARY
                # for a mismatch and NOT sufficient - 624 pairs have one and 25 of those agree
                # anyway: a zero operand gives 0 either way, and 32768 * 32768 agrees because BOTH
                # operands are extended and the two signs cancel. Stated that way because a peer
                # review caught the first wording claiming "every pair where bit 15 is set", which
                # is the stronger claim and false. That is the step the refusal below protects.
                #
                # WHAT THIS DELIBERATELY DOES NOT USE. Not a native 16-bit mul_hi - none is
                # measured through this path - not the signed form, whose sign extension would
                # corrupt an unsigned operand, and not a vector spelling for what the source
                # writes as a scalar call.
                assert 0xFFFF * 0xFFFF < (1 << 32), "the full product must fit the word result"
                args_ = re.findall(r"i16\s+(%?[\w.]+)", rhs)
                if len(args_) != 2:
                    raise Unsupported("a call to %s with %d arguments" % (fname_, len(args_)))
                xs = []
                for token in args_:
                    v = operand(token)
                    if isinstance(v, ir.Imm) or getattr(v, "type", None) is not ir.I16:
                        raise Unsupported(
                            "a call to %s on %s: the composition zero-extends a SIXTEEN-bit "
                            "register, and this operand is not one - a wider operand's product "
                            "would not fit the word result the upper half is taken from"
                            % (fname_, token))
                    xs.append(b.u16_to_u32(v, name="mw"))
                product = b.mul(xs[0], xs[1], name="mfull")
                val[dest] = b.low16(b.shr(product, ir.Imm(16), name="mhi"), name="m16")
                continue
            if fname_ == "air.convert.u.i32.f.f16" and not _NO_SCALAR_HALF:
                # half -> uint, as the widening plus the f32 decomposition already used for
                # air.convert.u.i32.f.f32. Exact: the widening loses nothing, so truncating the
                # binary32 value toward zero gives the same integer the half conversion would.
                args_ = re.findall(r"half\s+(%?[\w.+-]+)", rhs)
                if len(args_) < 1:
                    raise Unsupported("a call to %s with %d arguments" % (fname_, len(args_)))
                v = operand(args_[0])
                if isinstance(v, ir.Imm) or getattr(v, "type", None) is not ir.I16:
                    raise Unsupported(
                        "a call to %s on %s: the composition widens a SIXTEEN-bit register holding "
                        "the binary16 encoding, and this operand is not one" % (fname_, args_[0]))
                w_ = b.f16_to_f32(v, name="hw")
                val[dest] = _f32_to_u32(b, w_, ir) if _NO_F2I_OP9320 else b.f32_to_u32(w_, name="fu")
                continue
            if fname_ == "air.trunc.f16" and not _NO_SCALAR_HALF:
                # half truncation toward zero, as widen -> the measured 32-bit trunc -> narrow.
                #
                # AND THE NARROWING BACK IS EXACT, which is the part that needs an argument rather
                # than an assumption. For |h| >= 2^10 a binary16 value is ALREADY an integer - the
                # spacing there is at least 1 - so truncation returns it unchanged. For |h| < 2^10
                # the truncated integer has magnitude below 1024, and every integer up to 2048 is
                # representable in binary16. So trunc(h) is always a binary16 value, and
                # round-to-nearest-even returns an exactly representable value unchanged. No
                # double rounding arises, which is precisely why the half FMA candidates are NOT
                # admissible by the same argument and stay refused.
                args_ = re.findall(r"half\s+(%?[\w.+-]+)", rhs)
                if len(args_) < 1:
                    raise Unsupported("a call to %s with %d arguments" % (fname_, len(args_)))
                _fast_math(rhs, "a call to %s" % fname_)
                v = operand(args_[0])
                if isinstance(v, ir.Imm) or getattr(v, "type", None) is not ir.I16:
                    raise Unsupported(
                        "a call to %s on %s: the composition widens a SIXTEEN-bit register holding "
                        "the binary16 encoding, and this operand is not one" % (fname_, args_[0]))
                val[dest] = b.f32_to_f16_rte(b.trunc(b.f16_to_f32(v, name="tw"), name="t32"),
                                             name="tn")
                continue
            if fname_ == "air.convert.s.i32.f.f32" and not _NO_F2I_OP9320:
                # (int)x: op9320 code 5, mode operand 1 - truncate, saturate to int32, NaN -> 0 (MM 25.186, 25.196)
                args_ = re.findall(r"float\s+(%?[\w.+-]+)", rhs)
                if len(args_) < 1:
                    raise Unsupported("a call to %s with %d arguments" % (fname_, len(args_)))
                v = operand(args_[0])
                if isinstance(v, ir.Imm) or getattr(v, "type", None) not in (None, ir.I32):
                    raise Unsupported("a call to %s on %s: op9320 reads a 32-bit register holding the binary32 "
                                      "encoding" % (fname_, args_[0]))
                val[dest] = b.f32_to_i32(v, name="fs")
                continue
            if fname_ == "air.convert.u.i32.f.f32" and not _NO_F32_TO_U32:
                args_ = re.findall(r"float\s+(%?[\w.+-]+)", rhs)
                if len(args_) < 1:
                    raise Unsupported("a call to %s with %d arguments" % (fname_, len(args_)))
                v = operand(args_[0])
                if isinstance(v, ir.Imm) or getattr(v, "type", None) not in (None, ir.I32):
                    raise Unsupported("a call to %s on %s: the decomposition reads a 32-bit "
                                      "register holding the binary32 encoding" % (fname_, args_[0]))
                # op9320's plain cast (MM 25.196), Apple's instruction for (uint)x; the 34-instruction bit decomposition
                # stays as the control (_NO_F2I_OP9320), which differs on negatives (it drops the sign) and past 2^32
                val[dest] = _f32_to_u32(b, v, ir) if _NO_F2I_OP9320 else b.f32_to_u32(v, name="fu")
                continue
            if fname_ in _FAST_SELECT_CALLS and not _NO_FLOAT_SELECT:
                method, argc = _FAST_SELECT_CALLS[fname_]
                _fast_math(rhs, "a call to %s" % fname_)
                args_ = re.findall(r"float\s+(%?[\w.+-]+)", rhs)
                if len(args_) < argc:
                    raise Unsupported("a call to %s with %d arguments" % (fname_, len(args_)))
                val[dest] = getattr(b, method)(*[operand(a_) for a_ in args_[:argc]], name="s")
                continue
            if fname_ == "air.fast_sqrt.f32" and not _NO_DOT_SQRT:
                # APPLE'S fast::sqrt IS x * rsqrt(x), read off Apple's code (MM 25.180): op3978
                # (rsqrt2 here) and an fmul (op3290) of the source by it. The same two instructions,
                # so sqrt(0) is Apple's 0 * inf as well - the `fast` contract excludes it anyway.
                _fast_math(rhs, "a call to %s" % fname_)
                args_ = re.findall(r"float\s+(%[\w.]+)", rhs)
                x = operand(args_[0]) if args_ else None
                if x is None or isinstance(x, ir.Imm):
                    raise Unsupported("a call to %s on %s: the lowering reads a register"
                                      % (fname_, args_[0] if args_ else "nothing"))
                val[dest] = b.fmul(x, b.rsqrt2(x, type=ir.I32, name="rs"), name="sq")
                continue
            if fname_ in _SATURATE_CALLS and not _NO_FLOAT_SATURATE:
                _fast_math(rhs, "a call to %s" % fname_)
                args_ = re.findall(r"float\s+(%?[\w.+-]+)", rhs)
                if len(args_) < 1:
                    raise Unsupported("a call to %s with no argument" % fname_)
                x = operand(args_[0])
                if isinstance(x, ir.Imm):
                    raise Unsupported("a call to %s on the literal %s: folding it would ask what a "
                                      "compile-time clamp does, which is not what the instructions "
                                      "do" % (fname_, args_[0]))
                zero = b.const(0x00000000, type=ir.F32, name="sat_lo")
                one = b.const(0x3F800000, type=ir.F32, name="sat_hi")
                val[dest] = b.fmin(b.fmax(x, zero, name="sat_max"), one, name="sat")
                continue
            if fname_ == "air.ctz.i32" and not _NO_CTZ:
                # COUNT TRAILING ZEROS, SANITIZED SO NEITHER `reverse` NOR `msb` EVER SEES ZERO.
                #
                # The call carries `i1 false` - LLVM's is_zero_poison=false - so ctz(0) is DEFINED
                # and must be 32. Root's composition removes the dependency on clz(0) entirely:
                #
                #     z    = (x == 0)        a VALUE, 0 or 1 (Builder.icmp, op11372, not the
                #                            exec-mask Builder.cmp)
                #     safe = x | z           always nonzero: x when x != 0, else 1
                #
                # ROOT'S SKETCH USED clz AND THIS BACKEND HAS NO clz, which is the one correction
                # to it. op9986 is named `msb` in UNARY_OPCODE and a comment beside it called it
                # "clz"; the executed sweep settles which (isa/g17-execution-sweep-results.json,
                # cb_status 0, status ok, 3 runs):
                #
                #     4096 -> 12    4097 -> 12    4352 -> 12    8192 -> 13
                #
                # 4096 is 2^12 and 8192 is 2^13, so op9986 returns the INDEX OF THE MOST
                # SIGNIFICANT SET BIT - 31 - clz(x), not clz(x). With msb the composition is:
                #
                #     t      = msb(reverse(safe))       = 31 - ctz(safe)
                #     k      = t ^ 31                   = ctz(safe)      (both sides < 32)
                #     result = k | (z << 5)
                #
                # For x != 0 that is ctz(x) with z = 0. For x == 0: safe = 1, reverse(1) is
                # 0x80000000, t = 31, k = 0, and the 32 comes from z << 5. The two terms occupy
                # disjoint bits - k needs five, z << 5 is bit five - so the OR is exactly the
                # addition root wrote.
                #
                # EVERY FORM HERE HAS EXECUTED, in that same sweep:
                #   op11372 icmp     [4096,4096] -> 1, [4097,4096] -> 0   the 0/1 representation
                #   op14047 reverse  4096 -> 524288 (bit 12 to bit 19), 4097 -> 2148007936
                #                    (bits 0 and 12 to bits 31 and 19) - a 32-bit bit reversal,
                #                    exactly, on two informative inputs
                #   op9986  msb      the four values above
                #   op17770 xor imm  4096 -> 4127, which is 0x1000 ^ 0x1F: the immediate is 31,
                #                    the very one this composition needs
                #   op14391 shl imm  4096 -> 2097152 (a nine-bit shift in that probe)
                #   op13575 or reg   the reg-reg form BITWISE_REG_OPCODE already lowers
                #
                # WHAT IS NOT ESTABLISHED, and is not claimed: clz(0) - which this no longer needs -
                # and op9986's value at every point of its range. The sweep measured it at 12 and
                # 13; this composition reaches its endpoints 0 and 31. That is a domain gap in the
                # MEASUREMENT of an existing form, not a guess in this lowering, and it is the
                # first thing a dispatch of these bytes would close.
                args_ = re.findall(r"i32\s+(%?[\w.+-]+)", rhs)
                if not args_:
                    raise Unsupported("a call to air.ctz.i32 with no integer argument")
                if "i1 true" in rhs:
                    raise Unsupported(
                        "a call to air.ctz.i32 with is_zero_poison=TRUE: ctz(0) is undefined there, "
                        "and this composition defines it as 32. Admitting it would make this "
                        "backend's answer differ from another correct one on an input the source "
                        "says nothing about")
                x = operand(args_[0])
                if isinstance(x, ir.Imm):
                    raise Unsupported("a call to air.ctz.i32 on the literal %s: folding it would "
                                      "ask what a compile-time count does, which is not what these "
                                      "instructions do" % args_[0])
                z = b.icmp(x, b.const(0, name="ctz_zero"), rel="eq", name="ctz_is0")
                bit_or = getattr(b, "or")
                safe = bit_or(x, z, name="ctz_safe")
                t = b.msb(b.reverse(safe, name="ctz_rev"), name="ctz_msb")
                k = b.xor(t, ir.Imm(31), name="ctz_k")
                hi = b.shl(z, ir.Imm(5), name="ctz_hi")
                val[dest] = bit_or(k, hi, name="ctz")
                continue
            if fname_ == "air.ctz.i32":
                # REFUSED, WITH THE MISSING FACT NAMED. The call carries `i1 false`, which is
                # LLVM's is_zero_poison=false: ctz(0) is DEFINED and must be 32.
                #
                # This backend's route would be reverse(op14047) then clz(op9986), and that makes
                # ctz(0) equal to clz(0) - a value nothing here has measured. Apple does not take
                # that route either: for this source it emits sub(op11666), op397, op465 and
                # op11363 per call, and none of those four is named in this project's tables, so
                # neither the formula nor the opcodes are available to copy.
                #
                # The one measurement that would close it is clz(0) on this hardware, a single
                # point. That needs a dispatch, which this batch does not do.
                raise Unsupported(
                    "a call to air.ctz.i32 with is_zero_poison=false, so ctz(0) must be 32: the "
                    "only route here is reverse(op14047) then clz(op9986), which makes ctz(0) equal "
                    "to clz(0) - unmeasured. Apple computes it as op11666, op397, op465 and "
                    "op11363, none of them named in this project's tables. The missing fact is "
                    "clz(0), one point, and it needs a dispatch")
            if fname_ in COMPOSED_CALLS and not _NO_U32_TO_F16:
                steps, argc = COMPOSED_CALLS[fname_]
                _cls = r"[\w.-]" if _NO_FP32_LITERALS else r"[\w.+-]"
                args_ = re.findall(r"(?:float|i32|half|double)\s+(%?" + _cls + r"+)", rhs)
                if len(args_) < argc:
                    raise Unsupported("a call to %s with %d arguments" % (fname_, len(args_)))
                v = operand(args_[0])
                if isinstance(v, ir.Imm):
                    # Folding it would ask which rounding a compile-time conversion uses, which is
                    # a different question from what the instructions do. Refused, not folded.
                    raise Unsupported("a call to %s on the literal %s: a compile-time conversion "
                                      "is not the instruction this lowers" % (fname_, args_[0]))
                for step in steps:
                    v = getattr(b, step)(v, name="c")
                val[dest] = v
                continue
            if fname_ in CALLS:
                # `+` IS IN THE CLASS NOW, AND ITS ABSENCE WAS ASYMMETRIC. `[\w.-]` admitted the
                # minus of a negative exponent and not the plus of a positive one, so
                # 5.000000e-01 arrived whole and 1.000000e+00 arrived as '1.000000e'. That is the
                # nastiest possible truncation: for the exponent +00 the dropped text changes
                # nothing, so a lenient parser reads 1.0 and is right by accident - and then reads
                # 1.000000e+03 as 1.0, silently, which syn-s47f3984d60 actually contains. The
                # parser below refuses '1.000000e' outright, and the test keeps that pair.
                _cls = r"[\w.-]" if _NO_FP32_LITERALS else r"[\w.+-]"
                args_ = re.findall(r"(?:float|i32|half|double)\s+(%?" + _cls + r"+)", rhs)
                want = CALLS[fname_]
                if len(args_) < want[1]:
                    raise Unsupported("a call to %s with %d arguments" % (fname_, len(args_)))
                val[dest] = getattr(b, want[0])(*[operand(a_) for a_ in args_[:want[1]]], name="c")
                continue
            raise Unsupported("a call to %s" % (fname_ or "an unnamed function"))

        if head == "fneg" and not _NO_SIGN_OPS:
            # A SIGN-BIT FLIP, NOT A MULTIPLICATION. Root's instruction is the whole design note:
            # "Source-level fneg must preserve its bit semantics, including signed zero; a sign-bit
            # operation is a candidate, multiplication by -1 is not automatically equivalent."
            #
            # xor with 0x80000000 is EXACT for every binary32 encoding - it flips the sign of a
            # zero, of a subnormal, of an infinity and of a NaN while preserving the payload, and it
            # cannot round because it is not arithmetic. Multiplying by -1.0f would be an FMUL: it
            # agrees on normals, and on a NaN it is entitled to return a different NaN, on a
            # subnormal it depends on flush behaviour this backend has not measured, and it consumes
            # a float unit for a bit operation. So the xor is not merely cheaper, it is the only one
            # of the two whose bit semantics are known.
            #
            # The AIR here carries `fast`, which would have LICENSED the looser lowering. The xor
            # meets the STRICT contract as well, so the flag is not relied on - stated because a
            # lowering that happens to be stricter than its licence should say so rather than let a
            # reader assume the licence was needed.
            mm = re.match(r"fneg(?: [a-z]+)* (\S+) (%?\S+)$", rhs)
            if not mm:
                raise Unsupported("fneg %r" % rhs)
            ty, arg = mm.group(1), mm.group(2)
            if ty != "float":
                raise Unsupported("fneg of a %s: the sign bit is at a different position in a "
                                  "narrower float and this front end flips binary32's only" % ty)
            v = operand(arg)
            if isinstance(v, ir.Imm) or getattr(v, "type", None) not in (None, ir.I32):
                raise Unsupported("fneg of %s: the sign flip reads a 32-bit register" % arg)
            val[dest] = getattr(b, "xor")(v, b.const(0x80000000, name="sgn"), name="ng")
            continue

        if head == "ashr" and not _NO_SIGN_OPS:
            # ARITHMETIC shift right: op16805/12 `sar`, which sign-extends. The logical `shr` would
            # be wrong for a negative value, which is the entire point of the AIR distinguishing
            # them - and this front end already maps AIR's `lshr` to the logical one.
            #
            # THE SHIFT-COUNT DOMAIN, stated as root asked. LLVM makes `ashr` POISON when the count
            # is at or above the operand width, so the defined domain is a count in [0, 31]. A
            # constant is checked here and refused outside it. A VARIABLE count refuses: op16806
            # `sarv` exists, but what this hardware does for a count of 32 or more is unmeasured,
            # so a lowering that accepted one could not state its own domain. Both sources in the
            # population shift by the constant 3.
            mm = re.match(r"ashr(?: \w+)* i32 (%?\S+), (%?\S+)$", rhs)
            if not mm:
                raise Unsupported("ashr %r" % rhs)
            a_, n_ = operand(mm.group(1)), operand(mm.group(2))
            if not isinstance(n_, ir.Imm):
                raise Unsupported("ashr by a register: op16806 exists, but a count of 32 or more is "
                                  "poison in AIR and unmeasured here, so this lowering cannot state "
                                  "its domain for one")
            if not 0 <= n_.v <= 31:
                raise Unsupported("ashr by %d: at or above the operand width the AIR result is "
                                  "poison, so there is nothing to preserve" % n_.v)
            if isinstance(a_, ir.Imm):
                a_ = b.const(a_.v, name="c%d" % a_.v)
            val[dest] = b.sar(a_, n_, name="sar")
            continue

        if head == "fcmp" and not _NO_FLOAT_ORDER:
            # AN ORDERED FLOAT COMPARE AS AN INTEGER COMPARE OF ORDER KEYS. See _f32_order_key for
            # why the keys order correctly, how signed zero is canonicalised, and why NaN is not
            # expressible this way - which is exactly why the `fast` flag is REQUIRED rather than
            # preferred: an unrestricted ordered compare must answer false when either operand is
            # NaN, and no integer order can do that.
            mm = re.match(r"fcmp (?:\w+ )*?(\w+) (\S+) (%?\S+), (%?\S+)$", rhs)
            if not mm:
                raise Unsupported("fcmp %r" % rhs)
            relation, ty, left_, right_ = (mm.group(1), mm.group(2), mm.group(3), mm.group(4))
            if ty != "float":
                raise Unsupported(
                    "an %s ordered compare: the order keys are the binary32 patterns, and a "
                    "different width is a different key" % ty)
            if relation not in ("oge", "ogt", "ole", "olt"):
                raise Unsupported(
                    "a %r float compare: only the ORDERED inequalities are keyed this way. An "
                    "equality or an unordered predicate needs semantics the integer order does "
                    "not carry" % relation)
            _fast_math(rhs, "an %r float compare" % relation)
            xs = []
            for token in (left_, right_):
                v = operand(token)
                if isinstance(v, ir.Imm) or getattr(v, "type", None) not in (None, ir.I32):
                    raise Unsupported(
                        "a float compare on %s: the key reads a 32-bit register holding the "
                        "binary32 encoding" % token)
                xs.append(v)
            keys = [_f32_order_key(b, xs[0], ir, "l"), _f32_order_key(b, xs[1], ir, "r")]
            # the ordered inequality becomes the same inequality on the unsigned keys
            wanted = {"oge": "uge", "ogt": "ugt", "ole": "ule", "olt": "ult"}[relation]
            val[dest] = b.icmp(keys[0], keys[1], rel=wanted, name="fo")
            booleans.add(dest)
            continue
        if head == "icmp" and not _NO_INTEGER_COMPARE:
            # THE COMPARISON THAT YIELDS A VALUE, not the one that gates a block. The backend has
            # both and they are different instructions: Builder.icmp is op11372 and produces 0 or 1
            # in a 32-bit register, while Builder.cmp sets the EXEC MASK. AIR's `icmp` in a
            # straight-line kernel is the first kind - its result is consumed by a zext, a select
            # or an arithmetic op, never by this front end as a branch, because a branch refuses.
            #
            # ALL TEN RELATIONS ARE AVAILABLE and only three are measured codes: eq, ult and slt
            # (read off execution on (10,20), (20,10), (10,10) and again on (-5,3), which is what
            # separates signed from unsigned). The rest the backend DERIVES - greater-than by
            # swapping the operands, the "or equal" forms and not-equal by complementing with one
            # xor, which is exact precisely because the value is 0 or 1. So signedness is carried
            # by the relation name rather than dropped: `ult` and `slt` reach different codes.
            mm = re.match(r"icmp (?:\w+ )*?(\w+) (\S+) (%?\S+), (%?\S+)$", rhs)
            if not mm:
                raise Unsupported("icmp %r" % rhs)
            rel, ty, a_, c_ = mm.group(1), mm.group(2), mm.group(3), mm.group(4)
            widen16 = ty == "i16" and not _NO_U16_TO_U32 and not _NARROW_ELEMENTS_AS_WORD
            if ty != "i32" and not widen16:
                raise Unsupported("an %s comparison: this front end compares 32-bit integers, and "
                                  "a narrower one needs a width change - for i16 that is now the "
                                  "measured op10283 zero-first widening, which %s" %
                                  (ty, "is switched off here" if ty == "i16"
                                   else "does not cover this width"))
            # A SIXTEEN-BIT COMPARISON IS THE 32-BIT ONE ON ZERO-EXTENDED OPERANDS.
            #
            # op11372's modelled layout requires all three registers in the 32-bit file, so the
            # sixteen-bit registers cannot be fed to it directly - feeding them would compare the
            # whole words containing them, whose high halves nothing wrote. Zero-extending both
            # first is exact for an EQUALITY on unsigned sixteen-bit values, and for `ult` too,
            # because zero extension preserves unsigned order. A SIGNED sixteen-bit relation is
            # NOT admitted here: `slt` on zero-extended operands compares them as unsigned, which
            # is a different answer for negatives, and the sign-extending fold belongs at the
            # source's own sext rather than being invented inside a comparison.
            if widen16 and rel in ("slt", "sle", "sgt", "sge"):
                raise Unsupported(
                    "a SIGNED sixteen-bit comparison (%s): zero extension preserves unsigned order "
                    "only, and sign-extending inside the comparison would invent a conversion the "
                    "source did not write" % rel)
            if rel not in _ICMP_RELATIONS:
                raise Unsupported("the comparison relation %r: the backend offers %s"
                                  % (rel, ", ".join(sorted(_ICMP_RELATIONS))))
            xs = []
            for t in (a_, c_):
                v = operand(t)
                if widen16 and isinstance(v, ir.Value) and getattr(v, "type", None) is ir.I16:
                    # the waiting widen consumes the load itself; see the zext arm below for the
                    # pair that measures it, and note there is no half-to-half copy here any more
                    v = b.u16_to_u32(v, name="c16")
                # op11372 compares two REGISTERS; an immediate is materialised the way the load
                # and store paths already materialise theirs.
                #
                # AT THE RELATION'S OWN WIDTH, IN TWO'S COMPLEMENT. AIR spells a literal signed -
                # `icmp eq i16 %7, -1` and `icmp ne i32 %8, -1` are both legal - and this passed
                # the Python int through, so materialisation raised "imm -1 is not a 32-bit value":
                # a program that LOWERED and then failed in the backend, which is the worst shape
                # a refusal can take. Root's review found it on the i16 arm; it was never specific
                # to i16, and the width is not cosmetic either. The register operand of an i16
                # comparison has been ZERO-extended, so -1 must become 0xFFFF and not 0xFFFFFFFF;
                # masking everything to 32 bits would compare 65535 against 4294967295 and answer
                # the wrong question. Signed i16 relations are refused above, so the i16 arm only
                # ever needs the unsigned reading, and for i32 the two's-complement bits are what
                # both the signed and the unsigned condition codes are defined on.
                if isinstance(v, ir.Imm):
                    k = v.v & (0xFFFF if ty == "i16" else 0xFFFFFFFF)
                    xs.append(b.const(k, name="k%d" % k))
                else:
                    xs.append(v)
            if dest in branch_conditions:
                # A BRANCH PREDICATE IS NOT A VALUE. op11372 yields a register the program can
                # store; a branch's compare is the immediate form op10369, whose result is mask
                # state - and the IR refuses a br_cond whose predicate did not come from Builder.cmp
                # ("br_cond predicate must come directly from a cmp"). Which one an icmp needs
                # depends on its USE, so the branch conditions were collected before the walk.
                if not isinstance(xs[1], ir.Value) and not isinstance(xs[1], ir.Imm):
                    raise Unsupported("a branch compare against %r" % (xs[1],))
                immediate = None
                second = operand(c_)
                if isinstance(second, ir.Imm):
                    immediate = second.v
                if immediate is None:
                    raise Unsupported(
                        "a branch on a comparison of two registers (%s): the branch's compare is "
                        "the IMMEDIATE form, and a register-register branch compare is not a form "
                        "this front end has" % rhs)
                if not 0 <= immediate <= 0xFF:
                    raise Unsupported("a branch compare against %d: the compare-immediate slot is "
                                      "eight bits" % immediate)
                # THE RELATION IS CHECKED HERE, NOT LEFT TO THE BACKEND. The branch compare's
                # relation field has two RECOVERED values, gt and lt - Apple's compiler never
                # emits >= or <=, so >= and <= reduce to them, and nothing else has been seen in
                # that field. Letting an `eq` branch through would compile the front end's half and
                # fail in cc.py, which the census then records as a BACKEND refusal for a fact the
                # front end already knew.
                # No reduction of `ge`/`le` here: AIR never spells a relation that way (it uses
                # sge/uge/sle/ule), so the arms that used to do it were unreachable and are gone.
                branch_rel, branch_imm = rel, immediate
                if branch_rel not in _cc().CMP_RELATIONS:
                    # THE RELATION THE MASK COMPARE LACKS, TAKEN THROUGH THE VALUE THAT HAS IT.
                    #
                    # I reported this case as a dead end - "composing eq from gt and lt would
                    # combine two predicates' mask state" - and root pointed out that nothing has
                    # to be composed. op11372 already computes the relation AS A VALUE, 0 or 1,
                    # and `eq` is one of its three measured condition codes; a value of 0 or 1 is
                    # greater than zero exactly when the relation held. So one op11372 followed by
                    # ONE immediate compare against 0 gates the block, using only forms that are
                    # already measured:
                    #
                    #     %p = icmp eq %x, 4      op11372 -> 1 when equal, 0 when not
                    #     cmp %p > 0              op10369, the recovered `gt`, sets the mask
                    #
                    # This writes the mask ONCE. It is not two predicates combined, and it is
                    # exact rather than approximate - precisely because the value is 0 or 1, which
                    # is the same property the backend's derived relations already rest on. Branch
                    # polarity is preserved: the mask is true on exactly the original condition.
                    #
                    # AND THIS IS WHY IT IS THE ROUTE FOR EVERY AIR RELATION, NOT A FALLBACK.
                    # AIR spells its predicates with signedness - sgt, ugt, slt, ult - while the
                    # mask compare's two recovered relations are spelled plainly, "gt" and "lt",
                    # with NO recorded signedness for either: nothing in the ledgers or the
                    # execution results says which of the two AIR orderings op10369 implements. So
                    # mapping `sgt` onto it would be an assumption about a field nobody measured,
                    # and it would answer a different question for negative operands. op11372's
                    # `ult` and `slt` ARE separately measured condition codes, so the value form
                    # carries the signedness the source wrote. The plainly-named immediate compare
                    # above still serves a caller that asks for "gt"/"lt" directly - the backend's
                    # own loop lowering does - so that path is untouched, and no program that
                    # already compiled emits a different byte either way.
                    value = b.icmp(xs[0], xs[1], rel=rel, name="pv")
                    val[dest] = b.cmp(value, 0, "gt", name="p")
                    booleans.add(dest)
                    continue
                val[dest] = b.cmp(xs[0], branch_imm, branch_rel, name="p")
                booleans.add(dest)
                continue
            val[dest] = b.icmp(xs[0], xs[1], rel=rel, name="p")
            booleans.add(dest)
            continue

        if head == "select":
            # `select i1 %c, T %x, T %y` where %c is a compare VALUE (0 or 1, op11372): the measured
            # compare-and-select op11375 with its `gt` code against zero - c > 0 ? x : y. NOT its
            # code 1, which is a bit test (ledger/g17-csel-code-one-is-a-bit-test.toml); the first
            # draft used it and chose the wrong arm on every lane. The chosen value keeps its type.
            mm = re.match(r"select (?:\w+ )*?i1 (%[\w.]+), (\S+) (\S+), (\S+) (\S+)$", rhs)
            if not mm or mm.group(2) != mm.group(4):
                raise Unsupported("select %r: the form taken is a 32-bit choice on an i1 value" % rhs)
            if mm.group(2) not in ("float", "i32"):
                raise Unsupported("a select of %s: only 32-bit values are chosen" % mm.group(2))
            if mm.group(1)[1:] not in booleans:
                raise Unsupported("a select on %s, which is not a compare this front end lowered to "
                                  "a value" % mm.group(1))
            regs_ = []
            for token in (mm.group(1), mm.group(3), mm.group(5)):
                v = operand(token)
                if isinstance(v, ir.Imm):
                    v = b.const(v.v, name="sel%d" % len(regs_))
                regs_.append(v)
            c, x, y = regs_
            zero = b.const(0, name="selz")
            val[dest] = b._def("csel", [c, zero, x, y], type=getattr(x, "type", ir.I32), name="sel", rel="gt")
            continue
        if head == "extractelement":
            # `extractelement <3 x i32> %2, i64 0` - the .x of a position builtin
            src = re.search(r"%(\S+),\s*i\d+ (\d+)", rhs)
            if not src:
                raise Unsupported("extractelement %r" % rhs)
            k, lane = src.group(1), int(src.group(2))
            held = _lanes_of(val.get(k))
            if held is not None:
                # A LANE READ OF A VECTOR VALUE. The lane index is constant in AIR here; a
                # dynamic one would be a different operation and refuses below.
                if not 0 <= lane < len(held):
                    raise Unsupported("lane %d of a %d-lane value" % (lane, len(held)))
                if isinstance(held[lane], _Poison):
                    raise Unsupported("extractelement reads lane %d, which is poison: an "
                                      "undefined lane is refused where it is CONSUMED" % lane)
                val[dest] = held[lane]
                continue
            if k not in argmap or argmap[k][0] != "builtin":
                raise Unsupported("extractelement from %%%s, which is not a position builtin" % k)
            if lane not in (0, 1, 2):
                raise Unsupported("component %d of a position builtin" % lane)
            val[dest] = b.builtin(argmap[k][1], name="txyz"[lane + 1] if lane else "t",
                                  axis="xyz"[lane])
        elif head in ("fptrunc", "fpext") and not _NO_HALF_CONVERSIONS:
            # THE TWO HALF CONVERSIONS THE BACKEND ALREADY HAS, and nothing else.
            #
            #     fptrunc float %x to half   ->  f32_to_f16_rte  ->  op1016/12 cvt.f32.f16
            #     fpext   half  %x to float  ->  f16_to_f32      ->  op1004/12 (an fadd.imm of
            #                                                        zero with a 16-bit source)
            #
            # Both lowerings are measured and both appear in the executed half-scan objects
            # (results/g17-half-runtime-executed-33 and g17-half-full-executed carry op1016/12 and
            # op1004/12 in their own bytes), so this arm adds a FRONT END mapping and no new ISA.
            #
            # EVERY OTHER WIDTH PAIR REFUSES BY NAME, because "narrow a float" and "narrow THIS
            # float to THIS width" are different claims and only the second is lowered. double is
            # not representable in this backend at all; bfloat has its own format and is a
            # different instruction; a vector fptrunc is a different form again. The IR's own
            # builders re-check the widths (f32_to_f16_rte demands a 32-bit value, f16_to_f32 a
            # 16-bit one), so a wrong mapping here raises rather than emitting a plausible wrong
            # instruction - but a REFUSAL BY NAME is what the census can read, so it is done here.
            mm = re.match(r"fp(?:trunc|ext) (\S+) (%?\S+) to (\S+)$", rhs)
            if not mm:
                raise Unsupported("%s %r" % (head, rhs))
            src_ty, arg, dst_ty = mm.group(1), mm.group(2), mm.group(3)
            if (head == "fpext" and (src_ty, dst_ty) == ("bfloat", "float")
                    and not _NO_BFLOAT_WIDEN):
                # BFLOAT TO BINARY32 IS A SHIFT, NOT A CONVERSION INSTRUCTION - which is why the
                # comment above says bfloat "is a different instruction" and why this arm does not
                # use one. bfloat16 and binary32 share a sign bit and an 8-bit exponent; bfloat's
                # 7 fraction bits are the top 7 of binary32's 23. So the binary32 pattern is the
                # bfloat pattern in the high half and zeros in the low half, which is exactly
                # `bits << 16`, and it is EXACT with no rounding for every input - normals,
                # subnormals, both zeros, both infinities and every NaN, whose sign and payload
                # ride along unchanged because no bit is examined.
                #
                # Composed from two measured pieces and no new opcode: the halfword load's
                # widening (op10283's zero-first form) puts the sixteen bits in a word, and the
                # word shift (op14391/op17013, whose execution layer is supported) moves them up.
                v = operand(arg)
                if isinstance(v, ir.Imm) or getattr(v, "type", None) is not ir.I16:
                    raise Unsupported(
                        "fpext bfloat on %s: the transport widens a SIXTEEN-bit register holding "
                        "the bfloat pattern, and this operand is not one" % arg)
                val[dest] = b.shl(b.u16_to_u32(v, name="bfw"), ir.Imm(16), name="bf32")
                continue
            want = ("float", "half") if head == "fptrunc" else ("half", "float")
            if (src_ty, dst_ty) != want:
                raise Unsupported("%s from %s to %s: this front end lowers only %s to %s, the two "
                                  "conversions the backend has measured instructions for"
                                  % (head, src_ty, dst_ty, want[0], want[1]))
            # A LITERAL OPERAND IS A CONSTANT FOLD, NOT A CONVERSION: the value would have to be
            # converted at compile time, and which rounding that uses is a separate question from
            # what op1016 does. Refused rather than folded - and checked BEFORE the width, because
            # a float literal is materialised with type F32 while a loaded float is a 32-bit
            # REGISTER (I32), so the width check below would otherwise refuse the literal with a
            # message about widths that says nothing about the actual reason.
            v = operand(arg)
            if isinstance(v, ir.Imm) or getattr(v, "type", None) in (ir.F32, ir.F16):
                raise Unsupported("%s of the literal %s: a compile-time conversion is not the "
                                  "instruction this lowers" % (head, arg))
            # AND THE OPERAND'S WIDTH IS CHECKED HERE SO THE FAILURE IS A REFUSAL AND NOT A CRASH.
            # The IR builders re-check it and raise IRError, which the census records as a
            # TOOLCHAIN failure - indistinguishable from a broken compiler and attributable to
            # nothing. `fpext half` whose value did not come from a sixteen-bit access is the
            # real case: two sources hit exactly that the moment the element width is switched off.
            got = getattr(v, "type", None)
            want_ty = ir.I32 if head == "fptrunc" else ir.I16
            if got != want_ty:
                raise Unsupported("%s of a %s-bit value: %s to %s needs a %s-bit operand, and this "
                                  "one came from an access or an operation of another width"
                                  % (head, str(got).lstrip("if"), src_ty, dst_ty,
                                     str(want_ty).lstrip("if")))
            val[dest] = (b.f32_to_f16_rte(v, name="h") if head == "fptrunc"
                         else b.f16_to_f32(v, name="f"))
        elif head == "zext" and rhs.split()[1] == "i1":
            # `zext i1 %p to i32` AFTER AN icmp IS THE IDENTITY, because op11372 already leaves 0
            # or 1 in a 32-bit register. That is only true of an i1 THIS front end produced: an i1
            # from anywhere else has no defined register representation here, and treating it as a
            # 0/1 word would be the same class of silent width assumption as the three this
            # campaign has already had to undo. So the widening is allowed only for a tracked
            # comparison result and refuses otherwise.
            mm = re.match(r"zext i1 (%\S+) to (\S+)$", rhs)
            if not mm:
                raise Unsupported("zext %r" % rhs)
            if mm.group(1).lstrip("%") not in booleans:
                raise Unsupported("zext of the i1 %s, which did not come from a comparison this "
                                  "front end lowered: an i1 has no register representation here "
                                  "except as a comparison's 0-or-1 result" % mm.group(1))
            if mm.group(2) == "i16" and not _NO_TRUNC_I32_I16:
                # A COMPARISON RESULT STORED AS SIXTEEN BITS. The i1 lives as a 0-or-1 value in a
                # 32-bit register, so its sixteen-bit form is its LOW HALF - exactly `low16`, and
                # exact because the only values are 0 and 1. That is what `h[200] = (ushort)(a == b)`
                # needs, and it is the same truncation the store side already uses.
                val[dest] = b.low16(operand(mm.group(1)), name="b16")
                continue
            if mm.group(2) != "i32":
                raise Unsupported("zext i1 to %s: only the 32-bit widening is the identity here"
                                  % mm.group(2))
            val[dest] = operand(mm.group(1))
        elif head in ("zext", "trunc", "bitcast", "sext"):
            # A PASS-THROUGH IS CORRECT ONLY WHEN THE WIDTH DOES NOT CHANGE IN A REGISTER.
            # `zext i32 %x to i64` is a no-op here because an index is one register either way,
            # and that is why this arm exists. But `sext i16 %x to i32` and `trunc i32 %x to i16`
            # are real operations - Apple emits op10284/12 for the first - and passing them
            # through DROPS them. That is the other half of the narrow-access defect: the four
            # baseline programs lost both their element width and their extension, and the second
            # loss is invisible because the value is already sitting in a 32-bit register.
            mm = re.match(r"(?:zext|trunc|bitcast|sext) (\S+) (%?\S+) to (\S+)$", rhs)
            # A ZEXT TO SIXTY-FOUR BITS ESTABLISHES BOTH WORDS, and that is the one shape whose
            # high word needs no measurement: zero extension makes it zero by definition. The
            # scalar pass-through below is KEPT unchanged, because the same value is what a
            # getelementptr index needs and every source that uses it that way must be unaffected;
            # this only records the pair alongside, for a later wide add, sub or store to read.
            # RECORDED LAZILY, because materialising the zero here MOVES BYTES. `r-shr64a`
            # compiles today and zero-extends to 64 bits without ever needing a high word; an
            # eager `b.const(0)` took it from 100 to 132 bytes, which is precisely the kind of
            # silent change to an accepted program this batch must not make. So the pair is stored
            # as a promise and the zero is created only where a wide operand actually reads it.
            if (mm is not None and head == "zext" and mm.group(3) == "i64"
                    and mm.group(1) == "i32" and not _NO_WIDE_ARITHMETIC
                    and not _NARROW_ELEMENTS_AS_WORD):
                words[dest] = ("zext32", operand(mm.group(2)))
            # A TRUNCATION OF AN ESTABLISHED PAIR IS ITS LOW WORD, and it is handled HERE because
            # this arm runs before the wide-data arm in the dispatch chain - `trunc i64 %x to i32`
            # would otherwise take the pass-through below and ask operand() for a value that only
            # ever existed as two words. That is exactly the "used before this front end defines
            # it" refusal, about a value that WAS defined, in the other half of the file.
            if (mm is not None and head == "trunc" and mm.group(1) == "i64"
                    and mm.group(3) == "i32" and not _NO_WIDE_ARITHMETIC
                    and mm.group(2).lstrip("%") in words):
                val[dest] = _wide_low_word(words[mm.group(2).lstrip("%")])
                continue
            # THE LEGACY ARM KEEPS ITS PASS-THROUGH. `_NARROW_ELEMENTS_AS_WORD` exists to reproduce
            # the four retained WRONG programs byte for byte, and they lost both their element width
            # AND their trunc - the second loss being invisible because the value already sits in a
            # 32-bit register. Lowering the trunc under that arm would make it reproduce something
            # else, and the arm's whole value is that it reproduces those exact bytes: that is how
            # they are known to be wrong rather than merely different.
            if (mm and head == "trunc" and mm.group(1) == "i32" and mm.group(3) == "i16"
                    and not _NO_TRUNC_I32_I16 and not _NARROW_ELEMENTS_AS_WORD):
                # TRUNCATION TO SIXTEEN BITS IS NOW A REAL OPERATION HERE, not a pass-through.
                #
                # `(short)x` and `(ushort)x` of a 32-bit expression keep the low half, and which
                # half 425+n names is established rather than assumed: the imageblock prologue
                # witnesses 425+n as the LOW half of word register n and 281+n as the HIGH half of
                # the same word, and root's retained op10283 measurement confirms the READ view by
                # execution - the observed words match a + low16(b) against a + high16(b) and
                # a + b on three discriminating pairs (results/g17-tensor-width-runtime-v1).
                #
                # ir.low16 lowers to op590/4, the half move whose two file bits select the half on
                # each side independently and whose four combinations were authored and read back;
                # the half-vector packing path already emits it to put components into halves, and
                # that packed layout has a validating hardware receipt.
                #
                # APPLE DOES IT IN ONE INSTRUCTION AND THIS DOES NOT PRETEND OTHERWISE: for
                # `(short)x -> s[i]` Apple selects op17193 at TEN bytes, storing from a word
                # register with no separate truncation at all. This backend has op17193/14, whose
                # value operand names the 425-based file, so it takes the low half explicitly first.
                # A different program computing the same function.
                #
                # EXTENSION IS STILL REFUSED - see below. Truncation needs only the read relation;
                # widening needs either op10284's unmeasured sign behaviour or an op10283 endpoint
                # (a zero first operand) that has not been measured either.
                val[dest] = b.low16(operand(mm.group(2)), name="t16")
                continue
            if (mm and head in ("zext", "sext") and mm.group(1) == "i16" and mm.group(3) == "i32"
                    and not _NO_U16_TO_U32 and not _NARROW_ELEMENTS_AS_WORD):
                x = operand(mm.group(2))
                if not isinstance(x, ir.Value) or getattr(x, "type", None) is not ir.I16:
                    raise Unsupported("%s takes a sixteen-bit value; got %r"
                                      % (head, getattr(x, "type", x)))
                # NO WAITING COPY. This inserted a half-to-half `add(h, 0)` while op10283's
                # half-load readiness was unmeasured; root's results/g17-integer16-half-load-v1
                # pair measures it directly (the wait bit is the only difference between a correct
                # arm and a wrong, unstable one), so the widening consumes the load itself and the
                # back end sets that bit. One instruction fewer, and the readiness now rests on a
                # receipt for THIS form instead of on a neighbouring one.
                wide = b.u16_to_u32(x, name="z16")
                if head == "zext":
                    val[dest] = wide
                else:
                    # sext16(v) = (zext16(v) ^ 0x8000) - 0x8000, the standard fold
                    sign = b.const(0x8000, name="sxsign")
                    val[dest] = b.sub(getattr(b, "xor")(wide, sign, name="sxflip"),
                                      b.const(0x8000, name="sxbias"), name="sx16")
                continue
            # A SAME-WIDTH BITCAST IS NOT A WIDTH CHANGE, and the refusal below is about width.
            #
            # `bitcast half %x to i16` re-spells sixteen bits as sixteen bits. The representation
            # already establishes that this is the identity: a half lives in the 16-bit register
            # file, and the IR's own narrowing `f32_to_f16_rte` yields an I16-typed value - the
            # half and the integer are the SAME register with the same bits. So the value passes
            # through, exactly as the 32-bit bitcasts already do.
            #
            # WHAT THIS DOES NOT ADMIT: any pair whose widths differ. Those still reach the refusal
            # below, which is correct - Apple emits a real instruction for the signed widening and
            # passing one through would silently drop it. Only the same-width pair is the identity,
            # and only because the register file says so rather than because it is convenient.
            if (mm and head == "bitcast" and not _NARROW_ELEMENTS_AS_WORD
                    and not _NO_SCALAR_HALF
                    and {mm.group(1), mm.group(3)} <= {"half", "i16"}
                    and mm.group(1) != mm.group(3)):
                x = operand(mm.group(2))
                if not isinstance(x, ir.Value) or getattr(x, "type", None) is not ir.I16:
                    raise Unsupported(
                        "a same-width bitcast %s to %s on %s: it is the identity only on a "
                        "value already in the sixteen-bit register file, and this operand is %r"
                        % (mm.group(1), mm.group(3), mm.group(2), getattr(x, "type", x)))
                val[dest] = x
                continue
            if mm and not _NARROW_ELEMENTS_AS_WORD:
                a, c = mm.group(1), mm.group(3)
                if "i16" in (a, c) or "i8" in (a, c):
                    raise Unsupported("%s from %s to %s: a 16- or 8-bit integer width change is a "
                                      "real operation here (Apple emits op10284/12 for the signed "
                                      "widening) and this front end has no lowering for it, so "
                                      "passing it through would silently drop it" % (head, a, c))
            # THE VALUE IS THE TOKEN BEFORE `to`, NOT THE THIRD ONE. `rhs.split()[2]` is right for
            # `bitcast i32 %x to float` and WRONG the moment the source type contains a space:
            #
            #     %12 = bitcast i32 addrspace(1)* %11 to float addrspace(1)*
            #
            # splits to [bitcast, i32, addrspace(1)*, %11, ...], so the arm took `addrspace(1)*` as
            # the operand and the kernel refused as `operand 'addrspace(1)*'` - a refusal naming a
            # fragment of a TYPE, which reads like a missing capability and is a parse bug. It
            # blocked mp-fabs.f-1 and mf-hf32_h.copysign-1 across two batches while I recorded it
            # as an unrelated downstream blocker. A POINTER bitcast is the case that exposes it, and
            # passing the pointer through is correct: the address is unchanged and only its element
            # type is re-spelled, which the buffer's own declared type still governs.
            vm = re.match(r"\w+ .*?(%\S+) to \S", rhs)
            if not vm:
                raise Unsupported("%s %r" % (head, rhs))
            source = vm.group(1).lstrip("%")
            if source in argmap and argmap[source][0] == "buffer":
                # A BARE BUFFER ARGUMENT RATHER THAN A VALUE. `bitcast float addrspace(1)* %2 to
                # i32 addrspace(1)*` re-spells the element type of the whole buffer, and %2 is a
                # kernel parameter, so asking operand() for it refused with "used before this front
                # end defines it" - a refusal about a definition for something that was never a
                # definition. The pointee passes through at index zero, exactly as a bare pointer
                # reaching a load does, and the ACCESS width still has to agree with the declared
                # element where the access happens.
                val[dest] = _pointee(source, val, argmap, b)
            else:
                val[dest] = operand(vm.group(1))
        elif head == "getelementptr":
            if tg_global is not None and tg_global in rhs:
                val[dest] = ("tgep", _tg_index(rhs, tg_global, tg_extent, b, operand,
                                              tg_defs, tg_threads, tg_kinds))
                continue
            # THE INDEX CLASS MUST NOT SWALLOW A TRAILING COMMA. `\S+` did, and it only showed up
            # once a getelementptr had something AFTER the index: an atomic's address is
            # `... %0, i64 %5, i32 0`, whose second index is the struct field, so the element index
            # came back as "%5," and resolving it raised "used before this front end defines it"
            # about a value that was defined. An ordinary buffer's gep ends at the index, which is
            # why every source until now matched.
            mm = re.search(r"addrspace\(\d+\)\*\s+%([\w.]+),\s*i\d+ (%?[-\w.]+)", rhs)
            if not mm:
                raise Unsupported("getelementptr %r" % rhs)
            base, idx = mm.group(1), mm.group(2)
            if base not in argmap or argmap[base][0] != "buffer":
                raise Unsupported("getelementptr from %%%s, which is not a kernel buffer" % base)
            val[dest] = ("gep", argmap[base][1], operand(idx))
        elif _wide_int_data(head, rhs, val, argmap, words, wide_results):
            # SIXTY-FOUR-BIT INTEGER DATA, as two explicit 32-bit words. See the composition
            # comment beside BINOPS: the words are reached with the word store this backend
            # already emits, and the carry/borrow are bitwise rather than a flag.
            ir = _ir()

            def pair_of(token):
                """The established (low, high) words of an i64 operand, or a named refusal."""
                token = token.strip().rstrip(",")
                if token.startswith("%") and token.lstrip("%") in words:
                    held = words[token.lstrip("%")]
                    if isinstance(held, tuple) and len(held) == 2 and held[0] == "zext32":
                        # the promise from the zext arm: the high word is zero BY DEFINITION, and
                        # the constant is materialised here, where it is actually read
                        return (_wide_low_word(held), b.const(0, name="whz"))
                    return held
                if re.match(r"^-?\d+$", token):
                    v = int(token) & 0xFFFFFFFFFFFFFFFF
                    return (b.const(v & _WORD_MASK, name="wk"),
                            b.const((v >> 32) & _WORD_MASK, name="wkh"))
                raise Unsupported(
                    "a 64-bit operand %s whose two words this front end has not established: a "
                    "wide value is usable only when BOTH words are known, which a component load, "
                    "a zext from 32 bits, or another wide add/sub establishes. Anything else would "
                    "need a guessed high word" % token)

            if head == "load":
                mm = re.search(r"addrspace\(\d+\)\*\s+%([\w.]+)", rhs)
                g = _pointee(mm.group(1), val, argmap, b) if mm else None
                if not (isinstance(g, tuple) and g[0] == "gep"):
                    raise Unsupported("a 64-bit load from %r, which is not a getelementptr of a "
                                      "buffer" % rhs)
                buf, index = g[1], g[2]
                _wide_declaration(buf, "load")
                if isinstance(index, ir.Imm):
                    index = b.const(index.v, name="k%d" % index.v)
                words[dest] = (b.load_word_component(buf, index, 0, name="lo"),
                               b.load_word_component(buf, index, 1, name="hi"))
                wide_results.add(dest)
                continue
            if head == "store":
                mm = re.search(r"store i64 (%[\w.]+|-?\d+),\s*i64 addrspace\(\d+\)\*\s+%([\w.]+)",
                               rhs if rhs.startswith("store") else "store " + rhs)
                if not mm:
                    raise Unsupported("a 64-bit store this front end cannot read: %r" % rhs)
                g = _pointee(mm.group(2), val, argmap, b)
                if not (isinstance(g, tuple) and g[0] == "gep"):
                    raise Unsupported("a 64-bit store whose address is not a getelementptr of a "
                                      "buffer: %r" % rhs)
                buf, index = g[1], g[2]
                _wide_declaration(buf, "store")
                if isinstance(index, ir.Imm):
                    index = b.const(index.v, name="k%d" % index.v)
                low, high = pair_of(mm.group(1))
                # LOW WORD FIRST, and the order is the contract rather than an accident: the
                # component argument names which half, so a reader does not have to infer it.
                b.store_word_component(buf, index, 0, low)
                b.store_word_component(buf, index, 1, high)
                continue
            if head == "trunc":
                mm = re.match(r"trunc i64 (%\S+) to i32$", rhs)
                if not mm:
                    raise Unsupported("a 64-bit truncation this front end cannot read: %r" % rhs)
                val[dest] = pair_of(mm.group(1))[0]       # the low word IS the truncation
                continue
            mm = re.match(r"(add|sub|or|and|xor|shl|lshr) (?:\w+ )*i64 (\S+), (\S+)$", rhs)
            if not mm:
                raise Unsupported(
                    "a 64-bit %r reading a value whose two words this front end established: the "
                    "pair lowerings are add, sub, or, and, xor and the CONSTANT shifts, and "
                    "letting anything else reach a 32-bit arm would silently compute on the low "
                    "word alone (%s)" % (head, rhs))
            what = mm.group(1)
            left = pair_of(mm.group(2))
            if what in ("shl", "lshr"):
                second = mm.group(3).strip().rstrip(",")
                amount = ir.Imm(int(second)) if re.match(r"^-?\d+$", second) else operand(second)
                words[dest] = _wide_shift_words(b, left, amount,
                                                "shr" if what == "lshr" else "shl")
            else:
                right = pair_of(mm.group(3))
                if what in ("add", "sub"):
                    words[dest] = (_wide_add_words if what == "add" else _wide_sub_words)(
                        b, left, right)
                else:
                    words[dest] = _wide_bitwise_words(b, left, right, what)
            wide_results.add(dest)
        elif head == "load":

            # `load i32, i32 addrspace(1)* %6, align 4` - the pointer is not last on the line
            #
            # THE LOADED TYPE IS IN THE INSTRUCTION AND IT SELECTS THE FORM. A half element is
            # op12646 where a word is op12682, and its destination is a 16-bit register in the
            # file based at 425 rather than the 32-bit one based at 105 (Builder.load). Reading
            # the width from the AIR rather than from the buffer declaration is deliberate: the
            # two must agree, and the cross-check below refuses them when they do not instead of
            # preferring one silently.
            if tg_global is not None and ("addrspace(3)" in rhs or tg_global in rhs):
                # THREADGROUP LOAD. Its result arrives late like any load's, which the backend
                # already knows from the op, so nothing here sets a wait.
                mm = re.search(r"addrspace\(3\)\*\s+%([\w.]+)", rhs)
                if mm:
                    held = val.get(mm.group(1))
                    if not (isinstance(held, tuple) and held[0] == "tgep"):
                        raise Unsupported("a threadgroup load from %%%s, which is not a "
                                          "getelementptr of the threadgroup array" % mm.group(1))
                    index = held[1]
                else:
                    index = _tg_index(rhs, tg_global, tg_extent, b, operand,
                                      tg_defs, tg_threads, tg_kinds)
                ty = _ir().F32 if tg_element == "float" else _ir().I32
                val[dest] = b.load_tg(index, type=ty, name="tg")
                continue
            vec = _vector_access(rhs, "load")
            if vec is not None:
                lanes, lty, lane_width = vec
                mm = re.search(r"addrspace\(\d+\)\*\s+%([\w.]+)", rhs)
                g = _pointee(mm.group(1), val, argmap, b) if mm else None
                if not (isinstance(g, tuple) and g[0] == "gep"):
                    raise Unsupported("a %d-lane load from %r, which is not a getelementptr of a "
                                      "buffer" % (lanes, rhs))
                _lane_declaration(g[1], lanes, lane_width, "load")
                extra = {} if lane_width == "word" else dict(width=lane_width)
                val[dest] = _lane_value(
                    b.load(g[1], _lane_index(b, g[2], lanes, k), name="v%d" % k, **extra)
                    for k in range(lanes))
                continue
            width = _element_width(rhs, "load")
            mm = re.search(r"addrspace\(\d+\)\*\s+%([\w.]+)", rhs)
            g = _pointee(mm.group(1), val, argmap, b) if mm else None
            if not (isinstance(g, tuple) and g[0] == "gep"):
                raise Unsupported("load from %r, which is not a getelementptr of a buffer" % rhs)
            # THE LOAD'S INDEX IS A REGISTER. `h[0]` reaches AIR as a bare pointer, so the
            # index is the constant zero, and handing an Imm to b.load refuses - which was the
            # single biggest backend refusal in the census, 44 of 296 kernels, for a constant this
            # front end can simply materialise.
            idxv = g[2]
            if isinstance(idxv, ir.Imm):
                idxv = b.const(idxv.v, name="k%d" % idxv.v)
            _agree(g[1], width, "load")
            val[dest] = b.load(g[1], idxv, name="v", **({} if width == "word" else dict(width=width)))
        elif head == "store":
            # THE SAME MISSING `+`, AT A SECOND SITE IN THIS FILE. `store float 1.000000e+03, ...`
            # failed the whole match - the class stopped at the `+`, so `, ` never lined up - and the
            # kernel refused as "store %r" rather than as a literal. Fixing the call parser alone
            # left this one refusing unchanged, which is why both are in one edit: a defect in a
            # character class does not travel between two regexes that share it.
            _cls = r"[\w.-]" if _NO_FP32_LITERALS else r"[\w.+-]"
            # AND THE STORED ELEMENT'S WIDTH IS THE FORM TOO - op17193 for a half where op17229 is
            # the word. This was missing, and the first source it mattered for compiled anyway:
            # `store half %x, half addrspace(1)* %p` went through the WORD store, so the program
            # wrote four bytes where the source writes two and clobbered the neighbouring half.
            # It is the same class of defect as the constant-address store below - a form chosen
            # by default rather than by what the source says - and it does not announce itself,
            # because the bytes assemble and decode perfectly well.
            if tg_global is not None and ("addrspace(3)" in rhs or tg_global in rhs):
                sm = re.match(r"store \w+ (\S+), \w+ addrspace\(3\)\*\s*(.*)$", rhs)
                if not sm:
                    raise Unsupported("a threadgroup store this front end cannot read: %r" % rhs)
                value = operand(sm.group(1).rstrip(","))
                if isinstance(value, _ir().Imm):
                    value = b.const(value.v, name="c%d" % value.v)
                pointee = sm.group(2)
                pm = re.match(r"%([\w.]+)", pointee)
                if pm:
                    held = val.get(pm.group(1))
                    if not (isinstance(held, tuple) and held[0] == "tgep"):
                        raise Unsupported("a threadgroup store to %%%s, which is not a "
                                          "getelementptr of the threadgroup array" % pm.group(1))
                    index = held[1]
                else:
                    index = _tg_index(pointee, tg_global, tg_extent, b, operand,
                                      tg_defs, tg_threads, tg_kinds)
                b.store_tg(value, index)
                continue
            vec = _vector_access(rhs, "store")
            if vec is not None:
                lanes, lty, lane_width = vec
                vm = re.match(r"store <\d+ x \w+> (\S+), <\d+ x \w+> addrspace\(\d+\)\*\s+"
                              r"%([\w.]+)", rhs)
                if not vm:
                    raise Unsupported("a %d-lane store %r" % (lanes, rhs))
                g = _pointee(vm.group(2), val, argmap, b)
                if not (isinstance(g, tuple) and g[0] == "gep"):
                    raise Unsupported("a %d-lane store to %r, which is not a getelementptr of a "
                                      "buffer" % (lanes, rhs))
                _lane_declaration(g[1], lanes, lane_width, "store")
                stored = _lanes_of(val.get(vm.group(1).lstrip("%")))
                if stored is None:
                    raise Unsupported("a %d-lane store of %s, which this front end does not hold "
                                      "as lanes" % (lanes, vm.group(1)))
                if len(stored) != lanes:
                    raise Unsupported("a %d-lane store of a %d-lane value"
                                      % (lanes, len(stored)))
                for k, lane in enumerate(stored):
                    if isinstance(lane, _Poison):
                        raise Unsupported("a %d-lane store whose lane %d is poison: an undefined "
                                          "lane is not written rather than written as anything"
                                          % (lanes, k))
                    # A LOADED HALF LANE IS REFUSED, AND THE REASON IS IN THE BACK END.
                    #
                    # cc.py's _wait_for_load copies a stored value that came from a load through an
                    # ALU that waits - measured, and the reason a range store reads its members
                    # after the loads land. But it hardcodes src1_w=1, dest_w=1, srcb_w=1: a
                    # THIRTY-TWO-BIT copy. Given a half in the 425-based file that reads the whole
                    # word containing it, whose high half nothing wrote, and the delivered-byte
                    # interpreter says so - "32-bit read of reg:122 after only its lo half was
                    # written" on the program this arm produced before this refusal existed.
                    #
                    # WHAT THE REPAIR WOULD NEED, stated without overclaiming it. The half-to-half
                    # form IS identified - op10289, harvested from Apple's decoder into
                    # isa/g17-form-opcodes.json as alu.12|12|op=3|mode=1|src1_w=0|srcb_w=1|dest_w=0
                    # - and op10289/12 has an executed record (isa/g17-execution-sweep-results.json,
                    # D10289.l12, cb_status 0, three runs). But all four executed encodings there
                    # carry byte0[3] CLEAR, so the load WAIT has never executed on this opcode;
                    # ledger/g17-alu-load-use-wait.toml settled that bit causally on the WORD form
                    # (`uint x = B[i]` through op12682 and a 32-bit add). Using it here would be
                    # proving-of-an-opcode what was proven of an encoding. So this refuses, and the
                    # concrete thing that would lift it is an executed op10289 with byte0[3] set -
                    # root's to decide, not this front end's to assume.
                    if (lane_width == "half" and isinstance(lane, ir.Value)
                            and getattr(lane, "op", None) is not None
                            and lane.op.kind in ("load", "load_tg")):
                        raise Unsupported(
                            "a %d-lane half store whose lane %d came from a load: the back end's "
                            "load-wait copy (cc.py _wait_for_load) is thirty-two bits wide and "
                            "would read the word containing this half, whose high half nothing "
                            "wrote. The half-to-half form op10289 is identified and executed "
                            "WITHOUT the load wait, so its waiting variant is unmeasured; this "
                            "refuses rather than emit either width on an unproven bit"
                            % (lanes, k))
                    value = lane
                    if isinstance(value, ir.Imm):
                        value = b.const(value.v, name="c%d" % value.v)
                    b.store_at(g[1], _lane_index(b, g[2], lanes, k), value, width=lane_width)
                continue
            width = _element_width(rhs, "store")
            mm = re.match(r"store \w+ (%?" + _cls + r"+), .*?addrspace\(\d+\)\*\s+%([\w.]+)", rhs)
            if not mm:
                raise Unsupported("store %r" % rhs)
            g = _pointee(mm.group(2), val, argmap, b)
            if not (isinstance(g, tuple) and g[0] == "gep"):
                raise Unsupported("store to %r, which is not a getelementptr of a buffer" % rhs)
            # A CONSTANT INDEX IS A DIFFERENT STORE. store_at carries a register and refuses an
            # immediate; `store` carries a slot. Emitting the wrong one made 31 of 296 corpus
            # kernels refuse in the BACKEND for a choice the front end had already got wrong.
            # THE STORED VALUE IS A REGISTER TOO. `C[400] = 1000` hands the IR an immediate and
            # it raises IRError rather than refusing - a crash, not a refusal, which is worse:
            # nine of the first three hundred corpus kernels died there and the refusal census
            # could not attribute them to anything. Materialise it, the way the index already is.
            v = operand(mm.group(1))
            if isinstance(v, ir.Imm):
                v = b.const(v.v, name="c%d" % v.v)
            # A CONSTANT ADDRESS IS MATERIALISED AND GOES THROUGH store_at, WHICH WRITES ONE WORD.
            # The legacy `store` carries a SLOT and its documented semantics are a two-member range
            # store: "writes slot k, RESERVES slot k+1 (=0)". For an authored, explicitly padded IR
            # workload that is the intended API and its delivered bytes depend on it - so `store` is
            # untouched here. But Apple's source says `b1[8] = x` and nothing else, and routing it
            # to `store` CLEARED b1[9], which root's frozen twelve-word reference caught: an
            # unchanged source must leave every other word alive. store_at refuses an Imm index by
            # name, so the constant is materialised exactly as the LOAD path above already does.
            # Root's independent CPU diagnostic on committed code, results/g17-source-admission-v1/
            # scalar-store-control.json: legacy store takes [8, 9] to [2, 0]; store_at with const(8)
            # takes [8] to 2 and leaves word 9 at 123.
            # THE OLD MAPPING STAYS MEASURABLE. A census taken before this change is only
            # comparable to one taken after if the change can be switched off at the point of
            # measurement - the same reason g17cc keeps _NO_BITWISE_ISOLATION.
            idx = g[2]
            if isinstance(idx, ir.Imm) and not _NO_CONSTANT_STORE_AT:
                idx = b.const(idx.v, name="k%d" % idx.v)
            _agree(g[1], width, "store")
            kw = {} if width == "word" else dict(width=width)
            if isinstance(idx, ir.Imm):
                b.store(g[1], idx, v, **kw)
            else:
                b.store_at(g[1], idx, v, **kw)
        elif head == "insertelement" and not _NO_VECTOR_LANES:
            # `insertelement <4 x i32> poison, i32 %7, i64 0` - one lane of an otherwise
            # undefined value. The poison lanes are CARRIED, not invented: the splat idiom below
            # reads lane 0 and never touches them, and anything that does read one refuses.
            im = re.match(r"insertelement <(\d+) x (\w+)> (\S+), \w+ (\S+), i\d+ (\d+)", rhs)
            if not im:
                raise Unsupported("insertelement %r" % rhs)
            lanes, into, what, at = int(im.group(1)), im.group(3).rstrip(","), im.group(4).rstrip(","), int(im.group(5))
            if lanes not in _LANE_COUNTS:
                raise Unsupported("a %d-lane insertelement" % lanes)
            if into in ("poison", "undef"):
                base = [POISON] * lanes
            else:
                base = _lanes_of(val.get(into.lstrip("%")))
                if base is None:
                    raise Unsupported("insertelement into %s, which this front end does not hold "
                                      "as lanes" % into)
            if not 0 <= at < lanes:
                raise Unsupported("insertelement at lane %d of %d" % (at, lanes))
            base = list(base)
            base[at] = operand(what)
            val[dest] = _lane_value(base)
        elif head == "shufflevector" and not _NO_VECTOR_LANES:
            # `shufflevector <4 x i32> %10, <4 x i32> poison, <4 x i32> zeroinitializer` is the
            # splat, and `<i32 1, i32 0>` is a lane SWAP. Both are permutations of the lane list
            # and emit no instruction at all. A mask that is not a constant list of lane numbers
            # refuses: a dynamic shuffle is a different operation and nothing here measures one.
            sm = re.match(r"shufflevector <(\d+) x (\w+)> (\S+), <(\d+) x \w+> (\S+), "
                          r"<(\d+) x i32> (.+?)(?:,\s*!.*)?$", rhs)
            if not sm:
                raise Unsupported("shufflevector %r" % rhs)
            n_a, first, second = int(sm.group(1)), sm.group(3).rstrip(","), sm.group(5).rstrip(",")
            out_lanes, mask_text = int(sm.group(6)), sm.group(7).strip()
            def _side(token):
                if token in ("poison", "undef"):
                    return [POISON] * n_a
                held = _lanes_of(val.get(token.lstrip("%")))
                if held is None:
                    raise Unsupported("shufflevector of %s, which this front end does not hold as "
                                      "lanes" % token)
                return held
            pool = _side(first) + _side(second)
            if mask_text.startswith("zeroinitializer"):
                mask = [0] * out_lanes
            else:
                mm2 = re.match(r"<(.+)>$", mask_text)
                if not mm2:
                    raise Unsupported("shufflevector mask %r, which is not a constant lane list"
                                      % mask_text)
                mask = []
                for part in mm2.group(1).split(","):
                    part = part.strip()
                    if not re.match(r"i32 -?\d+$", part):
                        raise Unsupported("shufflevector mask entry %r: only constant lane "
                                          "numbers are indexed, and undef or a dynamic mask "
                                          "refuses" % part)
                    mask.append(int(part.split()[1]))
            if len(mask) != out_lanes or out_lanes not in _LANE_COUNTS:
                raise Unsupported("a %d-lane shufflevector with a %d-entry mask"
                                  % (out_lanes, len(mask)))
            picked = []
            for position, source_lane in enumerate(mask):
                if not 0 <= source_lane < len(pool):
                    raise Unsupported("shufflevector lane %d reads source lane %d of %d"
                                      % (position, source_lane, len(pool)))
                lane = pool[source_lane]
                if isinstance(lane, _Poison):
                    raise Unsupported("shufflevector lane %d reads source lane %d, which is "
                                      "poison: an undefined lane is refused where it is CONSUMED"
                                      % (position, source_lane))
                picked.append(lane)
            val[dest] = _lane_value(picked)
        elif head in BINOPS:
            # THE ARITHMETIC'S WIDTH IS PART OF THE OPERATION, and AIR states it on the line.
            # `fadd fast half %a, %b` is a BINARY16 add: it rounds to eleven significand bits, and
            # a chain of them is not the same value as a chain of binary32 adds narrowed once at
            # the end. This backend's fadd/fmul are the 32-bit forms, so routing a half add
            # through them changes the result of every program that chains two.
            #
            # FOUR SOURCES IN ROOT'S SIX COMPILED THIS WAY BEFORE THIS CHECK EXISTED, emitting
            # op998/12 - the 32-bit float add - for `fadd fast half`. They compiled, they decoded,
            # and they were wrong: the same shape as the element-width defect one batch ago and the
            # dropped integer extension beside it. Third instance of one class in one campaign.
            #
            # WHAT WOULD LIFT IT: op775 is the measured f16 add-with-immediate (op767 saturating),
            # reachable as `faddi(ty="f16")`, so the immediate case has a form - but these sources
            # also chain REGISTER-REGISTER half adds, for which no form is measured here, and
            # op775's f16 mode keeps its value in the LOW HALF of a 32-bit register rather than in
            # the 16-bit file this front end's half loads and literals use. Mixing the two
            # representations would be a fourth instance of the same class, so both refuse.
            wm = re.match(r"\w+(?: \w+)* (half|double|bfloat) ", rhs)
            if wm and head.startswith("f"):
                if wm.group(1) == "half" and head in _COMPOSED_HALF and not _NO_COMPOSED_HALF_ARITH:
                    # THE FLAGS ARE PART OF THE DOMAIN, and the first version of this arm did not
                    # read them - root's review of c8760654 caught it. Two reasons, and the second
                    # is the one I had wrong:
                    #
                    #   nnan/ninf  the verified domain is FINITE inputs. Root's own checker refuses
                    #              a nonfinite input through op1004 as "outside retained model".
                    #   nsz        THE WIDENING IS NOT A BIT-PRESERVING CAST. op1004 is an fadd of
                    #              +0, so a half -0.0 widens to f32 +0.0 - measured: 0x8000 in,
                    #              0x00000000 out where a cast would give 0x80000000. So strict
                    #              (-0.0h) + (-0.0h) is -0.0 and this composition returns +0.0.
                    #              My exhaustive sweep used numpy's CAST and therefore never tested
                    #              the instruction this lowers to; the corrected sweep uses the
                    #              add-of-zero.
                    #
                    # `fast` implies nnan, ninf and nsz, and every half operation in the admitted
                    # population carries it. An operation without them is the STRICT contract and
                    # is refused rather than approximated.
                    flags = set(re.match(r"\w+((?: [a-z]+)*) (?:half|double|bfloat) ", rhs)
                                .group(1).split())
                    if not ({"fast"} <= flags or {"nnan", "ninf", "nsz"} <= flags):
                        raise Unsupported(
                            "a half-typed %s without the flags its composed lowering needs "
                            "(has %s): the widening op1004 is an fadd of +0, so a half -0.0 "
                            "becomes +0.0 and `nsz` is required; and the verified domain is finite, "
                            "so `nnan` and `ninf` are required. `fast` carries all three. The "
                            "strict contract needs either a bit-preserving widening or a "
                            "measurement of op1004 on -0 and subnormals"
                            % (head, ", ".join(sorted(flags)) or "no flags"))
                    mm = re.match(r"\w+(?: \w+)* \w+ (%?\S+), (%?\S+)$", rhs)
                    if not mm:
                        raise Unsupported("%s %r" % (head, rhs))
                    xs = [operand(mm.group(1)), operand(mm.group(2))]
                    for x in xs:
                        if getattr(x, "type", None) != ir.I16:
                            raise Unsupported(
                                "a half-typed %s of a %s value: both operands must come from a "
                                "16-bit access, literal or half operation, and this one did not"
                                % (head, getattr(x, "type", None)))
                    wide = [b.f16_to_f32(x, name="w") for x in xs]
                    val[dest] = b.f32_to_f16_rte(
                        getattr(b, _COMPOSED_HALF[head])(wide[0], wide[1], name="a"), name="h")
                    continue
                raise Unsupported("a %s-typed %s: this backend's %s is the 32-bit form and no "
                                  "register-register f16 arithmetic form is measured here. "
                                  "%s"
                                  % (wm.group(1), head, head,
                                     "The composed lowering covers half fadd and fmul; half fsub "
                                     "is not run on hardware (fp32 fsub is the add with a negate modifier)"
                                     if wm.group(1) == "half" else
                                     "Only `half` has a composed lowering"))
            vt = _vector_type(rhs[len(head):])
            if vt is not None:
                # ELEMENTWISE, PER LANE, WITH THE SAME MEASURED SCALAR OPERATION. Nothing is
                # composed and nothing is approximated: lane k runs the operation the scalar
                # sources already run, and a head with no scalar arm never reaches here.
                lanes, lty = vt
                if _NO_VECTOR_LANES:
                    raise Unsupported("a %d-lane %s: the vector-lane capability is switched off "
                                      "here" % (lanes, head))
                # A HALF LANE GETS THE SAME REFUSAL A HALF SCALAR GETS. The scalar arm below
                # refuses `fadd fast half` because this backend's fadd is the 32-bit form and no
                # register-register f16 arithmetic form is measured - routing half LANES through
                # it would reintroduce exactly that defect four sources deep in a vector.
                if lty == "half" and head in ("fadd", "fmul"):
                    raise Unsupported("a %d-lane %s of half: this backend's %s is the 32-bit form "
                                      "and no register-register f16 arithmetic form is measured, "
                                      "so half lanes refuse here for the same reason half scalars "
                                      "do" % (lanes, head, head))
                vm = re.match(r"\w+(?: \w+)* <\d+ x \w+> (%[\w.]+|<[^>]*>), (%[\w.]+|<[^>]*>)$", rhs)
                if not vm:
                    raise Unsupported("a %d-lane %s %r" % (lanes, head, rhs))
                sides = []
                for token in (vm.group(1).rstrip(","), vm.group(2)):
                    held = _lanes_of(val.get(token.lstrip("%")))
                    if held is None and token.startswith("<"):
                        # A CONSTANT VECTOR OPERAND, which is how AIR spells `v + 1` on a vector:
                        # `<i32 1, i32 1>`. Each element is read with the scalar parser, so a
                        # literal this front end cannot read refuses there rather than here.
                        inner = re.match(r"<(.+)>$", token)
                        if inner is None:
                            raise Unsupported("a %d-lane %s of the constant %s"
                                              % (lanes, head, token))
                        parts = [p.strip() for p in inner.group(1).split(",")]
                        if len(parts) != lanes:
                            raise Unsupported("a %d-lane %s of a %d-element constant"
                                              % (lanes, head, len(parts)))
                        held = [operand(p.split()[-1]) for p in parts]
                    if held is None:
                        raise Unsupported("a %d-lane %s of %s, which this front end does not hold "
                                          "as lanes" % (lanes, head, token))
                    if len(held) != lanes:
                        raise Unsupported("a %d-lane %s of a %d-lane value"
                                          % (lanes, head, len(held)))
                    sides.append(held)
                for side, token in zip(sides, (vm.group(1), vm.group(2))):
                    for k, lane in enumerate(side):
                        if isinstance(lane, _Poison):
                            raise Unsupported("a %d-lane %s reads lane %d of %s, which is poison"
                                              % (lanes, head, k, token))
                out = []
                for k, (x, y) in enumerate(zip(*sides)):
                    for i, v in enumerate((x, y)):
                        if isinstance(v, ir.Imm) and not 0 <= v.v <= 0xFF:
                            materialised = b.const(v.v & 0xFFFFFFFF, name="k%x" % (v.v & 0xFFFFFFFF))
                            x, y = (materialised, y) if i == 0 else (x, materialised)
                    out.append(_binop(b, head, x, y, "w%d" % k))
                val[dest] = _lane_value(out)
                continue
            mm = re.match(r"\w+(?: \w+)* \w+ (%?\S+), (%?\S+)$", rhs)
            if not mm:
                raise Unsupported("%s %r" % (head, rhs))
            a, c = operand(mm.group(1)), operand(mm.group(2))
            # AN IMMEDIATE THE SLOT CANNOT HOLD BECOMES A REGISTER. The bitwise-immediate form's
            # slot is eight bits, and AIR spells a sign mask as a negative i32 - `and %x,
            # -2147483648` is 0x80000000 - so the backend refused `and immediate -2147483648
            # exceeds the 8-bit slot` and a copysign-shaped source refused with it. Materialising
            # is the same repair the f32->u32 masks needed, applied where AIR supplies the constant
            # instead of where this file writes one.
            #
            # STRICTLY ADDITIVE: only values the slot cannot hold are moved, and those REFUSED
            # before, so no program that compiles today changes a byte. The 59 frozen identities
            # are the check on that claim, not this comment.
            for i, v in enumerate((a, c)):
                if isinstance(v, ir.Imm) and not 0 <= v.v <= 0xFF:
                    materialised = b.const(v.v & 0xFFFFFFFF, name="k%x" % (v.v & 0xFFFFFFFF))
                    if i == 0:
                        a = materialised
                    else:
                        c = materialised
            val[dest] = _binop(b, head, a, c, "w")
        elif head == "br" and ir_blocks:
            two = re.match(r"br i1 %([\w.]+), label %(\d+), label %(\d+)$", rhs)
            if two:
                condition = val.get(two.group(1))
                if condition is None:
                    raise Unsupported("a branch on %%%s, which this front end does not hold"
                                      % two.group(1))
                if two.group(1) not in booleans:
                    raise Unsupported("a branch on %%%s, which did not come from a comparison: a "
                                      "predicate this front end did not produce has no measured "
                                      "compare state" % two.group(1))
                b.br_cond(condition, ir_blocks[two.group(2)], ir_blocks[two.group(3)])
                continue
            one = re.match(r"br label %(\d+)$", rhs)
            if not one:
                raise Unsupported("a branch %r this front end cannot read" % rhs)
            b.br(ir_blocks[one.group(1)])
        elif head == "phi":
            raise Unsupported("a phi: the only sources in this population whose phis are acyclic "
                              "are none - every retained phi sits on a loop back edge - so this "
                              "front end refuses one rather than shipping a lowering no source "
                              "exercises")
        elif head == "ret":
            b.ret()
        else:
            raise Unsupported("AIR instruction %r" % head)
    return fn


def from_metal(path, name=None):
    return to_ir(air_of(path), name)


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return 0
    if argv[0] == "--air":
        air = open(argv[1]).read()
    else:
        air = air_of(argv[0])
    try:
        fn = to_ir(air)
    except Unsupported as ex:
        print("REFUSED: %s" % ex)
        return 2
    print(fn)
    for blk in fn.blocks:
        for op in blk.ops:
            print("   %s" % op)
    return 0


if __name__ == "__main__":
    sys.exit(main())
