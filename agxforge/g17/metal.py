#!/usr/bin/env python3
"""Name opcodes BY CONSTRUCTION: compile Metal that must contain an operation, and count.

Structural naming has saturated. Interpolation adds zero, and 2,897 of the unnamed opcodes sit in
scheduling classes with no named member at all, so there is no anchor for any rule to propagate
from. The peer's execution harness can reach some of them, but their dispatch is the scarcest
thing either of us has and their reach measurement said only 39 of 6,131 were fully controllable
at the time.

This is the third route and it needs no hardware at all: Apple's own compiler is an oracle that
maps source constructs to encodings, and `xcrun metal` is installed. It is how op998 was named
fadd in the first place, and the method generalises.

COUNT SCALING, not presence. Differencing says which opcodes an operation brings in; only scaling
says which of them IS the operation, because a lowering emits plumbing too - address arithmetic,
loads, stores, the store's own conversion. So each probe is compiled three times with the SAME
expression evaluated once, twice and four times over independent operands, and an opcode is part
of the operation only if its count goes k, 2k, 4k. Everything flat is setup.

    c(1)=1 c(2)=2 c(4)=4     the operation, or something it always emits
    c(1)=3 c(2)=3 c(4)=3     prologue, address setup, the store's conversion
    c(1)=1 c(2)=3 c(4)=5     partly shared - not attributable, and reported as such

UNIQUENESS ACROSS PROBES is what turns a scaling set into a name. `fabs(x)` scales the absolute
value AND whatever the float store emits; so does `floor(x)`. An opcode is attributed to a probe
only if it scales in that probe and in NO other probe of the same operand type, which subtracts
the shared lowering without needing to model it.

WHAT THIS CANNOT DO, stated because a naming method that overreaches is worse than a narrow one.
It names what Apple's compiler SELECTS for a construct. If two opcodes are interchangeable and the
compiler always picks one, the other stays unnamed - that is the reason op10272 could not be named
from the corpus despite 371 instances, since Apple emits add+cmp+add for a 64-bit addition rather
than selecting the widening add. Reachability is not frequency, and this method is bounded by
reachability.

    python3 tools/g17metal.py --build          compile every probe (slow, once)
    python3 tools/g17metal.py --name           attribute opcodes and print candidates
"""
import collections, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# ANCHORED ON THE CHECKOUT ROOT: two levels up from agxforge/g17/ where one sufficed from
# tools/. The decoder binary stays in tools/ where the Makefile builds it.
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS = os.path.join(ROOT, "tools")
from agxforge.g17 import target as g17target
from agxforge.g17 import corpus as g17corpus

DIS = os.path.join(TOOLS, "agx3dis")
# Same override as g17corpus.CACHE; kept in step so a corpus built under AGXFORGE_CACHE is
# the corpus every tool reads.
CACHE = os.environ.get("AGXFORGE_CACHE") or os.path.expanduser("~/.cache/agxforge/agx")
COUNTS = os.path.expanduser("~/.cache/agxforge/g17metal-counts.json")
SCALES = (1, 2, 4)

HEAD = """#include <metal_stdlib>
using namespace metal;
kernel void k(device uint *u [[buffer(0)]], device int *s [[buffer(1)]],
              device float *f [[buffer(2)]], device half *h [[buffer(3)]],
              device long *l [[buffer(4)]], device ulong *L [[buffer(5)]],
              device short *w [[buffer(6)]], device ushort *W [[buffer(7)]],
              device float2 *f2 [[buffer(8)]], device float4 *f4 [[buffer(9)]],
              device half2 *h2 [[buffer(10)]], device half4 *h4 [[buffer(11)]],
              device uint2 *u2 [[buffer(12)]], device uint4 *u4 [[buffer(13)]],
              device int4 *i4 [[buffer(14)]], device short2 *w2 [[buffer(15)]],
              device bfloat *bf [[buffer(16)]], device bfloat2 *bf2 [[buffer(17)]],
              device bfloat4 *bf4 [[buffer(18)]],
              uint3 tg [[threadgroup_position_in_grid]]) {
"""

# type tag -> (C++ type, buffer name). Every probe stores its results into its OWN buffer, so the
# store emits no conversion and the only type-dependent plumbing is the store width itself - which
# is shared by every probe of that type and therefore subtracted by the uniqueness filter.
TYPES = {
    "u":  ("uint",   "u"),   "s":  ("int",    "s"),
    "f":  ("float",  "f"),   "h":  ("half",   "h"),
    "l":  ("long",   "l"),   "L":  ("ulong",  "L"),
    "w":  ("short",  "w"),   "W":  ("ushort", "W"),
    "f2": ("float2", "f2"),  "f4": ("float4", "f4"),
    # BFLOAT IS A WHOLE TYPE AXIS THE CORPUS DOES NOT CONTAIN. Apple's compiler accepts it and it
    # is a different width and exponent layout from half, so where a half form has its own opcode
    # a bfloat form very likely does too - and none of them can appear in any probe written so far.
    "bf": ("bfloat", "bf"), "bf2": ("bfloat2", "bf2"), "bf4": ("bfloat4", "bf4"),
    "h2": ("half2",  "h2"),  "h4": ("half4",  "h4"),
    "u2": ("uint2",  "u2"),  "u4": ("uint4",  "u4"),
    "i4": ("int4",   "i4"),  "w2": ("short2", "w2"),
}


def kernel(n, ty, expr):
    """`expr` uses {a} and {b} for two independent operands of the probe's type."""
    decl, buf = TYPES[ty]
    body = []
    for i in range(n):
        a = "%s[tg.x + %du]" % (buf, 2 * i)
        b = "%s[tg.x + %du]" % (buf, 2 * i + 1)
        body.append("  %s v%d = (%s)(%s);" % (decl, i, decl, expr.format(a=a, b=b)))
    for i in range(n):
        body.append("  %s[%d] = v%d;" % (buf, 300 + i, i))
    return HEAD + "\n".join(body) + "\n}\n"


# THE TYPE AXIS IS WHERE THE UNNAMED OPCODES ARE. The corpus is overwhelmingly 32-bit scalar, so
# the half, 16-bit-integer, 64-bit and vector forms of operations whose 32-bit form is named make
# up a large part of what is left - and the biggest nameless scheduling classes have GPR32tup2 and
# GPR16tup2 operands, which is what a 64-bit or two-component value is.
FLOATS = ("f", "h", "f2", "f4", "h2", "h4", "bf", "bf2", "bf4")
INTS = ("u", "s", "w", "W", "l", "L", "u2", "u4", "i4", "w2")
NARROW = ("u", "s", "w", "W", "u2", "u4", "i4", "w2")       # no 64-bit: not every builtin has it
WIDE = ("l", "L")


def probes():
    """{name: (type, expression)}. One construct each, in every type it is defined for."""
    P = {}
    def add(tag, tys, expr):
        for t in tys:
            P["%s.%s" % (tag, t)] = (t, expr)
    for fn in ("fabs", "floor", "ceil", "trunc", "rint", "sqrt", "rsqrt", "exp2", "log2",
               "sin", "cos", "tan", "exp", "log", "fract", "sign", "saturate"):
        add(fn, FLOATS, "%s({a})" % fn)
    for fn in ("fmin", "fmax", "fmod", "pow", "atan2", "step", "copysign", "ldexp", "powr"):
        add(fn, FLOATS, "%s({a}, {b})" % fn)
    add("fdiv", FLOATS, "{a} / {b}")
    add("fadd", FLOATS, "{a} + {b}")
    add("fsub", FLOATS, "{a} - {b}")
    add("fmul", FLOATS, "{a} * {b}")
    add("fma", FLOATS, "fma({a}, {b}, {a})")
    add("mix", FLOATS, "mix({a}, {b}, {a})")
    add("smoothstep", FLOATS, "smoothstep({a}, {b}, {a})")
    add("clampf", FLOATS, "clamp({a}, {b}, {b})")
    add("rcp", FLOATS, "1.0f / {a}")
    add("fneg", FLOATS, "-{a}")
    # geometric and vector-only, which is the only route to several tuple forms
    add("dot", ("f2", "f4", "h2", "h4", "bf2", "bf4"), "dot({a}, {b})")
    add("length", ("f2", "f4", "h2", "h4", "bf2", "bf4"), "length({a})")
    add("normalize", ("f2", "f4", "h2", "h4", "bf2", "bf4"), "normalize({a})")
    add("distance", ("f2", "f4", "h2", "h4", "bf2", "bf4"), "distance({a}, {b})")
    add("cross", ("f4",), "float4(cross({a}.xyz, {b}.xyz), 0.0f)")
    add("reverse", ("f4", "u4", "i4", "h4"), "{a}.wzyx")
    add("swizzle", ("f4", "u4", "i4", "h4"), "{a}.xxzz")
    # integer
    for fn in ("abs", "clz", "ctz", "popcount"):
        add(fn, INTS, "%s({a})" % fn)
    add("reverse_bits", NARROW, "reverse_bits({a})")
    for fn in ("min", "max", "absdiff", "addsat", "subsat", "hadd", "rhadd"):
        add(fn, INTS, "%s({a}, {b})" % fn)
    add("rotate", NARROW, "rotate({a}, {b})")
    add("mulhi", NARROW, "mulhi({a}, {b})")
    add("madhi", NARROW, "mulhi({a}, {b}) + {a}")
    add("iadd", INTS, "{a} + {b}")
    add("isub", INTS, "{a} - {b}")
    add("imul", INTS, "{a} * {b}")
    add("idiv", INTS, "{a} / {b}")
    add("imod", INTS, "{a} % {b}")
    add("shl", INTS, "{a} << ({b} & 15)")
    add("shr", INTS, "{a} >> ({b} & 15)")
    add("and", INTS, "{a} & {b}")
    add("or", INTS, "{a} | {b}")
    add("xor", INTS, "{a} ^ {b}")
    add("andn", INTS, "{a} & ~{b}")
    add("orn", INTS, "{a} | ~{b}")
    add("nand", INTS, "~({a} & {b})")
    add("nor", INTS, "~({a} | {b})")
    add("xnor", INTS, "~({a} ^ {b})")
    add("not", INTS, "~{a}")
    add("neg", ("s", "w", "l", "i4", "w2"), "-{a}")
    add("madd", INTS, "{a} * {b} + {a}")
    add("clampi", INTS, "clamp({a}, {b}, {b})")
    add("extract_bits", ("u", "s"), "extract_bits({a}, 3u, 5u)")
    add("insert_bits", ("u", "s"), "insert_bits({a}, {b}, 3u, 5u)")
    add("select", INTS, "select({a}, {b}, {a} > {b})")
    add("cmplt", INTS, "({a} < {b}) ? {a} : {b}")
    add("cmpeq", INTS, "({a} == {b}) ? {a} : {b}")
    # cross-lane
    for fn in ("simd_sum", "simd_product", "simd_min", "simd_max",
               "simd_prefix_exclusive_sum", "simd_prefix_inclusive_sum", "simd_broadcast_first"):
        add(fn, ("u", "s", "f", "h", "l"), "%s({a})" % fn)
    for fn in ("simd_and", "simd_or", "simd_xor"):
        add(fn, ("u", "s"), "%s({a})" % fn)
    for fn in ("quad_sum", "quad_product", "quad_min", "quad_max"):
        add(fn, ("u", "s", "f", "h"), "%s({a})" % fn)
    add("simd_shuffle_xor", ("u", "f", "h"), "simd_shuffle_xor({a}, 1u)")
    # THE VOTE AND BROADCAST BLOCK, and quad_broadcast in particular: this table carried thirteen
    # opcodes named quad.broadcast0 or quad.broadcast3 which turned out to be the 16-bit forms of
    # the min, max and or reductions. What the REAL quad_broadcast compiles to was never probed.
    add("quad_broadcast", ("u", "s", "f", "h"), "quad_broadcast({a}, 1u)")
    add("quad_shuffle", ("u", "f", "h"), "quad_shuffle({a}, 2u)")
    add("quad_shuffle_down", ("u", "f", "h"), "quad_shuffle_down({a}, 1u)")
    add("quad_shuffle_up", ("u", "f", "h"), "quad_shuffle_up({a}, 1u)")
    add("simd_broadcast", ("u", "s", "f", "h"), "simd_broadcast({a}, 3u)")
    add("simd_shuffle", ("u", "f", "h"), "simd_shuffle({a}, 5u)")
    add("simd_shuffle_and_fill_up", ("u", "f"), "simd_shuffle_and_fill_up({a}, {b}, 1u)")
    add("simd_shuffle_and_fill_down", ("u", "f"), "simd_shuffle_and_fill_down({a}, {b}, 1u)")
    add("simd_all", ("u",), "(uint)simd_all({a} > {b})")
    add("simd_any", ("u",), "(uint)simd_any({a} > {b})")
    add("quad_all", ("u",), "(uint)quad_all({a} > {b})")
    add("quad_any", ("u",), "(uint)quad_any({a} > {b})")
    add("simd_active_mask", ("u",), "(uint)((ulong)simd_active_threads_mask())")
    add("simd_prefix_product", ("u", "f"), "simd_prefix_exclusive_product({a})")
    # MORE OF THE MATH LIBRARY. Each of these is a distinct entry point and any of them may have
    # its own opcode where the ones already probed do not.
    for fn in ("sinh", "cosh", "tanh", "asin", "acos", "atan", "asinh", "acosh", "atanh",
               "exp10", "log10", "cbrt", "rsqrt", "sinpi", "cospi", "tanpi", "erf", "erfc"):
        add(fn, ("f", "h"), "%s({a})" % fn)
    for fn in ("hypot", "fdim", "remainder", "nextafter", "maxmag", "minmag"):
        add(fn, ("f", "h"), "%s({a}, {b})" % fn)
    add("fma3_max", ("f", "h"), "fmax3({a}, {b}, {a})")
    add("fma3_min", ("f", "h"), "fmin3({a}, {b}, {a})")
    add("median3", ("f", "h"), "median3({a}, {b}, {a})")
    add("frexp", ("f", "h"), "frexp({a}, *(thread int *)nullptr)" if False else "{a}")
    # THE fast:: AND precise:: NAMESPACES select different lowerings for the same function, which
    # is exactly the kind of thing that has its own opcode.
    for fn in ("divide", "sqrt", "rsqrt", "sin", "cos", "tan", "exp", "exp2", "log", "log2",
               "normalize", "length", "distance"):
        arity2 = fn in ("divide", "distance")
        e = "fast::%s({a}, {b})" % fn if arity2 else "fast::%s({a})" % fn
        add("fast_%s" % fn, ("f",), e)
        e2 = "precise::%s({a}, {b})" % fn if arity2 else "precise::%s({a})" % fn
        add("precise_%s" % fn, ("f",), e2)
    add("simd_shuffle_up", ("u", "f", "h"), "simd_shuffle_up({a}, 1u)")
    add("simd_shuffle_down", ("u", "f", "h"), "simd_shuffle_down({a}, 1u)")
    add("quad_shuffle_xor", ("u", "f", "h"), "quad_shuffle_xor({a}, 1u)")
    add("simd_is_first", ("u",), "(uint)simd_is_first()")
    # CONVERSIONS, one direction per probe so the scaling set is the conversion itself. Every
    # narrowing and widening pair the language admits, because a conversion opcode has no other
    # route into a probe.
    CV = [("f", "float", "s", "int"), ("f", "float", "u", "uint"), ("f", "float", "h", "half"),
          ("f", "float", "l", "long"), ("h", "half", "s", "int"), ("h", "half", "u", "uint"),
          ("h", "half", "w", "short"), ("s", "int", "f", "float"), ("s", "int", "h", "half"),
          ("s", "int", "l", "long"), ("s", "int", "w", "short"), ("u", "uint", "f", "float"),
          ("u", "uint", "h", "half"), ("u", "uint", "L", "ulong"), ("u", "uint", "W", "ushort"),
          ("f", "float", "bf", "bfloat"), ("bf", "bfloat", "f", "float"),
          ("h", "half", "bf", "bfloat"), ("bf", "bfloat", "h", "half"),
          ("u", "uint", "bf", "bfloat"), ("bf", "bfloat", "u", "uint"),
          ("l", "long", "f", "float"), ("l", "long", "s", "int"), ("w", "short", "s", "int"),
          ("w", "short", "f", "float"), ("W", "ushort", "u", "uint")]
    for st, sn, dt, dn in CV:
        P["cvt.%s2%s.%s" % (sn, dn, dt)] = (dt, "(%s)(%s[tg.x])" % (dn, TYPES[st][1]))
    return P


def opcodes(tag):
    """Multiset of opcodes in a built object, main plus constant program."""
    from agxforge.g17 import machobj, agxdis, ref as g17ref
    g17ref.binary()
    d = os.path.join(CACHE, tag)
    arc, obj = d + "/s.arc.metallib", d + "/out/object/0-0"
    if not (os.path.exists(arc) and os.path.exists(obj)):
        return None
    try:
        loc = machobj.locate(arc, obj)
        fo, sz = agxdis.sections(loc["obj"])
        t = bytes(loc["obj"][fo:fo + sz])
        e = loc["syms"]["_agc.main"]
    except Exception:
        return None
    cp = loc["syms"].get("_agc.main.constant_program")
    spans = [(e, len(t))]
    if cp is not None and cp < e:
        spans.append((cp, e))
    out = collections.Counter()
    with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
        fh.write(t); fh.flush()
        for start, stop in spans:
            r = subprocess.run([DIS, fh.name, str(start), str(stop - start), "--pc", str(start)],
                               capture_output=True, text=True)
            for line in r.stdout.splitlines():
                p = line.split()
                if len(p) >= 3 and p[1] != "bad":
                    try: out[int(p[2])] += 1
                    except ValueError: pass
    return out


def build():
    P = probes()
    ok = fail = 0
    msgs = collections.Counter()
    for name, (ty, expr) in sorted(P.items()):
        for n in SCALES:
            tag = "mp-%s-%d" % (name, n)
            r = g17corpus.build(tag, kernel(n, ty, expr))
            if r in ("built", "cached"):
                ok += 1
            else:
                fail += 1; msgs[name] = r
    print("probe objects built or cached: %d   failed: %d" % (ok, fail))
    for k, v in list(msgs.items())[:12]:
        print("   %-26s %s" % (k, v[:80]))
    return msgs


def all_probes():
    """{name: (uniqueness group, source)} over both families.

    The uniqueness group is the operand type for the expression family. The special family gets
    one group of its own: an atomic add and a texture sample share no lowering, so requiring an
    opcode to scale in exactly one of the 48 special probes is the same test in a coarser space.
    """
    P = {k: (t, e) for k, (t, e) in probes().items()}
    for k, body in special_probes().items():
        P["sp:" + k] = ("special", body)
    for k, (grp, body) in stage_probes().items():
        P[k] = (grp, body)
    for k, (grp, body) in rt_probes().items():
        P[k] = (grp, body)
    for k, (grp, body) in tg_probes().items():
        P[k] = (grp, body)
    for k, (grp, body) in tx_probes().items():
        P[k] = (grp, body)
    for k, (grp, body) in hx_probes().items():
        P[k] = (grp, body)
    for k, (grp, body) in fa_probes().items():
        P[k] = (grp, body)
    for k, (grp, body) in fi_probes().items():
        P[k] = (grp, body)
    for k, (grp, body) in in_probes().items():
        P[k] = (grp, body)
    for k, (grp, body) in mx_probes().items():
        P[k] = (grp, body)
    for k, (grp, body) in pv_probes().items():
        P[k] = (grp, body)
    for k, (grp, body) in ib_probes().items():
        P[k] = (grp, body)
    for k, (grp, body) in cs_probes().items():
        P[k] = (grp, body)
    for k, (grp, body) in uf_probes().items():
        P[k] = (grp, body)
    return P


def _tag(name, n):
    if name.startswith("sp:"):
        return "ms-%s-%d" % (name[3:], n)
    if name.startswith(("fs:", "vs:")):
        return "mg-%s-%d" % (name.replace(":", "_"), n)
    if name.startswith("rt:"):
        return "mr-%s-%d" % (name[3:], n)
    if name.startswith("tg:"):
        return "mt-%s-%d" % (name[3:], n)
    if name.startswith("tx:"):
        return "mx-%s-%d" % (name[3:], n)
    if name.startswith("hx:"):
        return "mh-%s-%d" % (name[3:], n)
    if name.startswith("fa:"):
        return "mf-%s-%d" % (name[3:], n)
    if name.startswith("fi:"):
        return "mi-%s-%d" % (name[3:], n)
    if name.startswith("in:"):
        return "mn-%s-%d" % (name[3:], n)
    if name.startswith("mx:"):
        return "mm-%s-%d" % (name[3:], n)
    if name.startswith("pv:"):
        return "mv-%s-%d" % (name[3:], n)
    if name.startswith("ib:"):
        return "mb-%s-%d" % (name[3:], n)
    if name.startswith("cs:"):
        return "mc-%s-%d" % (name[3:], n)
    if name.startswith("uf:"):
        return "mu-%s-%d" % (name[3:], n)
    return "mp-%s-%d" % (name, n)


def counts(refresh=False):
    if not refresh and os.path.exists(COUNTS):
        raw = json.load(open(COUNTS))
        return {k: {int(n): collections.Counter({int(o): c for o, c in v.items()})
                    for n, v in d.items()} for k, d in raw.items()}
    out = {}
    for name in sorted(all_probes()):
        d = {}
        for n in SCALES:
            c = opcodes(_tag(name, n))
            if c is not None:
                d[n] = c
        if len(d) == len(SCALES):
            out[name] = d
    os.makedirs(os.path.dirname(COUNTS), exist_ok=True)
    json.dump({k: {str(n): {str(o): c for o, c in v.items()} for n, v in d.items()}
               for k, d in out.items()}, open(COUNTS, "w"))
    return out


def scaling(d):
    """Opcodes whose count is exactly proportional to the number of copies."""
    out = {}
    for o, c1 in d[1].items():
        if c1 <= 0:
            continue
        if all(d[n].get(o, 0) == c1 * n for n in SCALES):
            out[o] = c1
    return out


def name(refresh=False):
    C = counts(refresh)
    P = all_probes()
    print("probes with all %d scales built: %d of %d" % (len(SCALES), len(C), len(P)))
    S = {k: scaling(d) for k, d in C.items()}
    bytype = collections.defaultdict(list)
    for k in S:
        bytype[P[k][0]].append(k)
    from agxforge.g17 import slice as g17slice
    named = g17slice.KNOWN_OPS
    hits = {}
    for ty, ks in sorted(bytype.items()):
        for k in ks:
            others = set()
            for k2 in ks:
                if k2 != k:
                    others |= set(S[k2])
            uniq = sorted(set(S[k]) - others)
            if uniq:
                hits[k] = uniq
    print("\nprobes with an opcode that scales in them and in NO other probe of the same type:")
    fresh = 0
    for k in sorted(hits):
        u = hits[k]
        tags = ["op%d%s" % (o, "" if o not in named else "=" + named[o]) for o in u]
        new = [o for o in u if o not in named]
        fresh += len(new)
        print("   %-30s %s" % (k, " ".join(tags[:6])))
    print("\nuniquely attributed opcodes: %d, of which UNNAMED: %d"
          % (sum(len(v) for v in hits.values()), fresh))
    out = os.path.join(HERE, "..", "isa", "g17-metal-attribution.jsonl")
    with open(out, "w") as fh:
        fh.write(json.dumps({"_note": "Opcodes attributed to a Metal construct by COUNT SCALING - "
                 "the expression is compiled once, twice and four times and only opcodes whose "
                 "count scales exactly are the operation. `unique` opcodes scale in this probe and "
                 "in no other probe of the same operand type, which subtracts the shared lowering. "
                 "This names what Apple SELECTS for a construct; an opcode the compiler never "
                 "chooses stays unnamed however often it appears elsewhere."},
                 separators=(",", ":")) + "\n")
        for k in sorted(S):
            fh.write(json.dumps({"probe": k, "type": P[k][0], "expr": P[k][1],
                                 "scaling": {str(o): c for o, c in sorted(S[k].items())},
                                 "unique": hits.get(k, [])}, separators=(",", ":")) + "\n")
    print("wrote %s" % out)
    return hits


# ---------------------------------------------------------------------------------------------
# THE SECOND FAMILY: constructs that are not an expression over two buffer loads. Atomics, texture
# access, threadgroup memory, barriers, the simdgroup matrix block and control flow each need
# their own kernel shape, and between them they are most of the Metal language that the first
# family cannot express. They matter because the first family reached only 259 opcodes: Apple's
# compiler has emitted just 417 of the 6,718 admitted opcodes across the entire corpus AND every
# probe, so the whole naming problem is bounded by what the compiler can be made to select, and
# every construct not yet probed is a piece of that bound.
SPECIAL_HEAD = """#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void k(device uint *u [[buffer(0)]], device int *s [[buffer(1)]],
              device float *f [[buffer(2)]], device half *h [[buffer(3)]],
              device atomic_uint *au [[buffer(4)]], device atomic_int *ai [[buffer(5)]],
              texture2d<float, access::sample> t2 [[texture(0)]],
              texture2d<float, access::write> tw [[texture(1)]],
              texture2d<half, access::sample> th [[texture(2)]],
              texture3d<float, access::sample> t3 [[texture(3)]],
              texture2d_array<float, access::sample> ta [[texture(4)]],
              depth2d<float, access::sample> td [[texture(5)]],
              sampler sm [[sampler(0)]],
              uint3 tg [[threadgroup_position_in_grid]],
              uint3 tp [[thread_position_in_threadgroup]],
              uint sl [[thread_index_in_simdgroup]]) {
  threadgroup uint tgm[256];
  threadgroup float tgf[256];
"""


def special(n, body):
    """`body` is a format string using {i} for the copy index."""
    return SPECIAL_HEAD + "\n".join(body.format(i=i) for i in range(n)) + "\n}\n"


def special_probes():
    P = {}
    A = ("relaxed", "memory_order_relaxed")
    for op in ("add", "sub", "and", "or", "xor", "min", "max"):
        P["atomic_%s.u" % op] = (
            "  atomic_fetch_%s_explicit(&au[tg.x + {i}u], u[{i}], memory_order_relaxed);" % op)
        P["atomic_%s.s" % op] = (
            "  atomic_fetch_%s_explicit(&ai[tg.x + {i}u], s[{i}], memory_order_relaxed);" % op)
    P["atomic_store.u"] = "  atomic_store_explicit(&au[tg.x + {i}u], u[{i}], memory_order_relaxed);"
    P["atomic_load.u"] = "  u[400 + {i}] = atomic_load_explicit(&au[tg.x + {i}u], memory_order_relaxed);"
    P["atomic_xchg.u"] = "  u[400 + {i}] = atomic_exchange_explicit(&au[tg.x + {i}u], u[{i}], memory_order_relaxed);"
    P["atomic_cmpxchg.u"] = ("  {{ uint e{i} = u[{i}]; atomic_compare_exchange_weak_explicit("
                             "&au[tg.x + {i}u], &e{i}, 7u, memory_order_relaxed, "
                             "memory_order_relaxed); u[400 + {i}] = e{i}; }}")
    # threadgroup memory and barriers
    P["tgmem.store"] = "  tgm[tp.x + {i}u] = u[{i}];"
    P["tgmem.load"] = "  u[400 + {i}] = tgm[tp.x + {i}u];"
    P["tgmem.float"] = "  tgf[tp.x + {i}u] = f[{i}]; f[400 + {i}] = tgf[tp.x + {i}u + 1u];"
    P["barrier.tg"] = "  threadgroup_barrier(mem_flags::mem_threadgroup); u[400 + {i}] = tgm[{i}];"
    P["barrier.dev"] = "  threadgroup_barrier(mem_flags::mem_device); u[400 + {i}] = u[{i}];"
    P["simdbarrier"] = "  simdgroup_barrier(mem_flags::mem_threadgroup); u[400 + {i}] = tgm[{i}];"
    # textures - the whole sampling and access block, which the corpus barely contains
    P["tex.sample"] = "  f4store({i}, t2.sample(sm, float2(f[{i}], f[{i} + 1])));"
    P["tex.read"] = "  f4store({i}, t2.read(uint2(u[{i}], u[{i} + 1])));"
    P["tex.write"] = "  tw.write(float4(f[{i}]), uint2(u[{i}], u[{i} + 1]));"
    P["tex.gather"] = "  f4store({i}, t2.gather(sm, float2(f[{i}], f[{i} + 1])));"
    P["tex.sample_lod"] = "  f4store({i}, t2.sample(sm, float2(f[{i}], f[{i}+1]), level(2.0f)));"
    P["tex.sample_bias"] = "  f4store({i}, t2.sample(sm, float2(f[{i}], f[{i}+1]), bias(0.5f)));"
    P["tex.sample_grad"] = ("  f4store({i}, t2.sample(sm, float2(f[{i}], f[{i}+1]), "
                            "gradient2d(float2(0.1f), float2(0.1f))));")
    P["tex.half"] = "  f[400 + {i}] = (float)th.sample(sm, float2(f[{i}], f[{i} + 1])).x;"
    P["tex.3d"] = "  f4store({i}, t3.sample(sm, float3(f[{i}], f[{i}+1], f[{i}+2])));"
    P["tex.array"] = "  f4store({i}, ta.sample(sm, float2(f[{i}], f[{i}+1]), {i}u));"
    P["tex.depth"] = "  f[400 + {i}] = td.sample(sm, float2(f[{i}], f[{i} + 1]));"
    P["tex.depth_cmp"] = ("  f[400 + {i}] = td.sample_compare(sm, float2(f[{i}], f[{i}+1]), "
                          "0.5f);")
    P["tex.size"] = "  u[400 + {i}] = t2.get_width({i}u) + t2.get_height({i}u);"
    # control flow
    P["branch.if"] = "  if (u[{i}] > 3u) {{ u[400 + {i}] = u[{i}] + 1u; }} else {{ u[400 + {i}] = 2u; }}"
    P["branch.loop"] = ("  {{ uint a{i} = 0u; for (uint j = 0u; j < u[{i}]; ++j) a{i} += j; "
                        "u[400 + {i}] = a{i}; }}")
    P["branch.while"] = ("  {{ uint a{i} = u[{i}]; while (a{i} > 1u) a{i} >>= 1; "
                         "u[400 + {i}] = a{i}; }}")
    P["branch.switch"] = ("  switch (u[{i}] & 3u) {{ case 0: u[400+{i}] = 1u; break; "
                          "case 1: u[400+{i}] = 2u; break; default: u[400+{i}] = 3u; }}")
    # simdgroup matrix - the tensor block
    P["sgmatrix.f32"] = ("  {{ simdgroup_float8x8 a{i}, b{i}, c{i}; "
                         "simdgroup_load(a{i}, f + {i} * 64); simdgroup_load(b{i}, f + 512); "
                         "simdgroup_multiply_accumulate(c{i}, a{i}, b{i}, c{i}); "
                         "simdgroup_store(c{i}, f + 1024 + {i} * 64); }}")
    P["sgmatrix.f16"] = ("  {{ simdgroup_half8x8 a{i}, b{i}, c{i}; "
                         "simdgroup_load(a{i}, h + {i} * 64); simdgroup_load(b{i}, h + 512); "
                         "simdgroup_multiply_accumulate(c{i}, a{i}, b{i}, c{i}); "
                         "simdgroup_store(c{i}, h + 1024 + {i} * 64); }}")
    P["sgmatrix.mul"] = ("  {{ simdgroup_float8x8 a{i}, b{i}, c{i}; "
                         "simdgroup_load(a{i}, f + {i} * 64); simdgroup_load(b{i}, f + 512); "
                         "c{i} = simdgroup_multiply(a{i}, b{i}); "
                         "simdgroup_store(c{i}, f + 1024 + {i} * 64); }}")
    # address spaces and pointer forms
    P["addr.constant"] = "  u[400 + {i}] = u[tg.x * 4u + {i}u];"
    P["addr.strided"] = "  u[400 + {i}] = u[tg.x * u[{i}] + {i}u];"
    P["addr.long"] = "  u[400 + {i}] = u[(ulong)tg.x * 8ul + {i}ul];"
    P["pack.uchar4"] = ("  {{ uchar4 p{i} = as_type<uchar4>(u[{i}]); "
                        "u[400 + {i}] = (uint)(p{i}.x + p{i}.w); }}")
    P["pack.ushort2"] = ("  {{ ushort2 p{i} = as_type<ushort2>(u[{i}]); "
                         "u[400 + {i}] = (uint)(p{i}.x + p{i}.y); }}")
    return P


F4STORE = ("static inline void f4store_impl(device float *f, uint i, float4 v) {\n"
           "  f[400 + i * 4] = v.x + v.y + v.z + v.w;\n}\n")


def special_source(name, n):
    body = special_probes()[name]
    src = special(n, body)
    if "f4store(" in src:
        src = src.replace("f4store({0}".format("").join([]), "")
        src = src.replace("  f4store(", "  f4store_helper(f, ")
        src = src.replace("using namespace metal;",
                          "using namespace metal;\nstatic inline void f4store_helper("
                          "device float *f, uint i, float4 v) { f[400 + i] = "
                          "v.x + v.y + v.z + v.w; }")
    return src


def build_special():
    P = special_probes()
    ok = fail = 0
    msgs = {}
    for name in sorted(P):
        for n in SCALES:
            r = g17corpus.build("ms-%s-%d" % (name, n), special_source(name, n))
            if r in ("built", "cached"): ok += 1
            else:
                fail += 1; msgs[name] = r
    print("special probe objects built or cached: %d   failed: %d" % (ok, fail))
    for k, v in sorted(msgs.items()):
        print("   %-22s %s" % (k, v[:78]))
    return msgs


# ---------------------------------------------------------------------------------------------
# THE THIRD FAMILY: FRAGMENT AND VERTEX STAGES. Everything above is a compute kernel, and a kernel
# cannot express the parts of this GPU that only a raster pipeline uses - screen-space derivatives,
# attribute interpolation, discard, per-sample masks and render-target access. Those are whole
# blocks of the ISA with no route in from a kernel, so their opcodes were unreachable by
# construction rather than merely unprobed.
# THE VERTEX STAGE IS REACHABLE, THE FRAGMENT STAGE IS NOT. libaccel archives a render pipeline
# through ac_archive_vertex, and that path builds it with rasterization DISABLED - Metal then
# requires the vertex shader to return void, and there is no fragment stage in the pipeline at all:
#
#     "RasterizationEnabled is false but the vertex shader's return type is not void"
#
# So screen-space derivatives, attribute interpolation, discard and render-target access have no
# route in from this harness. Their opcodes are unreachable BY CONSTRUCTION here, not merely
# unprobed, and that is a limit of the instrument which should not be read as a property of the
# ISA. What a void vertex shader does reach is the vertex-stage prologue: vertex_id, instance_id
# and the base offsets, which are fetched differently from a kernel's thread position.
STAGE_HEAD = """#include <metal_stdlib>
using namespace metal;
vertex void kv(uint vid [[vertex_id]], uint iid [[instance_id]],
               uint bv [[base_vertex]], uint bi [[base_instance]],
               device float4 *p [[buffer(0)]], device float *f [[buffer(1)]],
               device uint *u [[buffer(2)]]) {
"""
STAGE_TAIL = """}
"""


def stage_probes():
    P = {}
    def vert(tag, body): P["vs:" + tag] = ("vert", body)
    vert("vertexid", "  u[400 + {i}] = vid + {i}u;")
    vert("instanceid", "  u[400 + {i}] = iid + {i}u;")
    vert("basevertex", "  u[400 + {i}] = bv + {i}u;")
    vert("baseinstance", "  u[400 + {i}] = bi + {i}u;")
    vert("fetch", "  f[400 + {i}] = p[vid + {i}u].x;")
    vert("fetch.instanced", "  f[400 + {i}] = p[iid * 4u + {i}u].y;")
    vert("transform", "  f[400 + {i}] = dot(p[vid + {i}u], p[{i}]);")
    vert("scale", "  f[400 + {i}] = p[vid + {i}u].z * f[{i}];")
    vert("index", "  u[400 + {i}] = u[vid * 3u + {i}u];")
    return P


def stage_source(name, n):
    grp, body = stage_probes()[name]
    return STAGE_HEAD + "\n".join(body.format(i=i) for i in range(n)) + "\n" + STAGE_TAIL


def build_render(tag, src):
    """Like g17corpus.build but archives a RENDER pipeline. ac_archive takes one compute function;
    a void vertex shader needs ac_archive_vertex, whose signature is (function, output path)
    and not the three arguments the failing calls assumed."""
    import ctypes
    d = os.path.join(CACHE, tag)
    os.makedirs(d, exist_ok=True)
    if os.path.exists(d + "/out/object/0-0"):
        return "cached"
    open(d + "/s.metal", "w").write(src)
    r = subprocess.run(["xcrun", "metal", "-o", d + "/s.metallib", d + "/s.metal"],
                       capture_output=True, text=True)
    if r.returncode:
        return "COMPILE FAILED: " + (r.stderr.strip().splitlines() or ["?"])[-1][:90]
    L = ctypes.CDLL(g17corpus.LIBACCEL)
    L.ac_init()
    L.ac_lib_from_data.argtypes = [ctypes.c_char_p]
    L.ac_archive_vertex.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    if L.ac_lib_from_data((d + "/s.metallib").encode()) != 0:
        return "ac_lib_from_data failed"
    if L.ac_archive_vertex(b"kv", (d + "/s.arc.metallib").encode()) != 0:
        return "ac_archive_vertex failed"
    r = subprocess.run(["xcrun", "metal-lipo", "-thin", g17target.ARCH, "-output", d + "/s.nat",
                        d + "/s.arc.metallib"], capture_output=True, text=True)
    if r.returncode:
        return "lipo failed: " + r.stderr.strip()[:80]
    subprocess.run(["rm", "-rf", d + "/out"], check=True)
    os.makedirs(d + "/out")
    r = subprocess.run(["xcrun", "metal-source", "--flatbuffers=json", "-f", "-o=" + d + "/out",
                        d + "/s.nat"], capture_output=True, text=True)
    if r.returncode:
        return "metal-source failed: " + r.stderr.strip()[:80]
    return "built"


def build_stage():
    P = stage_probes()
    ok = fail = 0
    msgs = {}
    for name in sorted(P):
        for n in SCALES:
            r = build_render("mg-%s-%d" % (name.replace(":", "_"), n), stage_source(name, n))
            if r in ("built", "cached"): ok += 1
            else:
                fail += 1; msgs[name] = r
    print("stage probe objects built or cached: %d   failed: %d" % (ok, fail))
    for k, v in sorted(msgs.items()):
        print("   %-22s %s" % (k, v[:78]))
    return msgs


# ---------------------------------------------------------------------------------------------
# THE FOURTH FAMILY: the Metal surface none of the first three touches. Ray tracing, atomic
# floats, the texture types beyond texture2d, argument buffers, and the wider simdgroup matrix
# shapes. Reachability is what bounds this whole problem - Apple's compiler had emitted only 434
# of 6,718 admitted opcodes across the corpus and 750 probes - so every construct not yet probed
# is a piece of that bound, and bfloat alone moved it by 34.
RT_HEAD = """#include <metal_stdlib>
#include <metal_raytracing>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void k(device uint *u [[buffer(0)]], device float *f [[buffer(1)]],
              device half *h [[buffer(2)]], device atomic_float *af [[buffer(3)]],
              device atomic_uint *au [[buffer(4)]],
              raytracing::instance_acceleration_structure as [[buffer(5)]],
              raytracing::primitive_acceleration_structure ps [[buffer(6)]],
              texturecube<float, access::sample> tc [[texture(0)]],
              texture1d<float, access::sample> t1 [[texture(1)]],
              texture2d_ms<float, access::read> tms [[texture(2)]],
              texture3d<float, access::write> t3w [[texture(3)]],
              texturecube_array<float, access::sample> tca [[texture(4)]],
              texture2d<uint, access::read> tu [[texture(5)]],
              texture_buffer<float, access::read> tb [[texture(6)]],
              device bfloat *bf [[buffer(7)]],
              sampler sm [[sampler(0)]],
              uint3 tg [[threadgroup_position_in_grid]],
              uint3 tp [[thread_position_in_threadgroup]]) {
"""
RT_TAIL = "}\n"


def rt_probes():
    P = {}
    def add(tag, body): P["rt:" + tag] = ("rt", body)
    add("atomic_float_add",
        "  atomic_fetch_add_explicit(&af[tg.x + {i}u], f[{i}], memory_order_relaxed);")
    add("atomic_float_min",
        "  atomic_fetch_min_explicit(&af[tg.x + {i}u], f[{i}], memory_order_relaxed);")
    add("atomic_float_max",
        "  atomic_fetch_max_explicit(&af[tg.x + {i}u], f[{i}], memory_order_relaxed);")
    add("atomic_float_store",
        "  atomic_store_explicit(&af[tg.x + {i}u], f[{i}], memory_order_relaxed);")
    add("atomic_float_load",
        "  f[400 + {i}] = atomic_load_explicit(&af[tg.x + {i}u], memory_order_relaxed);")
    add("tex.cube", "  f[400 + {i}] = tc.sample(sm, float3(f[{i}], f[{i}+1], f[{i}+2])).x;")
    add("tex.cube_array",
        "  f[400 + {i}] = tca.sample(sm, float3(f[{i}], f[{i}+1], f[{i}+2]), {i}u).x;")
    add("tex.1d", "  f[400 + {i}] = t1.sample(sm, f[{i}]).x;")
    add("tex.ms", "  f[400 + {i}] = tms.read(uint2(u[{i}], u[{i}+1]), {i}u).x;")
    add("tex.3d_write", "  t3w.write(float4(f[{i}]), uint3(u[{i}], u[{i}+1], u[{i}+2]));")
    add("tex.uint_read", "  u[400 + {i}] = tu.read(uint2(u[{i}], u[{i}+1])).x;")
    add("tex.buffer", "  f[400 + {i}] = tb.read(u[{i}]).x;")
    add("tex.lod_query", "  f[400 + {i}] = tc.calculate_unclamped_lod(sm, float3(f[{i}]));")
    add("rt.intersect", "  {{ raytracing::ray r{i}; r{i}.origin = float3(f[{i}]); "
        "r{i}.direction = float3(0.0f, 0.0f, 1.0f); r{i}.min_distance = 0.0f; "
        "r{i}.max_distance = 100.0f; "
        "raytracing::intersector<raytracing::instancing, raytracing::triangle_data> it{i}; "
        "f[400 + {i}] = it{i}.intersect(r{i}, as).distance; }}")
    add("rt.intersect_prim", "  {{ raytracing::ray r{i}; r{i}.origin = float3(f[{i}]); "
        "r{i}.direction = float3(0.0f, 0.0f, 1.0f); r{i}.min_distance = 0.0f; "
        "r{i}.max_distance = 100.0f; "
        "raytracing::intersector<raytracing::triangle_data> it{i}; "
        "f[400 + {i}] = it{i}.intersect(r{i}, ps).distance; }}")
    add("rt.intersect_any", "  {{ raytracing::ray r{i}; r{i}.origin = float3(f[{i}]); "
        "r{i}.direction = float3(0.0f, 0.0f, 1.0f); r{i}.min_distance = 0.0f; "
        "r{i}.max_distance = 100.0f; "
        "raytracing::intersector<raytracing::instancing, raytracing::triangle_data> it{i}; "
        "it{i}.accept_any_intersection(true); "
        "f[400 + {i}] = it{i}.intersect(r{i}, as).distance; }}")
    add("sgmatrix.f32.load_store", "  {{ simdgroup_float8x8 m{i}; "
        "simdgroup_load(m{i}, f + {i} * 64); simdgroup_store(m{i}, f + 2048 + {i} * 64); }}")
    # MORE MATRIX SHAPES. sched 106 holds 81 opcodes with nothing named and its only candidate is
    # simdgroup.mma - the same operand shape as op2842 in sched 118 - so a matrix variant that
    # lands in a different scheduling class is exactly what would name it. Four axes are reachable:
    # accumulate against plain multiply, and float, half, bfloat and MIXED (half inputs into a
    # float accumulator) element types.
    add("sgmatrix.mul_noacc", "  {{ simdgroup_float8x8 a{i}, b{i}, c{i}; "
        "simdgroup_load(a{i}, f + {i} * 64); simdgroup_load(b{i}, f + 512); "
        "simdgroup_multiply(c{i}, a{i}, b{i}); "
        "simdgroup_store(c{i}, f + 1024 + {i} * 64); }}")
    add("sgmatrix.bf16.mac", "  {{ simdgroup_bfloat8x8 a{i}, b{i}, c{i}; "
        "simdgroup_load(a{i}, bf + {i} * 64); simdgroup_load(b{i}, bf + 512); "
        "simdgroup_multiply_accumulate(c{i}, a{i}, b{i}, c{i}); "
        "simdgroup_store(c{i}, bf + 1024 + {i} * 64); }}")
    add("sgmatrix.mixed.mac", "  {{ simdgroup_half8x8 a{i}, b{i}; simdgroup_float8x8 c{i}; "
        "simdgroup_load(a{i}, h + {i} * 64); simdgroup_load(b{i}, h + 512); "
        "simdgroup_multiply_accumulate(c{i}, a{i}, b{i}, c{i}); "
        "simdgroup_store(c{i}, f + 1024 + {i} * 64); }}")
    add("sgmatrix.mixed.mul", "  {{ simdgroup_half8x8 a{i}, b{i}; simdgroup_float8x8 c{i}; "
        "simdgroup_load(a{i}, h + {i} * 64); simdgroup_load(b{i}, h + 512); "
        "simdgroup_multiply(c{i}, a{i}, b{i}); "
        "simdgroup_store(c{i}, f + 1024 + {i} * 64); }}")
    add("sgmatrix.f16.mac", "  {{ simdgroup_half8x8 a{i}, b{i}, c{i}; "
        "simdgroup_load(a{i}, h + {i} * 64); simdgroup_load(b{i}, h + 1024); "
        "simdgroup_multiply_accumulate(c{i}, a{i}, b{i}, c{i}); "
        "simdgroup_store(c{i}, h + 2048 + {i} * 64); }}")
    return P


def rt_source(name, n):
    body = rt_probes()[name][1]
    return RT_HEAD + "\n".join(body.format(i=i) for i in range(n)) + "\n" + RT_TAIL


def build_rt():
    P = rt_probes()
    ok = fail = 0
    msgs = {}
    for name in sorted(P):
        for n in SCALES:
            r = g17corpus.build("mr-%s-%d" % (name[3:], n), rt_source(name, n))
            if r in ("built", "cached"): ok += 1
            else:
                fail += 1; msgs[name] = r
    print("ray/texture/atomic probe objects built or cached: %d   failed: %d" % (ok, fail))
    for k, v in sorted(msgs.items()):
        print("   %-24s %s" % (k, v[:74]))
    return msgs


# ---------------------------------------------------------------------------------------------
# THE FIFTH FAMILY: threadgroup atomics, memory orderings, the barrier flag variants, argument
# buffers and a visible-function call. THREADGROUP ATOMICS are the reason it exists - the device
# and threadgroup forms of an ordinary load are different opcodes here (load and load.tg were
# separated by exactly that isolation), so the atomics almost certainly split the same way, and
# every atomic probe so far has been on device memory.
TG_HEAD = """#include <metal_stdlib>
using namespace metal;
struct AB { device float *p [[id(0)]]; texture2d<float, access::sample> t [[id(1)]]; };
[[visible]] float vf(float x) { return x * 2.0f + 1.0f; }
kernel void k(device uint *u [[buffer(0)]], device float *f [[buffer(1)]],
              device atomic_uint *au [[buffer(2)]], constant AB &ab [[buffer(3)]],
              sampler sm [[sampler(0)]],
              uint3 tg [[threadgroup_position_in_grid]],
              uint3 tp [[thread_position_in_threadgroup]]) {
  threadgroup atomic_uint ta[128];
  threadgroup atomic_int ti[128];
"""
TG_TAIL = "}\n"


def tg_probes():
    P = {}
    def add(tag, body): P["tg:" + tag] = ("tgat", body)
    for op in ("add", "sub", "and", "or", "xor", "min", "max"):
        add("tgatomic_%s" % op,
            "  atomic_fetch_%s_explicit(&ta[tp.x + {i}u], u[{i}], memory_order_relaxed);" % op)
    add("tgatomic_store",
        "  atomic_store_explicit(&ta[tp.x + {i}u], u[{i}], memory_order_relaxed);")
    add("tgatomic_load",
        "  u[400 + {i}] = atomic_load_explicit(&ta[tp.x + {i}u], memory_order_relaxed);")
    add("tgatomic_xchg",
        "  u[400 + {i}] = atomic_exchange_explicit(&ta[tp.x + {i}u], u[{i}], memory_order_relaxed);")
    add("tgatomic_cmpxchg", "  {{ uint e{i} = u[{i}]; atomic_compare_exchange_weak_explicit("
        "&ta[tp.x + {i}u], &e{i}, 7u, memory_order_relaxed, memory_order_relaxed); "
        "u[400 + {i}] = e{i}; }}")
    add("tgatomic_int_min",
        "  atomic_fetch_min_explicit(&ti[tp.x + {i}u], (int)u[{i}], memory_order_relaxed);")
    # ORDERING. A relaxed atomic and a sequentially consistent one differ in the fences around
    # them, and a fence is an instruction.
    add("atomic_seqcst",
        "  atomic_fetch_add_explicit(&au[tg.x + {i}u], u[{i}], memory_order_seq_cst);")
    add("tgatomic_seqcst",
        "  atomic_fetch_add_explicit(&ta[tp.x + {i}u], u[{i}], memory_order_seq_cst);")
    # BARRIER FLAGS. Each names a different thing to make visible, and they need not share an
    # encoding.
    for fl in ("mem_none", "mem_device", "mem_threadgroup", "mem_texture"):
        add("barrier_%s" % fl,
            "  threadgroup_barrier(mem_flags::%s); u[400 + {i}] = u[{i}];" % fl)
        add("simdbarrier_%s" % fl,
            "  simdgroup_barrier(mem_flags::%s); u[400 + {i}] = u[{i}];" % fl)
    # SPLIT, because the first version did both a buffer load and a texture sample through the
    # argument buffer and attributed op14665 to the pair - which says nothing about which half
    # emits it. One construct per probe is the whole discipline of this method.
    add("argbuffer_load", "  f[400 + {i}] = ab.p[{i}];")
    add("argbuffer_texture", "  f[400 + {i}] = ab.t.sample(sm, float2(f[{i}], 0.0f)).x;")
    add("visible_call", "  f[400 + {i}] = vf(f[{i}]);")
    return P


def tg_source(name, n):
    body = tg_probes()[name][1]
    return TG_HEAD + "\n".join(body.format(i=i) for i in range(n)) + "\n" + TG_TAIL


def build_tg():
    P = tg_probes()
    ok = fail = 0
    msgs = {}
    for name in sorted(P):
        for n in SCALES:
            r = g17corpus.build("mt-%s-%d" % (name[3:], n), tg_source(name, n))
            if r in ("built", "cached"): ok += 1
            else:
                fail += 1; msgs[name] = r
    print("threadgroup/ordering probe objects built or cached: %d   failed: %d" % (ok, fail))
    for k, v in sorted(msgs.items()):
        print("   %-26s %s" % (k, v[:72]))
    return msgs


# ---------------------------------------------------------------------------------------------
# THE SIXTH FAMILY: THE SYSTEMATIC TEXTURE MATRIX, 64-BIT INTEGERS, PACKING, AND - the reason this
# family exists at all - THE fast:: / precise:: SPLIT.
#
# Reachability, not sample size, is the bound on naming: 914 probes and the whole shipped corpus
# together emit only 530 of 6,718 opcodes, so a rule can only ever redistribute what the probes
# already reached. The way to name more opcodes is to make Apple's compiler emit more of them.
#
# Two blind spots dominate what is left.
#
# THE TEXTURE MATRIX. Metal's texture surface is a product of four axes - dimensionality (1d, 2d,
# 3d, cube, their arrays, multisample, depth, buffer), element type (float, half, int, uint),
# access (sample, read, write, read_write), and the modifier on the access itself (level, bias,
# gradient, offset, min_lod_clamp, compare, the gather component). The existing probes sample that
# space at maybe twenty points. sched 49 alone holds 192 opcodes with nothing named and the only
# constructs that reach any of them are texture ones.
#
# fast:: AND precise::. This is the instrument the transcendental classes needed. Classes 49, 50,
# 52 and 79 hold 464 unnamed opcodes, all sharing one flags word, each selected by a HANDFUL of
# different transcendental lowerings - so they are shared lowering steps and naming one after
# `cos` is exactly the error that cost this table 794 names. But Metal exposes the same function
# at two precisions: `fast::sin` and `precise::sin` are different lowerings of one operation.
# An opcode that scales in `precise::sin` but not `fast::sin` is a step in the refinement, not the
# core evaluation, and that distinction comes from the language, NOT from this project's name
# table - which is the property the fsat retraction says every new instrument must have.
TX_HEAD = """#include <metal_stdlib>
using namespace metal;
constexpr sampler smp(coord::normalized, address::repeat, filter::linear, mip_filter::linear);
constexpr sampler smc(coord::normalized, compare_func::less);
constexpr sampler smn(coord::pixel, address::clamp_to_edge, filter::nearest);
kernel void k(device uint *u [[buffer(0)]], device float *f [[buffer(1)]],
              device half *h [[buffer(2)]], device ulong *w [[buffer(3)]],
              texture1d<float, access::sample> T1 [[texture(0)]],
              texture1d_array<float, access::sample> T1A [[texture(1)]],
              texture2d<float, access::sample> T2 [[texture(2)]],
              texture2d_array<float, access::sample> T2A [[texture(3)]],
              texture3d<float, access::sample> T3 [[texture(4)]],
              texturecube<float, access::sample> TC [[texture(5)]],
              texturecube_array<float, access::sample> TCA [[texture(6)]],
              texture2d_ms<float, access::read> TMS [[texture(7)]],
              depth2d<float, access::sample> D2 [[texture(8)]],
              depth2d_array<float, access::sample> D2A [[texture(9)]],
              depthcube<float, access::sample> DC [[texture(10)]],
              texture2d<half, access::sample> H2 [[texture(11)]],
              texture2d<int, access::read> I2 [[texture(12)]],
              texture2d<uint, access::read> U2 [[texture(13)]],
              texture2d<float, access::read> R2 [[texture(14)]],
              texture2d<float, access::write> W2 [[texture(15)]],
              texture3d<float, access::write> W3 [[texture(16)]],
              texture1d<float, access::write> W1 [[texture(17)]],
              texture2d_array<float, access::write> W2A [[texture(18)]],
              texture2d<half, access::write> WH2 [[texture(19)]],
              texture2d<uint, access::write> WU2 [[texture(20)]],
              texture2d<float, access::read_write> RW2 [[texture(21)]],
              uint3 tg [[threadgroup_position_in_grid]],
              uint3 tp [[thread_position_in_threadgroup]]) {
  float2 c = float2(f[0], f[1]);
"""
TX_TAIL = "}\n"


def tx_probes():
    P = {}
    def add(tag, body): P["tx:" + tag] = ("tex", body)

    # -- SAMPLE, one modifier per probe so an opcode can be attributed to the modifier itself.
    add("s.2d",          "  f[400 + {i}] = T2.sample(smp, c + float2({i})).x;")
    add("s.2d.level",    "  f[400 + {i}] = T2.sample(smp, c + float2({i}), level(1.0f)).x;")
    add("s.2d.bias",     "  f[400 + {i}] = T2.sample(smp, c + float2({i}), bias(0.5f)).x;")
    add("s.2d.grad",     "  f[400 + {i}] = T2.sample(smp, c + float2({i}), "
                         "gradient2d(float2(0.1f), float2(0.2f))).x;")
    add("s.2d.offset",   "  f[400 + {i}] = T2.sample(smp, c + float2({i}), int2(1, 1)).x;")
    add("s.2d.lodclamp", "  f[400 + {i}] = T2.sample(smp, c + float2({i}), "
                         "min_lod_clamp(1.0f)).x;")
    add("s.2d.nearest",  "  f[400 + {i}] = T2.sample(smn, c + float2({i})).x;")
    add("s.1d",          "  f[400 + {i}] = T1.sample(smp, f[{i}]).x;")
    add("s.1darray",     "  f[400 + {i}] = T1A.sample(smp, f[{i}], {i}u).x;")
    add("s.2darray",     "  f[400 + {i}] = T2A.sample(smp, c + float2({i}), {i}u).x;")
    add("s.3d",          "  f[400 + {i}] = T3.sample(smp, float3(c, f[{i}])).x;")
    add("s.3d.level",    "  f[400 + {i}] = T3.sample(smp, float3(c, f[{i}]), level(1.0f)).x;")
    add("s.3d.grad",     "  f[400 + {i}] = T3.sample(smp, float3(c, f[{i}]), "
                         "gradient3d(float3(0.1f), float3(0.2f))).x;")
    add("s.cube",        "  f[400 + {i}] = TC.sample(smp, float3(c, f[{i}])).x;")
    add("s.cube.grad",   "  f[400 + {i}] = TC.sample(smp, float3(c, f[{i}]), "
                         "gradientcube(float3(0.1f), float3(0.2f))).x;")
    add("s.cubearray",   "  f[400 + {i}] = TCA.sample(smp, float3(c, f[{i}]), {i}u).x;")
    add("s.half",        "  h[400 + {i}] = H2.sample(smp, c + float2({i})).x;")

    # -- GATHER. Four texels instead of one, and the component is part of the encoding.
    add("g.2d",          "  f[400 + {i}] = T2.gather(smp, c + float2({i})).x;")
    add("g.2d.offset",   "  f[400 + {i}] = T2.gather(smp, c + float2({i}), int2(1, 1)).x;")
    add("g.2d.comp",     "  f[400 + {i}] = T2.gather(smp, c + float2({i}), int2(0), "
                         "component::y).x;")
    add("g.2darray",     "  f[400 + {i}] = T2A.gather(smp, c + float2({i}), {i}u).x;")
    add("g.cube",        "  f[400 + {i}] = TC.gather(smp, float3(c, f[{i}])).x;")

    # -- DEPTH AND COMPARE. A compare sampler is a different instruction, not a different operand.
    add("d.sample",      "  f[400 + {i}] = D2.sample(smp, c + float2({i}));")
    add("d.compare",     "  f[400 + {i}] = D2.sample_compare(smc, c + float2({i}), 0.5f);")
    add("d.compare.lod", "  f[400 + {i}] = D2.sample_compare(smc, c + float2({i}), 0.5f, "
                         "level(0.0f));")
    add("d.gather",      "  f[400 + {i}] = D2.gather(smp, c + float2({i})).x;")
    add("d.gathercmp",   "  f[400 + {i}] = D2.gather_compare(smc, c + float2({i}), 0.5f).x;")
    add("d.array",       "  f[400 + {i}] = D2A.sample(smp, c + float2({i}), {i}u);")
    add("d.cube",        "  f[400 + {i}] = DC.sample(smp, float3(c, f[{i}]));")

    # -- READ. No sampler, so no filtering step - the pure fetch.
    add("r.2d",          "  f[400 + {i}] = R2.read(uint2(u[{i}], u[{i} + 1])).x;")
    add("r.2d.lod",      "  f[400 + {i}] = R2.read(uint2(u[{i}], u[{i} + 1]), 1u).x;")
    add("r.int",         "  u[400 + {i}] = (uint)I2.read(uint2(u[{i}], u[{i} + 1])).x;")
    add("r.uint",        "  u[400 + {i}] = U2.read(uint2(u[{i}], u[{i} + 1])).x;")
    add("r.ms",          "  f[400 + {i}] = TMS.read(uint2(u[{i}], u[{i} + 1]), {i}u).x;")
    add("r.rw",          "  f[400 + {i}] = RW2.read(uint2(u[{i}], u[{i} + 1])).x;")

    # -- WRITE, across dimensionality and element type.
    add("w.2d",          "  W2.write(float4(f[{i}]), uint2(u[{i}], u[{i} + 1]));")
    add("w.2d.lod",      "  W2.write(float4(f[{i}]), uint2(u[{i}], u[{i} + 1]), 1u);")
    add("w.3d",          "  W3.write(float4(f[{i}]), uint3(u[{i}], u[{i} + 1], u[{i} + 2]));")
    add("w.1d",          "  W1.write(float4(f[{i}]), u[{i}]);")
    add("w.2darray",     "  W2A.write(float4(f[{i}]), uint2(u[{i}], u[{i} + 1]), {i}u);")
    add("w.half",        "  WH2.write(half4(h[{i}]), uint2(u[{i}], u[{i} + 1]));")
    add("w.uint",        "  WU2.write(uint4(u[{i}]), uint2(u[{i}], u[{i} + 1]));")
    add("w.rw",          "  RW2.write(float4(f[{i}]), uint2(u[{i}], u[{i} + 1]));")

    # -- QUERIES. Descriptor reads, not memory accesses, and they should not share an opcode with
    # a sample - which is a prediction this family can falsify.
    add("q.width",       "  u[400 + {i}] = T2.get_width();")
    add("q.width.lod",   "  u[400 + {i}] = T2.get_width({i}u);")
    add("q.height",      "  u[400 + {i}] = T2.get_height();")
    add("q.depth",       "  u[400 + {i}] = T3.get_depth();")
    add("q.arraysize",   "  u[400 + {i}] = T2A.get_array_size();")
    add("q.miplevels",   "  u[400 + {i}] = T2.get_num_mip_levels();")
    add("q.samples",     "  u[400 + {i}] = TMS.get_num_samples();")
    add("q.lod",         "  f[400 + {i}] = T2.calculate_clamped_lod(smp, c + float2({i}));")
    add("q.lod.unclamp", "  f[400 + {i}] = T2.calculate_unclamped_lod(smp, c + float2({i}));")
    add("fence.tex",     "  RW2.fence(); f[400 + {i}] = f[{i}];")

    # -- fast:: AND precise::, THE INDEPENDENT DISCRIMINATOR. Same mathematical function, two
    # lowerings named by the LANGUAGE. Whatever separates these two columns is a refinement step
    # and whatever they share is the core evaluation - a split this project's own name table had
    # no way to make.
    for fn in ("sin", "cos", "tan", "exp", "exp2", "exp10", "log", "log2", "log10", "sqrt",
               "rsqrt", "sinh", "cosh", "tanh", "asin", "acos", "atan"):
        add("fast." + fn,    "  f[400 + {i}] = fast::%s(f[{i}]);" % fn)
        add("precise." + fn, "  f[400 + {i}] = precise::%s(f[{i}]);" % fn)
    for fn in ("divide", "powr", "pow"):
        add("fast." + fn,    "  f[400 + {i}] = fast::%s(f[{i}], f[{i} + 1]);" % fn)
        add("precise." + fn, "  f[400 + {i}] = precise::%s(f[{i}], f[{i} + 1]);" % fn)

    # -- 64-BIT INTEGERS. A width the expression family never covered, and one where a single
    # source operation must become several instructions.
    for tag, ex in (("add", "w[{i}] + w[{i} + 1]"), ("sub", "w[{i}] - w[{i} + 1]"),
                    ("mul", "w[{i}] * w[{i} + 1]"), ("div", "w[{i}] / (w[{i} + 1] | 1ul)"),
                    ("mod", "w[{i}] % (w[{i} + 1] | 1ul)"), ("shl", "w[{i}] << (w[{i}+1] & 63ul)"),
                    ("shr", "w[{i}] >> (w[{i}+1] & 63ul)"), ("and", "w[{i}] & w[{i} + 1]"),
                    ("or", "w[{i}] | w[{i} + 1]"), ("xor", "w[{i}] ^ w[{i} + 1]"),
                    ("clz", "(ulong)clz(w[{i}])"), ("popcount", "(ulong)popcount(w[{i}])"),
                    ("cmp", "(ulong)(w[{i}] < w[{i} + 1])"),
                    ("from.u32", "(ulong)u[{i}]"), ("to.u32", "(ulong)(uint)w[{i}]"),
                    ("from.f32", "(ulong)f[{i}]"), ("to.f32", "(ulong)(float)w[{i}]")):
        add("i64." + tag, "  w[400 + {i}] = " + ex + ";")

    # -- PACKING. Format conversion in one instruction is the kind of thing a GPU has and a
    # general-purpose ISA does not, so these are opcodes nothing else would reach.
    add("pk.unorm4x8",   "  u[400 + {i}] = pack_float_to_unorm4x8(float4(f[{i}]));")
    add("pk.snorm4x8",   "  u[400 + {i}] = pack_float_to_snorm4x8(float4(f[{i}]));")
    add("pk.unorm2x16",  "  u[400 + {i}] = pack_float_to_unorm2x16(float2(f[{i}]));")
    add("pk.snorm2x16",  "  u[400 + {i}] = pack_float_to_snorm2x16(float2(f[{i}]));")
    add("pk.srgb",       "  u[400 + {i}] = pack_float_to_srgb_unorm4x8(float4(f[{i}]));")
    add("up.unorm4x8",   "  f[400 + {i}] = unpack_unorm4x8_to_float(u[{i}]).x;")
    add("up.snorm4x8",   "  f[400 + {i}] = unpack_snorm4x8_to_float(u[{i}]).x;")
    add("up.unorm2x16",  "  f[400 + {i}] = unpack_unorm2x16_to_float(u[{i}]).x;")
    add("up.srgb",       "  f[400 + {i}] = unpack_unorm4x8_srgb_to_float(u[{i}]).x;")
    add("up.unorm4x8.h", "  h[400 + {i}] = unpack_unorm4x8_to_half(u[{i}]).x;")
    add("up.unorm10a2",  "  f[400 + {i}] = unpack_unorm10a2_to_float(u[{i}]).x;")
    add("up.unorm565",   "  f[400 + {i}] = unpack_unorm565_to_float((ushort)u[{i}]).x;")

    # -- HALF TRANSCENDENTALS. Every f16 name this table has came from an f32 name plus a form
    # bit; these are measured instead.
    for fn in ("sin", "cos", "exp", "log", "sqrt", "rsqrt", "tanh", "floor", "ceil", "rint",
               "trunc", "fabs"):
        add("h." + fn, "  h[400 + {i}] = %s(h[{i}]);" % fn)
    return P


def tx_source(name, n):
    body = tx_probes()[name][1]
    return TX_HEAD + "\n".join(body.format(i=i) for i in range(n)) + "\n" + TX_TAIL


def build_tx():
    P = tx_probes()
    ok = fail = 0
    msgs = {}
    for name in sorted(P):
        for n in SCALES:
            r = g17corpus.build("mx-%s-%d" % (name[3:], n), tx_source(name, n))
            if r in ("built", "cached"): ok += 1
            else:
                fail += 1; msgs[name] = r
    print("texture/precision/i64 probe objects built or cached: %d   failed: %d" % (ok, fail))
    for k, v in sorted(msgs.items()):
        print("   %-24s %s" % (k, v[:74]))
    return msgs


# ---------------------------------------------------------------------------------------------
# THE SEVENTH FAMILY: five probes that TEST A PREDICTION rather than survey a surface.
# The predictions are written down in isa/g17-half-alu-predictions.txt BEFORE these were built,
# because the 794-name retraction happened when rules were scored against a table that already
# held the error, and a prediction recorded first cannot be fitted to its own result.
HX_HEAD = """#include <metal_stdlib>
using namespace metal;
kernel void k(device float *f [[buffer(0)]], device half *h [[buffer(1)]],
              uint3 tg [[threadgroup_position_in_grid]]) {
"""
HX_TAIL = "}\n"


def hx_probes():
    P = {}
    def add(tag, body): P["hx:" + tag] = ("halfalu", body)
    # P1 and P2: is the +8 block offset saturation?
    add("mul.sat",   "  h[400 + {i}] = saturate(h[{i}] * h[{i} + 1]);")
    add("add.sat",   "  h[400 + {i}] = saturate(h[{i}] + h[{i} + 1]);")
    # P3, P4 and the P5 control: is the mixed-width block arithmetic, not transcendence?
    add("mul.f2h",   "  h[400 + {i}] = (half)(f[{i}] * f[{i} + 1]);")
    add("fma.f2h",   "  h[400 + {i}] = (half)(f[{i}] * f[{i} + 1] + f[{i} + 2]);")
    add("mul.f2f",   "  f[400 + {i}] = f[{i}] * f[{i} + 1];")
    # the neighbouring mixed shapes, to see whether the operand widths pick the opcode
    add("mul.h_f",   "  h[400 + {i}] = h[{i}] * (half)f[{i} + 1];")
    add("mul.hf2h",  "  h[400 + {i}] = (half)((float)h[{i}] * f[{i} + 1]);")
    add("add.f2h",   "  h[400 + {i}] = (half)(f[{i}] + f[{i} + 1]);")
    add("fma.h",     "  h[400 + {i}] = fma(h[{i}], h[{i} + 1], h[{i} + 2]);")
    add("mul.sat.f", "  f[400 + {i}] = saturate(f[{i}] * f[{i} + 1]);")
    return P


def hx_source(name, n):
    body = hx_probes()[name][1]
    return HX_HEAD + "\n".join(body.format(i=i) for i in range(n)) + "\n" + HX_TAIL


def build_hx():
    P = hx_probes()
    ok = fail = 0
    msgs = {}
    for name in sorted(P):
        for n in SCALES:
            r = g17corpus.build("mh-%s-%d" % (name[3:], n), hx_source(name, n))
            if r in ("built", "cached"): ok += 1
            else:
                fail += 1; msgs[name] = r
    print("half-ALU prediction probes built or cached: %d   failed: %d" % (ok, fail))
    for k, v in sorted(msgs.items()):
        print("   %-24s %s" % (k, v[:74]))
    return msgs


# ---------------------------------------------------------------------------------------------
# THE EIGHTH FAMILY: sweep the MIXED-WIDTH FLOAT ALU now that the prediction probes identified it.
#
# Classes 49, 50, 52 and 79 hold 464 opcodes with nothing named, and they were the largest dark
# region in the ISA. The five recorded predictions confirmed what they are: two- and three-source
# float arithmetic where the operand widths differ from the destination width. The scheduling
# class is the WIDTH SIGNATURE - c49 is (f32, f32) -> f16, c50 is (f32, f16) -> f16, c52 is
# (f16, f32) -> f16, c79 is the three-source form - and the OPCODE picks the operation, exactly as
# op774 (add) and op862 (multiply) share class 438 and differ only by opcode.
#
# Each signature group holds thirty opcodes. Thirty is a product, not a coincidence, and the
# factors are visible in the f16 ALU next door: an operation, a saturating variant at +8, and
# source modifiers. This family varies operation and modifier one at a time against a fixed width
# combination so each opcode is attributed to a single construct rather than to a lowering.
FA_HEAD = """#include <metal_stdlib>
using namespace metal;
kernel void k(device float *f [[buffer(0)]], device half *h [[buffer(1)]],
              uint3 tg [[threadgroup_position_in_grid]]) {
"""
FA_TAIL = "}\n"

# (name, C expression in terms of A and B) - the two-source operations and the modifier patterns.
FA_OPS = [("mul", "%s * %s"), ("add", "%s + %s"), ("sub", "%s - %s"),
          ("min", "min(%s, %s)"), ("max", "max(%s, %s)"),
          ("fmin", "fmin(%s, %s)"), ("fmax", "fmax(%s, %s)"),
          ("copysign", "copysign(%s, %s)"), ("fdim", "fdim(%s, %s)"),
          ("step", "step(%s, %s)"), ("pow", "powr(%s, %s)")]
FA_MODS = [("", "%s", "%s"), ("absa", "fabs(%s)", "%s"), ("nega", "(-%s)", "%s"),
           ("absb", "%s", "fabs(%s)"), ("negb", "%s", "(-%s)"),
           ("absab", "fabs(%s)", "fabs(%s)")]
# (name, A source, B source, how the result is stored) - the width signatures.
FA_WIDTH = [("f32f32_h", "f[{i}]", "f[{i} + 1]", "h[400 + {i}] = (half)(%s);"),
            ("f32h_h", "f[{i}]", "(float)h[{i} + 1]", "h[400 + {i}] = (half)(%s);"),
            ("hf32_h", "(float)h[{i}]", "f[{i} + 1]", "h[400 + {i}] = (half)(%s);"),
            ("hh_h", "h[{i}]", "h[{i} + 1]", "h[400 + {i}] = (half)(%s);"),
            ("f32f32_f", "f[{i}]", "f[{i} + 1]", "f[400 + {i}] = (%s);")]


def fa_probes():
    P = {}
    def add(tag, body): P["fa:" + tag] = ("mixalu", body)
    for wname, A, B, store in FA_WIDTH:
        for oname, expr in FA_OPS:
            for mname, ma, mb in FA_MODS:
                # only sweep the modifiers on the two widths where the class is densest
                if mname and wname not in ("f32f32_h", "hh_h"):
                    continue
                e = expr % (ma % A, mb % B)
                add("%s.%s%s" % (wname, oname, "." + mname if mname else ""),
                    "  " + store % e)
        # saturation, the +8 offset the f16 ALU showed
        for oname, expr in (("mul", "%s * %s"), ("add", "%s + %s"), ("sub", "%s - %s")):
            add("%s.%s.sat" % (wname, oname), "  " + store % ("saturate(%s)" % (expr % (A, B))))
    # THREE SOURCE, for class 79 and its neighbours.
    for wname, A, B, store in FA_WIDTH:
        C = A.replace("{i}]", "{i} + 2]").replace("{i} + 1]", "{i} + 2]")
        add("%s.fma" % wname, "  " + store % ("fma(%s, %s, %s)" % (A, B, C)))
        add("%s.fma.nega" % wname, "  " + store % ("fma(-%s, %s, %s)" % (A, B, C)))
        add("%s.fma.negc" % wname, "  " + store % ("fma(%s, %s, -%s)" % (A, B, C)))
        add("%s.mad.sat" % wname,
            "  " + store % ("saturate(fma(%s, %s, %s))" % (A, B, C)))
        add("%s.mix" % wname, "  " + store % ("mix(%s, %s, %s)" % (A, B, C)))
        add("%s.clamp" % wname, "  " + store % ("clamp(%s, %s, %s)" % (A, B, C)))
    return P


def fa_source(name, n):
    body = fa_probes()[name][1]
    return FA_HEAD + "\n".join(body.format(i=i) for i in range(n)) + "\n" + FA_TAIL


def build_fa():
    P = fa_probes()
    ok = fail = 0
    msgs = {}
    for name in sorted(P):
        for n in SCALES:
            r = g17corpus.build("mf-%s-%d" % (name[3:], n), fa_source(name, n))
            if r in ("built", "cached"): ok += 1
            else:
                fail += 1; msgs[name] = r
    print("mixed-width ALU probe objects built or cached: %d   failed: %d" % (ok, fail))
    for k, v in sorted(msgs.items())[:10]:
        print("   %-28s %s" % (k, v[:70]))
    return msgs


# ---------------------------------------------------------------------------------------------
# THE NINTH FAMILY: the IMMEDIATE axis of the mixed-width ALU.
#
# Each width signature in class 49 holds thirty opcodes and the operation sweep named four of
# them. Class 439 shows what the rest are: it is class 438's four blocks again, one per block,
# in the form where a source is an immediate rather than a register - op767 is the immediate form
# of op766 and op775 of op774. So the remaining opcodes are operand FORMS, and the axis that
# generates them is which source is a constant and how wide that constant is.
#
# The float immediate is the 8-bit miniature format this project already decoded (bit 7 sign,
# bits 6:4 exponent, bits 3:0 mantissa), so a constant that fits it and one that does not should
# reach different opcodes. Both are probed.
FI_HEAD = """#include <metal_stdlib>
using namespace metal;
kernel void k(device float *f [[buffer(0)]], device half *h [[buffer(1)]],
              device bfloat *b [[buffer(2)]],
              uint3 tg [[threadgroup_position_in_grid]]) {
"""
FI_TAIL = "}\n"

# 0.5 is exactly representable in the 8-bit float immediate; 0.1 is not, and 1e30 is out of its
# range entirely - so if the immediate width picks the opcode, these three separate.
FI_CONST = [("half", "0.5"), ("odd", "0.1"), ("big", "1e30")]


def fi_probes():
    P = {}
    def add(tag, body): P["fi:" + tag] = ("mixstore", body)
    for cname, c in FI_CONST:
        # EVERY width combination, because retracting 106 names left holes that only the
        # uncovered combinations can fill: an f16 source into an f32 destination, the bfloat
        # forms, and each of them saturating and not.
        for wname, A, store in (("f32_h", "f[{i}]", "h[400 + {i}] = (half)(%s);"),
                                ("h_h", "h[{i}]", "h[400 + {i}] = (half)(%s);"),
                                ("f32_f", "f[{i}]", "f[400 + {i}] = (%s);"),
                                ("h_f", "(float)h[{i}]", "f[400 + {i}] = (%s);"),
                                ("bf_bf", "b[{i}]", "b[400 + {i}] = (bfloat)(%s);"),
                                ("bf_f", "(float)b[{i}]", "f[400 + {i}] = (%s);"),
                                ("f32_bf", "f[{i}]", "b[400 + {i}] = (bfloat)(%s);"),
                                ("h_bf", "(float)h[{i}]", "b[400 + {i}] = (bfloat)(%s);")):
            k = c + ("h" if wname == "h_h" else "f")
            if wname == "bf_bf":
                k = "(bfloat)" + c + "f"
            add("%s.%s.mul" % (wname, cname), "  " + store % ("%s * %s" % (A, k)))
            add("%s.%s.add" % (wname, cname), "  " + store % ("%s + %s" % (A, k)))
            add("%s.%s.rsub" % (wname, cname), "  " + store % ("%s - %s" % (k, A)))
            add("%s.%s.mul.sat" % (wname, cname),
                "  " + store % ("saturate(%s * %s)" % (A, k)))
            add("%s.%s.add.sat" % (wname, cname),
                "  " + store % ("saturate(%s + %s)" % (A, k)))
            # the three-source form with the constant in each position in turn
            B = A.replace("{i}]", "{i} + 1]")
            add("%s.%s.fma_a" % (wname, cname), "  " + store % ("fma(%s, %s, %s)" % (k, A, B)))
            add("%s.%s.fma_c" % (wname, cname), "  " + store % ("fma(%s, %s, %s)" % (A, B, k)))
    return P


def fi_source(name, n):
    body = fi_probes()[name][1]
    return FI_HEAD + "\n".join(body.format(i=i) for i in range(n)) + "\n" + FI_TAIL


def build_fi():
    P = fi_probes()
    ok = fail = 0
    msgs = {}
    for name in sorted(P):
        for n in SCALES:
            r = g17corpus.build("mi-%s-%d" % (name[3:], n), fi_source(name, n))
            if r in ("built", "cached"): ok += 1
            else:
                fail += 1; msgs[name] = r
    print("immediate-form probe objects built or cached: %d   failed: %d" % (ok, fail))
    for k, v in sorted(msgs.items())[:10]:
        print("   %-28s %s" % (k, v[:70]))
    return msgs


# ---------------------------------------------------------------------------------------------
# THE TENTH FAMILY: the INTEGER ALU, swept the way the float one was.
#
# Class 312 holds 342 opcodes: 140 named `madd`, 40 named `mul`, 162 nothing. Its commonest
# operand shape is GPR16 <- GPR32, which is the mixed-width immediate form - the same shape that
# made class 51 look unary until the immediate was accounted for. So this is the integer twin of
# the block that just resolved, and `mul` sitting beside `madd` in one class is the degenerate-case
# trap again: a multiply is a multiply-add with a zero addend, and whichever the opcode really is,
# one of those two names is the special case rather than the operation.
#
# Same discipline: one construct per probe, every width combination, and the constant chosen to
# be either inside or outside the small-immediate range so the immediate and register forms
# separate on their own.
IN_HEAD = """#include <metal_stdlib>
using namespace metal;
kernel void k(device uint *u [[buffer(0)]], device int *s [[buffer(1)]],
              device ushort *uh [[buffer(2)]], device short *sh [[buffer(3)]],
              uint3 tg [[threadgroup_position_in_grid]]) {
"""
IN_TAIL = "}\n"

# (name, A, B, C, store) - the width and signedness combinations.
IN_WIDTH = [
    ("s32", "s[{i}]", "s[{i} + 1]", "s[{i} + 2]", "s[400 + {i}] = (int)(%s);"),
    ("u32", "u[{i}]", "u[{i} + 1]", "u[{i} + 2]", "u[400 + {i}] = (uint)(%s);"),
    ("s16", "sh[{i}]", "sh[{i} + 1]", "sh[{i} + 2]", "sh[400 + {i}] = (short)(%s);"),
    ("u16", "uh[{i}]", "uh[{i} + 1]", "uh[{i} + 2]", "uh[400 + {i}] = (ushort)(%s);"),
    ("s32_s16", "s[{i}]", "s[{i} + 1]", "s[{i} + 2]", "sh[400 + {i}] = (short)(%s);"),
    ("u32_u16", "u[{i}]", "u[{i} + 1]", "u[{i} + 2]", "uh[400 + {i}] = (ushort)(%s);"),
    ("s16_s32", "(int)sh[{i}]", "s[{i} + 1]", "s[{i} + 2]", "s[400 + {i}] = (int)(%s);"),
    ("u16_u32", "(uint)uh[{i}]", "u[{i} + 1]", "u[{i} + 2]", "u[400 + {i}] = (uint)(%s);"),
]
IN_OPS = [("mul", "%s * %s"), ("add", "%s + %s"), ("sub", "%s - %s"),
          ("and", "%s & %s"), ("or", "%s | %s"), ("xor", "%s ^ %s"),
          ("shl", "%s << (%s & 15)"), ("shr", "%s >> (%s & 15)"),
          ("min", "min(%s, %s)"), ("max", "max(%s, %s)"),
          ("mulhi", "mulhi(%s, %s)"), ("addsat", "addsat(%s, %s)"),
          ("subsat", "subsat(%s, %s)"), ("absdiff", "absdiff(%s, %s)"),
          ("hadd", "hadd(%s, %s)"), ("rhadd", "rhadd(%s, %s)")]


def in_probes():
    P = {}
    def add(tag, body): P["in:" + tag] = ("intalu", body)
    for wname, A, B, Cc, store in IN_WIDTH:
        for oname, expr in IN_OPS:
            add("%s.%s" % (wname, oname), "  " + store % (expr % (A, B)))
        # THE THREE-SOURCE FORM, which is what distinguishes madd from mul.
        add("%s.madd" % wname, "  " + store % ("%s * %s + %s" % (A, B, Cc)))
        add("%s.madhi" % wname, "  " + store % ("mulhi(%s, %s) + %s" % (A, B, Cc)))
        add("%s.clamp" % wname, "  " + store % ("clamp(%s, %s, %s)" % (A, B, Cc)))
        # IMMEDIATE FORMS. 3 fits any small immediate field; 100000 fits none, so it must be
        # materialised into a register and reach the register form instead.
        for cname, k in (("small", "3"), ("big", "100000")):
            add("%s.mul.%s" % (wname, cname), "  " + store % ("%s * %s" % (A, k)))
            add("%s.add.%s" % (wname, cname), "  " + store % ("%s + %s" % (A, k)))
            add("%s.madd.%s" % (wname, cname), "  " + store % ("%s * %s + %s" % (A, k, B)))
            add("%s.and.%s" % (wname, cname), "  " + store % ("%s & %s" % (A, k)))
    return P


def in_source(name, n):
    body = in_probes()[name][1]
    return IN_HEAD + "\n".join(body.format(i=i) for i in range(n)) + "\n" + IN_TAIL


def build_in():
    P = in_probes()
    ok = fail = 0
    msgs = {}
    for name in sorted(P):
        for n in SCALES:
            r = g17corpus.build("mn-%s-%d" % (name[3:], n), in_source(name, n))
            if r in ("built", "cached"): ok += 1
            else:
                fail += 1; msgs[name] = r
    print("integer ALU probe objects built or cached: %d   failed: %d" % (ok, fail))
    for k, v in sorted(msgs.items())[:10]:
        print("   %-28s %s" % (k, v[:70]))
    return msgs


# ---------------------------------------------------------------------------------------------
# THE ELEVENTH FAMILY: the SIMDGROUP MATRIX block, varying the axes the earlier probes held fixed.
#
# Four classes have the identical operand shape and only one of them has a name:
#
#     c118  81 opcodes, tup2 <- three sources, three named    simdgroup.mma.f32/.f16.f32/.bf16
#     c106  81 opcodes, THE SAME SHAPE, nothing named
#     c119  the two-source form, two named                    simdgroup.mul.f32/.f16.f32
#     c108  54 opcodes and c120 54 opcodes, the same two-source shape, nothing named
#
# So 270 opcodes sit in classes indistinguishable by shape from ones that are named, and no probe
# reaches them. Every matrix probe so far accumulated IN PLACE, loaded from device memory, and
# never transposed - three axes held fixed at once. This family moves each of them separately:
# a destination distinct from the accumulator, a transposed operand, threadgroup rather than
# device memory, and each accumulator element type against each input element type.
MX_HEAD = """#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void k(device float *f [[buffer(0)]], device half *h [[buffer(1)]],
              device bfloat *b [[buffer(2)]],
              uint3 tg [[threadgroup_position_in_grid]],
              uint sl [[thread_index_in_simdgroup]]) {
  threadgroup float tf[1024];
  threadgroup half th[1024];
"""
MX_TAIL = "}\n"


def mx_probes():
    P = {}
    def add(tag, body): P["mx:" + tag] = ("sgmat", body)
    # (tag, matrix type, element buffer) for the three element types.
    TY = [("f32", "simdgroup_float8x8", "f"), ("f16", "simdgroup_half8x8", "h"),
          ("bf16", "simdgroup_bfloat8x8", "b")]
    for tn, mt, buf in TY:
        # ACCUMULATE IN PLACE - the form every earlier probe used, kept as the control.
        add("%s.mac.inplace" % tn,
            "  {{ %s a{i}, b{i}, c{i}; simdgroup_load(a{i}, %s + {i} * 64); "
            "simdgroup_load(b{i}, %s + 512); simdgroup_load(c{i}, %s + 1024); "
            "simdgroup_multiply_accumulate(c{i}, a{i}, b{i}, c{i}); "
            "simdgroup_store(c{i}, %s + 2048 + {i} * 64); }}" % (mt, buf, buf, buf, buf))
        # A DESTINATION DISTINCT FROM THE ACCUMULATOR - four matrices, not three.
        add("%s.mac.dst" % tn,
            "  {{ %s a{i}, b{i}, c{i}, d{i}; simdgroup_load(a{i}, %s + {i} * 64); "
            "simdgroup_load(b{i}, %s + 512); simdgroup_load(c{i}, %s + 1024); "
            "simdgroup_multiply_accumulate(d{i}, a{i}, b{i}, c{i}); "
            "simdgroup_store(d{i}, %s + 2048 + {i} * 64); }}" % (mt, buf, buf, buf, buf))
        # TRANSPOSED OPERANDS, one side at a time.
        add("%s.load.transpose" % tn,
            "  {{ %s a{i}, b{i}, c{i}; simdgroup_load(a{i}, %s + {i} * 64, 8, 0, true); "
            "simdgroup_load(b{i}, %s + 512); "
            "simdgroup_multiply_accumulate(c{i}, a{i}, b{i}, c{i}); "
            "simdgroup_store(c{i}, %s + 2048 + {i} * 64); }}" % (mt, buf, buf, buf))
        add("%s.store.transpose" % tn,
            "  {{ %s a{i}, b{i}, c{i}; simdgroup_load(a{i}, %s + {i} * 64); "
            "simdgroup_load(b{i}, %s + 512); "
            "simdgroup_multiply(c{i}, a{i}, b{i}); "
            "simdgroup_store(c{i}, %s + 2048 + {i} * 64, 8, 0, true); }}" % (mt, buf, buf, buf))
        # A NON-DEFAULT STRIDE, which changes the address arithmetic but should not change the
        # multiply - a control on whether the class is about the multiply or about the load.
        add("%s.load.stride" % tn,
            "  {{ %s a{i}, b{i}, c{i}; simdgroup_load(a{i}, %s + {i} * 64, 16); "
            "simdgroup_load(b{i}, %s + 512, 16); "
            "simdgroup_multiply_accumulate(c{i}, a{i}, b{i}, c{i}); "
            "simdgroup_store(c{i}, %s + 2048 + {i} * 64, 16); }}" % (mt, buf, buf, buf))
        # MULTIPLY WITHOUT ACCUMULATE, and the identity accumulate, so the two-source and
        # three-source forms can be told apart by construction rather than by opcode arithmetic.
        add("%s.mul" % tn,
            "  {{ %s a{i}, b{i}, c{i}; simdgroup_load(a{i}, %s + {i} * 64); "
            "simdgroup_load(b{i}, %s + 512); simdgroup_multiply(c{i}, a{i}, b{i}); "
            "simdgroup_store(c{i}, %s + 2048 + {i} * 64); }}" % (mt, buf, buf, buf))
    # THREADGROUP MEMORY rather than device - the axis that split load from load.tg.
    add("f32.tg", "  {{ simdgroup_float8x8 a{i}, b{i}, c{i}; simdgroup_load(a{i}, tf + {i} * 64); "
        "simdgroup_load(b{i}, tf + 512); "
        "simdgroup_multiply_accumulate(c{i}, a{i}, b{i}, c{i}); "
        "simdgroup_store(c{i}, tf + 2048 + {i} * 64); }}")
    add("f16.tg", "  {{ simdgroup_half8x8 a{i}, b{i}, c{i}; simdgroup_load(a{i}, th + {i} * 64); "
        "simdgroup_load(b{i}, th + 512); "
        "simdgroup_multiply_accumulate(c{i}, a{i}, b{i}, c{i}); "
        "simdgroup_store(c{i}, th + 2048 + {i} * 64); }}")
    add("f32.tg.store", "  {{ simdgroup_float8x8 a{i}, b{i}, c{i}; "
        "simdgroup_load(a{i}, f + {i} * 64); simdgroup_load(b{i}, f + 512); "
        "simdgroup_multiply_accumulate(c{i}, a{i}, b{i}, c{i}); "
        "simdgroup_store(c{i}, tf + 2048 + {i} * 64); }}")
    # MIXED ELEMENT TYPES: narrow inputs into a wide accumulator, each combination.
    add("mixed.h.f32", "  {{ simdgroup_half8x8 a{i}, b{i}; simdgroup_float8x8 c{i}; "
        "simdgroup_load(a{i}, h + {i} * 64); simdgroup_load(b{i}, h + 512); "
        "simdgroup_multiply_accumulate(c{i}, a{i}, b{i}, c{i}); "
        "simdgroup_store(c{i}, f + 2048 + {i} * 64); }}")
    add("mixed.bf.f32", "  {{ simdgroup_bfloat8x8 a{i}, b{i}; simdgroup_float8x8 c{i}; "
        "simdgroup_load(a{i}, b + {i} * 64); simdgroup_load(b{i}, b + 512); "
        "simdgroup_multiply_accumulate(c{i}, a{i}, b{i}, c{i}); "
        "simdgroup_store(c{i}, f + 2048 + {i} * 64); }}")
    add("mixed.h.f32.mul", "  {{ simdgroup_half8x8 a{i}, b{i}; simdgroup_float8x8 c{i}; "
        "simdgroup_load(a{i}, h + {i} * 64); simdgroup_load(b{i}, h + 512); "
        "simdgroup_multiply(c{i}, a{i}, b{i}); "
        "simdgroup_store(c{i}, f + 2048 + {i} * 64); }}")
    add("mixed.bf.f32.mul", "  {{ simdgroup_bfloat8x8 a{i}, b{i}; simdgroup_float8x8 c{i}; "
        "simdgroup_load(a{i}, b + {i} * 64); simdgroup_load(b{i}, b + 512); "
        "simdgroup_multiply(c{i}, a{i}, b{i}); "
        "simdgroup_store(c{i}, f + 2048 + {i} * 64); }}")
    # THE LOAD AND STORE ALONE, so a matrix move can be told from a matrix multiply.
    for tn, mt, buf in TY:
        add("%s.move" % tn, "  {{ %s m{i}; simdgroup_load(m{i}, %s + {i} * 64); "
            "simdgroup_store(m{i}, %s + 2048 + {i} * 64); }}" % (mt, buf, buf))
    return P


def mx_source(name, n):
    body = mx_probes()[name][1]
    return MX_HEAD + "\n".join(body.format(i=i) for i in range(n)) + "\n" + MX_TAIL


def build_mx():
    P = mx_probes()
    ok = fail = 0
    msgs = {}
    for name in sorted(P):
        for n in SCALES:
            r = g17corpus.build("mm-%s-%d" % (name[3:], n), mx_source(name, n))
            if r in ("built", "cached"): ok += 1
            else:
                fail += 1; msgs[name] = r
    print("simdgroup matrix probe objects built or cached: %d   failed: %d" % (ok, fail))
    for k, v in sorted(msgs.items()):
        print("   %-26s %s" % (k, v[:70]))
    return msgs


# ---------------------------------------------------------------------------------------------
# THE TWELFTH FAMILY: THREAD-PRIVATE MEMORY, aimed at two holes at once.
#
# 132 memory opcodes implicitly read SR_LOCAL_X and SR_LOCAL_Y - Apple declares it at MCInstrDesc
# +24 - so their address is formed from the thread's position without any operand saying so. None
# of the 132 appears in 1,795 corpus objects and no probe has ever reached one, which is why they
# are still named plain `load` and `store`.
#
# A THREAD-PRIVATE ARRAY IS EXACTLY THAT SHAPE. An array in the `thread` address space that is too
# large for registers, or indexed by a value the compiler cannot fold, has to live in memory, and
# every lane needs its own copy - so the address is base plus something derived from where the
# thread sits. That is an implicit thread-position index by construction rather than by analogy.
#
# The same probes should reach op10234 and op11140, the only two opcodes in the ISA that touch SP
# (Apple's implicit-operand table again, and they are the entire membership of class 288). If a
# private array is stack-allocated they appear; if it is not, their absence here is itself
# evidence about what SP is for.
PV_HEAD = """#include <metal_stdlib>
using namespace metal;
kernel void k(device uint *u [[buffer(0)]], device float *f [[buffer(1)]],
              uint3 tg [[threadgroup_position_in_grid]],
              uint3 tp [[thread_position_in_threadgroup]]) {
"""
PV_TAIL = "}\n"


def pv_probes():
    P = {}
    def add(tag, body): P["pv:" + tag] = ("private", body)
    # A dynamic index the compiler cannot fold forces the array out of registers.
    for n in (8, 32, 128, 512):
        add("arr.u32.%d" % n,
            "  {{ uint a{i}[%d]; for (uint j = 0; j < %du; ++j) a{i}[j] = j + u[{i}]; "
            "u[400 + {i}] = a{i}[u[{i}] %% %du]; }}" % (n, n, n))
    add("arr.f32.128",
        "  {{ float a{i}[128]; for (uint j = 0; j < 128u; ++j) a{i}[j] = (float)j * f[{i}]; "
        "f[400 + {i}] = a{i}[u[{i}] %% 128u]; }}")
    add("arr.half.128",
        "  {{ half a{i}[128]; for (uint j = 0; j < 128u; ++j) a{i}[j] = (half)j; "
        "f[400 + {i}] = (float)a{i}[u[{i}] %% 128u]; }}")
    add("arr.struct.64",
        "  {{ struct S {{ float a; uint b; }} s{i}[64]; "
        "for (uint j = 0; j < 64u; ++j) {{ s{i}[j].a = (float)j; s{i}[j].b = j; }} "
        "u[400 + {i}] = s{i}[u[{i}] %% 64u].b; }}")
    # WRITE THEN READ at a dynamic index - the store and the load of the same private array, which
    # is the matched pair that is missing for these opcodes.
    add("arr.rw.256",
        "  {{ uint a{i}[256]; for (uint j = 0; j < 256u; ++j) a{i}[j] = 0u; "
        "a{i}[u[{i}] %% 256u] = u[{i} + 1]; u[400 + {i}] = a{i}[u[{i} + 1] %% 256u]; }}")
    # A pointer INTO the private array, so the address is a value rather than an index.
    add("ptr.private",
        "  {{ uint a{i}[64]; thread uint *p{i} = a{i} + (u[{i}] %% 64u); "
        "*p{i} = u[{i} + 1]; u[400 + {i}] = a{i}[{i}]; }}")
    # A big enough array that it cannot be anything but memory.
    add("arr.huge",
        "  {{ uint a{i}[2048]; for (uint j = 0; j < 2048u; ++j) a{i}[j] = j ^ u[{i}]; "
        "u[400 + {i}] = a{i}[u[{i}] %% 2048u]; }}")
    # A CALL with a private array passed by pointer - if there is a stack, this needs it.
    add("call.private",
        "  {{ uint a{i}[64]; for (uint j = 0; j < 64u; ++j) a{i}[j] = j; "
        "u[400 + {i}] = sum64(a{i}, u[{i}] %% 64u); }}")
    # THE CONTROLS. A threadgroup array indexed by the thread position is the access these 132
    # opcodes would be confused with, and it is already known to use different opcodes.
    add("ctl.tg",
        "  {{ threadgroup uint t{i}[256]; t{i}[tp.x] = u[{i}]; "
        "threadgroup_barrier(mem_flags::mem_threadgroup); u[400 + {i}] = t{i}[tp.x ^ 1u]; }}")
    add("ctl.device", "  u[400 + {i}] = u[tp.x + {i}u];")
    return P


PV_HELPER = ("static uint sum64(thread uint *a, uint k) {\n"
             "  uint s = 0; for (uint j = 0; j < 64u; ++j) s += a[j] * (j == k ? 2u : 1u);\n"
             "  return s;\n}\n")


def pv_source(name, n):
    # Bodies built with a % operator have already collapsed their %% to %; the ones built as
    # plain literals have not, and a stray %% is a syntax error in Metal rather than a warning.
    body = pv_probes()[name][1].replace("%%", "%")
    head = PV_HEAD.replace("kernel void k(", PV_HELPER + "kernel void k(")
    return head + "\n".join(body.format(i=i) for i in range(n)) + "\n" + PV_TAIL


def build_pv():
    P = pv_probes()
    ok = fail = 0
    msgs = {}
    for name in sorted(P):
        for n in SCALES:
            r = g17corpus.build("mv-%s-%d" % (name[3:], n), pv_source(name, n))
            if r in ("built", "cached"): ok += 1
            else:
                fail += 1; msgs[name] = r
    print("private-memory probe objects built or cached: %d   failed: %d" % (ok, fail))
    for k, v in sorted(msgs.items()):
        print("   %-22s %s" % (k, v[:78]))
    return msgs


# ---------------------------------------------------------------------------------------------
# THE THIRTEENTH FAMILY: IMAGEBLOCKS, which is what the 132 implicitly thread-indexed memory
# opcodes turned out to be.
#
# Apple declares SR_LOCAL_X and SR_LOCAL_Y as implicit uses of 132 memory opcodes at MCInstrDesc
# +24. None appears in 1,795 corpus objects, no probe had ever reached one, and 128 of them were
# named plain `load` or `store` - so a compiler selecting one for an ordinary load would compute
# an address nobody intended.
#
# Thread-private arrays were the first guess and they are NOT it: a private array too big for
# registers lowers to ordinary device memory, which is worth knowing on its own since it means
# this ISA has no stack-relative addressing for Metal to reach.
#
# It is the imageblock. `threadgroup_imageblock` is its own address space, an imageblock is
# indexed by the thread's position in the tile BY DEFINITION, and a single explicit-layout
# imageblock kernel emits op13075 - class 329, one of the 132, implicitly thread-indexed. That is
# a construct reaching one of them, which is what was missing.
IB_HEAD = """#include <metal_stdlib>
using namespace metal;
struct IBS { half4 c; float d; uint k; ushort s; uchar b; };
kernel void k(imageblock<IBS, imageblock_layout_explicit> ib,
              device uint *u [[buffer(0)]], device float *f [[buffer(1)]],
              device half *h [[buffer(2)]],
              ushort2 tp [[thread_position_in_threadgroup]]) {
"""
IB_TAIL = "}\n"


def ib_probes():
    P = {}
    def add(tag, body): P["ib:" + tag] = ("imgblk", body)
    C = "ushort2(tp.x + {i}, tp.y)"
    # ONE FIELD AT A TIME, so an opcode is attributed to an element width rather than to a struct.
    for fn, ty, dev, cast in (("k", "uint", "u", "(uint)"), ("d", "float", "f", "(float)"),
                              ("s", "ushort", "u", "(ushort)"), ("b", "uchar", "u", "(uchar)")):
        add("store.%s" % ty,
            "  ib.data(%s)->%s = %s%s[{i}];" % (C, fn, cast, dev))
        add("load.%s" % ty,
            "  %s[400 + {i}] = (%s)ib.data(%s)->%s;"
            % (dev, "uint" if dev == "u" else "float", C, fn))
    add("store.half4", "  ib.data(%s)->c = half4(h[{i}]);" % C)
    add("load.half4", "  h[400 + {i}] = ib.data(%s)->c.x;" % C)
    # READ THEN WRITE THE SAME SLOT - the matched pair the peer needs, from one kernel, so the
    # store's address operands and the load's are known to mean the same thing by construction.
    add("rw.uint", "  {{ threadgroup_imageblock IBS *p{i} = ib.data(%s); "
        "uint v{i} = p{i}->k; p{i}->k = v{i} + u[{i}]; u[400 + {i}] = v{i}; }}" % C)
    add("rw.float", "  {{ threadgroup_imageblock IBS *p{i} = ib.data(%s); "
        "float v{i} = p{i}->d; p{i}->d = v{i} + f[{i}]; f[400 + {i}] = v{i}; }}" % C)
    # THE WHOLE STRUCT AT ONCE, which should be the widest access the block supports.
    add("rw.struct", "  {{ threadgroup_imageblock IBS *p{i} = ib.data(%s); "
        "IBS v{i} = *p{i}; v{i}.k += u[{i}]; *p{i} = v{i}; u[400 + {i}] = v{i}.k; }}" % C)
    # THE COORDINATE AXIS. If the address really is the thread position, a coordinate the
    # compiler cannot fold and one it can should reach different opcodes - or the same one with a
    # different operand, which is just as informative.
    add("coord.const", "  ib.data(ushort2({i}, 0))->k = u[{i}];")
    add("coord.dyn", "  ib.data(ushort2((ushort)u[{i}], (ushort)u[{i} + 1]))->k = u[{i}];")
    add("coord.y", "  ib.data(ushort2(tp.x, tp.y + {i}))->k = u[{i}];")
    # PER-SAMPLE RATE rather than per-pixel, and the block's own descriptor queries.
    add("rate.sample", "  ib.data(%s, 0, imageblock_data_rate::sample)->k = u[{i}];" % C)
    add("q.width", "  u[400 + {i}] = (uint)ib.get_width() + {i}u;")
    add("q.height", "  u[400 + {i}] = (uint)ib.get_height() + {i}u;")
    add("q.samples", "  u[400 + {i}] = (uint)ib.get_num_samples() + {i}u;")
    add("q.colors", "  u[400 + {i}] = (uint)ib.get_num_colors(%s);" % C)
    add("q.coverage", "  u[400 + {i}] = (uint)ib.get_color_coverage_mask(%s, 0);" % C)
    return P


def ib_source(name, n):
    body = ib_probes()[name][1]
    return IB_HEAD + "\n".join(body.format(i=i) for i in range(n)) + "\n" + IB_TAIL


def build_ib():
    P = ib_probes()
    ok = fail = 0
    msgs = {}
    for name in sorted(P):
        for n in SCALES:
            r = g17corpus.build("mb-%s-%d" % (name[3:], n), ib_source(name, n))
            if r in ("built", "cached"): ok += 1
            else:
                fail += 1; msgs[name] = r
    print("imageblock probe objects built or cached: %d   failed: %d" % (ok, fail))
    for k, v in sorted(msgs.items()):
        print("   %-20s %s" % (k, v[:80]))
    return msgs


# ---------------------------------------------------------------------------------------------
# THE FOURTEENTH FAMILY: is `step` a name or a degenerate case?
#
# op9796, op9806 and op9836 are selected ONLY by step, and execution verifies they compute
# step(edge, x) - the probe that checked out contains no other arithmetic instruction, so the one
# instruction did the comparison and produced 1.0 or 0.0 by itself. op9832 and op9881 are the same
# story for min, max, fmin, fmax and clamp: ONE opcode computes min in one probe and max in
# another, so the comparison direction is an operand rather than part of the opcode.
#
# That is exactly the shape that made `saturate` the wrong name for op904 - the instruction was
# add-immediate-with-saturation and saturate was the case with the immediate at zero. So before
# naming any of these `step` or `min`, vary the thing that would be the degenerate part: the two
# values selected between, and the direction of the comparison.
CS_HEAD = """#include <metal_stdlib>
using namespace metal;
kernel void k(device float *f [[buffer(0)]], device half *h [[buffer(1)]],
              uint3 tg [[threadgroup_position_in_grid]]) {
"""
CS_TAIL = "}\n"


def cs_probes():
    P = {}
    def add(tag, body): P["cs:" + tag] = ("cmpsel", body)
    A, B = "f[{i}]", "f[{i} + 1]"
    HA, HB = "h[{i}]", "h[{i} + 1]"
    # step itself, as the control.
    add("step.f32_h", "  h[400 + {i}] = (half)step(%s, %s);" % (A, B))
    add("step.hh", "  h[400 + {i}] = step(%s, %s);" % (HA, HB))
    # THE SAME COMPARISON WITH OTHER CONSTANTS. If step is a degenerate case, these reach the same
    # opcode and the name is a compare-and-select-immediate; if they reach a different one, the
    # constants 0 and 1 are part of the instruction and `step` is the operation.
    for tag, t, e in (("01", "1.0f", "0.0f"), ("37", "3.0f", "7.0f"),
                      ("10", "0.0f", "1.0f"), ("neg", "-1.0f", "1.0f")):
        add("sel.f32_h.%s" % tag,
            "  h[400 + {i}] = (half)((%s >= %s) ? %s : %s);" % (B, A, t, e))
        add("sel.hh.%s" % tag,
            "  h[400 + {i}] = (%s >= %s) ? (half)%s : (half)%s;" % (HB, HA, t, e))
    # THE COMPARISON DIRECTION, which is what min and max would differ by if it is an operand.
    for tag, op in (("ge", ">="), ("gt", ">"), ("le", "<="), ("lt", "<"),
                    ("eq", "=="), ("ne", "!=")):
        add("cc.hh.%s" % tag,
            "  h[400 + {i}] = (%s %s %s) ? (half)3.0f : (half)7.0f;" % (HA, op, HB))
    # SELECTING BETWEEN TWO REGISTERS rather than two constants - the same comparison, a different
    # kind of second operand, which is how min and max are written by hand.
    add("sel.reg.hh", "  h[400 + {i}] = (%s < %s) ? %s : %s;" % (HA, HB, HA, HB))
    add("sel.reg.f32", "  f[400 + {i}] = (%s < %s) ? %s : %s;" % (A, B, A, B))
    add("min.hh", "  h[400 + {i}] = min(%s, %s);" % (HA, HB))
    add("max.hh", "  h[400 + {i}] = max(%s, %s);" % (HA, HB))
    return P


def cs_source(name, n):
    body = cs_probes()[name][1]
    return CS_HEAD + "\n".join(body.format(i=i) for i in range(n)) + "\n" + CS_TAIL


def build_cs():
    P = cs_probes()
    ok = fail = 0
    msgs = {}
    for name in sorted(P):
        for n in SCALES:
            r = g17corpus.build("mc-%s-%d" % (name[3:], n), cs_source(name, n))
            if r in ("built", "cached"): ok += 1
            else:
                fail += 1; msgs[name] = r
    print("compare-select probe objects built or cached: %d   failed: %d" % (ok, fail))
    for k, v in sorted(msgs.items()): print("   %-20s %s" % (k, v[:70]))
    return msgs


# ---------------------------------------------------------------------------------------------
# THE FIFTEENTH FAMILY: UNIFORM OPERANDS, aimed at the forms the device-buffer probes cannot reach.
#
# The classes with the most unnamed members - 49, 5, 312, 51, 50, 52 - all have a verified anchor
# now, so what is left in them is FORMS rather than functions. Their operand signatures say which
# forms: alongside the GPR32 shapes there are IRGPR32 ones, and IR is the uniform register file.
#
# Every probe written so far reads its operands from `device` buffers, which produce ordinary
# per-thread registers. A value that is the same for every thread in the dispatch - a scalar in
# the `constant` address space, or a struct field, or a value the compiler can prove uniform -
# lands in an IR register instead, and the instruction that consumes it is a different opcode.
#
# That is not a guess about the encoding: it is what the operand class in Apple's own table says,
# and it is the one axis fourteen families have held constant.
UF_HEAD = """#include <metal_stdlib>
using namespace metal;
struct P { float a; float b; uint u; int s; half h; };
kernel void k(device float *f [[buffer(0)]], device half *h [[buffer(1)]],
              device uint *u [[buffer(2)]], device int *s [[buffer(3)]],
              constant P &p [[buffer(4)]], constant float *cf [[buffer(5)]],
              constant uint *cu [[buffer(6)]],
              uint3 tg [[threadgroup_position_in_grid]],
              uint3 tp [[thread_position_in_threadgroup]]) {
"""
UF_TAIL = "}\n"


def uf_probes():
    P = {}
    def add(tag, body): P["uf:" + tag] = ("uniform", body)
    # ONE uniform source, then the other, then both - so an opcode is attributed to WHICH operand
    # is uniform and not merely to uniformity being present.
    F, H, U, S = "f[{i}]", "h[{i}]", "u[{i}]", "s[{i}]"
    CF, CH, CU, CS = "p.a", "p.h", "p.u", "p.s"
    for tag, op in (("mul", "%s * %s"), ("add", "%s + %s"), ("sub", "%s - %s")):
        add("f32.%s.ua" % tag, "  f[400 + {i}] = " + (op % (CF, F)) + ";")
        add("f32.%s.ub" % tag, "  f[400 + {i}] = " + (op % (F, CF)) + ";")
        add("f32.%s.uu" % tag, "  f[400 + {i}] = " + (op % (CF, "cf[2]")) + ";")
        add("h.%s.ua" % tag, "  h[400 + {i}] = " + (op % (CH, H)) + ";")
        add("h.%s.ub" % tag, "  h[400 + {i}] = " + (op % (H, CH)) + ";")
        add("f32h.%s.ua" % tag, "  h[400 + {i}] = (half)(" + (op % (CF, "(float)" + H)) + ");")
        add("hf32.%s.ub" % tag, "  h[400 + {i}] = (half)(" + (op % ("(float)" + H, CF)) + ");")
    for tag, op in (("mul", "%s * %s"), ("add", "%s + %s"), ("sub", "%s - %s"),
                    ("and", "%s & %s"), ("or", "%s | %s"), ("xor", "%s ^ %s"),
                    ("shl", "%s << (%s & 15u)"), ("shr", "%s >> (%s & 15u)"),
                    ("min", "min(%s, %s)"), ("max", "max(%s, %s)")):
        add("u32.%s.ua" % tag, "  u[400 + {i}] = " + (op % (CU, U)) + ";")
        add("u32.%s.ub" % tag, "  u[400 + {i}] = " + (op % (U, CU)) + ";")
        add("u32.%s.uu" % tag, "  u[400 + {i}] = " + (op % (CU, "cu[2]")) + ";")
    add("s32.mul.ua", "  s[400 + {i}] = %s * %s;" % (CS, S))
    add("s32.madd.ua", "  s[400 + {i}] = %s * %s + %s;" % (CS, S, S))
    add("s32.madd.ub", "  s[400 + {i}] = %s * %s + %s;" % (S, CS, S))
    add("s32.madd.uc", "  s[400 + {i}] = %s * %s + %s;" % (S, S, CS))
    # THREE-SOURCE float with each operand uniform in turn.
    add("f32.fma.ua", "  f[400 + {i}] = fma(%s, %s, %s);" % (CF, F, "f[{i} + 1]"))
    add("f32.fma.ub", "  f[400 + {i}] = fma(%s, %s, %s);" % (F, CF, "f[{i} + 1]"))
    add("f32.fma.uc", "  f[400 + {i}] = fma(%s, %s, %s);" % (F, "f[{i} + 1]", CF))
    add("f32h.fma.ua", "  h[400 + {i}] = (half)fma(%s, %s, %s);" % (CF, F, "f[{i} + 1]"))
    add("f32h.fma.uc", "  h[400 + {i}] = (half)fma(%s, %s, %s);" % (F, "f[{i} + 1]", CF))
    # SATURATING and comparison forms with a uniform operand.
    add("f32h.mul.sat.ua", "  h[400 + {i}] = (half)saturate(%s * %s);" % (CF, F))
    add("f32h.add.sat.ua", "  h[400 + {i}] = (half)saturate(%s + %s);" % (CF, F))
    add("f32.cmpsel.ua", "  f[400 + {i}] = (%s < %s) ? %s : %s;" % (CF, F, F, CF))
    add("h.cmpsel.ua", "  h[400 + {i}] = (%s < %s) ? %s : %s;" % (CH, H, H, CH))
    # A uniform INDEX rather than a uniform value, which is what an address computation consumes.
    add("addr.uniform", "  f[400 + {i}] = f[p.u + {i}u];")
    add("addr.uniform.scaled", "  f[400 + {i}] = f[p.u * 4u + {i}u];")
    return P


def uf_source(name, n):
    body = uf_probes()[name][1]
    return UF_HEAD + "\n".join(body.format(i=i) for i in range(n)) + "\n" + UF_TAIL


def build_uf():
    P = uf_probes()
    ok = fail = 0
    msgs = {}
    for name in sorted(P):
        for n in SCALES:
            r = g17corpus.build("mu-%s-%d" % (name[3:], n), uf_source(name, n))
            if r in ("built", "cached"): ok += 1
            else:
                fail += 1; msgs[name] = r
    print("uniform-operand probe objects built or cached: %d   failed: %d" % (ok, fail))
    for k, v in sorted(msgs.items())[:8]: print("   %-22s %s" % (k, v[:70]))
    return msgs


def main(argv=None):
    """The command-line entry, callable. It was inline under `if __name__`, so after the
    move neither the compatibility module nor the library could reach it: a shim IMPORTS
    this file, it does not execute it, and the result is an entry point that exits zero
    having done nothing. Sixth, seventh and eighth instance of that shape in this
    migration, which is why it is now the first thing checked per module.
    """
    argv = list(sys.argv if argv is None else argv)
    if "--build-uf" in argv: build_uf()
    elif "--build-cs" in argv: build_cs()
    elif "--build-ib" in argv: build_ib()
    elif "--build-pv" in argv: build_pv()
    elif "--build-mx" in argv: build_mx()
    elif "--build-in" in argv: build_in()
    elif "--build-fi" in argv: build_fi()
    elif "--build-fa" in argv: build_fa()
    elif "--build-hx" in argv: build_hx()
    elif "--build-tx" in argv: build_tx()
    elif "--build-tg" in argv: build_tg()
    elif "--build-rt" in argv: build_rt()
    elif "--build-stage" in argv: build_stage()
    elif "--build-special" in argv: build_special()
    elif "--build" in argv: build()
    elif "--name" in argv: name("--refresh" in argv)
    else: print(__doc__)


if __name__ == "__main__":
    main()
