#!/usr/bin/env python3
"""Which bytes of a metadata class are TRUE CONSTANTS and which are generated from signature state.

A class in isa/g17-mdclass.json was recorded from ONE Apple kernel, so every byte in it that the
generator does not compute is an Apple-derived byte being copied forward. The mission is to shrink
that set until the linker generates the whole image from a structured signature. This tool says
which bytes are candidates for being constant and which certainly are not.

THE METHOD IS CONDITIONAL, and that is not a detail. Grouping first and comparing within the
group is what makes a weak global signal readable: a byte that varies across unrelated classes
says nothing, while the same byte constant within a class and varying across it is a class
constant. Scoring features globally across unrelated groups is exactly how this project got
register maps wrong for a session, and how an atomic's "reduction identity" survived three
kernels that turned out to be in different metadata classes.

WHAT `constant` MEANS HERE, precisely: constant across every cached kernel of that size. That is a
CANDIDATE, not a proof. A byte constant across 40 kernels that all happen to bind two device
buffers is a fact about those 40 kernels. Each group's population is reported next to its verdict
so the claim can be weighed, and a byte only earns "constant" once its group has been widened.

    python3 tools/g17classbytes.py                  per size-class, constant vs varying bytes
    python3 tools/g17classbytes.py --varying 456    where the varying bytes are, for one class
"""
import collections, json, os, re, struct, sys

from . import metal as g17metal

ISA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "isa")


def metadata(tag):
    """The __GPU_METADATA section bytes of a cached kernel, or None.

    __GPU_METADATA is the SEGMENT name of a section_64 whose section name is __compute, so the
    section record starts sixteen bytes before the string.
    """
    obj = os.path.join(g17metal.CACHE, tag, "out", "object", "0-0")
    if not os.path.exists(obj):
        return None
    raw = open(obj, "rb").read()
    i = raw.find(b"__GPU_METADATA")
    if i < 0:
        return None
    _addr, size = struct.unpack_from("<QQ", raw, i + 16)
    off, = struct.unpack_from("<I", raw, i + 32)
    return raw[off:off + size] if off + size <= len(raw) and size else None


# RESOURCE KINDS BEYOND BUFFERS. The signature named buffer argument TYPES and that was worth
# eight points; it still only COUNTED textures. A cube texture and a 1d texture are different
# kinds, and mr-tex.cube took its class from mr-tex.1d and differed in two bytes. Same for a
# raytracing intersector variant. Naming them is the same measurement, not a new idea.
RESKIND = re.compile(r"\b(texture\w*|depth\w*|sampler|acceleration_structure|"
                     r"intersect\w*|primitive_acceleration_structure|instance_acceleration_"
                     r"structure|imageblock|visible_function_table|ray_data)\b")

TYPED = re.compile(r"(device|constant|threadgroup)\s+([A-Za-z_][\w:<>]*)\s*[*&]?\s*(\w+)"
                   r"\s*\[\[\s*buffer\s*\(\s*(\d+)")

NAMED = re.compile(r"(?:device|constant|threadgroup)\s+[^,()]*?[*&]?\s*(\w+)\s*\[\[\s*buffer\s*\(\s*(\d+)\s*\)\s*\]\]")
_NAMED_SWAP = True

ARG = re.compile(r"(device|constant|threadgroup)\s+[^,()]*?\[\[\s*buffer\s*\(\s*(\d+)\s*\)\s*\]\]")


def source_signature(tag):
    """The signature a BACKEND would hand the linker, read from the kernel's own Metal source.

    Deliberately not read from the kernel's metadata. That is the thing being reproduced, and
    taking the key out of it would make any rule derived here unfalsifiable - the same circularity
    that let an operand map be verified against the witness it was fitted to. The source is what
    stands in for a backend that compiled the signature and knows it.

    Returns (bound buffer count, starts at buffer 0, any constant binding, declared resources).
    """
    from . import classgen as g17classgen
    src = g17classgen.source_text(tag)
    if not src:
        return None
    args = ARG.findall(src)
    if not args:
        return None
    # DECLARED versus BOUND, and they are different keys. The class key counts the buffers the
    # CODE refers to; the declared count is every resource in the signature. A parameter that is
    # never mentioned in the body is declared and not bound, and conflating them put every class
    # but one out of reach of this analysis.
    body = src[src.index("{", src.index("kernel")):] if "kernel" in src else src
    names = {num: nm for nm, num in NAMED.findall(src)}
    bound = []
    for kind, num in args:
        nm = names.get(num)
        if nm is None or re.search(r"\b%s\b" % re.escape(nm), body):
            bound.append((kind, int(num)))
    if not bound:
        return None
    idx = sorted(n for _k, n in bound)
    # THE ARGUMENT TYPE SET is part of the key. The section carries type reflection, so a
    # `device half *` and a `device uint *` do not describe alike; measured over 7,548 kernels
    # it takes structural-shape determination from 78.6% to 86.6%, which is more than every other
    # candidate dimension put together. Counting the types is worth nothing - it is knowing which.
    types = tuple(sorted({t for _k, t, _n, _i in TYPED.findall(src)}))
    # SCAN THE PARAMETER LIST, not the file. These probe sources declare every texture type and
    # use one, so scraping the whole file gave mr-tex.cube-4 and mr-tex.1d-4 identical signatures
    # - all sixteen kinds each - and they took the same donor and differed in two bytes.
    m = re.search(r"kernel\s+void\s+\w+\s*\(([^)]*)\)", src, re.S)
    kinds = tuple(sorted(set(RESKIND.findall(m.group(1) if m else "")))) 
    return (len(bound), idx[0] == 0,
            any(k == "constant" for k, _n in bound), len(args), types + kinds)


def kernel_param_list(src):
    r"""The kernel's parameter list, matched with BALANCED parentheses.

    source_signature uses r"kernel\s+void\s+\w+\s*\(([^)]*)\)" and [^)]* stops at the first
    close paren, which is inside `buffer(0)`. For mr-tex.1d-1 the whole parameter list it sees is
    'device uint *u [[buffer(0'.
    """
    m = re.search(r"kernel\s+void\s+\w+\s*\(", src)
    if not m:
        return ""
    i, depth, out = m.end(), 1, []
    while i < len(src) and depth:
        c = src[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if not depth:
                break
        out.append(c)
        i += 1
    return "".join(out)


def used_resource_kinds(tag):
    r"""The resource kinds the kernel USES - textures, samplers, acceleration structures.

    THE VALIDATION TARGET FOR RECOVERING `kinds` FROM A SECTION, and it exists because the reader
    already in source_signature does not work. Two defects compound there:

        the parameter list is truncated at the first `)`, which sits inside `buffer(0)`, so
        RESKIND scans one partial parameter. 1,322 corpus sources declare a resource kind and 85
        signatures carry one.

        and the fix for that alone re-creates the problem the RESKIND comment describes: the
        mr-tex probes declare every texture type in their parameter list and use one, so a
        DECLARED reading gives mr-tex.cube-4 and mr-tex.1d-4 identical kind sets.

    So this applies the consumes rule, which source_signature already applies to buffers - "A
    parameter that is never mentioned in the body is declared and not bound" - to resources. Under
    it cube-4 is (sampler, texture, texturecube) and 1d-4 is (sampler, texture, texture1d), which
    is what that comment wanted. 423 corpus kernels carry a kind set, against 85 today and 1,322
    read as declared.

    NOT WIRED INTO source_signature. That would move `sig` and therefore the contract key for
    1,237 kernels, and the change should be earned by a section-based reader that needs it rather
    than taken on the strength of the source reader being wrong.
    """
    from . import classgen as g17classgen
    src = g17classgen.source_text(tag)
    if not src:
        return None
    params = kernel_param_list(src)
    body = src[src.index("{", src.index("kernel")):] if "kernel" in src else src
    # SPLIT AT DEPTH ZERO. A plain split(",") cuts `texturecube<float, access::sample> tc` in half,
    # leaving the kind keyword in one piece and the parameter name in the other, and every texture
    # then reads as used.
    pieces, depth, cur = [], 0, []
    for c in params:
        if c in "<(":
            depth += 1
        elif c in ">)":
            depth -= 1
        if c == "," and depth == 0:
            pieces.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    pieces.append("".join(cur))
    out = set()
    for piece in pieces:
        if not RESKIND.search(piece):
            continue
        nm = re.findall(r"(\w+)\s*\[\[", piece)
        if not nm or re.search(r"\b%s\b" % re.escape(nm[-1]), body):
            out |= set(RESKIND.findall(piece))
    return tuple(sorted(out))


def collect():
    """{size: {tag: bytes}} over every cached kernel that has the section."""
    out = collections.defaultdict(dict)
    for d in sorted(os.listdir(g17metal.CACHE)):
        s = metadata(d)
        if s:
            out[len(s)][d] = s
    return out


def analyse(group):
    """Per byte offset: constant across the group, or the set of values it takes."""
    tags = sorted(group)
    n = len(next(iter(group.values())))
    const, vary = {}, {}
    for j in range(n):
        vals = {group[t][j] for t in tags}
        if len(vals) == 1:
            const[j] = vals.pop()
        else:
            vary[j] = vals
    return const, vary


def by_signature():
    """{signature key: {tag: metadata bytes}} - the grouping the class model actually uses."""
    out = collections.defaultdict(dict)
    for d in sorted(os.listdir(g17metal.CACHE)):
        sig = source_signature(d)
        if sig is None:
            continue
        s = metadata(d)
        if s:
            out[sig + (len(s),)][d] = s
    return out


def class_key(sig):
    """The class-model key for a source signature: "{n}{z}{c}" with an optional ".d{declared}".

    n is the bound-buffer count, z is present when the signature starts at buffer 0, c when any
    binding is `constant`. This is the whole-program layer's own key, reproduced here so the
    grouping is the one the class model uses rather than one I invented.
    """
    n, z, c, decl = sig[:4]
    return "%d%s%s" % (n, "z" if z else "", "c" if c else ""), decl


def per_class():
    """{class name: {tag: metadata}} - kernels grouped by the key that SELECTS the class,
    which is the grouping the mission asks for and the one that makes a weak signal readable."""
    classes = json.load(open(os.path.join(ISA, "g17-mdclass.json")))
    out = collections.defaultdict(dict)
    for d in sorted(os.listdir(g17metal.CACHE)):
        sig = source_signature(d)
        md = metadata(d)
        if sig is None or not md:
            continue
        key, decl = class_key(sig)
        for name, spec in classes.items():
            base, _, dd = name.partition(".d")
            if base != key:
                continue
            if dd and int(dd) != decl:
                continue
            if spec.get("size") != len(md):
                continue
            out[name][d] = md
    return out


def branches():
    """{(n, starts@0, has-constant, declared): {tag: metadata}} over the whole cache.

    This is the full signature branch set, not just the branches the recorded classes cover. A
    branch that contains more than one metadata SIZE is ambiguous: the signature does not
    determine the class there, and the linker must either gain a dimension or refuse the
    signature rather than pick one and hope.
    """
    out = collections.defaultdict(dict)
    for d in sorted(os.listdir(g17metal.CACHE)):
        sig = source_signature(d)
        md = metadata(d)
        if sig and md:
            out[sig][d] = md
    return out


def runs_of(offsets):
    runs, start, prev = [], None, None
    for j in sorted(offsets):
        if start is None:
            start = prev = j
        elif j == prev + 1:
            prev = j
        else:
            runs.append((start, prev))
            start = prev = j
    if start is not None:
        runs.append((start, prev))
    return runs


def norm(d):
    """A description as comparable fields, indexed by ORDER position rather than byte offset.

    Absolute offsets move when a variable-length region grows, so comparing them makes two
    identical structures look different - which is what made the byte-level view report 19
    classes where there is one.
    """
    out = {"ntables": len(d["tables"]), "nvectors": len(d["vectors"]),
           "extra": repr(sorted((d.get("extra") or {}).items()))}
    for i, off in enumerate(d["order"]):
        t = d["tables"][off]
        out[("vlen", i)] = t.get("vlen")
        out[("tlen", i)] = t.get("tlen")
        out[("slots", i)] = tuple(sorted((int(k), v) for k, v in (t.get("slots") or {}).items()))
        out[("fields", i)] = tuple(sorted((int(k), tuple(v))
                                          for k, v in (t.get("fields") or {}).items()))
        out[("tail", i)] = t.get("tail")
    for i, (_off, rows) in enumerate(sorted(d["vectors"].items())):
        # the CONTENTS, not just the length - a vector carries the binding records, and their
        # count and order are exactly the signature-derived part of the description
        out[("vec", i)] = len(rows)
    return out


def generative(group, verify=True):
    """How many description fields a branch needs beyond its constants, and whether transplanting
    exactly those reproduces every member byte-for-byte."""
    import copy
    from . import mdgen as M
    descs = {}
    for t, md in group.items():
        try:
            descs[t] = M.describe(md)
        except Exception:
            pass
    if len(descs) < 2:
        return None
    shapes = {(len(d["tables"]), len(d["vectors"])) for d in descs.values()}
    if len(shapes) > 1:
        return {"shapes": len(shapes)}
    N = {t: norm(d) for t, d in descs.items()}
    keys = set()
    for v in N.values():
        keys |= set(v)
    vary = [k for k in keys if len({N[t].get(k) for t in N}) > 1]
    res = {"kernels": len(descs), "fields": len(keys), "varying": sorted(map(str, vary)),
           "shapes": 1}
    if not verify:
        return res
    tags = sorted(descs)
    base = descs[tags[0]]
    ok = 0
    for t in tags:
        gen = copy.deepcopy(base)
        d = descs[t]
        for i in range(min(len(gen["order"]), len(d["order"]))):
            bo, do = gen["order"][i], d["order"][i]
            for what in ("vlen", "tlen", "fields", "slots", "tail"):
                gen["tables"][bo][what] = copy.deepcopy(d["tables"][do].get(what))
        # Vectors and `extra` are transplanted WHOLESALE rather than only when they differ:
        # a vector holds the binding records, which are the signature's own content and never a
        # class constant, and `extra` is by definition what the structure did not account for.
        gb = sorted(gen["vectors"]); db = sorted(d["vectors"])
        if len(gb) == len(db):
            for a, b in zip(gb, db):
                gen["vectors"][a] = copy.deepcopy(d["vectors"][b])
        gen["extra"] = copy.deepcopy(d.get("extra") or {})
        # RELAYOUT, not just transplant. build_from() writes at the description's recorded
        # offsets, so a transplanted table that is longer produces a section short by the
        # difference. g17classgen.relayout recomputes every offset from the block contents.
        try:
            from . import classgen as G
            if bytes(M.build_from(G.relayout(gen))) == bytes(group[t]):
                ok += 1
        except Exception:
            pass
    res["regenerated"] = ok
    return res


FAMILY = {"at-": "atomics", "cf-": "control flow", "bp_": "buffer patterns",
          "ad-": "addressing", "ac2-": "tensor matmul", "ab_": "device load/store",
          "bar_": "barriers", "mc-": "compare/select", "mf-": "float alu",
          "ms-": "scalar", "mt-": "threadgroup", "mr-": "reductions",
          "cm-": "tensor variants", "fm_": "fused multiply", "tg-": "threadgroup dims"}


def family_of(tag):
    for pre, name in FAMILY.items():
        if tag.startswith(pre):
            return name
    return None


def main():
    if "--families" in sys.argv:
        # THE MATRIX. One row per program family rather than per signature branch, because a
        # family is what the backend produces and what a regression suite can cover. A family
        # spanning several structural shapes is one the class model cannot yet build from a
        # signature, and saying which is more useful than an average.
        # Conditioned on family AND signature. A family legitimately contains several
        # signatures - atomics come with two and three buffers - so a family spanning several
        # shapes is not by itself a missing dimension. Grouping by the established key first is
        # the whole method.
        groups = collections.defaultdict(dict)
        for d in sorted(os.listdir(g17metal.CACHE)):
            f = family_of(d)
            sig = source_signature(d) if f else None
            md = metadata(d) if sig else None
            if md:
                groups[(f, sig)][d] = md
        print("THE MATRIX: every program family, and whether one class description covers it\n")
        print("   family             signature        kernels shapes fields  prog  regenerated")
        done = tot = 0
        for f in sorted(groups, key=str):
            g = groups[f]
            r = generative(g)
            name, sig = f
            lab = "%-18s %-24s" % (name, "%d/%s/%s/%d/%s" % (
                sig[0], "y" if sig[1] else "n", "y" if sig[2] else "n", sig[3],
                "+".join(t[:4] for t in (sig[4] if len(sig) > 4 else ()))[:12]))
            if r is None:
                print("   %s %7d  (one kernel)" % (lab, len(g)))
                continue
            tot += r.get("kernels", len(g))
            if r.get("shapes", 1) > 1:
                print("   %s %7d %6d %6s %5s  needs another dimension"
                      % (lab, len(g), r["shapes"], "-", "-"))
                continue
            done += r.get("regenerated", 0)
            print("   %s %7d %6d %6d %5d  %d of %d%s"
                  % (lab, len(g), 1, r["fields"], len(r["varying"]),
                     r.get("regenerated", 0), r["kernels"],
                     "" if r.get("regenerated") == r["kernels"] else "  <-- incomplete"))
        print("\n   %d of %d kernels in these families regenerate byte-identically" % (done, tot))
        return 0
    if "--generate" in sys.argv:
        print("EVERY BRANCH: how many description fields are class constants, how many the")
        print("backend must supply, and whether transplanting exactly those reproduces Apple's")
        print("own metadata byte for byte.\n")
        print("   signature            kernels  fields  program-derived  regenerated")
        tot = good = 0
        for sig, g in sorted(branches().items()):
            r = generative(g)
            if r is None:
                continue
            if r.get("shapes", 1) > 1:
                print("  %-20s %7d  %6s  %15s  %s"
                      % (str(sig), len(g), "-", "-", "%d shapes - not one class" % r["shapes"]))
                continue
            tot += r["kernels"]
            good += r.get("regenerated", 0)
            print("  %-20s %7d  %6d  %15d  %d of %d%s"
                  % (str(sig), r["kernels"], r["fields"], len(r["varying"]),
                     r.get("regenerated", 0), r["kernels"],
                     "" if r.get("regenerated") == r["kernels"] else "   <-- incomplete"))
        print("\n  %d of %d kernels regenerated byte-identically from their branch's constant"
              " description" % (good, tot))
        return 0
    if "--rules" in sys.argv:
        br = branches()
        print("EVERY SIGNATURE BRANCH IN THE CACHE, and whether it determines a class.")
        print("A branch carrying more than one metadata size is AMBIGUOUS - the linker must gain")
        print("a dimension there or refuse the signature.\n")
        print("   n  @0 const decl  kernels  sizes  verdict        constant/size  varying")
        det = amb = thin = 0
        for sig in sorted(br):
            g = br[sig]
            n, z, c, decl = sig[:4]
            sizes = collections.Counter(len(m) for m in g.values())
            if len(sizes) > 1:
                amb += 1
                print("  %2d  %-2s %-5s %-4d %7d  %5d  AMBIGUOUS       %-13s %s"
                      % (n, "y" if z else "n", "y" if c else "n", decl, len(g), len(sizes), "-",
                         " ".join("%d(x%d)" % (s, k) for s, k in sizes.most_common(4))))
                continue
            size = next(iter(sizes))
            if len(g) < 2:
                thin += 1
                print("  %2d  %-2s %-5s %-4d %7d  %5d  one kernel      %-13s -"
                      % (n, "y" if z else "n", "y" if c else "n", decl, len(g), 1, size))
                continue
            det += 1
            const, vary = analyse(g)
            print("  %2d  %-2s %-5s %-4d %7d  %5d  determines      %4d/%-8d %d in %d runs"
                  % (n, "y" if z else "n", "y" if c else "n", decl, len(g), 1,
                     len(const), size, len(vary), len(runs_of(vary))))
        print("\n  %d branches determine a single class, %d are ambiguous, %d have one kernel"
              % (det, amb, thin))
        return 0
    if "--class" in sys.argv:
        groups = per_class()
        classes = json.load(open(os.path.join(ISA, "g17-mdclass.json")))
        print("bytes of each CLASS, over the cached kernels whose signature selects it")
        print("  (constant means constant across those kernels - a candidate, not a proof)\n")
        print("  class     size  kernels  constant  varying  varying runs")
        for name in sorted(classes):
            g = groups.get(name, {})
            size = classes[name]["size"]
            if len(g) < 2:
                print("  %-9s %5d  %7d  %8s  %7s   (too few to say anything)"
                      % (name, size, len(g), "-", "-"))
                continue
            const, vary = analyse(g)
            runs, prev = 0, -2
            for j in sorted(vary):
                if j != prev + 1:
                    runs += 1
                prev = j
            print("  %-9s %5d  %7d  %8d  %7d  %5d"
                  % (name, size, len(g), len(const), len(vary), runs))
        return 0
    if "--signature" in sys.argv:
        groups = by_signature()
        print("kernels grouped by SOURCE signature (bound, starts@0, has constant, declared) + size")
        print("  n  @0  const  decl  size  kernels  constant  varying  runs")
        for key in sorted(groups):
            g = groups[key]
            n, z, c, decl, size = key
            if len(g) < 2:
                print("  %2d  %-3s %-5s  %-4d %5d  %7d  %8s  %7s" % (n, z, c, decl, size, len(g), "-", "-"))
                continue
            const, vary = analyse(g)
            runs = 0
            prev = -2
            for j in sorted(vary):
                if j != prev + 1:
                    runs += 1
                prev = j
            print("  %2d  %-3s %-5s  %-4d %5d  %7d  %8d  %7d  %4d"
                  % (n, z, c, decl, size, len(g), len(const), len(vary), runs))
        return 0
    groups = collect()
    if "--varying" in sys.argv:
        want = int(sys.argv[sys.argv.index("--varying") + 1])
        g = groups.get(want)
        if not g:
            print("no cached kernel has a %d-byte metadata section" % want)
            return 1
        const, vary = analyse(g)
        print("%d-byte class: %d kernels, %d constant bytes, %d varying"
              % (want, len(g), len(const), len(vary)))
        runs, start, prev = [], None, None
        for j in sorted(vary):
            if start is None:
                start = prev = j
            elif j == prev + 1:
                prev = j
            else:
                runs.append((start, prev))
                start = prev = j
        if start is not None:
            runs.append((start, prev))
        print("  varying RUNS (offset..offset, length, distinct values in the first byte):")
        for a, b in runs:
            print("     %4d..%-4d  len %-4d  %d" % (a, b, b - a + 1, len(vary[a])))
        return 0
    print("metadata sizes across the cache, and how much of each is constant WITHIN its size")
    print("  (constant here means constant across the kernels listed - a candidate, not a proof)\n")
    print("  size  kernels  constant  varying   zero-valued constants")
    for size in sorted(groups):
        g = groups[size]
        if len(g) < 2:
            print("  %4d  %7d  %8s  %7s   (single kernel - says nothing)" % (size, len(g), "-", "-"))
            continue
        const, vary = analyse(g)
        zeros = sum(1 for v in const.values() if v == 0)
        print("  %4d  %7d  %8d  %7d   %d" % (size, len(g), len(const), len(vary), zeros))
    return 0


if __name__ == "__main__":
    sys.exit(main())
