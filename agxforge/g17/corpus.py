#!/usr/bin/env python3
"""Generate corpus objects aimed at forms the existing corpus cannot resolve.

Some opcodes resist not because their encoding is obscure but because the sample is thin. op11437
has 156 instances of its 8-byte form and four residual bits whose minority sits near 75; op11375's
8-byte form has 37. Five register operands in eight bytes means the HIGH bits of some register
fields never vary at that sample size, and no amount of analysis on a corpus that does not exercise
them will recover their positions.

The lever is the corpus. These opcodes are reachable from Metal - op11437 and op999 appear in
om-div and om-mod, op11375 in the min/max probes - so the fix is to compile more kernels that
emit them, with enough live values that the register allocator is forced to reach high registers.

    python3 tools/g17corpus.py --build          compile and cache the generated kernels
    python3 tools/g17corpus.py --list           print the sources without building

Objects land in ~/.cache/agxforge/agx/gen-* and are picked up by every tool that walks the cache.
"""
import ctypes, os, subprocess, sys

# siblings come from the package
from agxforge.g17 import target as g17target

# THE CORPUS AS A POPULATION, in the module that owns the corpus. Every capped measurement here
# used to read the first N lines of this file, which is the first N programs of a SORTED listing,
# and a sorted listing is not a sample. Reading it from one place means a tool cannot cap itself
# back into the front of the corpus by accident.
# ANCHORED ON THE CHECKOUT ROOT: two levels up from agxforge/g17/ where one sufficed from
# tools/. Native helpers stay in tools/ where the Makefile builds them.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ISA = os.path.join(ROOT, "isa")
CORPUS = os.path.join(ISA, "g17-corpus-programs.jsonl")


def corpus_programs(limit=None, path=None):
    """Programs sampled ACROSS the corpus, not taken from the front.

    The peer found this hazard on the metadata side and it is what made me check here: a static
    check capped at sixty families walked the cache sorted, stopped at `divlan`, and had never
    once looked at tgm- - a whole class of kernels outside every measurement that used it.

    It had already bitten this side without being recognised. The held-out score read 26.4% until
    composition was checked, because small kernels are two thirds padding and the front of the
    corpus is small kernels. And g17why --bits over the first 200 programs blames op586, op10372,
    op12646, op10295 and op11462; over 200 sampled across the same corpus it blames select.cc,
    select, fselect, texture.read, texture.sample and texture.gather - six opcodes that appear
    nowhere in the first view.

    A stride costs nothing: the file is read either way, and the cap was never protecting anything
    but the loop body.
    """
    lines = open(path or CORPUS).read().splitlines()
    if not limit or limit >= len(lines):
        return lines
    stride = max(1, len(lines) // limit)
    return lines[::stride][:limit]

# PORTABILITY. What is specific to one machine rather than to one GPU lives here; what is specific
# to the GPU lives in g17target, which is the single place the target is named. LIBACCEL used to
# be an absolute path into one user's home directory, so the corpus could not be rebuilt anywhere
# else at all - and a corpus that cannot be rebuilt makes every number measured against it
# unfalsifiable by anyone but its author.
#
# The rest of this file is generation-neutral, because Metal source and the archive container are.
# `xcrun metal-lipo -info` on any archived metallib lists the slices the toolchain knows about.
CACHE = os.environ.get("AGXFORGE_CACHE") or os.path.expanduser("~/.cache/agxforge/agx")
ARCH = g17target.ARCH
LIBACCEL = os.environ.get("AGXFORGE_LIBACCEL") or os.path.join(
    ROOT, "spike", "accel", "libaccel.dylib")

HEAD = """#include <metal_stdlib>
using namespace metal;
kernel void k(device uint *u [[buffer(0)]], device int *s [[buffer(1)]],
              uint3 tg [[threadgroup_position_in_grid]]) {
"""


def pressure(n, expr, decl="uint"):
    """A kernel with n independent results live at once, so the allocator must spread them.

    Each result is computed from a different pair of loaded values and all n are stored at the
    end, which keeps every one live across the whole body. That is what pushes the allocator into
    high register numbers, which is what makes the high bits of a register field vary.
    """
    body = ["  %s v%d = %s;" % (decl, i, expr % (i, i + 1)) for i in range(n)]
    body += ["  %s[%d] = (%s)v%d;" % ("u" if decl != "int" else "s", 300 + i,
                                      "uint" if decl != "int" else "int", i) for i in range(n)]
    return HEAD + "\n".join(body) + "\n}\n"


def sources():
    """{tag: source}. Each family targets opcodes that are stuck for want of register variety."""
    out = {}
    for n in (4, 8, 12, 16, 24, 32):
        # op11437 and op999 - the division and modulo lowering
        out["gen-div%d" % n] = pressure(n, "u[tg.x + %du] / u[tg.x + %du]")
        out["gen-mod%d" % n] = pressure(n, "u[tg.x + %du] %% u[tg.x + %du]")
        out["gen-sdiv%d" % n] = pressure(n, "s[tg.x + %du] / s[tg.x + %du]", "int")
        out["gen-smod%d" % n] = pressure(n, "s[tg.x + %du] %% s[tg.x + %du]", "int")
        # op11375 and op13576 - min, max and clamp
        out["gen-smax%d" % n] = pressure(n, "max(s[tg.x + %du], s[tg.x + %du])", "int")
        out["gen-umin%d" % n] = pressure(n, "min(u[tg.x + %du], u[tg.x + %du])")
        out["gen-clamp%d" % n] = pressure(
            n, "clamp(u[tg.x + %du], 1u, u[tg.x + %du])")
        # op424 and its family - bitwise register-register. The 10-byte reg-form never exercises
        # slot bits 4 and 5 of operand 4 in the existing corpus (its registers top out at slot 14),
        # so the field's upper half is CONSISTENT but untested there. High pressure forces them.
        out["gen-and%d" % n] = pressure(n, "u[tg.x + %du] & u[tg.x + %du]")
        out["gen-or%d" % n] = pressure(n, "u[tg.x + %du] | u[tg.x + %du]")
        out["gen-xor%d" % n] = pressure(n, "u[tg.x + %du] ^ u[tg.x + %du]")
        # op11462 (carry, 14-byte form) - signed COMPARISONS whose result is stored. The corpus
        # has one instance per c2-s-* kernel and fourteen in total, which is why its b2[6] and
        # b11[2] resist. Many independent comparisons live at once forces both variety and volume.
        for rel in ("==", "!=", "<", "<=", ">", ">="):
            out["gen-cmp%s%d" % ({"==": "eq", "!=": "ne", "<": "lt", "<=": "le",
                                  ">": "gt", ">=": "ge"}[rel], n)] = pressure(
                n, "s[tg.x + %%du] %s s[tg.x + %%du]" % rel, "int")
        # op12691 (load, 14-byte form) - the grid-dimension probes reach it. Vary how many
        # dimensions are read and how they combine, with the results held live.
        out["gen-grid%d" % n] = HEAD.replace(
            "uint3 tg [[threadgroup_position_in_grid]]",
            "uint3 tg [[threadgroup_position_in_grid]], uint3 tgg [[threadgroups_per_grid]],\n"
            "              uint3 tpt [[threads_per_threadgroup]]") + "\n".join(
            ["  uint g%d = tgg.%s * tpt.%s + %du;" % (i, "xyz"[i % 3], "xyz"[(i + 1) % 3], i)
             for i in range(n)] +
            ["  u[%d] = g%d;" % (400 + i, i) for i in range(n)]) + "\n}\n"
        # op11491 - the predicate conjunction form
        out["gen-pred%d" % n] = pressure(
            n, "uint((u[tg.x + %du] > 9u) && (u[tg.x + %du] > 9u))")
        # op11365 - abs, compare and select
        out["gen-abs%d" % n] = pressure(n, "abs(s[tg.x + %du] - s[tg.x + %du])", "int")
        out["gen-sel%d" % n] = pressure(
            n, "u[tg.x + %du] > u[tg.x + %du] ? u[tg.x] : u[tg.x + 1u]")
    return out


def build(tag, src):
    d = os.path.join(CACHE, tag)
    os.makedirs(d, exist_ok=True)
    if os.path.exists(d + "/out/object/0-0"):
        # A tag identifies a directory, not a program. Never report an old
        # object's success for new source, and preserve the old evidence.
        try:
            with open(d + "/s.metal", "rb") as cached_source:
                retained = cached_source.read()
        except OSError:
            return "CACHE REFUSED: retained Metal source is unavailable"
        if retained != src.encode("utf-8"):
            return "CACHE REFUSED: requested Metal source differs; use a new probe tag"
        return "cached"
    open(d + "/s.metal", "w").write(src)
    r = subprocess.run(["xcrun", "metal", "-o", d + "/s.metallib", d + "/s.metal"],
                       capture_output=True, text=True)
    if r.returncode:
        return "COMPILE FAILED: " + (r.stderr.strip().splitlines() or ["?"])[-1][:90]
    L = ctypes.CDLL(LIBACCEL)
    L.ac_init()
    if L.ac_lib_from_data((d + "/s.metallib").encode()) != 0:
        return "ac_lib_from_data failed"
    # THE STAGE DECIDES WHICH ARCHIVE CALL. ac_archive builds a COMPUTE pipeline descriptor, and
    # handing it a vertex function fails a Metal assertion that takes the whole process down -
    # `computeFunction must not be nil` - rather than returning an error. ac_archive_vertex has
    # existed in accel.mm since the blit_vertex_* driver shaders were found, for exactly this
    # reason, and nothing here called it: every vertex source in the corpus was therefore
    # unwitnessable, and five of the last singletons were vertex shaders.
    import re as _re2
    m = _re2.search(r"\bvertex\s+[\w:<>,\s]*?\s+(\w+)\s*\(", src)
    frag = _re2.search(r"\bfragment\s+[\w:<>,\s]*?\s+(\w+)\s*\(", src)
    if m and frag:
        # A vertex AND a fragment function: the rasterizing path, which ac_archive_vertex cannot take.
        if L.ac_archive_render(m.group(1).encode(), frag.group(1).encode(),
                               (d + "/s.arc.metallib").encode()) != 0:
            return "ac_archive_render failed"
    elif m:
        if L.ac_archive_vertex(m.group(1).encode(),
                               (d + "/s.arc.metallib").encode()) != 0:
            return "ac_archive_vertex failed"
    elif L.ac_archive(b"k", (d + "/s.arc.metallib").encode()) != 0:
        return "ac_archive failed"
    for cmd in (["xcrun", "metal-lipo", "-thin", ARCH, "-output", d + "/s.nat",
                 d + "/s.arc.metallib"],):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            return "lipo failed: " + r.stderr.strip()[:80]
    subprocess.run(["rm", "-rf", d + "/out"], check=True)
    os.makedirs(d + "/out")
    r = subprocess.run(["xcrun", "metal-source", "--flatbuffers=json", "-f", "-o=" + d + "/out",
                        d + "/s.nat"], capture_output=True, text=True)
    if r.returncode:
        return "metal-source failed: " + r.stderr.strip()[:80]
    return "built"


def main():
    src = sources()
    if "--list" in sys.argv:
        for tag in sorted(src):
            print("=== %s ===\n%s" % (tag, src[tag]))
        return
    ok = fail = cached = 0
    for tag in sorted(src):
        r = build(tag, src[tag])
        if r == "built":
            ok += 1
        elif r == "cached":
            cached += 1
        else:
            fail += 1
            print("%-18s %s" % (tag, r))
    print("built %d, cached %d, failed %d, of %d kernels" % (ok, cached, fail, len(src)))


if __name__ == "__main__":
    main()
