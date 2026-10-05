#!/usr/bin/env python3
"""The register aliasing model: which registers overlap, and what a def actually kills.

G17 code mixes widths freely - R3H, R3L, R2L and R4H all appear in one driver shader - so
"defines R3H" and "defines R3" are not the same statement and neither kills the other outright.
Any dataflow over this ISA has to know that, and Apple's MCRegisterInfo says it exactly.

The atomic locations are the LEAF registers, the ones with no sub-registers. There are exactly
410 of them and MCRegisterInfo reports NumRegUnits as 410, which is what licenses treating
leaves as the units; tools/agx3meta.c enforces that equality and refuses to emit if it stops
holding. So:

    R0        leaves {R0L, R0H}                      a 32-bit register, two locations
    R0H       leaves {R0H}                           one location
    R36_R37   leaves {R36L, R36H, R37L, R37H}        a 64-bit tuple, four locations
    CTLFLOWST leaves {CTLFLOWST}                     a leaf in its own right

and a def of R3H kills one of R3's two leaves while leaving the other live.

    python3 tools/g17regs.py R0 R36_R37 FLAG0    show leaves and overlap for named registers
"""
import functools, os, subprocess, sys

# THE HELPER STAYS WHERE THE MAKEFILE BUILDS IT. This module does not merely locate agx3meta, it
# COMPILES it when the source is newer, so the anchor decides where a BINARY is written. Under
# tools/ `HERE` was the checkout's tools directory; from agxforge/g17/ the same expression would
# compile into the library tree from a source that is not there. A misanchored read fails loudly -
# a misanchored build does not, which is why the test for this move looks for an executable under
# agxforge/ after a forced rebuild rather than only checking that the path string is right.
#
# The sys.path insertion is gone with it: it existed so siblings under tools/ could be imported,
# this module imports none, and the library must not put tools/ on the import path.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS = os.path.join(ROOT, "tools")
META = os.path.join(TOOLS, "agx3meta")


@functools.lru_cache(maxsize=1)
def _model():
    src = META + ".c"
    if not os.path.exists(META) or os.path.getmtime(src) > os.path.getmtime(META):
        subprocess.run(["clang", "-O2", "-o", META, src], check=True)
    out = subprocess.run([META, "regmodel"], capture_output=True, text=True, check=True).stdout
    name, subs = {}, {}
    for line in out.splitlines():
        if line.startswith("#"):
            continue
        head, sub = line.split("|", 1)
        p = head.split()
        rid = int(p[0])
        name[rid] = p[1] if len(p) > 1 else ""
        subs[rid] = tuple(sub.split())
    by_name = {v: k for k, v in name.items() if v}
    return name, subs, by_name


def name(rid):
    return _model()[0].get(rid, "reg%d" % rid)


def ident(reg):
    """Accept a register id or an Apple register name."""
    if isinstance(reg, int):
        return reg
    return _model()[2].get(reg)


@functools.lru_cache(maxsize=None)
def leaves(reg):
    """The set of atomic locations this register covers, as register NAMES.

    Transitive, because a tuple lists intermediate registers as well as leaves: R36_R37 lists
    R36, R36L, R36H, R37, R37L, R37H and three overlapping triples, and only the four L/H
    entries are locations.
    """
    rid = ident(reg)
    if rid is None:
        return frozenset()
    _, subs, by_name = _model()
    direct = subs.get(rid, ())
    if not direct:
        return frozenset([name(rid)])
    out = set()
    for s in direct:
        out |= leaves(s)
    return frozenset(out)


def overlaps(a, b):
    """True when a def of `a` disturbs `b`."""
    return bool(leaves(a) & leaves(b))


def main():
    args = sys.argv[1:] or ["R0", "R3H", "R36_R37", "CTLFLOWST"]
    for r in args:
        ls = sorted(leaves(r))
        print("%-14s id=%-6s %2d leaves  %s" % (r, ident(r), len(ls), " ".join(ls)))
    if len(args) >= 2:
        print()
        for i in range(len(args)):
            for j in range(i + 1, len(args)):
                print("%-14s overlaps %-14s %s" % (args[i], args[j], overlaps(args[i], args[j])))


if __name__ == "__main__":
    main()
