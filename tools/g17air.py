#!/usr/bin/env python3
"""THE THIRD VIEW: AIR semantics, next to the parsed G17 and the executed behaviour.

Most of this project's wrong conclusions came from having only two views. Today's was exact: with
a parse and an execution but no AIR, I spent an afternoon treating an exec-mask write as a branch,
because a masked-off block and a jumped-over block are indistinguishable in both of those views.
AIR distinguishes them immediately - it has the source-level CFG, before Apple's backend chooses
how to realise it.

`xcrun metal -S` emits it as LLVM IR text, so this needs no LLVM installation at all.

The comparison this supports:

    AIR basic blocks and conditional edges     what the program MEANS
    native instruction stream and branches     what Apple's backend EMITTED
    dispatch                                   what the hardware DID

A native branch count that does not match the AIR edge count is a lead in one direction or the
other: an unmodelled control-flow form, or a branch recogniser producing false positives.
"""
import os, re, sys, subprocess, collections
_T = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _T); sys.path.insert(0, os.path.join(_T, "..", "spike", "accel", "re"))
import g17dis, g17cover, machobj, agxdis

def air_of(metal_path, out=None):
    """AIR (LLVM IR) text for a .metal source."""
    out = out or metal_path.replace(".metal", ".air")
    r = subprocess.run(["xcrun", "metal", "-S", "-o", out, metal_path],
                       capture_output=True, text=True, env=dict(os.environ))
    if r.returncode: raise RuntimeError(r.stderr[:400])
    return open(out).read()

_LABEL = re.compile(r"^(\d+):", re.M)
_BRCOND = re.compile(r"^\s*br i1 [^,]+, label %(\w+), label %(\w+)", re.M)
_BRUNC = re.compile(r"^\s*br label %(\w+)", re.M)
_ICMP  = re.compile(r"^\s*%(\w+) = icmp (\w+) i(\d+) %?(\S+), (\S+)", re.M)

def cfg(air):
    """Basic blocks and edges of the AIR function, plus the comparisons that feed them."""
    entry = "0"
    blocks = [entry] + _LABEL.findall(air)
    cond = [(a, b) for a, b in _BRCOND.findall(air)]
    return dict(blocks=blocks, cond_edges=cond, uncond=_BRUNC.findall(air),
                cmps=[dict(name=n, pred=p, width=int(w), lhs=l, rhs=r) for n, p, w, l, r in _ICMP.findall(air)])

def native(cachedir):
    """Instruction families and branch sites of the compiled kernel."""
    loc = machobj.locate(cachedir + "/s.arc.metallib", cachedir + "/out/object/0-0")
    f, sz = agxdis.sections(loc["obj"]); t = bytes(loc["obj"][f:f+sz])
    ins = list(g17dis.walk(t, loc["syms"]["_agc.main"]))
    fam = collections.Counter(g17cover.family(t, o, l, k) for o, l, k in ins)
    br = [o for o, l, k in ins if g17cover.family(t, o, l, k) == "branch.4"]
    mask = [o for o, l, k in ins if l == 4 and t[o:o+4] == bytes.fromhex("1e00000e")]
    rest = [o for o, l, k in ins if l == 4 and t[o:o+4] == bytes.fromhex("3e03400e")]
    return dict(text=t, ins=ins, fam=fam, branches=br, masks=mask, restores=rest)

def compare(metal_path, cachedir):
    c = cfg(air_of(metal_path))
    n = native(cachedir)
    print("AIR      %d basic blocks, %d conditional edges, %d unconditional, %d icmp"
          % (len(c["blocks"]), len(c["cond_edges"]), len(c["uncond"]), len(c["cmps"])))
    for m in c["cmps"]:
        print("           icmp %-5s i%-3d %s, %s" % (m["pred"], m["width"], m["lhs"], m["rhs"]))
    print("NATIVE   %d instructions, %d branch.4, %d exec-mask writes, %d exec.restore"
          % (len(n["ins"]), len(n["branches"]), len(n["masks"]), len(n["restores"])))
    ce, br = len(c["cond_edges"]), len(n["branches"])
    print("VERDICT  %d AIR conditional edge(s) -> %d native branch(es) + %d masked region(s): %s"
          % (ce, br, len(n["masks"]),
             "consistent" if ce == max(br, len(n["masks"])) else "MISMATCH - a lead"))
    return c, n

if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])
