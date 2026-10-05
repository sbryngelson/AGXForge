#!/usr/bin/env python3
"""Control flow graph over decoded G17 instructions.

Branches are scheduling class 6. Both corpus branch opcodes are 10 bytes with exactly two
operands and operand 0 is imm:0 in every instance, so the branch itself carries NO condition:
the target is the last immediate, and it is signed already.

    target = branch offset + displacement          1867 of 1867 corpus branches land on a real
                                                   instruction boundary under this rule, and
                                                   12.9% under offset+size+displacement

WHAT THIS DELIBERATELY DOES NOT CLAIM. Whether a branch is taken depends on mask state that the
branch does not name, so every branch here gets BOTH a branch edge and a fallthrough edge. That
over-approximates, which is the safe direction for dataflow.

It is also the direction the evidence demands. The flag instruction preceding a branch was
measured, in the other line of work here, NOT to be interchangeable: op582 gates a forward
branch on the compare predicate and op579 does not gate on it at all, so lowering a loop by
copying Apple's compare/op579/back-edge sequence produces something that never exits. A CFG
that inferred conditionality from the preceding flag op would have encoded that mistake. So the
guard is REPORTED - the nearest preceding instruction that defines a FLAGR register - and no
semantics are attached to it.

CONTROL DEPENDENCE is computed as well as dominance, and for this ISA it is the more important
of the two. A predicated GPU guards work by mask, so an instruction can be conditional with no
branch wrapped around it, and plain dominance cannot express that. Postdominators and control
dependence can: a block is control dependent on a branch when that branch decides whether the
block runs.

    python3 tools/g17cfg.py <applegpu-object>     blocks, edges, guards and control dependence
"""
import collections, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The sys.path inserts are gone: every sibling this module reaches - model, regs, agxdis, machobj -
# is a package module now, so they are imported rather than located by path. Nothing here anchors
# data or a native helper, so there is no anchor to re-level.
from agxforge.g17 import model as g17model, regs as g17regs

BRANCH_SCHED = 6
FLAG_CLASS = "FLAGR"


def is_branch(inst):
    return inst.opcode is not None and inst.opcode.sched == BRANCH_SCHED


def target_of(inst):
    """Branch target: offset plus the displacement operand, which the decoder signs."""
    imms = [v for k, v in inst.values if k == "imm"]
    return inst.offset + imms[-1] if imms else None


def defines_flag(inst):
    """True when the instruction writes a FLAGR-class operand. Structural, not semantic."""
    if inst.opcode is None:
        return False
    sig = inst.opcode.signature()
    return any(i < inst.opcode.ndefs and sig[i] == FLAG_CLASS for i in range(len(sig)))


class Block:
    __slots__ = ("start", "insts", "succs", "preds", "guard")

    def __init__(self, start):
        self.start, self.insts = start, []
        self.succs, self.preds, self.guard = [], [], None

    @property
    def end(self):
        return self.insts[-1].offset + self.insts[-1].size if self.insts else self.start

    def __repr__(self):
        return "block@%08x" % self.start


def build(insts):
    """Split into basic blocks and connect them. Returns {start: Block} in address order."""
    insts = sorted(insts, key=lambda i: i.offset)
    if not insts:
        return {}
    by_offset = {i.offset: i for i in insts}
    leaders = {insts[0].offset}
    for n, inst in enumerate(insts):
        if not is_branch(inst):
            continue
        t = target_of(inst)
        if t in by_offset:
            leaders.add(t)
        if n + 1 < len(insts):
            leaders.add(insts[n + 1].offset)

    blocks, current = {}, None
    for inst in insts:
        if inst.offset in leaders or current is None:
            current = Block(inst.offset)
            blocks[current.start] = current
        current.insts.append(inst)

    order = sorted(blocks)
    for n, start in enumerate(order):
        b = blocks[start]
        last = b.insts[-1]
        if is_branch(last):
            t = target_of(last)
            if t in blocks:
                b.succs.append(("branch", t))
            # Fallthrough is kept because the branch names no condition. See the docstring.
            if n + 1 < len(order):
                b.succs.append(("fallthrough", order[n + 1]))
            for i in reversed(b.insts[:-1]):
                if defines_flag(i):
                    b.guard = i
                    break
        elif n + 1 < len(order):
            b.succs.append(("fallthrough", order[n + 1]))
    for b in blocks.values():
        for _, t in b.succs:
            blocks[t].preds.append(b.start)
    return blocks


def dominators(blocks, entry):
    """Iterative dominator sets, keyed by block start."""
    order = sorted(blocks)
    dom = {b: set(order) for b in order}
    dom[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for b in order:
            if b == entry:
                continue
            preds = blocks[b].preds
            if not preds:
                new = {b}
            else:
                new = set(order)
                for p in preds:
                    new &= dom[p]
                new = new | {b}
            if new != dom[b]:
                dom[b] = new
                changed = True
    return dom


def idoms(blocks, dom, entry):
    """Immediate dominator per block."""
    out = {}
    for b in sorted(blocks):
        if b == entry:
            continue
        cands = dom[b] - {b}
        best = None
        for c in cands:
            if all(c == o or c not in dom[o] or o not in cands for o in cands):
                pass
            if best is None or (best in dom[c] and best != c):
                best = c
        # pick the candidate dominated by every other candidate
        for c in cands:
            if all(o in dom[c] for o in cands):
                best = c
                break
        out[b] = best
    return out


def frontiers(blocks, idom):
    """Dominance frontiers, the phi-placement sets."""
    df = collections.defaultdict(set)
    for b in sorted(blocks):
        preds = blocks[b].preds
        if len(preds) < 2:
            continue
        for p in preds:
            runner = p
            while runner is not None and runner != idom.get(b):
                df[runner].add(b)
                runner = idom.get(runner)
    return df


def postdominators(blocks):
    """Dominator sets on the reverse graph, rooted at a virtual exit.

    Blocks with no successors are the real exits; a virtual exit keyed None is their common
    root so that a function with several exits still has a single postdominator tree.
    """
    order = sorted(blocks)
    exits = [b for b in order if not blocks[b].succs]
    succs = {b: [t for _, t in blocks[b].succs] for b in order}
    preds_rev = {b: succs[b] for b in order}          # reverse graph: pred of b is its succ
    universe = set(order) | {None}
    pdom = {b: set(universe) for b in order}
    pdom[None] = {None}
    changed = True
    while changed:
        changed = False
        for b in order:
            ins = preds_rev[b] or [None] if b in exits else preds_rev[b]
            if not ins:
                new = {b, None}
            else:
                new = set(universe)
                for p in ins:
                    new &= pdom[p]
                new |= {b}
            if new != pdom[b]:
                pdom[b] = new
                changed = True
    return pdom


def ipostdoms(blocks, pdom):
    """Immediate postdominator per block: the candidate postdominated by all the others."""
    out = {}
    for b in sorted(blocks):
        cands = pdom[b] - {b}
        best = None
        for c in cands:
            if all(o == c or c in pdom.get(o, {o}) for o in cands):
                best = c
                break
        out[b] = best
    return out


def control_dependence(blocks):
    """{block: set of branch blocks that decide whether it runs}.

    Standard formulation: for every edge A->B where B does not postdominate A, every block from
    B up to but excluding the immediate postdominator of A is control dependent on A.
    """
    pdom = postdominators(blocks)
    ipdom = ipostdoms(blocks, pdom)
    cd = collections.defaultdict(set)
    for a in sorted(blocks):
        # Only a block with more than one successor decides anything. Without this, every
        # block on a straight line is reported as controlling its successor.
        if len(blocks[a].succs) < 2:
            continue
        for _, b in blocks[a].succs:
            if a in pdom.get(b, set()):
                continue                       # b postdominates a: not a decision point
            stop = ipdom.get(a)
            runner = b
            while runner is not None and runner != stop:
                cd[runner].add(a)
                runner = ipdom.get(runner)
    return cd


def load(path):
    from agxforge.g17 import machobj, agxdis
    blob = open(path, "rb").read()
    f, sz = agxdis.sections(blob)
    t = blob[f:f + sz]
    return t, list(g17model.decode(t, 0))


def main():
    from agxforge.g17 import machobj, agxdis
    path = sys.argv[1]
    if os.path.isdir(path):
        loc = machobj.locate(path + "/s.arc.metallib", path + "/out/object/0-0")
        f, sz = agxdis.sections(loc["obj"])
        t = loc["obj"][f:f + sz]
        insts = list(g17model.decode(t, loc["syms"]["_agc.main"]))
    else:
        t, insts = load(path)
    blocks = build(insts)
    entry = min(blocks) if blocks else None
    dom = dominators(blocks, entry)
    idom = idoms(blocks, dom, entry)
    df = frontiers(blocks, idom)
    cd = control_dependence(blocks)
    print("%d instructions, %d blocks, %d control-dependent blocks"
          % (len(insts), len(blocks), sum(1 for b in cd if cd[b])))
    for start in sorted(blocks):
        b = blocks[start]
        succ = " ".join("%s->%08x" % (k, t) for k, t in b.succs) or "exit"
        g = ("  guard=op%d@%08x" % (b.guard.opcode.id, b.guard.offset)) if b.guard else ""
        print("\n%08x  %2d insts  preds=%d  %s%s" % (start, len(b.insts), len(b.preds), succ, g))
        if start in df and df[start]:
            print("          frontier: %s" % " ".join("%08x" % x for x in sorted(df[start])))
        if cd.get(start):
            guards = []
            for a in sorted(cd[start]):
                g = blocks[a].guard
                guards.append("%08x%s" % (a, "(op%d)" % g.opcode.id if g else ""))
            print("          control-dependent on: %s" % " ".join(guards))


if __name__ == "__main__":
    main()
