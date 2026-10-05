#!/usr/bin/env python3
"""SSA form over decoded G17 code.

The point is to stop looking at instructions one at a time. Once every value is connected
through defs, uses and control flow, an unknown opcode stops being an isolated encoding and
becomes a node with a role: something that consumes a grid index and produces an address,
something that feeds only a flag that only a branch reads, something whose result nothing uses.

Three things this rests on, none of them inferred:

    defs and uses     MCInstrDesc.NumDefs - the first NumDefs operands are definitions
    aliasing          tools/g17regs.py - versioning is per LEAF register, so a def of R3H
                      versions one location and leaves R3L alone, while a use of R3 reads both
    control flow      tools/g17cfg.py - branch target is offset+displacement, 1867/1867

MEMORY is versioned in the same graph, as one chain per RESOURCE rather than one global blob.
The memory family is scheduling classes 286, 287, 426 and 335, and two things about it are
mechanical rather than modelled:

    load vs store     MCInstrDesc.NumDefs. A load defines a register, a store defines none.
                      Over the corpus: 14,499 loads and 5,497 stores.
    the resource      the first GPR32tup2 operand that is a USE is the address, and in 18,673
                      of 19,996 accesses it is an MCExpr - a relocation naming a binding, which
                      is a compile-time resource identity rather than a computed address.

    load   uses    M:res_k
    store  uses    M:res_k  and defines  M:res_k+1

The 1,323 accesses whose address is a register instead of a relocation have no known resource,
so they are treated as touching EVERY resource in the function, in both directions. That
over-approximates, which is the safe direction: it can order two accesses that never actually
meet, and it cannot miss a pair that does. Address-expression refinement comes later; the
question this answers now is only which store can reach a given load.

EXEC is the third state domain, and it is the one with no architectural evidence behind it yet.
Scheduling class 25 - opcodes 575, 579, 582 and 577, about 4,000 instructions - consumes a FLAGR
and defines NO register. Those instructions are not nops, and the only architectural state left
for them to touch is the lane mask, so they are modelled as transforming a synthetic EXEC
location:

    EXEC_k+1 = op582(EXEC_k, FLAG0_3)

and branches READ EXEC without defining it, since whether a branch is taken depends on mask
state the branch does not name. That is the entire claim. Nothing here says push, pop, and-mask
or restore, and nothing should until execution proves it: op582 and op579 have IDENTICAL
descriptors and were measured to behave differently, so the tables cannot distinguish what these
instructions do, only that they do something the register model cannot see.

Instructions other than these are NOT given an EXEC use, even though every instruction executes
under the mask. Modelling that would make EXEC an operand of everything and drown the graph
without adding information.

Versioning per leaf rather than per register is what makes the mixed widths in real code
tractable. R3H, R3L, R2L and R4H all appear in one driver shader, and a model that versioned
whole registers would either lose the independence of the halves or falsely kill one on a
write to the other.

CONSERVATISM, deliberately. Every branch contributes BOTH edges, because a G17 branch names no
condition and the mask state that decides it is not in the instruction. So this over-approximates
reachability, which is the safe direction: it can merge values that never actually meet, and it
cannot miss a value that does.

SIMPLIFICATION. Raw phi placement over LEAF locations produces a lot of phis, because every
leaf is versioned independently and every branch contributes both edges. Two standard passes run
by default and neither is heuristic:

    trivial phi folding    a phi whose arguments all name one value (ignoring itself) IS that
                           value. Replace and iterate to a fixed point.
    dead phi elimination   a phi whose result no live instruction or live phi reads is removed.
                           Iterated, because dropping one phi can kill its arguments' only use.

Pass simplify=False to build() to see the unsimplified graph.

    python3 tools/g17ssa.py <object-or-cache-dir>          SSA listing
    python3 tools/g17ssa.py <object> --chains              def-use chains, most-used first
    python3 tools/g17ssa.py <object> --raw                 skip simplification
    python3 tools/g17ssa.py <object> --memory              per load, which stores can reach it
"""
import collections, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The sys.path inserts are gone: every sibling this module reaches - model, regs, agxdis, machobj -
# is a package module now, so they are imported rather than located by path. Nothing here anchors
# data or a native helper, so there is no anchor to re-level.
from agxforge.g17 import cfg as g17cfg, model as g17model, regs as g17regs


class Phi:
    """A merge point for one leaf location. Not a real instruction."""
    __slots__ = ("leaf", "version", "args", "block")

    def __init__(self, leaf, block):
        self.leaf, self.block = leaf, block
        self.version, self.args = None, []


MEMORY_SCHED = frozenset((286, 287, 426, 335))
EXEC_SCHED = frozenset((25,))
BRANCH_SCHED = 6
EXEC = "EXEC"
ADDRESS_CLASS = "GPR32tup2"
UNKNOWN_RESOURCE = "M:?"


def memory_access(inst):
    """(kind, resource) for a memory instruction, else None.

    kind is "load" or "store" from NumDefs. resource is "M:0x..." when the address operand is a
    relocation, and None when it is a register, which means the resource is not known.
    """
    e = inst.opcode
    if e is None or e.sched not in MEMORY_SCHED:
        return None
    sig = e.signature()
    for i, (kind, value) in enumerate(inst.values):
        if i < len(sig) and sig[i] == ADDRESS_CLASS and not e.is_def(i):
            return ("load" if e.ndefs else "store",
                    "M:0x%x" % value if kind == "expr" else None)
    return ("load" if e.ndefs else "store", None)


def resources(insts):
    """Every named memory resource the function touches."""
    out = set()
    for i in insts:
        a = memory_access(i)
        if a and a[1]:
            out.add(a[1])
    return out


def _locations(inst, all_resources):
    """(defs, uses) as abstract location names: register leaves and memory resources."""
    defs, uses = [], []
    if inst.opcode is None:
        return defs, uses
    names = g17model.registers()
    for i, (kind, value) in enumerate(inst.values):
        if kind != "reg":
            continue
        reg = names.get(value)
        if reg is None:
            continue
        target = defs if inst.opcode.is_def(i) else uses
        target.extend(sorted(g17regs.leaves(reg)))
    if inst.opcode.sched in EXEC_SCHED:
        uses.append(EXEC)
        defs.append(EXEC)
    elif inst.opcode.sched == BRANCH_SCHED:
        uses.append(EXEC)
    access = memory_access(inst)
    if access:
        kind, res = access
        touched = [res] if res else sorted(all_resources) + [UNKNOWN_RESOURCE]
        uses.extend(touched)
        if kind == "store":
            defs.extend(touched)
    return defs, uses


def _fold_trivial(phis, uses):
    """A phi whose arguments name a single value, ignoring itself, is that value.

    Returns the replacement map. Iterated to a fixed point because folding one phi can make
    another trivial. A phi with no arguments left at all is unreachable and folds to version 0,
    the live-in value, rather than being silently dropped.
    """
    repl = {}

    def find(lv):
        seen = set()
        while lv in repl and lv not in seen:
            seen.add(lv)
            lv = repl[lv]
        return lv

    changed = True
    while changed:
        changed = False
        for table in phis.values():
            for leaf, phi in list(table.items()):
                me = (leaf, phi.version)
                args = {find((leaf, v)) for _, v in phi.args}
                args.discard(me)
                if len(args) <= 1:
                    repl[me] = args.pop() if args else (leaf, 0)
                    del table[leaf]
                    changed = True
    for off in uses:
        uses[off] = [find(lv) for lv in uses[off]]
    for table in phis.values():
        for leaf, phi in table.items():
            phi.args = [(b, find((leaf, v))[1]) for b, v in phi.args]
    return repl


def _drop_dead(phis, uses):
    """Remove phis nothing reads. Iterated: dropping one can remove its arguments' only use."""
    consumed = set()
    for us in uses.values():
        consumed |= set(us)
    live = set()
    changed = True
    while changed:
        changed = False
        for table in phis.values():
            for leaf, phi in table.items():
                me = (leaf, phi.version)
                if me in live or me not in consumed:
                    continue
                live.add(me)
                changed = True
                for _, v in phi.args:
                    if (leaf, v) not in consumed:
                        consumed.add((leaf, v))
    removed = 0
    for table in phis.values():
        for leaf, phi in list(table.items()):
            if (leaf, phi.version) not in live:
                del table[leaf]
                removed += 1
    return removed


def build(insts, simplify=True):
    """Returns (blocks, phis, defs, uses) with every value numbered.

    defs[inst] and uses[inst] are lists of (leaf, version). Version 0 of a leaf means live on
    entry - used before any definition reaches it.
    """
    blocks = g17cfg.build(insts)
    if not blocks:
        return {}, {}, {}, {}
    entry = min(blocks)
    dom = g17cfg.dominators(blocks, entry)
    idom = g17cfg.idoms(blocks, dom, entry)
    df = g17cfg.frontiers(blocks, idom)

    all_resources = resources(insts)
    per_inst = {i.offset: _locations(i, all_resources) for i in insts}
    defining = collections.defaultdict(set)          # leaf -> blocks that define it
    for b in blocks.values():
        for i in b.insts:
            for leaf in per_inst[i.offset][0]:
                defining[leaf].add(b.start)

    # phi placement: iterate the dominance frontier to a fixed point
    phis = collections.defaultdict(dict)             # block -> leaf -> Phi
    for leaf, sites in defining.items():
        work, seen = list(sites), set()
        while work:
            b = work.pop()
            for target in df.get(b, ()):
                if leaf in phis[target]:
                    continue
                phis[target][leaf] = Phi(leaf, target)
                if target not in sites and target not in seen:
                    seen.add(target)
                    work.append(target)

    children = collections.defaultdict(list)
    for b, p in idom.items():
        if p is not None:
            children[p].append(b)

    counter = collections.Counter()
    stack = collections.defaultdict(lambda: [0])     # leaf -> versions, 0 = live-in
    out_defs, out_uses = {}, {}

    def rename(b):
        pushed = []
        for leaf, phi in sorted(phis[b].items()):
            counter[leaf] += 1
            phi.version = counter[leaf]
            stack[leaf].append(phi.version)
            pushed.append(leaf)
        for i in blocks[b].insts:
            d, u = per_inst[i.offset]
            out_uses[i.offset] = [(leaf, stack[leaf][-1]) for leaf in u]
            versioned = []
            for leaf in d:
                counter[leaf] += 1
                stack[leaf].append(counter[leaf])
                pushed.append(leaf)
                versioned.append((leaf, counter[leaf]))
            out_defs[i.offset] = versioned
        for _, succ in blocks[b].succs:
            for leaf, phi in phis[succ].items():
                phi.args.append((b, stack[leaf][-1]))
        for c in sorted(children[b]):
            rename(c)
        for leaf in pushed:
            stack[leaf].pop()

    sys.setrecursionlimit(10000)
    rename(entry)
    if simplify:
        _fold_trivial(phis, out_uses)
        _drop_dead(phis, out_uses)
    return blocks, phis, out_defs, out_uses


def _load(path):
    from agxforge.g17 import machobj, agxdis
    if os.path.isdir(path):
        loc = machobj.locate(path + "/s.arc.metallib", path + "/out/object/0-0")
        f, sz = agxdis.sections(loc["obj"])
        t = loc["obj"][f:f + sz]
        return list(g17model.decode(t, loc["syms"]["_agc.main"]))
    blob = open(path, "rb").read()
    f, sz = agxdis.sections(blob)
    return list(g17model.decode(blob[f:f + sz], 0))


def _fmt(pairs):
    return ", ".join("%s_%d" % (l, v) for l, v in pairs)


def _memory_report(insts, blocks, phis, defs, uses):
    """For each load, the store or stores whose memory version it reads.

    A load reading M:res_k is reached by whatever defined version k: a store, a phi merging
    several stores, or version 0, meaning the resource was live on entry and no store in this
    function reaches the load.
    """
    by_offset = {i.offset: i for i in insts}
    store_of = {}
    for off, ds in defs.items():
        for loc, ver in ds:
            if loc.startswith("M:"):
                store_of[(loc, ver)] = off
    phi_of = {}
    for block, table in phis.items():
        for loc, phi in table.items():
            if loc.startswith("M:"):
                phi_of[(loc, phi.version)] = (block, phi)

    def sources(lv, seen=None):
        """Resolve a memory version to the set of stores that can produce it."""
        seen = seen or set()
        if lv in seen:
            return set()
        seen.add(lv)
        if lv in store_of:
            return {store_of[lv]}
        if lv in phi_of:
            block, phi = phi_of[lv]
            out = set()
            for _, v in phi.args:
                out |= sources((lv[0], v), seen)
            return out
        return {None}          # version 0: live on entry

    loads = reached = entry_only = 0
    lines = []
    for i in insts:
        a = memory_access(i)
        if not a or a[0] != "load":
            continue
        loads += 1
        for lv in uses.get(i.offset, []):
            if not lv[0].startswith("M:"):
                continue
            src = sources(lv)
            real = sorted(x for x in src if x is not None)
            if real:
                reached += 1
            else:
                entry_only += 1
            if len(lines) < 12:
                where = ", ".join("op%d@%08x" % (by_offset[o].opcode.id, o) for o in real) or "live-in"
                lines.append("  load op%-6d @%08x  reads %s_%d  from %s"
                             % (i.opcode.id, i.offset, lv[0], lv[1], where))
    print("loads %d   memory reads resolved to a store %d   reads of entry state %d"
          % (loads, reached, entry_only))
    for l in lines:
        print(l)


def main():
    insts = _load(sys.argv[1])
    blocks, phis, defs, uses = build(insts, simplify="--raw" not in sys.argv)
    by_offset = {i.offset: i for i in insts}
    if "--memory" in sys.argv:
        _memory_report(insts, blocks, phis, defs, uses)
        return
    if "--chains" in sys.argv:
        producer = {}
        for off, ds in defs.items():
            for lv in ds:
                producer[lv] = ("inst", off)
        for block, table in phis.items():
            for leaf, phi in table.items():
                producer[(leaf, phi.version)] = ("phi", block)
        consumers = collections.defaultdict(list)
        for off, us in uses.items():
            for lv in us:
                consumers[lv].append(off)
        live_in = sum(1 for lv in consumers if lv[1] == 0)
        dead = [lv for lv in producer if lv not in consumers]
        print("values defined %d   values used %d   live-in uses %d   defined-but-unused %d"
              % (len(producer), len(consumers), live_in, len(dead)))
        print("\nmost-consumed values:")
        for lv, cs in sorted(consumers.items(), key=lambda kv: -len(kv[1]))[:12]:
            src = producer.get(lv)
            if src is None:
                where = "live-in"
            elif src[0] == "phi":
                where = "phi@%08x" % src[1]
            else:
                where = "op%d@%08x" % (by_offset[src[1]].opcode.id, src[1])
            print("  %-12s %2d uses   from %s" % ("%s_%d" % lv, len(cs), where))
        return
    print("%d instructions, %d blocks, %d phi nodes"
          % (len(insts), len(blocks), sum(len(v) for v in phis.values())))
    for start in sorted(blocks):
        b = blocks[start]
        print("\n%08x:  preds=%s" % (start, " ".join("%08x" % p for p in b.preds) or "entry"))
        for leaf, phi in sorted(phis[start].items()):
            print("        %s_%d = phi(%s)" % (leaf, phi.version,
                  ", ".join("%s_%d" % (leaf, v) for _, v in phi.args)))
        for i in b.insts:
            d, u = defs.get(i.offset, []), uses.get(i.offset, [])
            arrow = (_fmt(d) + " <- ") if d else ""
            print("  %08x op%-6d %s%s" % (i.offset, i.opcode.id if i.opcode else -1, arrow, _fmt(u)))


if __name__ == "__main__":
    main()
