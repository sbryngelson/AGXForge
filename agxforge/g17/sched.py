"""The first instruction scheduler: list scheduling of each block by latency-weighted critical path.

WHAT IT IS FOR, from measurement (agxforge/g17/schedmodel.py). ALU dependences are interlocked, so
order never changes a result, only time: a simple op's result is ready two issue intervals after
its producer issues, a slow op's (multiply, shift, msb, reverse) about five. A program that writes
one dependence chain after another stalls on every link; the same instructions interleaved issue
back to back. That is the whole of what this pass does - it reorders, and adds and removes nothing.

WHAT MAY MOVE. Only PURE operations: arithmetic, unary, ternary, constants and position builtins.
Every other op - loads, stores, atomics, barriers, imageblock and texture access, compares feeding
branches, phis, machine ops - is a FENCE, and fences keep their relative order exactly. A pure op
has no effects, so it may go anywhere between its operands' definitions and its first use. Loads
are fences and are never moved past each other or past stores, so memory order is untouched; among
ready ops they are PRIORITISED, since a late value waits anyway.

WHAT IT DOES NOT DO. It does not cross blocks, rename, or consider register pressure: a schedule
that raises pressure past the allocator's reach is refused by the allocator, and `schedule` is
opt-in for that reason (compile the result; if it refuses, compile the original).

    schedule(fn) -> number of ops whose position changed (fn is modified in place)
"""
from agxforge.g17 import ir, schedmodel

PURE = (ir.ARITH | ir.UNARY | ir.TERNARY) - {"fneg", "fabs"} | {"const", "builtin"}
SLOW_KINDS = {"mul", "mulhi", "shl", "shr", "sar", "sarv", "msb", "reverse"}
LOAD_KINDS = {"load", "load_tg", "load_vec_at", "imageblock_read", "uniform_load"}


def _latency(op):
    if op.kind in LOAD_KINDS:
        return 8                                  # late: start it early, its consumer waits anyway
    lat, _issue = schedmodel.SLOW if op.kind in SLOW_KINDS else schedmodel.SIMPLE
    return lat


def _schedule_block(blk):
    ops = blk.ops
    head = [o for o in ops if o.kind == "phi"]
    tail = [ops[-1]] if ops and ops[-1].kind in ("br", "br_cond", "ret") else []
    body = [o for o in ops if o not in head and o not in tail]
    if len(body) < 2:
        return 0
    idx = {id(o): i for i, o in enumerate(body)}
    by_dest = {id(o.dest): o for o in body if o.dest is not None}
    preds = {id(o): set() for o in body}
    last_fence = None
    for o in body:
        for a in o.args:
            p = by_dest.get(id(a))
            if p is not None:
                preds[id(o)].add(id(p))
        if o.kind not in PURE:
            if last_fence is not None:
                preds[id(o)].add(id(last_fence))
            last_fence = o
    # the terminator's operands must be defined before it; everything in body precedes it anyway
    succs = {id(o): set() for o in body}
    for o in body:
        for p in preds[id(o)]:
            succs[p].add(id(o))
    # latency-weighted longest path to the end of the block
    prio = {}
    for o in reversed(body):
        prio[id(o)] = _latency(o) + max((prio[s] for s in succs[id(o)]), default=0)
    ready_at = {id(o): 0 for o in body}
    remaining = {id(o): len(preds[id(o)]) for o in body}
    ready = [o for o in body if remaining[id(o)] == 0]
    out, clock = [], 0
    while ready:
        # the op whose operands are ready soonest, then the longest remaining path, then source order
        o = min(ready, key=lambda x: (max(ready_at[id(x)], clock), -prio[id(x)], idx[id(x)]))
        ready.remove(o)
        clock = max(clock, ready_at[id(o)]) + (2 if o.kind in SLOW_KINDS else 1)
        out.append(o)
        for s in succs[id(o)]:
            ready_at[s] = max(ready_at[s], clock - 1 + _latency(o))
            remaining[s] -= 1
            if remaining[s] == 0:
                ready.append(body[idx[s]])
    assert len(out) == len(body), "the dependence graph had a cycle"
    moved = sum(1 for a, b in zip(out, body) if a is not b)
    blk.ops[:] = head + out + tail
    return moved


def schedule(fn):
    """Schedule every block of `fn` in place. -> the number of ops that changed position."""
    return sum(_schedule_block(b) for b in fn.blocks)
