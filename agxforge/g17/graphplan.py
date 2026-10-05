"""P1, finite graph admission: a planner that DERIVES regions, lifetimes, cuts and dispatch boundaries for
a finite tensor-op graph, and emits only what an existing measured cc route already admits.

Machine model 25.124.4 (design approved by Set C). The principle is Set C's: emit only MEASURED classes,
never infer. The planner decides where things go; cc's own route predicates (`cc.tensor_route`), cc's
compilation, the object author (`scanlink.author`, which runs mdgen/tensormetadata) and the hazard
reader (`tensorview`) decide whether that is admitted. A graph no route admits is refused with every
failing check named; nothing partial is emitted. tlower is never called or modified here.

INPUT. `Graph(tensors, nodes)`: `Tensor(name, rows, cols, dtype, role)` with dtype "half" or "float" and
role "input" (a half activation), "weight" (a half B operand) or "value" (a float intermediate or
output); `MatMul(name, a, b, out, accumulate=False)`, D = A . B with M = a.rows, K = a.cols = b.rows,
N = b.cols. Row stages (softmax, gelu, layernorm, scalar) are refused in v1 by name: the released
attention and stream classes keep their own routes.

REGIONS (Set C's constraints 1, 2 and 4). The binding signature is the one `tensormetadata.validate`
admits: buffer 1 half inputs, buffer 2 half weights, buffer 3 every float value.
- buffer 1: whole-row A offsets, offset % (2 K) == 0 for every half A read (25.102.2);
- buffer 2: B offsets 0 or even 2..47,104, cc's `_measured_tensor_stream_offset` (never widened);
- buffer 3: fp32, offset % 4 == 0, and pairwise DISJOINT: bump allocation, no reuse in v1;
- every offset element-aligned, the precondition of tlower's displacement-to-base fold.
An offset outside its rule refuses; there is no inferred placement.

CUTS AND FEED MODES (constraint 3). Every edge that crosses a dispatch boundary is a memory bridge
(25.114). Inside a dispatch, an A-feed edge asks `tensorsched.choose_cuts`; the production bodies are
one simdgroup with short K, outside every measured cut domain, so the answer is the known-safe cut and
the edge is a memory bridge. A register feed is emitted only when `choose_cuts` says "chain.fused" AND the
dispatch is exactly cc's in-place adjacent chain (the released feed form), never otherwise.

DISPATCHES. Nodes in deterministic topological order; a dispatch grows while cc admits it (route,
compile, author, hazards); a node that breaks admission starts the next dispatch; a node that is not
admitted alone refuses the plan.

RUNTIME. Each dispatch says "compile-admitted, runtime-unadmitted: P3/P8" unless its code is byte-
identical to a `gemm_generic` generic.json class the common worker already runs, which is checked by
building that class, never assumed.
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field

from agxforge.g17 import cc, ir, tensorsched

TENSOR_STREAM_OFFSET_MIN = cc.TENSOR_STREAM_OFFSET_MIN
TENSOR_STREAM_OFFSET_MAX = cc.TENSOR_STREAM_OFFSET_MAX
BINDINGS = ((1, 0, False), (2, 2, False), (3, 4, True))     # tensormetadata's measured signature
BUFFER_OF_ROLE = {"input": 1, "weight": 2, "value": 3}
ELEM = {"half": 2, "float": 4}
ROW_STAGES = ("softmax", "gelu", "layernorm", "scalar")


@dataclass(frozen=True)
class Tensor:
    name: str
    rows: int
    cols: int
    dtype: str
    role: str

    @property
    def bytes(self):
        return self.rows * self.cols * ELEM.get(self.dtype, 0)


@dataclass(frozen=True)
class MatMul:
    name: str
    a: str
    b: str
    out: str
    accumulate: bool = False


@dataclass(frozen=True)
class RowStage:
    """Accepted as input so a graph can SAY it has one; v1 refuses it by name."""
    name: str
    kind: str
    tensor: str


@dataclass(frozen=True)
class Graph:
    tensors: tuple
    nodes: tuple


@dataclass(frozen=True)
class Region:
    buffer: int
    offset: int
    bytes: int
    dtype: str


@dataclass
class Dispatch:
    nodes: tuple
    route: str
    code_sha256: str
    bindings: tuple
    metadata_class: str
    runtime: str
    program: object = None


@dataclass
class Plan:
    dispatches: list
    regions: dict
    lifetimes: dict
    feeds: dict                 # (producer, consumer) -> "register:A" | "memory_bridge"
    cut_decisions: dict         # (producer, consumer) -> the choose_cuts Decision
    boundaries: list            # (after node, reason)


class PlanRefused(Exception):
    def __init__(self, checks):
        self.checks = tuple(checks)
        super().__init__("graph refused: " + "; ".join(f"{n}: {d} [{s}]" for n, d, s in self.checks))


# ------------------------------------------------------------------------------------------ validation

def validate(graph):
    """Every check the graph fails, as (name, detail, source); empty when it is inside the v1 domain."""
    fails = []
    tensors = {}
    for t in graph.tensors:
        if t.name in tensors:
            fails.append(("tensor", f"{t.name} declared twice", "graphplan"))
        tensors[t.name] = t
        if t.dtype not in ELEM:
            fails.append(("dtype", f"{t.name}: {t.dtype} (half and float only)", "cc route predicates"))
        if t.role not in BUFFER_OF_ROLE:
            fails.append(("role", f"{t.name}: role {t.role!r}", "graphplan"))
        if (t.role in ("input", "weight") and t.dtype != "half") or (t.role == "value" and t.dtype != "float"):
            fails.append(("dtype", f"{t.name}: a {t.role} is {'half' if t.role != 'value' else 'float'}, "
                          f"not {t.dtype}", "tensormetadata binding signature; cc route predicates"))
        if t.rows <= 0 or t.cols <= 0 or t.rows % 16 or t.cols % 16:
            fails.append(("tiles", f"{t.name}: {t.rows}x{t.cols} is not whole 16x16 tiles", "cc route predicates"))
    produced = {}
    for n in graph.nodes:
        if isinstance(n, RowStage):
            fails.append(("row_stage", f"{n.name}: row stage {n.kind!r} is refused in v1 (the released "
                          "attention and stream classes keep their own routes)", "MM 25.124.4"))
            continue
        if not isinstance(n, MatMul):
            fails.append(("node", f"{n!r} is not a MatMul", "graphplan"))
            continue
        a, b, o = (tensors.get(x) for x in (n.a, n.b, n.out))
        for label, t, name in (("a", a, n.a), ("b", b, n.b), ("out", o, n.out)):
            if t is None:
                fails.append(("tensor", f"{n.name}: {label} {name!r} is not declared", "graphplan"))
        if None in (a, b, o):
            continue
        if b.role != "weight":
            fails.append(("feed", f"{n.name}: B must be a half weight in buffer 2; a value fed as B is a "
                          "register mode (B, Bt) admitted only in cc's two-body square chain", "MM 25.103"))
        if a.role == "weight":
            fails.append(("feed", f"{n.name}: A may not be a weight", "cc route predicates"))
        if o.role != "value":
            fails.append(("output", f"{n.name}: out must be a float value in buffer 3", "tensormetadata"))
        if a.cols != b.rows:
            fails.append(("shape", f"{n.name}: A is {a.rows}x{a.cols}, B is {b.rows}x{b.cols}", "graphplan"))
        if a.cols > 256 or b.cols > 128:
            fails.append(("extent", f"{n.name}: K {a.cols}, N {b.cols}; multi-body tensor programs are admitted "
                          "for K <= 256 and N <= 128 (the runtime's gemm_generic stage rule); larger bodies "
                          "need the K loop, a single-body class", "agxforge.g17.runtime.TensorSpec"))
        if (o.rows, o.cols) != (a.rows, b.cols):
            fails.append(("shape", f"{n.name}: out is {o.rows}x{o.cols}, not {a.rows}x{b.cols}", "graphplan"))
        if n.out in produced and not n.accumulate:
            fails.append(("single_assignment", f"{n.out} written by {produced[n.out]} and {n.name}", "graphplan"))
        if n.accumulate and n.out not in produced:
            fails.append(("accumulate", f"{n.name} accumulates into {n.out}, which nothing wrote before it",
                          "cc memory-stream route (a body may accumulate into its C region)"))
        produced.setdefault(n.out, n.name)
    return fails


def order(graph):
    """Deterministic topological order: a node follows every writer of its inputs; ties keep the order
    the graph lists them in."""
    nodes = [n for n in graph.nodes if isinstance(n, MatMul)]
    writers = {}
    for i, n in enumerate(nodes):
        writers.setdefault(n.out, []).append(i)
    done, out = set(), []
    while len(out) < len(nodes):
        progressed = False
        for i, n in enumerate(nodes):
            if i in done:
                continue
            deps = [w for x in (n.a, n.b) for w in writers.get(x, []) if w != i]
            deps += [w for w in writers.get(n.out, []) if w < i]      # an accumulate follows the first write
            if all(d in done for d in deps):
                done.add(i); out.append(n); progressed = True
                break
        if not progressed:
            raise PlanRefused([("cycle", "the graph is not a DAG", "graphplan")])
    return out


# ----------------------------------------------------------------------------------------- allocation

def allocate(graph, nodes):
    """Regions by role; offsets follow Set C's constraints or the plan refuses."""
    tensors = {t.name: t for t in graph.tensors}
    cursor = {1: 0, 2: 0, 3: 0}
    regions, fails = {}, []
    first_use = []
    for n in nodes:
        for x in (n.a, n.b, n.out):
            if x not in first_use:
                first_use.append(x)
    for name in first_use + [t.name for t in graph.tensors if t.name not in first_use]:
        t = tensors[name]
        buf = BUFFER_OF_ROLE[t.role]
        if buf == 1:
            # BUFFER 1 HOLDS ONLY HALF A OPERANDS, M x K with cols = K: transposes are refused at entry
            # and a chained fp32 A goes to buffer 3. So 2 * cols IS 2K; route anything else here and
            # the whole-row modulus would be derived from the wrong extent (Set C review, 25.102.4).
            assert t.role == "input" and t.dtype == "half", (t.name, t.dtype, t.role)
            align = 2 * t.cols                      # whole rows of A (offsetA % 2K, K = t.cols)
        elif buf == 2:
            align = 2                               # even
        else:
            align = 4                               # fp32
        off = math.ceil(cursor[buf] / align) * align
        regions[name] = Region(buf, off, t.bytes, t.dtype)
        cursor[buf] = off + t.bytes
    for name, r in regions.items():
        t = tensors[name]
        if r.offset % ELEM[r.dtype]:
            fails.append(("alignment", f"{name}: offset {r.offset} is not a whole {r.dtype} element "
                          "(tlower.based folds a displacement to a base register in elements)", "tlower.based"))
        if r.buffer == 2 and not cc._measured_tensor_stream_offset(r.offset):
            fails.append(("offsetB", f"{name}: B offset {r.offset} is outside 0 or even "
                          f"{TENSOR_STREAM_OFFSET_MIN}..{TENSOR_STREAM_OFFSET_MAX}", "MM 25.102.2; cc"))
        if r.buffer == 1 and r.offset % (2 * t.cols):
            fails.append(("offsetA", f"{name}: A offset {r.offset} is not whole rows", "MM 25.102.2"))
        if r.buffer == 3 and r.offset % 4:
            fails.append(("offsetC", f"{name}: C offset {r.offset} is not fp32-aligned", "MM 25.102.2"))
    return regions, fails


def disjoint(regions, buffer=3):
    """True when the regions of one buffer are pairwise non-overlapping (25.102.2's independent-group rule)."""
    spans = sorted((r.offset, r.offset + r.bytes) for r in regions.values() if r.buffer == buffer)
    return all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))


# ------------------------------------------------------------------------------------------- emission

def _build(nodes, tensors, regions, name, in_place=False, tail=False):
    """The cc IR for one dispatch. `in_place`: the released adjacent-chain form (every body writes C
    at offset 0 and each later body reads it as fp32 A), used only for a choose_cuts "chain.fused"."""
    a = ir.Buffer("A", 1, elem=ir.F16); b = ir.Buffer("B", 2, elem=ir.F16); c = ir.Buffer("C", 3, elem=ir.F32)
    fn = ir.Function(name, [a, b, c])
    bl = ir.Builder(fn, fn.block("entry"))
    bufs = {1: a, 2: b, 3: c}
    for i, n in enumerate(nodes):
        ta, tb = tensors[n.a], tensors[n.b]
        ra, rb, ro = regions[n.a], regions[n.b], regions[n.out]
        at = dict(M=ta.rows, N=tb.cols, K=ta.cols)
        if ta.dtype == "float":
            at["a_dtype"], at["b_dtype"] = "float", "half"
        if n.accumulate:
            at["accumulate"] = True
        if not in_place:
            for key, r in (("offsetA", ra), ("offsetB", rb), ("offsetC", ro)):
                if r.offset:
                    at[key] = r.offset
        elif rb.offset:
            at["offsetB"] = rb.offset
        bl.tensor_matmul(bufs[ra.buffer], bufs[rb.buffer], bufs[ro.buffer], **at)
    if tail:
        # the common worker's gemm_generic tail, C[0,0] += 1, which supplies SR156 (tools/
        # g17tensorcommonruntime.build_generic_program): only for a plan that is to match that class
        pos = bl.builtin("threadgroup_position_in_grid", name="group_x")
        loaded = bl.load(c, pos, type=ir.F32, name="c_value")
        bl.store_at(c, pos, bl.fadd(loaded, bl.const(struct.unpack("<I", struct.pack("<f", 1.0))[0]),
                                    name="c_plus_one"))
    bl.ret()
    ir.verify(fn)
    return fn


def admit(fn):
    """(route, program, author image, None) or (None, None, None, reason): cc's route, cc's compile,
    the object author (mdgen/tensormetadata), the binding signature and the hazard reader."""
    route = cc.tensor_route(fn)
    if isinstance(route, cc.TensorRouteRefusal):
        return None, None, None, f"route: {route}"
    try:
        program = cc.compile_function(fn)
    except Exception as e:                                    # cc's named refusal
        return None, None, None, f"compile: {type(e).__name__}: {e}"
    abi = program.abi()
    sig = tuple((b["index"], b["offset"], bool(b["written"])) for b in abi["bindings"])
    if sig != BINDINGS:
        return None, None, None, f"metadata: binding signature {sig} is not the measured {BINDINGS}"
    from agxforge.g17 import scanlink, tensorview
    try:
        image = scanlink.author(program)
    except Exception as e:
        return None, None, None, f"metadata: {type(e).__name__}: {e}"
    found = tensorview.hazards(tensorview.view(program.code))
    if found:
        return None, None, None, f"hazards: {len(found)} unwaited reads (tensorview.hazards)"
    return route, program, image, None


def offsets_reach_code(nodes, tensors, regions, name, tail, program):
    """A control on every admitted dispatch that carries an offset: the same dispatch built with every
    offset zeroed must compile to DIFFERENT bytes. cc's single-body path was measured dropping offsets
    silently (same bytes with and without, MM 25.124.4); this asks the same of every route."""
    if not any(regions[x].offset for n in nodes for x in (n.a, n.b, n.out)):
        return True
    zero = {k: Region(r.buffer, 0, r.bytes, r.dtype) for k, r in regions.items()}
    try:
        other = cc.compile_function(_build(nodes, tensors, zero, name, tail=tail))
    except Exception:
        return True                     # the zeroed variant is not even a program: offsets matter
    return other.code != program.code


def _edge_decision(producer, consumer, tensors):
    """The choose_cuts decision for one A-feed edge inside a dispatch."""
    stages = tuple(tensorsched.Stage(n.name, "fp16", tensors[n.a].rows * tensors[n.b].cols // 256 *
                                     (tensors[n.a].cols // 16)) for n in (producer, consumer))
    return tensorsched.choose_cuts(tensorsched.Chain(stages, fused_spill_stores=None, sg_per_core=1,
                                                     k_blocks=math.ceil(tensors[producer.a].cols / 32)))


def plan(graph, name="planned", tail=False):
    """Plan `graph` or raise PlanRefused with every failing check."""
    fails = validate(graph)
    if fails:
        raise PlanRefused(fails)
    nodes = order(graph)
    tensors = {t.name: t for t in graph.tensors}
    regions, fails = allocate(graph, nodes)
    if not disjoint(regions):
        fails.append(("disjoint", "fp32 value regions overlap", "MM 25.102.2"))
    if fails:
        raise PlanRefused(fails)

    # cut decisions on every A-feed edge between consecutive nodes
    decisions, feeds = {}, {}
    for p, q in zip(nodes, nodes[1:]):
        if q.a == p.out:
            d = _edge_decision(p, q, tensors)
            decisions[(p.name, q.name)] = d
    all_fused = bool(decisions) and len(decisions) == len(nodes) - 1 and all(
        d.schedule.kind == "chain.fused" for d in decisions.values())

    dispatches, boundaries, groups = [], [], []
    if all_fused:
        fn = _build(nodes, tensors, regions, name, in_place=True, tail=tail)
        route, program, image, why = admit(fn)
        if route == "adjacent_chain":
            groups.append((tuple(nodes), route, program, image))
            for k in decisions:
                feeds[k] = "register:A"
            # the in-place chain: every value lives at buffer-3 offset 0, overwritten body by body
            regions = {k: (Region(3, 0, r.bytes, r.dtype) if r.buffer == 3 else r) for k, r in regions.items()}
    if not groups:
        # a dispatch grows while cc admits it; a node that breaks admission closes the dispatch (if the
        # dispatch so far WAS admitted) and opens the next one, which must itself be admitted before it
        # closes. A dispatch that never became admitted refuses the plan.
        current, last, why_last = [], None, None

        def attempt(trial):
            fn = _build(trial, tensors, regions, name, tail=tail)
            route, program, image, why = admit(fn)
            if route is not None and not offsets_reach_code(trial, tensors, regions, name, tail, program):
                route, why = None, "offsets: the emitted bytes do not change when every offset is zeroed"
            return (route, program, image) if route is not None else None, why

        for n in nodes:
            got, why = attempt(current + [n])
            if got is not None:
                current, last, why_last = current + [n], got, None
                continue
            if last is None:
                if current:
                    raise PlanRefused([("admission", f"{', '.join(x.name for x in current + [n])} is not "
                                        f"admitted: {why}", "cc")])
                current, why_last = [n], why          # not admitted alone: the next node may complete it
                continue
            groups.append((tuple(current), *last))
            boundaries.append((current[-1].name, f"{n.name} breaks admission: {why}"))
            got, why = attempt([n])
            current, last, why_last = [n], got, (None if got else why)
        if last is None:
            raise PlanRefused([("admission", f"{', '.join(x.name for x in current)} is not admitted: {why_last}",
                                "cc")])
        groups.append((tuple(current), *last))
        for k in decisions:
            feeds[k] = "memory_bridge"
    where = {n.name: i for i, g in enumerate(groups) for n in g[0]}
    for (p, q), f in feeds.items():
        if where[p] != where[q] and f != "memory_bridge":
            raise PlanRefused([("feed", f"{p} -> {q} crosses a dispatch boundary with a register feed", "MM 25.114")])
    for g_nodes, route, program, image in groups:
        dispatches.append(Dispatch(tuple(n.name for n in g_nodes), route,
                                   hashlib.sha256(program.code).hexdigest(), BINDINGS,
                                   image.field_ledger.get("metadata class", ""),
                                   "compile-admitted, runtime-unadmitted: P3/P8", program))
    # lifetimes: ((dispatch, position) of the first touch, (dispatch, position) of the last), in order
    lifetimes = {}
    for i, n in enumerate(nodes):
        at = (where[n.name], i)
        for x in (n.a, n.b, n.out):
            first, _ = lifetimes.get(x, (at, at))
            lifetimes[x] = (first, at)
    return Plan(dispatches, regions, lifetimes, feeds, decisions, boundaries)


def check(plan_):
    """The plan's invariants, as a guard a hand-built or altered plan must also pass: every region in
    its offset rule, fp32 regions disjoint, no register feed across a dispatch boundary, the measured
    binding signature. Returns the failing checks."""
    fails = []
    for name, r in plan_.regions.items():
        if r.buffer == 2 and not cc._measured_tensor_stream_offset(r.offset):
            fails.append(("offsetB", name))
        if r.buffer == 3 and r.offset % 4:
            fails.append(("offsetC", name))
        if r.offset % ELEM.get(r.dtype, 1):
            fails.append(("alignment", name))
    in_place = bool(plan_.dispatches) and all(d.route == "adjacent_chain" for d in plan_.dispatches)
    if not in_place and not disjoint(plan_.regions):
        fails.append(("disjoint", "buffer 3"))
    where = {n: i for i, d in enumerate(plan_.dispatches) for n in d.nodes}
    for (p, q), f in plan_.feeds.items():
        if f != "memory_bridge" and where.get(p) != where.get(q):
            fails.append(("feed", f"{p}->{q}"))
    for d in plan_.dispatches:
        if tuple(d.bindings) != BINDINGS:
            fails.append(("bindings", d.nodes))
    return fails


def runtime_status(dispatch, generic_program):
    """"runtime-admitted: gemm_generic" only when the dispatch's code is byte-identical to a program the
    common worker's generic class builds (`generic_program`, built by the caller from its spec)."""
    same = generic_program is not None and hashlib.sha256(generic_program.code).hexdigest() == dispatch.code_sha256
    return "runtime-admitted: gemm_generic (byte-identical)" if same else "compile-admitted, runtime-unadmitted: P3/P8"
