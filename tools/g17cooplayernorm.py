#!/usr/bin/env python3
"""COOPERATIVE FP32 LAYERNORM: 32 lanes per row, 12 columns per lane, threadgroup partials.

The four-buffer LayerNorm (tools/g17layernorm.py) is one thread per row, 384 values live through
the mean. This one gives each row a threadgroup of 32 lanes: lane l owns columns 12l..12l+11,
centres them on the row's first value exactly as the baseline does, and the two reductions - the
shifted sum and the squared sum - go through ONE 32-word threadgroup scratchpad: each lane writes
its partial to word `lane`, a barrier, then every lane reads all 32 partials and sums them in the
same order, so all 32 lanes hold the same total and no second exchange is needed. The scratchpad
is reused for the second reduction behind a barrier that ends the first interval's reads.

THE CONTRACT (contract()), stated rather than derived, is what the linker and runtime consume:
system registers SR160 (lane = thread_position_in_grid.x) and SR161 (row = .y); exact grid
[32, rows, 1] with threadgroup [32, 1, 1]; static threadgroup storage 32 words = 128 bytes, 4-byte
aligned, no dynamic storage, no internal binding; bindings (1,0,r)(2,2,r)(3,4,r)(4,6,w) as the
baseline; three barriers, all reached uniformly (the program has no control flow). Nothing here
is a measured instruction, metadata or launch fact until the linker and runtime say so.

The arithmetic is the baseline's, in a different summation order: the FP64 reference and the
2e-5 (1 + |ref|) budget are unchanged, and the FP32 result is expected to differ from the
one-thread program's by the reordering.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
LANES, COLUMNS = 32, 384
BARRIERS = 3            # the contract's barrier count: partial sums, scratch reuse, squared partials
PER_LANE = COLUMNS // LANES            # 12
EPSILON = 1e-12


def float_bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def contract(rows):
    return dict(program="minilm_layernorm_cooperative", rows=rows, columns=COLUMNS,
                grid=[LANES, rows, 1], threadgroup=[LANES, 1, 1], lanes_per_row=LANES, columns_per_lane=PER_LANE,
                system_registers=[160, 161], lane="SR160 = thread_position_in_grid.x", row="SR161 = thread_position_in_grid.y",
                threadgroup_storage=dict(kind="static", words=LANES, bytes=4 * LANES, alignment=4, dynamic=False,
                                         reuse="one scratchpad, both reductions, separated by a barrier"),
                internal_bindings=[], bindings=[[1, 0, "r"], [2, 2, "r"], [3, 4, "r"], [4, 6, "w"]],
                barriers=3, barrier_scope="threadgroup", uniform_control_flow=True,
                synchronization=["store partial sum to word[lane]", "barrier", "read 32 words",
                                 "barrier (ends the reads before the scratchpad is reused)",
                                 "store squared partial to word[lane]", "barrier", "read 32 words"],
                exact_grid_required=True, bounds_checked=False,
                note="a proposal for the linker and runtime to measure against, not a measured launch or metadata fact")


def cooperative_layernorm_ir(rows=32, columns=COLUMNS, epsilon=EPSILON,
                             packed_parameters=False):
    import g17ir as ir
    if type(rows) is not int or not 1 <= rows <= 128 or columns != COLUMNS:
        raise ValueError("cooperative LayerNorm: 1..128 rows of 384 columns")
    if packed_parameters:
        source, parameters, output = [ir.Buffer(n, s, elem=ir.F32) for s, n in
                                      enumerate(("source", "parameters", "output"), 1)]
        gamma = beta = parameters
        fn = ir.Function("minilm_layernorm_cooperative_packed", [source, parameters, output])
    else:
        source, gamma, beta, output = [ir.Buffer(n, s, elem=ir.F32) for s, n in
                                       enumerate(("source", "gamma", "beta", "output"), 1)]
        fn = ir.Function("minilm_layernorm_cooperative", [source, gamma, beta, output])
    fn.declare_threadgroup(LANES, size=(LANES, 1, 1), alignment=4)     # the contract, stated in the program
    b = ir.Builder(fn, fn.block("entry"))
    lane = b.builtin("thread_position_in_grid", axis="x", name="lane")
    row = b.builtin("thread_position_in_grid", axis="y", name="row")
    row_base = b.mul(row, b.const(columns), name="row_base")
    lane_base = b.add(row_base, b.mul(lane, b.const(PER_LANE), name="lane_off"), name="lane_base")
    idx = [b.add(lane_base, ir.Imm(j)) if j else lane_base for j in range(PER_LANE)]
    values = [b.load(source, i, width="word") for i in idx]
    # the baseline's anchor: the row's first value, loaded by every lane
    anchor = b.load(source, row_base, width="word", name="anchor")
    negative_anchor = b.fmul(anchor, b.const(float_bits(-1.0)))
    shifted = [b.fadd(v, negative_anchor) for v in values]
    partial = b.const(float_bits(0.0))
    for v in shifted:
        partial = b.fadd(partial, v)
    b.store_tg(partial, lane)
    b.barrier()
    total = b.const(float_bits(0.0))
    for l in range(LANES):
        total = b.fadd(total, b.load_tg(b.const(l), name="p%d" % l))
    reciprocal = b.const(float_bits(1.0 / columns))
    mean = b.fmul(total, reciprocal)
    negative_mean = b.fmul(mean, b.const(float_bits(-1.0)))
    centered = [b.fadd(v, negative_mean) for v in shifted]
    squares = b.const(float_bits(0.0))
    for v in centered:
        squares = b.fadd(squares, b.fmul(v, v))
    b.barrier()                                     # every lane has finished reading the partials
    b.store_tg(squares, lane)
    b.barrier()
    total2 = b.const(float_bits(0.0))
    for l in range(LANES):
        total2 = b.fadd(total2, b.load_tg(b.const(l), name="q%d" % l))
    variance = b.fmul(total2, reciprocal)
    inverse_std = b.rsqrt(b.fadd(variance, b.const(float_bits(epsilon))))
    col_base = b.mul(lane, b.const(PER_LANE), name="col_base")
    for j, (i, v) in enumerate(zip(idx, centered)):
        col = b.add(col_base, ir.Imm(j)) if j else col_base
        scale = b.load(gamma, col, width="word")
        bias_col = b.add(col, ir.Imm(columns)) if packed_parameters else col
        bias = b.load(beta, bias_col, width="word")
        b.store_at(output, i, b.fadd(b.fmul(b.fmul(v, inverse_std), scale), bias), width="word")
    b.ret()
    return fn


def reference(source, gamma, beta, epsilon=EPSILON):
    import g17layernorm
    return g17layernorm.reference(source, gamma, beta, epsilon)


def interpret(code, source, gamma, beta, rows):
    """The delivered bytes under the lockstep group driver. -> (outputs, confidence, notes)"""
    import numpy as np, g17packedcheck, g17normcheck as N
    dec = g17packedcheck.decode(code)
    bufs = {0: np.asarray(source, np.float32).reshape(-1).tolist(), 1: np.asarray(gamma, np.float32).tolist(),
            2: np.asarray(beta, np.float32).tolist(), 3: [float("nan")] * (rows * COLUMNS)}
    bind = [(1, 0, False), (2, 2, False), (3, 4, False), (4, 6, True)]
    out, conf, notes = N.simulate_group(dec, bufs, bind, LANES, rows, LANES)
    return np.array(out[3], np.float32).reshape(rows, COLUMNS), conf, notes


def hazards(code, rows):
    """Hazard FACTS on the delivered bytes under the lockstep group model, stated positively.

    The driver refuses races, unwritten reads, out-of-range words and divergent barrier counts;
    this reports what it saw so the review can say what holds, not only what was not refused:
    every lane of every group reaches every barrier (count per lane), the scratch extent in
    bytes against the declared 128, and that every REUSE of a word (its second and later writes)
    lands in a strictly later barrier interval than every read of the value it replaces - the
    read-completion barrier is between them. Codex's admission check for unread asynchronous
    load destinations (g17layernormimagecheck.check_load_reuse) is run on the same bytes.
    What still needs silicon is stated in `needs_silicon`."""
    import numpy as np, g17packedcheck, g17normcheck as N, g17layernormimagecheck
    dec = g17packedcheck.decode(code)
    rng = np.random.default_rng(7)
    source = rng.standard_normal((rows, COLUMNS)).astype(np.float32)
    gamma = rng.standard_normal(COLUMNS).astype(np.float32); beta = rng.standard_normal(COLUMNS).astype(np.float32)
    bufs = {0: source.reshape(-1).tolist(), 1: gamma.tolist(), 2: beta.tolist(), 3: [float("nan")] * (rows * COLUMNS)}
    bind = [(1, 0, False), (2, 2, False), (3, 4, False), (4, 6, True)]
    seen = []
    N.simulate_group(dec, bufs, bind, LANES, rows, LANES, observer=lambda g, tg: seen.append(tg))
    barrier_counts, max_word, reuse_ok, reuses, races, intervals = set(), -1, True, 0, 0, set()
    for tg in seen:
        barrier_counts |= set(tg.barriers.values())
        if len(tg.barriers) != LANES:
            raise ValueError("%d of %d lanes reached a barrier" % (len(tg.barriers), LANES))
        races += len(tg.race)
        intervals.add(tg.interval + 1)
        writes, reads = {}, {}
        for kind, lane, idx, interval, _off in tg.log:
            if kind == "store":
                if idx in writes:
                    reuses += 1
                    if not all(interval > r for r in reads.get(idx, [])):
                        reuse_ok = False
                    reads[idx] = []
                writes.setdefault(idx, []).append(interval)
                max_word = max(max_word, idx)
            elif kind == "load":
                reads.setdefault(idx, []).append(interval)
                max_word = max(max_word, idx)
    used = (max_word + 1) * 4
    return dict(groups=len(seen), lanes=LANES, barriers_per_lane=sorted(barrier_counts), barrier_intervals=sorted(intervals),
                all_lanes_reach_every_barrier=(barrier_counts == {BARRIERS} and len(seen) == rows),
                scratch_words_used=max_word + 1, scratch_bytes_used=used, scratch_bytes_declared=LANES * 4,
                scratch_within_declared=used <= LANES * 4,
                word_reuses=reuses, reuse_follows_read_completion_barrier=reuse_ok, races=races,
                load_reuse=g17layernormimagecheck.check_load_reuse(dec),
                needs_silicon=["that op447 [0,276] orders threadgroup memory across the 32 lanes as a barrier does (the "
                               "model asserts race-freedom under any interleaving between barriers; it does not measure "
                               "the hardware's ordering)",
                               "asynchronous load completion: the model completes loads synchronously; Codex's check only "
                               "refuses a destination overwritten before a printed read",
                               "divergence: every lane runs every instruction in the model; a lane skipping a barrier "
                               "is refused by the count, not modelled"])


def deliver(destination, rows_list=(1, 32)):
    import numpy as np, g17cc
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    fx = np.load(ROOT / "results/g17-layernorm-fixtures-v1/minilm_embeddings.npz")
    source, gamma, beta = (fx[k].astype(np.float32) for k in ("source", "gamma", "beta"))
    report = dict(gpu_dispatched=False, fixture="results/g17-layernorm-fixtures-v1/minilm_embeddings.npz",
                  fixture_sha256=hashlib.sha256(open(ROOT / "results/g17-layernorm-fixtures-v1/minilm_embeddings.npz", "rb").read()).hexdigest(),
                  error_budget="2e-5 * (1 + abs(FP64 reference))", programs={})
    for rows in rows_list:
        fn = cooperative_layernorm_ir(rows)
        prog = g17cc.compile_function(fn)
        x = source[:rows]
        want = reference(x, gamma, beta)
        got, conf, notes = interpret(prog.code, x, gamma, beta, rows)
        err = np.abs(got.astype(np.float64) - want); budget = 2e-5 * (1 + np.abs(want))
        name = "coop_%dx%d" % (rows, COLUMNS)
        (destination / (name + ".bin")).write_bytes(prog.code)
        (destination / (name + ".abi.json")).write_text(json.dumps(prog.abi_plain(prog.abi()), indent=2) + "\n")
        (destination / (name + ".contract.json")).write_text(json.dumps(contract(rows), indent=2) + "\n")
        np.savez(destination / (name + ".prediction.npz"), interpreted=got, fp64=want)
        report["programs"][name] = dict(rows=rows, bytes=len(prog.code), code_sha256=hashlib.sha256(prog.code).hexdigest(),
                                        instructions=len(prog.layout), register_count=prog.abi()["register_count"],
                                        interpreted_max_abs_error=float(err.max()), interpreted_max_error_over_budget=float((err / budget).max()),
                                        failing=int((err > budget).sum()), confidence=conf, notes=notes,
                                        prediction_sha256=hashlib.sha256(got.tobytes()).hexdigest(),
                                        hazards=hazards(prog.code, rows))
    (destination / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("destination", type=Path)
    r = deliver(ap.parse_args().destination)
    for n, p in r["programs"].items():
        print("%-14s %6d B %5d instr regs %3d  interp err/budget %.4f failing %d  %s" % (n, p["bytes"], p["instructions"], p["register_count"], p["interpreted_max_error_over_budget"], p["failing"], p["confidence"]))
