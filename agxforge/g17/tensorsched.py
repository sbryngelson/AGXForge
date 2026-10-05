"""Native tensor scheduler (ledger row P11): the section-150 cost model as code, its measured domain,
and a chooser that picks a layer schedule and says what it rests on.

THE MODEL. A kernel's time is the largest of three terms (recon section 150 part 1):

    T_latency    = max over simdgroups of (steps of that simdgroup) x L_step
    T_throughput = (total steps) x C_step / cores
    T_memory     = unique bytes / BW(footprint)

T_latency is the MAXIMUM over simdgroups, not a common steps-per-simdgroup value (recon 155.1: with a
common value the causal prediction is worse still). The form is the standard makespan LOWER BOUND; it is
tight only when every simdgroup does the same number of steps. Every workload it was validated on is
uniform. On a non-uniform one (a causal attention) it is optimistic by up to 37 percent (155.1). For
non-uniform work a fourth term, T_schedule = ``list_makespan_ns`` (a processor-sharing list schedule in
dispatch order, MM 25.124.1), charges the tail the bound ignores; it cuts the speedup optimism on the
155.1 points from 20.9 / 37.1 to 7.5 / 26.5 percent and stays under every measured time. It does not
close the gap, so a non-uniform prediction is still marked ``bound == "lower_bound"`` and ``speedup``
refuses to form a ratio from it.

THE DOMAIN. Every prediction carries a ``Domain``: one ``Check`` per measured condition the constants
depend on (the kernel family the constants were fitted to, dtype, head width or layer shape, register
count within the measured residency range, the memory term not depending on the unmeasured SLC-to-DRAM
step, the token or wave range the measurement covered, uniform work). A failed check does not produce a
number. ``choose`` then returns the known-safe schedule with ``fallback=True`` and the failing checks'
reasons, and ``predicted_ms`` is None.

THE CHOOSER. ``choose(FFN(...))`` and ``choose(Attention(...))`` enumerate the schedules the record has
measured (token-parallel against weight-stationary FFN, the three cut points, simdgroups per threadgroup,
the KV split, the diagonal skip), exclude the ones a measured rule excludes, compare predicted TIMES
(never ratios) among the rest, and return the decision with every citation it used.

TWO KERNEL FAMILIES (MM 25.126). The FFN is priced by ``FFN.kernel``: ``layerbench`` (the default), the
RETAINED rebuilt harness ``tools/g17layerbench.py``, whose M5 receipts fix every FFN constant from one
process per format, so the WS / TP choice is a within-process ratio; or ``recon146``, the recon's
unretained kernels. The recon constants name the wrong winner on the harness at fp8 48 and 64 and bf16 64
tokens; the layerbench constants pick the measured-fastest schedule at all 11 harness points, crossing
over between 32 and 48 fp8 tokens (blocked WS; a measured one-pass WS still wins at 48) and between 48 and
64 bf16 tokens. Attention head width scales both step constants by ``head_width_factor`` (the MMA count
ratio, with a 16-wide remainder costing a whole 32-wide block), not by section 150's affine per-MMA cost.

CUTS, FUSED-OR-CUT ATTENTION, KV SPLIT (MM 25.124.3). ``choose_cuts(Chain)`` picks fused or a cut for a
multi-stage chain from recon 142's measured cut/fused ratios, and a per-stage format mix (row M6) only on
explicit opt-in; ``choose_attention_cut(Attention, exposed)`` picks among the attention schedules a caller
can realize; ``choose_kv_split(Attention)`` splits the key axis only with ``allow_value_change=True``.

WHAT IT CANNOT DO. The layer schedules it picks between were realized by the recon harness
(``mxpipe.py``, ``mxlayer_ws.py``, ``mxattn_exp.py``, sections 141 to 155), which is not retained, and by
its rebuild ``tools/g17layerbench.py`` (MM 25.126), which is a measurement harness, not the runtime. The production runtime (``agxforge.g17.runtime.TensorSpec``, class ``gemm_generic``) cannot
express them: ``runtime_limits()`` reads its caps from the code, and every ``Schedule`` records whether the
runtime admits it and why not. This module decides; it does not lower. It imports nothing from tlower.

Reordering instructions inside a body is deliberately NOT a schedule dimension: the rotation measured
1.08x at 8 simdgroups and 0.99x at 12.8 per core (machine model 25.97); occupancy is the lever.

Section numbers: ``recon N`` is docs/g17-tensorops-accelerator-recon.md section N; ``MM N`` is
docs/g17-tensorops-machine-model.md section N.
"""
from __future__ import annotations

import math
import typing
from dataclasses import dataclass, field

CORES = 20

# ---------------------------------------------------------------------------------------------------
# Constants. Every one names the section that measured it and the receipt that section cites. The recon
# logs (mx*.log, mxcost_model_validation.json) were not retained in this repository; the section's table
# is then the retained form of the measurement.


@dataclass(frozen=True)
class Constant:
    value: float
    unit: str
    section: str
    receipt: str


C = {
    "cores": Constant(20, "cores", "recon 145, 146",
                      "mxlayer_ffn_time1.log (one wave = 20 threadgroups; not retained)"),
    # read bandwidth by footprint (unique bytes), recon 150 part 2 table, from recon 145
    "bw_l1_lower": Constant(3000.0, "GB/s, lower bound, footprint <= 5 MiB", "recon 145, 150",
                            "mxcal.py stream (not retained)"),
    "bw_slc": Constant(520.0, "GB/s, footprint <= 20 MiB", "recon 145, 150",
                       "mxcal.py stream (not retained)"),
    "bw_dram": Constant(280.0, "GB/s, footprint >= 40 MiB", "recon 145, 150; MM row M7 (283-294); "
                        "MM 25.121 (read 269.6-275.7, a third instrument)",
                        "mxcal.py stream (not retained); isa/g17-dram-bandwidth-results.json (MM 25.121)"),
    "tg_memory": Constant(32768, "bytes per threadgroup", "recon 145, 146",
                          "native compiler refusal text quoted in recon 146 part 6"),
    # attention step constants, recon 148 part 4 and recon 150 part 2 (fitted from 4 rows; 23 held out)
    "attn_L_fp8": Constant(5350.0, "ns per 32-key block per simdgroup (fp8, int8)", "recon 148 part 4, 150",
                           "mxattn_time_b1..b7.log (not retained)"),
    "attn_C_fp8": Constant(433.0, "ns per 32-key block per core at saturation (fp8, int8)",
                           "recon 148 part 4, 150", "mxattn_time_b1..b7.log (not retained)"),
    "attn_L_bf16": Constant(5000.0, "ns per 32-key block per simdgroup (bf16)", "recon 150 part 2",
                            "mxattn_time_b1..b7.log (not retained)"),
    "attn_C_bf16": Constant(381.0, "ns per 32-key block per core (bf16)", "recon 150 part 2",
                            "mxattn_time_b1..b7.log (not retained)"),
    # the causal harness's own kernels: the calibrated constants do not transfer to them (recon 155.1);
    # these are the best max() fit to that harness's three UNIFORM points (worst error 12.1 percent)
    "attn155_L": Constant(2868.0, "ns per 32-key block per simdgroup (recon-155 kernels, e4m3)", "recon 155.1",
                          "results/g17-tensorops-recon-v1/mxattn_causal.py (not retained)"),
    "attn155_C": Constant(327.0, "ns per 32-key block per core (recon-155 kernels, e4m3)", "recon 155.1",
                          "results/g17-tensorops-recon-v1/mxattn_causal.py (not retained)"),
    # gated FFN, d 2048, Hh 8192 (recon 146 part 4, 150 part 2): one wave of 20 threadgroups, two-kernel
    "ffn_wave_fp8": Constant(1.84, "ms per wave, token-parallel two-kernel, e4m3, 32 SG", "recon 146, 150",
                             "mxlayer_ffn_time3.log (not retained)"),
    "ffn_wave_int8": Constant(1.72, "ms per wave, token-parallel two-kernel, int8, 32 SG", "recon 146, 150",
                              "mxlayer_ffn_time3.log (not retained)"),
    "ffn_wave_bf16": Constant(2.06, "ms per wave, token-parallel two-kernel, bf16, 16 SG", "recon 146, 150",
                              "mxlayer_ffn_time3.log (not retained)"),
    # weight-stationary gated FFN (recon 149 part 3 and 4; the best decomposition of the measured grid)
    "ws_chunk_fp8": Constant(59000.0, "ns per hidden chunk per simdgroup, stage 1, e4m3, 16 tokens",
                             "recon 149 part 4", "mxlayer_ws.py time (not retained)"),
    "ws_fp8_mt1": Constant(0.442, "ms: A 0.148 (64x4) + B 0.287 (8x4) + C 0.007, 16 tokens", "recon 149 part 3",
                           "mxlayer_ws.py time (not retained)"),
    "ws_fp8_mt2": Constant(0.841, "ms: A 0.153 (32x8) + B 0.665 (32x1) + C 0.023, 32 tokens", "recon 149 part 3",
                           "mxlayer_ws.py time (not retained)"),
    "ws_bf16_mt1": Constant(0.388, "ms: A 0.234 (32x4) + B 0.147 (32x1) + C 0.007, 16 tokens", "recon 149 part 3",
                            "mxlayer_ws.py time (not retained)"),
    # the section-142 least-squares fit (321 timings): the per-MMA cost of a DEPENDENT chain
    "mma_dep_1sg_bf16": Constant(62.6, "ns per dependent MMA at 1 SG/core (bf16 class)", "recon 150 part 6",
                                 "mxpipe_cost_*.log (not retained)"),
    "mma_dep_64sg_bf16": Constant(5.8, "ns per MMA at 64 SG/core (bf16 class)", "recon 150 part 6",
                                  "mxpipe_cost_*.log (not retained)"),
    "mma_dep_1sg_fp8": Constant(68.2, "ns per dependent MMA at 1 SG/core (fp8, int8)", "recon 150 part 6",
                                "mxpipe_cost_*.log (not retained)"),
    "mma_dep_64sg_fp8": Constant(7.0, "ns per MMA at 64 SG/core (fp8, int8)", "recon 150 part 6",
                                 "mxpipe_cost_*.log (not retained)"),
    "instr_1sg": Constant(1.6, "ns per other instruction at 1 SG/core", "recon 150 part 6",
                          "mxpipe_cost_*.log (not retained)"),
    "instr_64sg": Constant(0.34, "ns per other instruction at 64 SG/core (0.21 bf16 class)", "recon 150 part 6",
                           "mxpipe_cost_*.log (not retained)"),
    # residency (MM 25.121, this compiler, straight-line) and recon 170 (Apple-compiled): the floor
    "resid_ours_min": Constant(64.0, "resident SG per core, 49..125 registers used (this compiler)", "MM 25.121",
                               "isa/g17-residency-sweep-results.json (MM 25.121)"),
    "resid_apple_floor": Constant(32.1, "resident SG per core at 126 registers (Apple-compiled)", "recon 170",
                                  "recon 170 (H4 closed by direct measurement)"),
    # THE RETAINED LAYER HARNESS (MM 25.126): tools/g17layerbench.py's kernels, one process per format,
    # clean medians (records whose reference dispatches stayed within 1.2x the run's median). Every FFN
    # constant of a format comes from the SAME process, so the WS / TP comparison is a within-process
    # ratio: absolute times of this harness moved 25 to 44 percent between its own processes.
    "lb_ws16_fp8": Constant(0.392354, "ms, WS 16 tokens (A 64x4, B 8x4, C), e4m3", "MM 25.126",
                            "results/g17-layer-v1/measure/ffn-e4m3-m5.json label ws16"),
    "lb_ws32_fp8": Constant(1.014604, "ms, WS 32 tokens (mt 2, A 32x8, B 32x1, C), e4m3", "MM 25.126",
                            "results/g17-layer-v1/measure/ffn-e4m3-m5.json label ws32"),
    "lb_tp_fp8": Constant(1.241208, "ms, TP two-kernel below one wave, 32 SG, e4m3: median of 1/3/4/6/8 "
                          "threadgroups (1.217 to 1.304, flat)", "MM 25.126",
                          "results/g17-layer-v1/measure/ffn-e4m3-m5.json labels tp16_1tg, tp48..tp128"),
    "lb_onepass48_fp8": Constant(1.119333, "ms, one-pass WS (mt 3), 48 tokens, e4m3; no model", "MM 25.126",
                                 "results/g17-layer-v1/measure/ffn-e4m3-m5.json label ws48_1pass"),
    "lb_onepass64_fp8": Constant(1.308458, "ms, one-pass WS (mt 4), 64 tokens, e4m3; no model", "MM 25.126",
                                 "results/g17-layer-v1/measure/ffn-e4m3-m5.json label ws64_1pass"),
    "lb_ws16_bf16": Constant(0.523167, "ms, WS 16 tokens (A 32x4, B 32x1, C), bf16", "MM 25.126",
                             "results/g17-layer-v1/measure/ffn-bf16-m5.json label ws16"),
    "lb_tp_bf16": Constant(1.812375, "ms, TP two-kernel below one wave, 16 SG, bf16: median of 1/3/4/6/8 "
                           "threadgroups (1.780 to 1.955, flat)", "MM 25.126",
                           "results/g17-layer-v1/measure/ffn-bf16-m5.json labels tp16_1tg, tp48..tp128"),
    # head-width scaling of attention (MM 25.126, same-process ratios to width 128)
    "hw_64": Constant(0.355, "x width-128 time at head width 64: median of 0.311, 0.355, 0.362", "MM 25.126",
                      "results/g17-layer-v1/measure/attn-m4.json, attn-decode64.json"),
    # the stated validation medians (recon 150 part 5; MM 16 'The cost model')
    "median_attn": Constant(2.7, "percent median error, 23 held-out attention rows", "recon 150 part 5",
                            "mxcost_model_validation.json (not retained)"),
    "median_ffn_tp": Constant(3.3, "percent median error, 16 held-out token-parallel rows", "recon 150 part 5",
                              "mxcost_model_validation.json (not retained)"),
    "median_ws_stage1": Constant(5.1, "percent median error, 10 held-out stage-1 configurations",
                                 "recon 150 part 5", "mxcost_model_validation.json (not retained)"),
}

MIB = 1 << 20
# the registers the residency measurements cover: 19..125 (MM 25.121) and 22..126 (recon 170).
# R126 and above squash the instruction (MM 17 rule 1); nothing past 126 is a legal count.
REG_RANGE = (19, 126)
RESIDENCY_OURS = {19: 95.0, 33: 71.0, 49: 67.0, 65: 66.0, 81: 65.0, 97: 64.0, 113: 68.0, 125: 64.0}


def cite(*keys):
    return tuple(f"{k}: {C[k].value} {C[k].unit} [{C[k].section}; {C[k].receipt}]" for k in keys)


# ---------------------------------------------------------------------------------------------------
# Domain


@dataclass(frozen=True)
class Check:
    name: str
    inside: bool
    detail: str
    source: str


@dataclass(frozen=True)
class Domain:
    checks: tuple[Check, ...]

    @property
    def inside(self):
        return all(c.inside for c in self.checks)

    @property
    def reasons(self):
        return tuple(f"{c.name}: {c.detail} [{c.source}]" for c in self.checks if not c.inside)

    def get(self, name):
        return next(c for c in self.checks if c.name == name)


def residency_floor(registers):
    """Resident simdgroups per core that both measurements guarantee at this register count, or None."""
    lo, hi = REG_RANGE
    if registers is None or not lo <= registers <= hi:
        return None
    ours = min(v for r, v in RESIDENCY_OURS.items() if r >= min(registers, 125)) if registers >= 19 else None
    # recon 170's Apple-compiled curve falls monotonically to 32.1 at 126 and starts at 22
    return min(ours, C["resid_apple_floor"].value) if registers >= 22 else ours


def check_registers(registers, needed_per_core):
    """The step constants were measured at saturation; they hold only if the machine can keep
    ``needed_per_core`` simdgroups resident at this register count."""
    floor = residency_floor(registers)
    src = "MM 25.121 (19..125 used: 95, then 64-69 per core); recon 170 (22..126: 44.6 falling to 32.1)"
    if floor is None:
        return Check("registers", False, f"{registers} registers is outside the measured range "
                     f"{REG_RANGE[0]}..{REG_RANGE[1]} (R126+ is squashed: MM 17 rule 1)", src)
    ok = floor >= needed_per_core
    return Check("registers", ok, f"{registers} registers: at least {floor} resident per core, "
                 f"{needed_per_core:.1f} needed", src)


def bandwidth(footprint_bytes):
    """(GB/s, regime) by footprint, or (None, 'step') inside the unmeasured 20..40 MiB step."""
    if footprint_bytes <= 5 * MIB:
        return C["bw_l1_lower"].value, "l1 (lower bound)"
    if footprint_bytes <= 20 * MIB:
        return C["bw_slc"].value, "slc"
    if footprint_bytes >= 40 * MIB:
        return C["bw_dram"].value, "dram"
    return None, "step"


# ---------------------------------------------------------------------------------------------------
# The cost function


@dataclass(frozen=True)
class Prediction:
    t_latency_ms: float
    t_throughput_ms: float
    t_memory_ms: float
    uniform: bool
    domain: Domain
    note: str = ""
    # the list-schedule makespan (MM 25.124.1); 0 for uniform work, where it equals the bound
    t_schedule_ms: float = 0.0

    @property
    def ms(self):
        return max(self.t_latency_ms, self.t_throughput_ms, self.t_memory_ms, self.t_schedule_ms)

    @property
    def bound(self):
        # recon 155.1: max(longest task, total/cores) is tight only for uniform work. The scheduling
        # term (25.124.1) tightens it but stays optimistic on the only non-uniform workload measured
        # (by 7.5 and 26.5 percent in speedup), so a non-uniform time is still a lower bound.
        return "estimate" if self.uniform else "lower_bound"

    @property
    def dominant(self):
        terms = {"latency": self.t_latency_ms, "throughput": self.t_throughput_ms, "memory": self.t_memory_ms,
                 "schedule": self.t_schedule_ms}
        return max(terms, key=terms.get)


class RatioRefused(ValueError):
    pass


def speedup(slow, fast):
    """slow.ms / fast.ms, refused when either side is a lower bound (recon 155.1: a non-uniform
    workload's predicted speedup is a ceiling, and its time a lower bound)."""
    for p in (slow, fast):
        if p.bound != "estimate":
            raise RatioRefused("a lower-bound prediction cannot form a ratio (recon 155.1)")
    return slow.ms / fast.ms


def memory_ms(unique_bytes, footprint_bytes):
    """(T_memory ms, check). In the 20..40 MiB step the DRAM plateau is used, and the check fails
    only if that choice could matter, which the caller decides against the other terms."""
    bw, regime = bandwidth(footprint_bytes)
    if bw is None:
        lo_ms = unique_bytes / (C["bw_slc"].value * 1e9) * 1e3
        hi_ms = unique_bytes / (C["bw_dram"].value * 1e9) * 1e3
        return (lo_ms, hi_ms), regime
    t = unique_bytes / (bw * 1e9) * 1e3
    return (t, t), regime


def predict(steps, L_step_ns, C_step_ns, unique_bytes=0, footprint_bytes=None, cores=CORES, checks=(),
            schedule_term=True):
    """The section-150 cost function. ``steps`` is one step count per simdgroup (a key block of 32 or a
    hidden chunk of 32). Returns a Prediction whose domain includes the given checks, a uniformity
    check and the memory-step check."""
    steps = tuple(int(s) for s in steps)
    if not steps:
        raise ValueError("no simdgroups")
    t_lat = max(steps) * L_step_ns / 1e6
    t_thr = sum(steps) * C_step_ns / cores / 1e6
    footprint = unique_bytes if footprint_bytes is None else footprint_bytes
    (m_lo, m_hi), regime = memory_ms(unique_bytes, footprint)
    uniform = len(set(steps)) == 1
    checks = list(checks)
    if regime == "step":
        matters = max(t_lat, t_thr, m_lo) != max(t_lat, t_thr, m_hi)
        checks.append(Check("memory", not matters,
                            f"footprint {footprint / MIB:.1f} MiB is in the unmeasured 20..40 MiB step and "
                            f"T_memory {'decides' if matters else 'does not decide'} the time between the SLC "
                            "and DRAM plateaus", "recon 145, 150 part 2"))
    else:
        checks.append(Check("memory", True, f"footprint {footprint / MIB:.1f} MiB on the {regime} plateau",
                            "recon 145, 150 part 2; MM row M7; MM 25.121"))
    checks.append(Check("uniform", uniform,
                        "every simdgroup does the same number of steps" if uniform else
                        f"steps range {min(steps)}..{max(steps)}: the prediction is a LOWER BOUND "
                        "(list-schedule term included; still optimistic by 7.5 to 26.5 percent in speedup on "
                        "the measured causal workload)", "recon 155.1; MM 16, 25.124.1"))
    t_sched = 0.0 if uniform or not schedule_term else list_makespan_ns(steps, L_step_ns, C_step_ns, cores) / 1e6
    return Prediction(t_lat, t_thr, m_hi, uniform, Domain(tuple(checks)), t_schedule_ms=t_sched)


def list_makespan_ns(steps, L_step_ns, C_step_ns, cores=CORES, slots=None, order="index", step_ns=None):
    """The scheduling term for NON-UNIFORM work (MM 25.124.1), replacing max(longest, total / cores).

    A list schedule with processor sharing, event by event. Tasks are dispatched in ``order`` --
    "index" is the hardware's threadgroup order, "lpt" longest first -- first round-robin over the
    cores up to ``slots`` resident per core (None: all resident), then each freed slot takes the next
    task. On a core with n resident tasks each progresses at min(1 / L_step, 1 / (n x C_step)) steps
    per ns: one simdgroup's dependent chain until the core saturates, then an equal share of the
    core's step throughput. For equal tasks spread evenly this IS the section-150 form (per core
    max(steps x L, n x steps x C)); for unequal ones it charges the tail the bound ignores -- long
    tasks finishing alone at the chain rate while the core's other slots sit idle.

    ``step_ns`` (optional, MM 25.144.12): a function n -> ns per step of each of n tasks resident on one core,
    replacing max(L_step, n x C_step), e.g. a mean-value-analysis curve between the two bounds. Without it the
    schedule is exactly the one above."""
    queue = sorted(steps, reverse=True) if order == "lpt" else list(steps)
    if order not in ("index", "lpt"):
        raise ValueError("order is 'index' or 'lpt'")
    queue = [float(s) for s in reversed(queue)]          # pop() takes the next task
    slots = len(queue) if slots is None else slots
    res = [[] for _ in range(cores)]
    c = 0
    while queue and any(len(r) < slots for r in res):
        if len(res[c]) < slots:
            res[c].append(queue.pop())
        c = (c + 1) % cores
    t = 0.0
    while any(res):
        rates = [(1.0 / (step_ns(len(r)) if step_ns else max(L_step_ns, len(r) * C_step_ns))) if r else 0.0
                 for r in res]
        dt = min(min(r) / rate for r, rate in zip(res, rates) if r)
        t += dt
        for i, (r, rate) in enumerate(zip(res, rates)):
            if not r:
                continue
            left = [x - dt * rate for x in r]
            r[:] = [x for x in left if x > 1e-9]
            for _ in range(len(left) - len(r)):
                if queue:
                    r.append(queue.pop())
    return t


def chain_ns(mmas, other_instructions, dtype, sg_per_core):
    """The recon-142 least-squares fit (recon 150 part 6): a dependent chain's time is
    a x MMAs + b x other instructions, at one or at 64 simdgroups per core only. It was fitted, not
    held out (median error 5.4 percent bf16 class, 12.3 fp8/int8), and its large per-MMA term is the
    cost of DEPENDENCE: independent-MMA ablations are its six worst outliers. Refuses anything else.
    NOT a head-width extrapolator: adding its per-MMA cost for a wider head under-predicts the width
    ratio by 16 to 32 percent above 128 (MM 25.126); use head_width_factor."""
    cls = "bf16" if dtype in ("bf16", "fp16") else "fp8" if dtype in ("fp8", "int8") else None
    if cls is None or sg_per_core not in (1, 64):
        raise ValueError("the chain fit covers bf16/fp16/fp8/int8 at 1 or 64 simdgroups per core (recon 150 part 6)")
    a = C[f"mma_dep_{'1sg' if sg_per_core == 1 else '64sg'}_{cls}"].value
    b = C["instr_1sg" if sg_per_core == 1 else "instr_64sg"].value
    return mmas * a + other_instructions * b


# ---------------------------------------------------------------------------------------------------
# What the production runtime can express


def runtime_limits():
    """The caps of the production tensor runtime, read from agxforge.g17.runtime.TensorSpec itself."""
    from agxforge.g17.runtime import TensorSpec
    f = TensorSpec.model_fields

    def le(name):
        return next(m.le for m in f[name].metadata if hasattr(m, "le"))
    grid = typing.get_args(typing.get_args(f["grid"].annotation)[0])
    return {"M": le("M"), "N": le("N"), "K": le("K"),
            "simdgroups": typing.get_args(f["simdgroups"].annotation),
            "max_grid_threads": max(grid),
            "compositions": typing.get_args(f["composition"].annotation)}


@dataclass(frozen=True)
class Schedule:
    kind: str
    params: tuple
    measured_by: str
    runtime_realizable: bool
    runtime_reason: str

    def param(self, key):
        return dict(self.params)[key]


def _attention_class_gaps(w):
    """What the runtime's attention class (agxforge.g17.runtime ATTENTION_*, MM 25.129) does not cover in
    workload ``w``. The class exists since P7, but only at its own shapes: naming the family is not
    realizing the workload. Two shapes: head 64 in one threadgroup (phases fused, project, attend, step),
    and phase grid (MM 25.135): head 128 on a grid of 1, 2, 4, 8 or 16 heads, ATTENTION_GRID_CAPACITY blocks
    in one dispatch, one query row at that length; 16 rows were measured at 4 blocks (64 keys), and more
    rows at more blocks exceed the 1,000,000-byte image contract."""
    from agxforge.g17 import runtime as R
    why = []
    if w.head_dim == 128 and hasattr(R, "ATTENTION_GRID_CAPACITY"):
        cap = R.ATTENTION_GRID_CAPACITY
        if w.heads not in R.ATTENTION_GRID_HEADS:
            why.append(f"{w.heads} heads: phase grid launches {', '.join(map(str, R.ATTENTION_GRID_HEADS))} heads")
        if w.seq > cap * R.ATTENTION_BLOCK:
            why.append(f"{w.seq} keys: phase grid holds {cap} blocks of {R.ATTENTION_BLOCK} keys "
                       f"({cap * R.ATTENTION_BLOCK}) per dispatch")
        if w.queries > R.ATTENTION_MAX_ROWS:
            why.append(f"{w.queries} query rows: the attention class measures 1..{R.ATTENTION_MAX_ROWS}")
        elif w.queries > 1 and w.seq > 4 * R.ATTENTION_BLOCK:
            why.append(f"{w.queries} query rows at {w.seq} keys: phase grid is measured at 1 row to "
                       f"{cap * R.ATTENTION_BLOCK} keys and 16 rows to 64 (the image contract)")
        return why
    if w.head_dim != R.ATTENTION_HEAD:
        why.append(f"head {w.head_dim}: the attention class is head {R.ATTENTION_HEAD} (and head 128 in phase grid)")
    # the register-offset cache reads (MM 25.114.4) take the class to ATTENTION_MAX_REGISTER_BLOCKS; the
    # immediate-offset programs stop at ATTENTION_MAX_BLOCKS
    # and the counted key-block loop (MM 25.114.5) to ATTENTION_MAX_LOOP_BLOCKS
    cap = max(R.ATTENTION_MAX_BLOCKS, getattr(R, "ATTENTION_MAX_REGISTER_BLOCKS", 0),
              getattr(R, "ATTENTION_MAX_LOOP_BLOCKS", 0))
    if w.seq > cap * R.ATTENTION_BLOCK:
        why.append(f"{w.seq} keys: the attention class holds {cap} blocks of "
                   f"{R.ATTENTION_BLOCK} keys ({cap * R.ATTENTION_BLOCK})")
    if w.queries > R.ATTENTION_MAX_ROWS:
        why.append(f"{w.queries} query rows: the attention class measures 1..{R.ATTENTION_MAX_ROWS}")
    if w.heads != 1:
        why.append(f"{w.heads} heads: the head-64 phases are one head (the head grid is phase grid, head 128)")
    return why


def grid_split(n):
    """The smallest grid_n that the N-tiled grid class admits AND tlower lowers for an output width of n
    (MM 25.134): grid_n a power of two, n divisible by 16 * grid_n, n / grid_n at most 256 columns, and a
    power-of-two tile count per threadgroup (tlower). None if there is none. grid_n 1 is the plain class."""
    from agxforge.g17.runtime import TensorSpec
    f = TensorSpec.model_fields
    if "grid_n" not in f:
        return None
    cap = next(m.le for m in f["grid_n"].metadata if hasattr(m, "le"))
    per_tg = 256                      # runtime._generic_class_rule("n_tiled_grid"): N // grid_n <= 256
    g = 1
    while g <= cap:
        tiles = n // (16 * g)
        # tlower also needs a power-of-two tile count per threadgroup (the column offset is a shift),
        # which the runtime's class rule does not check
        if n % (16 * g) == 0 and n // g <= per_tg and tiles and not tiles & (tiles - 1):
            return g
        g *= 2
    return None


def _runtime_verdict(widest_n, simdgroups, bodies, family, workload=None):
    lim = runtime_limits()
    why = []
    if not any(family in c for c in lim["compositions"]):
        why.append(f"no TensorSpec.composition names {family!r} (the runtime has no such class)")
    elif family == "attention":
        if workload is None:
            why.append("an attention schedule without its workload cannot be checked against the class")
        else:
            why += _attention_class_gaps(workload)
            if simdgroups != 1:
                why.append(f"{simdgroups} simdgroups: the attention class runs one")
    # TensorSpec.N is the TOTAL width since the N-tiled grid (MM 25.134); a threadgroup holds 256 columns
    if widest_n > lim["N"] or (widest_n > 256 and grid_split(widest_n) is None):
        why.append(f"an output width of {widest_n} exceeds 256 columns per threadgroup and no N-tiled grid "
                   f"split (power-of-two grid_n and tile count, TensorSpec.N <= {lim['N']}) lowers it")
    if simdgroups not in lim["simdgroups"]:
        why.append(f"{simdgroups} simdgroups per threadgroup is not in TensorSpec.simdgroups {lim['simdgroups']}")
    if bodies > 1:
        why.append(f"{bodies} kernels chained through device memory with a fused epilogue is not a runtime class")
    return (not why), ("; ".join(why) if why else "admitted by gemm_generic")


def _schedule(kind, params, measured_by, widest_n, simdgroups, bodies, workload=None):
    family = "attention" if kind.startswith("attention") else "ffn"
    ok, why = _runtime_verdict(widest_n, simdgroups, bodies, family, workload)
    return Schedule(kind, tuple(sorted(params.items())), measured_by, ok, why)


def _aschedule(w, *args):
    return _schedule(*args, workload=w)


# ---------------------------------------------------------------------------------------------------
# Workloads and decisions


@dataclass(frozen=True)
class FFN:
    """A gated FFN (gate and up projections, silu, product, quantize, down projection, residual, norm).

    ``kernel`` names the kernel family whose constants price it: ``layerbench`` (the default: the
    RETAINED rebuild, tools/g17layerbench.py, MM 25.126) or ``recon146`` (the recon's unretained
    mxpipe kernels, sections 146 and 149). The two families' absolute times differ by -27 to +29
    percent (25.126's anchors), so constants never cross between them."""
    tokens: int
    d: int = 2048
    hidden: int = 8192
    dtype: str = "fp8"          # fp8 (e4m3), int8, bf16
    registers: int | None = None
    kernel: str = "layerbench"


@dataclass(frozen=True)
class Attention:
    """Fused attention: heads x ceil(queries / 16) tasks, each over ``seq`` keys in blocks of 32."""
    heads: int
    head_dim: int
    seq: int
    queries: int = 16
    causal: bool = False
    dtype: str = "fp8"          # fp8 (e4m3), int8, bf16
    kernel: str = "recon148"    # the generator the constants were fitted to: recon148, or recon155
    sets: int = 1               # copies alternated between passes (footprint multiplier of the measurement)
    allow_value_change: bool = False   # a KV split is not value-preserving (recon 154)
    registers: int | None = None


@dataclass
class Decision:
    workload: object
    schedule: Schedule
    predicted_ms: float | None
    bound: str | None
    fallback: bool
    basis: str
    reasons: tuple
    citations: tuple
    candidates: list = field(default_factory=list)   # (Schedule, Prediction | None, excluded reason | None)


# ------------------------------------------------------------------ gated FFN

FFN_SHAPE = (2048, 8192)
FFN_KERNEL_REGISTERS = 112   # the cut kernels p12 / p3 use 80 and 112 registers (recon 146 part 5)
TP_MAX_TOKENS = 80 * 16      # measured to 80 threadgroups, four waves (recon 146 part 3)


LB_MAX_TOKENS = 128          # the layerbench M5 sweep: 16 to 128 tokens (MM 25.126)
# the margin a PREDICTED win must clear at a token count no schedule was dispatched at: the recon family's
# constants come from different processes (9 to 13 percent between them, recon 146 part 5); the layerbench
# family's come from one process per format, where the additive WS form held within 1 to 4 percent (25.126)
MARGIN = {"recon146": 0.13, "layerbench": 0.04}


def _ffn_tp_lb(w, nsg):
    blocks = math.ceil(w.tokens / 16)
    wave = C[f"lb_tp_{w.dtype}"].value
    checks = (check_registers(w.registers or FFN_KERNEL_REGISTERS, nsg),
              Check("tokens", w.tokens <= LB_MAX_TOKENS,
                    f"{w.tokens} tokens = {blocks} threadgroups; the layerbench sweep covers 16 to 128 tokens, "
                    "all below one wave, where TP is flat", "MM 25.126"),
              Check("memory", True, "flat below one wave: measured, not modelled", "MM 25.126"),
              Check("uniform", True, "every threadgroup does the same work", "recon 146"))
    return Prediction(wave, wave, 0.0, True, Domain(checks))


def _ffn_ws_lb(w):
    """Layerbench WS blocks in turn, the additive form 25.126 found within 1 to 4 percent in-process:
    fp8 blocks of 32 (mt 2) and a last block of 16; bf16 blocks of 16."""
    if w.dtype == "fp8":
        full, rem = divmod(w.tokens, 32)
        parts = [C["lb_ws32_fp8"].value] * full + ([] if rem == 0 else [C["lb_ws16_fp8"].value] if rem <= 16
                                                   else [C["lb_ws32_fp8"].value])
        key, measured = ("lb_ws16_fp8" if w.tokens <= 16 else "lb_ws32_fp8"), w.tokens in (16, 32)
    else:
        parts = [C["lb_ws16_bf16"].value] * math.ceil(w.tokens / 16)
        key, measured = "lb_ws16_bf16", w.tokens == 16
    checks = (check_registers(w.registers or FFN_KERNEL_REGISTERS, 8),
              Check("tokens", w.tokens <= LB_MAX_TOKENS, f"{w.tokens} tokens; the sweep covers 16 to 128",
                    "MM 25.126"),
              Check("memory", True, "inside the per-block constant", "MM 25.126"),
              Check("uniform", True, "every simdgroup of each kernel walks the same number of chunks", "recon 149"))
    note = "measured" if measured else "prediction: the sum of same-process blocks (MM 25.126: within 1 to 4 percent)"
    return Prediction(max(parts), sum(parts), 0.0, True, Domain(checks), note=note), key


def _ffn_tp(w, nsg):
    if w.kernel == "layerbench":
        return _ffn_tp_lb(w, nsg)
    blocks = math.ceil(w.tokens / 16)
    wave = C[f"ffn_wave_{w.dtype}"].value
    waves = math.ceil(blocks / CORES)
    # one threadgroup per core per wave; below one wave the time does not fall (recon 146 part 4)
    checks = [check_registers(w.registers or FFN_KERNEL_REGISTERS, nsg),
              Check("tokens", w.tokens <= TP_MAX_TOKENS,
                    f"{w.tokens} tokens = {blocks} threadgroups ({waves} waves); measured 1..80 threadgroups",
                    "recon 146 part 3, 149 part 3")]
    # a wave is the measured unit: T_latency = one wave, T_throughput = waves x wave
    p = Prediction(wave, waves * wave, 0.0, True, Domain(tuple(checks) + (
        Check("memory", True, "issue-bound: shared weights come from the caches", "recon 146 part 4"),
        Check("uniform", True, "every threadgroup does the same work", "recon 146"))))
    return p


def ws_stage1_ms(hg, nsg, dtype="fp8"):
    """Weight-stationary kernel A: 256 hidden chunks interleaved over hg x nsg simdgroups, down to the
    DRAM floor of the gate and up weights (42 MB in fp8 with the scale tables; recon 149 part 4)."""
    if dtype != "fp8":
        raise ValueError("the stage-1 chunk constant is measured in e4m3 only (recon 149 part 4)")
    sgs = hg * nsg
    steps = [256 // sgs + (1 if i < 256 % sgs else 0) for i in range(sgs)]
    return predict(steps, C["ws_chunk_fp8"].value, 0.0, unique_bytes=42_000_000, footprint_bytes=42_000_000)


def _ffn_ws(w):
    """Blocks run in turn (recon 150 part 4 step 2): fp8 in blocks of 32 tokens (mt 2) with a last
    block of 16 (mt 1) when at most 16 remain; bf16 in blocks of 16 (mt 1; mt 2 was not measured)."""
    if w.kernel == "layerbench":
        return _ffn_ws_lb(w)
    if w.dtype == "fp8":
        full, rem = divmod(w.tokens, 32)
        parts = [C["ws_fp8_mt2"].value] * full + ([] if rem == 0 else [C["ws_fp8_mt1"].value] if rem <= 16
                                                   else [C["ws_fp8_mt2"].value])
        key = "ws_fp8_mt1" if w.tokens <= 16 else "ws_fp8_mt2"
    else:
        parts = [C["ws_bf16_mt1"].value] * math.ceil(w.tokens / 16)
        key = "ws_bf16_mt1"
    measured = (w.dtype == "fp8" and w.tokens in (16, 32)) or (w.dtype == "bf16" and w.tokens == 16)
    checks = (check_registers(w.registers or FFN_KERNEL_REGISTERS, 8),
              Check("memory", True, "stage 1 at the DRAM floor is inside the per-block constant", "recon 149 part 4"),
              Check("uniform", True, "every simdgroup of each kernel walks the same number of chunks", "recon 149"))
    p = Prediction(max(parts), sum(parts), 0.0, True, Domain(checks),
                   note=("measured" if measured else "prediction: blocks x (A + B + C) at a token count "
                         "that was not dispatched (recon 149 part 7; ledger row M5)"))
    return p, key


def _ffn_candidates(w):
    lim_n = w.d
    tp_nsg = 16 if w.dtype == "bf16" else 32
    out = []
    tp = _schedule("ffn.token_parallel.two_kernel", {"nsg": tp_nsg, "mt": 1, "threadgroups": math.ceil(w.tokens / 16)},
                   "recon 146 (mxlayer_ffn_exp.py)", lim_n, tp_nsg, 2)
    out.append((tp, _ffn_tp(w, tp_nsg), None))
    fused_bytes = tp_nsg * 1 * (1088 if w.dtype == "bf16" else 640)
    fused = _schedule("ffn.token_parallel.fused", {"nsg": tp_nsg, "mt": 1, "tg_bytes": fused_bytes},
                      "recon 146", lim_n, tp_nsg, 1)
    out.append((fused, None, ("threadgroup memory %d > 32768 (recon 146 part 6)" % fused_bytes)
                 if fused_bytes > C["tg_memory"].value else
                 "measured no faster: fused / two 0.95..1.14 (median 1.03) in fp8/int8, 1.45..1.56x slower in "
                 "bf16; cut when the fused kernel spills (recon 146 part 5, 150 part 4 rules 1 and 7)"))
    three = _schedule("ffn.token_parallel.three_kernel", {"nsg": tp_nsg, "mt": 1}, "recon 146", lim_n, tp_nsg, 3)
    out.append((three, None, "measured 1.02..1.06x the two-kernel time in every row (recon 146 part 5)"))
    if w.dtype in ("fp8", "bf16"):
        p, key = _ffn_ws(w)
        mt = 1 if (w.dtype == "bf16" or w.tokens <= 16) else 2
        ws = _schedule("ffn.weight_stationary", {"mt": mt, "A": "64x4" if key == "ws_fp8_mt1" else "32x8" if
                                                   key == "ws_fp8_mt2" else "32x4",
                                                   "B": "8x4" if key == "ws_fp8_mt1" else "32x1", "C": "1x32"},
                       "recon 149 (mxlayer_ws.py)", lim_n, 4, 3)
        out.append((ws, p, None))
    else:
        ws = _schedule("ffn.weight_stationary", {}, "recon 149", lim_n, 4, 3)
        out.append((ws, None, "int8 weight-stationary was not measured (recon 149 labels)"))
    if w.kernel == "layerbench" and w.dtype == "fp8":
        # one WS pass holding every token (mt 3 or 4): bit-exact, spills 45 to 155 stores, and its committed
        # heuristic under-predicted it by 32 to 35 percent -- so it has NO model, only its two measurements
        mt = math.ceil(w.tokens / 16)
        one = _schedule("ffn.weight_stationary.one_pass", {"mt": mt, "A": "32x8", "B": "32x1", "C": "1x32"},
                        "MM 25.126 (g17layerbench)", lim_n, 8, 3)
        key = {48: "lb_onepass48_fp8", 64: "lb_onepass64_fp8"}.get(w.tokens)
        if key is None:
            out.append((one, None, "one-pass WS has no model (its heuristic under-predicted by 32 to 35 percent); "
                                   "measured only at 48 and 64 e4m3 tokens (MM 25.126)"))
        else:
            v = C[key].value
            out.append((one, Prediction(v, v, 0.0, True, Domain((
                check_registers(w.registers or FFN_KERNEL_REGISTERS, 8),
                Check("memory", True, "measured", "MM 25.126"),
                Check("uniform", True, "one pass, equal chunks per simdgroup", "MM 25.126"))), note="measured"), None))
    return out


def _choose_ffn(w):
    base = []
    if (w.d, w.hidden) != FFN_SHAPE:
        base.append(Check("shape", False, f"d {w.d}, Hh {w.hidden}: every FFN constant is d 2048, Hh 8192",
                          "recon 146, 149, 150 'not measured'"))
    if w.kernel not in MARGIN:
        base.append(Check("constants", False, f"no FFN constants for kernel family {w.kernel!r} "
                          "(layerbench or recon146)", "recon 146; MM 25.126"))
    elif w.kernel == "layerbench" and w.dtype not in ("fp8", "bf16"):
        base.append(Check("dtype", False, f"{w.dtype}: the layerbench M5 sweep timed e4m3 and bf16 only",
                          "MM 25.126"))
    if w.dtype not in ("fp8", "int8", "bf16"):
        base.append(Check("dtype", False, f"{w.dtype} FFN timing was not measured (fp8, int8, bf16 were)",
                          "recon 146"))
    cands = _ffn_candidates(w) if not base else []
    safe_nsg = 16 if w.dtype == "bf16" else 32
    safe = _schedule("ffn.token_parallel.two_kernel", {"nsg": safe_nsg, "mt": 1, "threadgroups": math.ceil(w.tokens / 16)},
                     "recon 146", w.d, safe_nsg, 2)
    safe_why = ("known-safe: the two-kernel cut is bit-exact at layer scale in fp8, int8 and bf16, needs no "
                "threadgroup memory beyond the norm's 2 KiB and does not spill (recon 146 parts 2 and 5)")
    keys = (("cores", "lb_ws16_fp8", "lb_ws32_fp8", "lb_tp_fp8", "lb_onepass48_fp8", "lb_onepass64_fp8",
             "lb_ws16_bf16", "lb_tp_bf16", "tg_memory") if w.kernel == "layerbench" else
            ("cores", "ffn_wave_fp8", "ffn_wave_int8", "ffn_wave_bf16", "ws_fp8_mt1", "ws_fp8_mt2",
             "ws_bf16_mt1", "tg_memory"))
    return _decide(w, cands, base, safe, safe_why, keys, margin=MARGIN.get(w.kernel, 0.13))


# ------------------------------------------------------------------ attention

def causal_steps(queries, seq):
    """Key blocks per 16-query tile with the loop stopped at the diagonal (recon 155):
    min(nch, ((qb + 15) >> 5) + 1), queries aligned to the end of the key axis."""
    nch = math.ceil(seq / 32)
    q0 = seq - queries
    return [min(nch, ((q0 + 16 * i + 15) >> 5) + 1) for i in range(math.ceil(queries / 16))]


def _attn_constants(w):
    if w.kernel == "recon148":
        if w.dtype in ("fp8", "int8"):
            return C["attn_L_fp8"].value, C["attn_C_fp8"].value, ("attn_L_fp8", "attn_C_fp8")
        if w.dtype == "bf16":
            return C["attn_L_bf16"].value, C["attn_C_bf16"].value, ("attn_L_bf16", "attn_C_bf16")
    if w.kernel == "recon155" and w.dtype == "fp8":
        return C["attn155_L"].value, C["attn155_C"].value, ("attn155_L", "attn155_C")
    return None


def _attn_base_checks(w):
    out = []
    if _attn_constants(w) is None:
        out.append(Check("constants", False, f"no measured step constants for kernel {w.kernel!r} in {w.dtype} "
                         "(constants belong to the generator they were fitted to: recon 155.1 found the "
                         "recon-148 constants over-predict another generator by 48 percent)",
                         "recon 148, 150, 155.1"))
    if head_width_factor(w.head_dim) is None:
        out.append(Check("head_dim", False, f"head width {w.head_dim}: timed at 64, 96, 128, 144, 192 and 256; "
                         "proportional scaling holds for multiples of 32 from 96 to 256, and a 16-wide remainder "
                         "is measured only at 144", "MM 25.126"))
    return out


def head_width_factor(dh):
    """The attention step constants at head width ``dh`` are the width-128 constants times this factor
    (MM 25.126, same-process ratios to width 128, at 12.8 and at 1.6 simdgroups per core alike):
    - dh a multiple of 32 from 96 to 256: dh / 128, the MMA count ratio (measured within -6 to +2 percent);
    - dh = 144, a 16-wide remainder: the remainder costs a whole 32-wide block, ceil(dh / 32) x 32 / 128 =
      1.25 (measured 1.26 to 1.32); other remainders were not measured;
    - dh = 64: its measured factor, 0.355 (0.31 to 0.36: well under proportional);
    - otherwise None (outside the domain).
    The affine per-MMA extrapolation (section 150 part 6's 68.2 / 7.0 ns per extra MMA) is RETIRED for width:
    25.126 measured its ratio error at -16 to -32 percent above 128."""
    if dh == 64:
        return C["hw_64"].value
    if dh % 32 == 0 and 96 <= dh <= 256:
        return dh / 128
    if dh == 144:
        return math.ceil(dh / 32) * 32 / 128
    return None


def _attn_predict(w, L, Cs, tasks_steps, nsg):
    bpe = 2.0 if w.dtype == "bf16" else 1.25          # fp8 / int8 carry block scales (recon 148 part 1)
    unique = w.heads * w.seq * w.head_dim * 2 * bpe   # K and V; tasks of a head share them (recon 150 part 3)
    tgs = math.ceil(len(tasks_steps) / nsg)
    knee = L / Cs
    checks = [check_registers(w.registers or 126, min(len(tasks_steps) / CORES, knee)),
              Check("threadgroups", tgs >= CORES or nsg == 1,
                    f"{tgs} threadgroups of {nsg}: fewer than 20 leaves cores idle (the 16-SG row is 6 percent slower)",
                    "recon 148 part 3, 150 part 3 step 5")]
    return predict(tasks_steps, L, Cs, unique_bytes=unique, footprint_bytes=unique * w.sets, checks=checks)


def _choose_attention(w):
    base = _attn_base_checks(w)
    cands = []
    ntiles = math.ceil(w.queries / 16)
    tasks = w.heads * ntiles
    if not base:
        L, Cs, _ = _attn_constants(w)
        f = head_width_factor(w.head_dim)      # MM 25.126: both step constants scale
        L, Cs = L * f, Cs * f
        nch = math.ceil(w.seq / 32)
        full = [nch] * tasks
        if w.causal:
            skip = causal_steps(w.queries, w.seq) * w.heads
            nsg = _attn_nsg(tasks)
            s_skip = _aschedule(w, "attention.fused.causal_skip", {"nsg": nsg, "mt": 1, "split": 1},
                               "recon 155 (mxattn_causal.py)", w.head_dim, nsg, 1)
            s_full = _aschedule(w, "attention.fused.causal_full_loop", {"nsg": nsg, "mt": 1, "split": 1},
                               "recon 155", w.head_dim, nsg, 1)
            cands.append((s_skip, _attn_predict(w, L, Cs, skip, nsg), None))
            cands.append((s_full, _attn_predict(w, L, Cs, full, nsg),
                          "measured slower than the diagonal skip at every point (recon 155: 1.05x at 32 tasks, "
                          "1.06x, 1.21x, 1.37x at 32, 256, 1,024); the skip is value-preserving by proof and "
                          "measurement (6 of 6 bit-exact)"))
        else:
            for nsg in (1, 2, 4, 8, 16):
                sch = _aschedule(w, "attention.fused", {"nsg": nsg, "mt": 1, "split": 1},
                                "recon 148 (mxattn_exp.py)", w.head_dim, nsg, 1)
                excl = None
                if nsg != _attn_nsg(tasks):
                    excl = ("the model does not separate simdgroups per threadgroup (measured 1.396..1.459 ms "
                            "for 1..8 at 256 tasks); the measured default is kept (recon 148 part 3)")
                if tasks / nsg < CORES and nsg != 1:
                    excl = f"{math.ceil(tasks / nsg)} threadgroups < 20 cores (recon 148 part 3)"
                cands.append((sch, _attn_predict(w, L, Cs, full, nsg) if excl is None else None, excl))
            mt2 = _aschedule(w, "attention.fused", {"nsg": 8, "mt": 2, "split": 1}, "recon 148", w.head_dim, 8, 1)
            cands.append((mt2, None, "two token tiles per task: 2.917 against 1.455 ms, 354 spilled stores "
                                     "(recon 148 part 3)"))
            knee_sgs = CORES * L / Cs
            for s in (2, 4, 8, 16):
                sch = _aschedule(w, "attention.fused.kv_split", {"nsg": 1, "mt": 1, "split": s},
                                "recon 148 part 5 (no merge), recon 154 (merge)", w.head_dim, 1, 2)
                if not w.allow_value_change:
                    cands.append((sch, None, "a KV split is not value-preserving: 0.4 to 4 percent of the row "
                                             "maximum against the sequential chain (recon 154)"))
                    continue
                if tasks >= knee_sgs:
                    cands.append((sch, None, f"{tasks} tasks already reach the {knee_sgs:.0f}-simdgroup knee "
                                             "(recon 150 part 3 step 4)"))
                    continue
                if s > 8 or nch % s:
                    cands.append((sch, None, "16 slices bought nothing after 8 (recon 148 part 5); the merged "
                                             "split reverses past 4 to 8 (recon 154)" if s > 8 else
                                  f"{nch} key blocks do not split {s} ways"))
                    continue
                p = _attn_predict(w, L, Cs, [nch // s] * (tasks * s), 1)
                # the per-slice merge is not in the section-150 model: the time is a lower bound
                p = Prediction(p.t_latency_ms, p.t_throughput_ms, p.t_memory_ms, False, Domain(
                    tuple(c for c in p.domain.checks if c.name != "uniform") + (
                        Check("uniform", False, "the merge of the slices is not in the model: a LOWER BOUND "
                              "(recon 154 measured it)", "recon 150 part 3, 154"),)))
                cands.append((sch, p, None))
    nsg0 = _attn_nsg(tasks)
    safe = _aschedule(w, "attention.fused" + (".causal_full_loop" if w.causal else ""),
                     {"nsg": nsg0, "mt": 1, "split": 1}, "recon 147, 148", w.head_dim, nsg0, 1)
    safe_why = ("known-safe: one fused kernel, one 16-query tile per simdgroup, no KV split; bit-exact in fp8, "
                "int8, bf16 and fp16 and the fastest schedule in all 27 measured shapes (recon 147, 148)")
    keys = ("cores", "bw_slc", "bw_dram") + (_attn_constants(w)[2] if _attn_constants(w) else ())
    return _decide(w, cands, base, safe, safe_why, keys)


def _attn_nsg(tasks):
    """Largest measured simdgroups-per-threadgroup up to the default 8 that keeps 20 threadgroups."""
    for nsg in (8, 4, 2, 1):
        if tasks / nsg >= CORES:
            return nsg
    return 1


# ------------------------------------------------------------------ the decision

def _decide(w, cands, base, safe, safe_why, keys, margin=None):
    margin = SPREAD if margin is None else margin
    live = [(s, p) for s, p, excl in cands if excl is None and p is not None]
    failing = list(base)
    for _, p in live:
        failing += [c for c in p.domain.checks if not c.inside and c.name not in ("uniform",)]
    if base or not live or any(not c.inside and c.name != "uniform" for _, p in live for c in p.domain.checks):
        reasons = tuple(dict.fromkeys(f"{c.name}: {c.detail} [{c.source}]" for c in failing))
        return Decision(w, safe, None, None, True, "fallback", reasons + (safe_why,), cite(*keys), cands)
    # compare predicted TIMES only; never a ratio (recon 155.1)
    best_s, best_p = min(live, key=lambda sp: sp[1].ms)
    basis = best_p.note or ("model estimate" if best_p.bound == "estimate" else
                            "model LOWER BOUND (non-uniform work); compared as a time, never as a ratio")
    reasons = tuple(f"{s.kind} {dict(s.params)}: {p.ms:.3f} ms ({p.bound}, {p.dominant}-bound)"
                    for s, p in sorted(live, key=lambda sp: sp[1].ms))
    reasons += tuple(f"{s.kind} {dict(s.params)} excluded: {excl}" for s, _, excl in cands if excl)
    # AT AN UNDISPATCHED POINT, SELECT CONSERVATIVELY: a predicted win over the known-safe schedule counts
    # only when it is larger than the family's margin (MARGIN: recon 146's 13 percent between-process
    # spread; layerbench's 4 percent in-process additivity, MM 25.126). Both are uniform estimates, so this
    # compares two times.
    safe_live = next(((s, p) for s, p in live if s.kind == safe.kind and p.bound == "estimate"), None)
    if (best_p.note.startswith("prediction") and safe_live is not None and best_s is not safe_live[0]
            and best_p.ms > safe_live[1].ms - margin * safe_live[1].ms):
        s, p = safe_live
        basis = (f"conservative: {best_s.kind} is predicted at {best_p.ms:.3f} ms against {p.ms:.3f} ms, "
                 f"a margin inside the family's {margin:.0%} (recon 146 part 5; MM 25.126), at a point "
                 "no schedule was dispatched at")
        return Decision(w, s, p.ms, p.bound, False, basis, reasons, cite(*keys), cands)
    return Decision(w, best_s, best_p.ms, best_p.bound, False, basis, reasons, cite(*keys), cands)


SPREAD = 0.13   # recon 146 part 5: the same configuration in two processes differed by 9 to 13 percent


def choose(workload):
    """Pick a schedule for an FFN or an Attention workload. See the module docstring."""
    if isinstance(workload, FFN):
        return _choose_ffn(workload)
    if isinstance(workload, Attention):
        return _choose_attention(workload)
    raise TypeError(f"no scheduler for {type(workload).__name__}")


# ===================================================================================================
# CUT POINTS, FUSED-OR-CUT ATTENTION, AND THE KV SPLIT (MM 25.124.3). Three entries that the production
# rows P5 (cost-based cuts), P7 (the selected fused/cut schedule) and P9 (merge numerics) cite. Each
# returns a Decision: a safe schedule with reasons and no number outside its measured domain.

# recon 142 part 2 and part 7 (recon 144 part 7): cut time over fused time, measured, (low, high), per
# regime. Ratios are within one run (1.7 points of noise; differences under 3 percent not claimed).
CUT_RATIO = {
    # fused spill-free, 64 simdgroups per core: cut after the quantize (p12|p3), before it (p1|p23), three
    ("spill_free", "many"): {"quantize": (1.05, 1.10), "pre_quantize": (1.06, 1.08), "all": (1.08, 1.20)},
    # fused spills 15 or more stores: 0.95..1.00, 0.95..0.98, 0.99..1.06 at 64 SG
    ("spills", "many"): {"quantize": (0.95, 1.00), "pre_quantize": (0.95, 0.98), "all": (0.99, 1.06)},
    # the same shapes at one simdgroup per core: the cuts need 1 to 17 percent less
    ("spills", "one"): {"quantize": (0.83, 0.95), "pre_quantize": (0.87, 0.99), "all": (0.85, 0.98)},
    # one simdgroup per core, long K (1,024 blocks or more), spill-free: 0.93 to 0.99
    ("spill_free_long_k", "one"): {"quantize": (0.97, 0.99), "pre_quantize": (0.93, 0.94), "all": (0.94, 0.94)},
}
SPILL_CUT = 15          # recon 142 part 7: 15 or more spilled stores and the cuts win
CUT_KINDS = {"quantize": "chain.cut.quantize", "pre_quantize": "chain.cut.pre_quantize", "all": "chain.cut.all"}
CHAIN_KEYS = ("instr_1sg", "instr_64sg", "mma_dep_1sg_fp8", "mma_dep_64sg_fp8", "mma_dep_1sg_bf16",
              "mma_dep_64sg_bf16")


@dataclass(frozen=True)
class Stage:
    """One tensor stage of a chain. ``fmt``: "fp8" (MX software-scaled e4m3), "bf16" or "int8".
    ``mmas`` and ``instructions`` are per chain per simdgroup (the recon-142 fit's units); ``bytes`` is
    what the stage streams. ``measured_ms``: measured standalone times per format, {fmt: ms}, when the
    stage was timed as its own dispatch (recon 149's kernels A, B and C); None otherwise."""
    name: str
    fmt: str
    mmas: int = 0
    instructions: int = 0
    bytes: int = 0
    measured_ms: dict | None = None


@dataclass(frozen=True)
class Chain:
    """Tensor stages joined at quantize boundaries.
    ``fused_spill_stores``: the compiled fused kernel's spill count, COUNTED after the compile (recon 150
    rule 7: not predictable from shapes); None = not counted. ``sg_per_core``: resident simdgroups per
    core. ``k_blocks``: the first stage's K loop in 32-element blocks. ``process``: {fmt: process id} of
    the stages' measured times: a format comparison needs ONE process (MM 25.126). ``block_tokens``: the
    block size the stage times were measured at and the chain will run. ``mix_built``: the mixed-format
    graph exists (the harness's ws_mix kernels, MM 25.126); ``allow_unbuilt``: select a mix that does
    not."""
    stages: tuple
    fused_spill_stores: int | None = None
    sg_per_core: float = 64
    k_blocks: int = 64
    process: dict | None = None
    allow_unbuilt: bool = False
    block_tokens: int | None = None
    mix_built: bool = False
    lowering: tuple | None = None     # the chain as tensorcuts sees it: stage dicts (M, N, K, a, b, epilogue)


def _chain_regime(c):
    if c.sg_per_core >= 32:
        occ = "many"
    elif c.sg_per_core <= 1:
        occ = "one"
    else:
        return None, f"{c.sg_per_core} simdgroups per core: the cut ratios were measured at 1 and at 64"
    if c.fused_spill_stores >= SPILL_CUT:
        return ("spills", occ), None
    if c.fused_spill_stores == 0:
        if occ == "one" and c.k_blocks >= 1024:
            return ("spill_free_long_k", "one"), None
        if occ == "many":
            return ("spill_free", "many"), None
        return None, "spill-free at one simdgroup per core with K under 1,024 blocks was not measured"
    return None, (f"{c.fused_spill_stores} spilled stores: between the measured spill-free and "
                  f"{SPILL_CUT}-or-more classes")


def _fallback(workload, safe, failing, safe_why, keys):
    reasons = tuple(f"{c.name}: {c.detail} [{c.source}]" for c in failing)
    return Decision(workload, safe, None, None, True, "fallback", reasons + (safe_why,), cite(*keys), [])


def _choose_cuts_unfiltered(chain):
    """P5's cost-based cut, and M6's per-stage format choice. Returns a Decision whose schedule kind is
    "chain.fused", "chain.cut.quantize" (p12|p3), "chain.cut.pre_quantize" (p1|p23), "chain.cut.all"
    (every boundary a dispatch) or "chain.cut.mixed_format". ``predicted_ms`` is per chain per simdgroup
    for the fused/cut choice (chain_ns's unit) and per layer pass for the format mix.

    Fused against cut: the fused time is ``chain_ns`` summed over the stages (recon 150 part 6, at 1 or 64
    simdgroups per core), and each cut costs that time times its MEASURED cut/fused ratio (CUT_RATIO,
    recon 142). A cut is chosen only when its whole measured range is below 1, the fused kernel only when
    every cut's whole range is above 1; a range straddling 1 selects the known-safe cut after the quantize.

    Format mix (row M6; corrected by 25.126's one-process M6 timing): when a stage carries measured
    times in more than one format, every stage is its own dispatch. The stage times must come from ONE
    recorded process and name the block size (``block_tokens``); a cross-process or block-less comparison
    is refused. Each stage keeps the previous stage's format unless another is more than 4 percent faster.
    A mix is selected when it is built (``mix_built``) or a prediction is allowed (``allow_unbuilt``)."""
    stages = tuple(chain.stages)
    fmts = {s.fmt for s in stages}
    safe = _schedule("chain.cut.quantize", {"cuts": "after each quantize"}, "recon 142, 146", 0, 1, len(stages))
    safe_why = ("known-safe: cut after the quantize boundary -- no fused spill risk, no threadgroup-memory "
                "bound, bit-exact at layer scale, within 0 to 10 percent of fused in every measured shape "
                "(recon 142 part 7, recon 146)")
    base = []
    if not 2 <= len(stages) <= 3:
        base.append(Check("stages", False, f"{len(stages)} stages: fused chains were measured with 2 and 3 GEMMs",
                          "recon 142"))
    if not fmts <= {"fp8", "bf16", "int8"}:
        base.append(Check("dtype", False, f"formats {sorted(fmts)}: fp8, bf16 and int8 were measured", "recon 142"))
    if base:
        return _fallback(chain, safe, base, safe_why, CHAIN_KEYS)
    if any(s.measured_ms and len(s.measured_ms) > 1 for s in stages):
        return _choose_stage_formats(chain, stages, safe_why)
    if len(fmts) > 1:
        return _fallback(chain, safe, [Check("fused", False, "a mixed-format register-resident chain was never "
                                             "built or timed", "recon 142; row M6")], safe_why, CHAIN_KEYS)
    if chain.fused_spill_stores is None:
        return _fallback(chain, safe, [Check("spills", False, "the fused kernel's spill count must be counted "
                                             "after the compile, not predicted", "recon 150 rule 7")],
                         safe_why, CHAIN_KEYS)
    regime, why = _chain_regime(chain)
    if regime is None:
        return _fallback(chain, safe, [Check("regime", False, why, "recon 142 part 7")], safe_why, CHAIN_KEYS)
    fmt = next(iter(fmts))
    sg = 1 if regime[1] == "one" else 64
    fused_ms = sum(chain_ns(s.mmas, s.instructions, fmt, sg) for s in stages) / 1e6
    ratios = dict(CUT_RATIO[regime])
    if len(stages) == 3:
        # the three-GEMM chain's cuts (a|bc, ab|c, a|b|c) were measured only spill-free at K1 <= 64, where
        # fused beat them by 1.14 to 1.76 (recon 142 part 7); anything else is outside the domain
        if regime != ("spill_free", "many") or chain.k_blocks > 2:
            return _fallback(chain, safe, [Check("regime", False, "a three-GEMM chain's cuts were measured only "
                                                 "spill-free with K1 <= 64 (2 blocks) at many simdgroups",
                                                 "recon 142 part 7")], safe_why, CHAIN_KEYS)
        ratios = {"all": (1.14, 1.76)}
    reasons = [f"chain.fused: {fused_ms:.6f} ms (chain_ns at {sg} SG per core)"]
    reasons += [f"{CUT_KINDS[k]}: {fused_ms * lo:.6f}..{fused_ms * hi:.6f} ms (measured ratio {lo:.2f}..{hi:.2f})"
                for k, (lo, hi) in ratios.items()]
    below = {k: r for k, r in ratios.items() if r[1] < 1.0}
    if all(r[0] > 1.0 for r in ratios.values()):
        kind, ms, basis = "chain.fused", fused_ms, "fused: every cut's measured range is above the fused time"
    elif below:
        k = min(below, key=lambda k: below[k][1])
        kind, ms = CUT_KINDS[k], fused_ms * below[k][1]
        basis = f"cut: its whole measured range {below[k][0]:.2f}..{below[k][1]:.2f} is below the fused time"
    else:
        kind, ms = safe.kind, fused_ms * ratios["quantize"][1]
        basis = "a measured range straddles the fused time: the known-safe cut after the quantize"
    sch = _schedule(kind, {"regime": "/".join(regime)}, "recon 142 part 7", 0, 1, 1 if kind == "chain.fused" else 2)
    return Decision(chain, sch, ms, "estimate", False, basis, tuple(reasons),
                    cite(*CHAIN_KEYS) + ("CUT_RATIO: recon 142 parts 2 and 7",), [])


FORMAT_TIE = 0.04       # MM 25.126: in one process the additive WS form held within 1 to 4 percent


def _choose_stage_formats(chain, stages, safe_why):
    """Each stage its own dispatch, each taking a format, from stage times measured IN ONE PROCESS at the
    block size the chain will run (MM 25.124.3, corrected by the M6 timing of 25.126): a stage keeps the
    previous stage's format unless another is more than FORMAT_TIE faster, so a format boundary is paid
    only for a real difference. Cross-process or block-less stage times are REFUSED: on the rebuilt harness
    they made e4m3 stage B look faster than bf16 (0.176 against 0.201 ms, two processes), while one process
    measured 0.184 to 0.200 in every format at 16 tokens and 0.81 against 0.49 at 32."""
    declared = _schedule("chain.cut.all", {"format": "as declared"}, "recon 149; MM 25.126", 0, 1, len(stages))
    multi = [s for s in stages if s.measured_ms and len(s.measured_ms) > 1]
    formats = set.intersection(*[set(s.measured_ms) for s in multi])
    fails = []
    if not all(s.measured_ms for s in stages) or not formats:
        fails.append(Check("measured", False, "every stage needs measured times in a common format", "recon 149"))
    if chain.block_tokens is None:
        fails.append(Check("block", False, "stage times must name the block size they were measured at: the "
                           "e4m3 stage B costs 0.20 ms at 16 tokens and 0.81 at 32 (MM 25.126, M6)", "MM 25.126"))
    procs = chain.process or {}
    ids = {procs.get(f) for f in formats}
    if None in ids or len(ids) != 1:
        fails.append(Check("process", False, f"the formats' stage times come from different or unrecorded "
                           f"processes ({procs or 'none recorded'}); a format comparison across processes is "
                           "refused (MM 25.126: absolute times moved 25 to 44 percent between processes, and "
                           "the cross-process M5 kernels inverted stage B)", "MM 25.124.3, 25.126"))
    if fails:
        return _fallback(chain, declared, fails, safe_why, CHAIN_KEYS)

    def at(s, f):
        return s.measured_ms[f] if f in s.measured_ms else min(s.measured_ms.values())
    pure = {f: sum(at(s, f) for s in stages) for f in formats}
    best = min(pure, key=pure.get)
    chosen, prev = [], None
    for s in stages:
        fastest = min(s.measured_ms, key=s.measured_ms.get)
        keep = prev if prev in s.measured_ms and s.measured_ms[prev] <= s.measured_ms[fastest] * (1 + FORMAT_TIE) else None
        prev = keep or fastest
        chosen.append(prev)
    mix = sum(s.measured_ms[f] for s, f in zip(stages, chosen))
    reasons = tuple(f"pure {f}: {ms:.3f} ms (same-process stage times, {chain.block_tokens}-token block)"
                    for f, ms in sorted(pure.items(), key=lambda x: x[1]))
    reasons += (f"per stage {tuple(chosen)}: {mix:.3f} ms (a format changes only beyond a {FORMAT_TIE:.0%} "
                "difference)",)
    if len(set(chosen)) == 1:
        f = chosen[0]
        return Decision(chain, _schedule("chain.cut.all", {"format": f}, "MM 25.126", 0, 1, len(stages)),
                        pure[f], "estimate", False, f"pure {f}: no stage differs by more than {FORMAT_TIE:.0%}",
                        reasons, cite(*CHAIN_KEYS), [])
    if not (chain.mix_built or chain.allow_unbuilt):
        return Decision(chain, _schedule("chain.cut.all", {"format": best}, "MM 25.126", 0, 1, len(stages)),
                        pure[best], "estimate", False, f"pure {best}; the mix {tuple(chosen)} is not built "
                        "(mix_built=True when it is, allow_unbuilt=True to select a prediction)",
                        reasons, cite(*CHAIN_KEYS), [])
    sch = _schedule("chain.cut.mixed_format", {"formats": tuple(chosen)}, "MM 25.126 (M6)", 0, 1, len(stages))
    return Decision(chain, sch, mix, "estimate", False,
                    "mixed formats: " + ("built" if chain.mix_built else "prediction; not built"),
                    reasons, cite(*CHAIN_KEYS), [])



# the order a replacement is taken in when a decided kind cannot be lowered: the known-safe cut first
LOWERABLE_ORDER = ("chain.cut.quantize", "chain.cut.all", "chain.fused")


def lowerable_kinds(chain):
    """({P11 kind: (lowerable, why)}, scope). With ``chain.lowering`` (tensorcuts stage dicts) the answer
    is tensorcuts.p11_kinds_for for THIS chain's boundaries; without it, tensorcuts.P11_KINDS' kind-level
    flags, which cannot see a boundary the chain lacks (the reasons say so)."""
    from agxforge.g17 import tensorcuts
    if chain.lowering is not None:
        return dict(tensorcuts.p11_kinds_for([dict(s) for s in chain.lowering])), "this chain's boundaries"
    return ({k: (built, why) for k, (built, _via, why) in tensorcuts.P11_KINDS.items()},
            "kind level only (pass Chain.lowering for this chain's boundaries)")


def choose_cuts(chain):
    """P5's cost-based cut (and M6's format choice), restricted to what the compiler can LOWER.

    The decision is ``_choose_cuts_unfiltered``'s (measured ratios, one-process format data, the domain);
    then, unless ``chain.allow_unbuilt`` is True, a decided kind the lowering cannot emit
    (tensorcuts.p11_kinds_for; P5, MM 25.128.3) is dropped. The replacement is the first lowerable of
    LOWERABLE_ORDER, returned as a fallback with no number, and the reasons NAME every kind the lowering
    cannot emit, with its reason. If none is lowerable the decided schedule is kept as a refusal-grade
    fallback naming all of them. Candidates chain.cut.pre_quantize and chain.cut.mixed_format are the
    kinds P5 reports unbuilt today."""
    d = _choose_cuts_unfiltered(chain)
    if chain.allow_unbuilt:
        return d
    kinds, scope = lowerable_kinds(chain)
    unbuilt = {k: why for k, (ok, why) in kinds.items() if not ok}
    named = tuple(f"unlowerable {k}: {why} [tensorcuts.p11_kinds_for, {scope}]" for k, why in sorted(unbuilt.items()))
    if d.schedule.kind not in unbuilt and d.schedule.kind in kinds:
        return Decision(d.workload, d.schedule, d.predicted_ms, d.bound, d.fallback, d.basis,
                        d.reasons + named, d.citations, d.candidates)
    dropped = d.schedule.kind
    for k in LOWERABLE_ORDER:
        if k in kinds and k not in unbuilt:
            sch = _schedule(k, dict(d.schedule.params, dropped=dropped), "tensorcuts (P5)", 0, 1,
                            1 if k == "chain.fused" else 2)
            return Decision(d.workload, sch, None, None, True,
                            f"fallback: {dropped} cannot be lowered; {k} is the first lowerable of {LOWERABLE_ORDER}",
                            (f"dropped {dropped} (decided: {d.basis})",) + named + d.reasons,
                            d.citations + ("tensorcuts.p11_kinds_for (P5, MM 25.128.3)",), d.candidates)
    return Decision(d.workload, d.schedule, None, None, True, f"fallback: no P11 kind can be lowered for this chain",
                    (f"dropped {dropped}",) + named + d.reasons, d.citations, d.candidates)

# recon 148 part 2: cut time over fused time over its 27 measured mt = 1 rows, (low, median, high)
ATTN_CUT_RATIO = {"p_cut": (1.03, 1.13, 1.30), "s_cut": (1.07, 1.28, 1.42), "both_cuts": (1.14, 1.65, 2.03)}
ATTN_SCHEDULES = ("fused", "p_cut", "s_cut", "both_cuts", "proj_cut")
# "proj_cut" is P7's runtime schedule attention.proj_cut (MM 25.129): K/V projected and stored in operand
# layout by one dispatch, attention in a second; bit-identical to attention.fused. It is none of the
# in-attention cuts above. NO RATIO IS RECORDED FOR IT: P7's receipts carry per-query gpu_seconds from
# its correctness runs (fused_n3 0.538 to 0.578 ms; project 0.048 + attend 0.499), but each arm is its
# own process at an unwarmed clock -- the cross-process comparison 25.124.3 refuses -- so it is unpriced.
ATTN_UNPRICED = {"proj_cut": ("proj_cut is unpriced: P7 recorded no same-process timing of attention.proj_cut "
                              "against attention.fused (its correctness receipts are separate processes at an "
                              "unwarmed clock, MM 25.121 and 25.124.3); fused is measured and preferred")}


def choose_attention_cut(workload, exposed=ATTN_SCHEDULES):
    """P7's fused-or-cut attention: pick among the schedules the caller can realize, ``exposed`` a subset
    of ("fused", "p_cut" = P materialized, "s_cut" = S materialized, "both_cuts", "proj_cut" = P7's
    attention.proj_cut, the projection cut). The fused time is ``choose(workload)``'s; an in-attention
    cut is priced at the LOW end of its measured ratio to fused (recon 148 part 2: fused was fastest in
    all 27 shapes), so choosing fused never rests on an optimistic cut. ``proj_cut`` has no recorded
    ratio, so it is never priced: fused is preferred when exposed, a priced cut next, and a proj_cut-only
    exposure returns it as a fallback with no number. For a causal workload the ratios were measured only
    on uniform work: every time is a lower bound. Outside the attention domain: the first exposed of
    fused, p_cut, s_cut, both_cuts, proj_cut, with no number."""
    exposed = tuple(exposed)
    if not exposed or any(e not in ATTN_SCHEDULES for e in exposed):
        raise ValueError(f"exposed must be a non-empty subset of {ATTN_SCHEDULES}; got {exposed}")
    order = [e for e in ATTN_SCHEDULES if e in exposed]
    safe = _aschedule(workload, "attention." + order[0], {"exposed": exposed}, "recon 147, 148; MM 25.129", workload.head_dim,
                     8, 1 if order[0] == "fused" else 2)
    base = choose(workload)
    safe_why = ("known-safe: fused when exposed (fastest in all 27 measured shapes, bit-exact), else the "
                "cheapest measured cut (recon 148 part 2), else an unpriced proj_cut (bit-identical, MM 25.129)")
    unpriced = tuple(ATTN_UNPRICED[k] for k in order if k in ATTN_UNPRICED)
    if base.fallback:
        return Decision(workload, safe, None, None, True, "fallback", base.reasons + unpriced + (safe_why,),
                        base.citations, [])
    t = base.predicted_ms
    times = {"fused": t, **{k: t * lo for k, (lo, _, _) in ATTN_CUT_RATIO.items()}}
    live = {k: times[k] for k in order if k in times}
    if not live:
        return Decision(workload, safe, None, None, True, "fallback: nothing exposed is priced",
                        unpriced + (safe_why,), base.citations, [])
    kind = min(live, key=live.get)
    reasons = tuple(f"{k}: {v:.3f} ms" + ("" if k == "fused" else f" (fused x {ATTN_CUT_RATIO[k][0]:.2f}, "
                                                                   "the low end of its measured range)")
                    for k, v in sorted(live.items(), key=lambda kv: kv[1])) + unpriced
    bound = base.bound
    if workload.causal:
        bound = "lower_bound"
        reasons += ("causal: the cut ratios were measured on uniform work; every time is a lower bound (recon 155.1)",)
    sch = _aschedule(workload, "attention." + kind, {"exposed": exposed, "under": base.schedule.kind}, "recon 148",
                    workload.head_dim, 8, 1 if kind == "fused" else 2)
    return Decision(workload, sch, live[kind], bound, False, "model: the fused time x the measured cut ratio",
                    reasons, base.citations + ("ATTN_CUT_RATIO: recon 148 part 2",), [])


# recon 154: the merged split's speedup over no split; one query tile, one threadgroup per tile, e4m3, dh 128
KV_MERGE_SPEEDUP = {16: {1: 1.00, 2: 1.60, 4: 2.52, 8: 1.59, 16: 0.54},
                    32: {1: 1.00, 2: 1.83, 4: 3.32, 8: 3.00, 16: 1.21}}


def choose_kv_split(workload):
    """P9's KV split and merge. A split is chosen ONLY when ``workload.allow_value_change`` is True:
    splitting the key axis is not value-preserving -- 0.4 to 4 percent of the row maximum against the
    sequential chain, from the rounding of exp2(s - m) at a per-slice running max, which no merge removes
    (recon 154) -- so its result must be validated against a reference that splits the same way.
    With the opt-in:
    - where recon 154 measured the merged split itself (one head, one 16-query tile, 16 or 32 key blocks,
      e4m3, head width 128, not causal), the measured best split is returned (4 ways at both, 2.52x and
      3.32x) with bound "measured_ratio";
    - otherwise ``choose(workload)`` decides: an s-way split is a candidate only below the saturation knee
      (tasks < 20 x L_step / C_step, recon 150 part 3), for s <= 8 and dividing the key blocks, and its time
      omits the merge, so it is a LOWER BOUND.
    Without the opt-in the answer is always unsplit."""
    nch = math.ceil(workload.seq / 32)
    measured = (workload.allow_value_change and workload.heads == 1 and workload.queries <= 16
                and nch in KV_MERGE_SPEEDUP and workload.dtype == "fp8" and workload.head_dim == 128
                and not workload.causal)
    if not measured:
        return choose(workload)
    table = KV_MERGE_SPEEDUP[nch]
    s = max(table, key=table.get)
    sch = _aschedule(workload, "attention.fused.kv_split", {"nsg": s, "mt": 1, "split": s}, "recon 154 (merge measured)",
                    workload.head_dim, s, 1)
    return Decision(workload, sch, None, "measured_ratio", False,
                    f"measured: the merged {s}-way split takes 1/{table[s]:.2f} of the unsplit time (recon 154)",
                    tuple(f"split {k}: {v:.2f}x" for k, v in sorted(table.items())),
                    ("KV_MERGE_SPEEDUP: recon 154",), [])


# ===================================================================================================
# ONE DECODE STEP (Bet 3's first milestone: d 2048, 16 heads, KV 256, one query row). Each stage gets
# the scheduler's choice where one exists, and every stage gets its MEMORY FLOOR: at one token the
# weights are read once, cold, from DRAM (a model larger than the SLC streams every step), so
# floor = weight bytes / bw_dram, with no footprint-plateau step. This is a floor, not a prediction.
# A stage with no scheduler is reported as such, with its floor, and never gets a time.

WEIGHT_BYTES = {"bf16": 2, "fp8": 1, "int8": 1}   # fp8/int8 scales not counted (under 1/32 of the bytes)


@dataclass(frozen=True)
class DecodeStep:
    """One decoder layer for one new token over ``kv`` cached positions. Multi-head attention with
    ``heads * head_dim == d``; a gated FFN (gate, up, down) of width ``hidden``. The KV cache is held
    in ``kv_dtype`` (bf16 by default: the attention constants were fitted on bf16 or e4m3 K/V)."""
    d: int = 2048
    heads: int = 16
    kv: int = 256
    hidden: int = 8192
    dtype: str = "bf16"
    kv_dtype: str = "bf16"


@dataclass
class DecodeStage:
    name: str
    bytes: int
    floor_ms: float
    decision: Decision | None
    note: str

    @property
    def predicted_ms(self):
        return self.decision.predicted_ms if self.decision else None


@dataclass
class DecodePlan:
    step: DecodeStep
    stages: list

    @property
    def floor_ms(self):
        return sum(s.floor_ms for s in self.stages)

    @property
    def unpriced(self):
        """Stages without a predicted time: the plan's time is incomplete while any remain."""
        return [s.name for s in self.stages if s.predicted_ms is None]


def decode_step(w):
    """Stage-by-stage schedule choices and memory floors for one decode step (see above)."""
    if w.dtype not in WEIGHT_BYTES or w.kv_dtype not in WEIGHT_BYTES:
        raise ValueError(f"dtype {w.dtype!r} / kv_dtype {w.kv_dtype!r}: one of {sorted(WEIGHT_BYTES)}")
    if w.d % w.heads:
        raise ValueError(f"d {w.d} does not split over {w.heads} heads")
    es, kes, dh = WEIGHT_BYTES[w.dtype], WEIGHT_BYTES[w.kv_dtype], w.d // w.heads
    bw = C["bw_dram"].value * 1e9

    def stage(name, nbytes, decision, note):
        return DecodeStage(name, nbytes, nbytes / bw * 1e3, decision, note)

    norm = "RMSNorm: no scheduler (the op is not emitted by the compiler yet)"
    proj = ("one-token GEMM: memory-bound; no scheduler constant for a GEMV-shaped projection, so the "
            "floor is the only number")
    attn = choose(Attention(heads=w.heads, head_dim=dh, seq=w.kv, queries=1,
                            dtype="bf16" if w.kv_dtype == "bf16" else "fp8"))
    ffn = choose(FFN(tokens=1, d=w.d, hidden=w.hidden, dtype=w.dtype))
    stages = [
        stage("attn_norm", w.d * es, None, norm),
        stage("qkv_proj", 3 * w.d * w.d * es, None, proj),
        stage("attention", 2 * w.kv * w.d * kes, attn, "fused attention over the cached K/V (P11)"),
        stage("o_proj", w.d * w.d * es, None, proj + "; the residual add is its epilogue"),
        stage("ffn_norm", w.d * es, None, norm),
        stage("ffn", 3 * w.d * w.hidden * es, ffn, "gated FFN, gate/up/down, residual (P11)"),
    ]
    return DecodePlan(w, stages)
