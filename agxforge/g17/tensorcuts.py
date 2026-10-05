"""The cut points this compiler can lower between consecutive GEMM stages (production row P5,
machine model 25.128.3), as data a scheduler can call. This module ENUMERATES; it does not choose.
Choosing a cut by cost is production row P11's: Piece B's scheduler, `tensorsched.choose_cuts(chain)`
(machine model 25.124.3), whose decision kinds P11_KINDS maps onto the hand-offs built here.

A chain is a list of stages, each a gemm_generic-style dict: M, N, K, operand types `a` and `b`,
and an optional `epilogue` word list. Stage i + 1's A is stage i's output, so it has the same M
and its K is stage i's N. A later stage's `a` may be None: the cut then decides it, and every A type
a hand-off can produce is enumerated. For each boundary, `cut_points` lists every hand-off the lowering admits
today, with how many dispatches it costs, the producer's last epilogue word it requires, the
consumer's A type, and the receipts that admitted it. Every hand-off not listed is refused, and the
refusals are listed by name beside the options.

The hand-off kinds:
  register  one dispatch; the producer's D stays in registers as the consumer's operand. Admitted
            only for a key of cc.FEED_TABLE (P2, machine model 25.125); an absent key compiles as
            the memory bridge.
  memory    one dispatch; D is stored and reloaded as the consumer's A (the gemm_generic chain:
            a half x half producer, a float or half consumer A, a half B).
  cut_fp32  two dispatches; the producer stores fp32 C and the consumer reads it as a float A.
  cut_fp8   two dispatches; the producer's last epilogue step is the fp8 quantize-out (fp8e4m3 or
            fp8e5m2: RNE, no saturation, section 25.105), and the consumer reads those bytes as an
            fp8 A (M x N row-major bytes are exactly an M x K' fp8 A with K' = N).
"""
from agxforge.g17 import cc

FP8 = ("fp8e4m3", "fp8e5m2")
FP8_OUT = {"fp8e4m3": "fp8e4m3", "fp8e5m2": "fp8e5m2"}   # quantize-out word -> the A type it produces

# P11's decision kinds (tensorsched.choose_cuts) -> (built by this lowering?, the hand-off kinds that
# realise it, and what is missing when it is not built). The mapping is the contract between the two
# rows; test_g17tensormxcodes checks that every realising kind is one cut_points can emit.
P11_KINDS = {
    "chain.fused": (True, ("register", "memory"),
                    "built for a half x half producer into a float or half A only; no fp8, bf16 or int8 "
                    "stage has a one-dispatch hand-off (the register feed is fp32 or half, P2)"),
    "chain.cut.quantize": (True, ("cut_fp8",),
                           "the producer's last epilogue step is the fp8 quantize-out; the next dispatch "
                           "reads the bytes as its fp8 A"),
    "chain.cut.pre_quantize": (False, (),
                               "not built: the next dispatch would quantize an fp32 A to fp8 before its MMA, "
                               "and tlower has no fp32-to-fp8 operand conversion (a float A is the fp32 MMA)"),
    "chain.cut.all": (True, ("cut_fp8", "cut_fp32"),
                      "every boundary a dispatch: built wherever each boundary has a cut option"),
    "chain.cut.mixed_format": (False, ("cut_fp8",),
                               "the boundary it needs (fp8 bytes into an fp8 x bfloat consumer) is built, "
                               "but no mixed-format graph has been dispatched (row M6)"),
}

EVIDENCE = {
    "memory": ("gemm_generic stages (test_g17tensorchain); results/g17-register-chain-v1 chain_memory",),
    "cut_fp32": ("any admitted gemm_generic GEMM stores fp32 C (test_g17tensorgeneric); "
                 "a float A is the measured fp32-truncating operand (section 136)",),
    "cut_fp8": ("fp8 quantize-out: results/g17-tensor-lowprec-v1 q8_e4m3, q8_e5m2 and the overflow arms",
                "fp8 MMA from memory: results/g17-tensor-fp8-v1; results/g17-tensor-p5-v1 kloop_fp8_K512",
                "fp8 input nonfinite policy: results/g17-tensor-p5-v1 fp8_nonfinite"),
}


def _form(stage):
    return "%s.%s" % (stage["a"], stage["b"])


def _check_chain(stages):
    if len(stages) < 2:
        raise ValueError("refused: a chain has at least two stages")
    for i, (p, c) in enumerate(zip(stages, stages[1:])):
        if c["M"] != p["M"] or c["K"] != p["N"]:
            raise ValueError("refused: stage %d's A is stage %d's output, so M %d = %d and K %d = N %d"
                             % (i + 1, i, c["M"], p["M"], c["K"], p["N"]))


def _register_receipts(producer, consumer):
    """FEED_TABLE receipts for a mode-A hand-off between these operand forms (any grid)."""
    out = []
    for key, entry in cc.FEED_TABLE.items():
        if (key.role == "A" and key.transpose == "N" and key.producer == _form(producer)
                and key.consumer.split("+")[0] == _form(consumer)):
            out.append(entry["receipt"])
    return out


def _boundary(b, p, c, options, refused):
    """The options and refusals for one boundary and one consumer A type."""
    pep = list(p.get("epilogue") or ())
    mx_consumer = "mx32e8m0" in (c.get("epilogue") or ()) or "mx32" in (c.get("epilogue") or ())
    notes = (["the consumer's MX scale codes come from outside the chain: no in-kernel scale "
              "derivation (block amax to E8M0) exists"] if mx_consumer else [])
    # register and memory: one dispatch, the gemm_generic chain's forms only
    fused_ok = (_form(p) == "half.half" and not pep and c["a"] in ("float", "half") and c["b"] == "half"
                and not c.get("epilogue"))
    receipts = _register_receipts(p, c) if fused_ok else []
    if receipts:
        options.append(dict(boundary=b, kind="register", dispatches=1, producer_last_epilogue=None,
                            consumer_a=c["a"], notes=["only for the FEED_TABLE key's exact grid; any "
                                                      "other grid compiles as the memory bridge"],
                            evidence=tuple(receipts)))
    else:
        refused.append((b, "register", "no cc.FEED_TABLE key for %s -> %s: the register feed is fp32 or "
                                       "op1016-narrowed half, never fp8 or bfloat (P2)" % (_form(p), _form(c))))
    if fused_ok:
        options.append(dict(boundary=b, kind="memory", dispatches=1, producer_last_epilogue=None,
                            consumer_a=c["a"], notes=[], evidence=EVIDENCE["memory"]))
    else:
        refused.append((b, "memory", "one-dispatch chains are a half x half producer into a float or half "
                                     "A with a half B, no epilogue (gemm_generic stages)"))
    # cut_fp32: the producer's plain fp32 C, read as a float A
    if pep and pep[-1] in FP8_OUT:
        refused.append((b, "cut_fp32", "the producer ends with an fp8 quantize-out, so its C is bytes"))
    elif c["a"] != "float" or c["b"] not in ("half", "float"):
        refused.append((b, "cut_fp32", "an fp32 hand-off is a float consumer A with a half or float B "
                                       "(an fp8 B pairs only with fp8 or bfloat)"))
    else:
        options.append(dict(boundary=b, kind="cut_fp32", dispatches=2, producer_last_epilogue=None,
                            consumer_a="float", notes=notes, evidence=EVIDENCE["cut_fp32"]))
    # cut_fp8: the producer quantizes out, the consumer reads fp8 bytes
    if c["a"] not in FP8:
        refused.append((b, "cut_fp8", "the consumer's A is %s, not fp8" % c["a"]))
    elif c["b"] not in FP8 + ("bfloat",):
        refused.append((b, "cut_fp8", "an fp8 A pairs with an fp8 or bfloat B"))
    elif p["a"] == "int8" or p.get("stages"):
        refused.append((b, "cut_fp8", "fp8 quantize-out is a single fp32-accumulating GEMM's last step"))
    else:
        options.append(dict(boundary=b, kind="cut_fp8", dispatches=2, producer_last_epilogue=c["a"],
                            consumer_a=c["a"],
                            notes=notes + ["gemm_generic runs the quantize-out on a grid split (two or "
                                           "more threadgroups)", "RNE without saturation: an overflow "
                                           "becomes NaN (e4m3fn) or infinity (e5m2), which the consumer's "
                                           "MMA then carries (section 25.128)"],
                            evidence=EVIDENCE["cut_fp8"]))


def p11_kinds_for(stages):
    """For a chain, each P11 decision kind -> whether this lowering can realise it for THIS chain now:
    fused needs a one-dispatch option at every boundary, quantize a cut_fp8 at every boundary, all a
    cut option (cut_fp8 or cut_fp32) at every boundary. The unbuilt kinds are False with their reason."""
    r = cut_points(stages)
    per = {}
    for o in r["options"]:
        per.setdefault(o["boundary"], set()).add(o["kind"])
    bounds = [(i, i + 1) for i in range(len(stages) - 1)]
    out = {}
    for kind, (built, via, why) in P11_KINDS.items():
        if kind == "chain.cut.all":
            ok = built and all(per.get(b, set()) & {"cut_fp8", "cut_fp32"} for b in bounds)
        else:
            ok = built and all(per.get(b, set()) & set(via) for b in bounds)
        out[kind] = (ok, why)
    return out


def cut_points(stages):
    """{'options': [...], 'refused': [...]} for every boundary of the chain. Each option is a dict:
    boundary (i, i + 1), kind, dispatches, producer_last_epilogue (a word the producer must end with,
    or None), consumer_a (the consumer's A type), notes, evidence. Each refusal is (boundary, kind,
    why). Nothing here is ranked: selection is P11's."""
    _check_chain(stages)
    options, refused = [], []
    for i, (p, c0) in enumerate(zip(stages, stages[1:])):
        b = (i, i + 1)
        for a in ([c0["a"]] if c0.get("a") else ["float", "half", "fp8e4m3", "fp8e5m2"]):
            _boundary(b, p, dict(c0, a=a), options, refused)
    return dict(options=options, refused=refused)
