#!/usr/bin/env python3
"""The GPU check section 25.88 requires before the pinned free-bits verdicts can be adopted.

`tools/g17freebits.py` re-derives every invisible-bit verdict from both committed Apple corpora and
the result differs from `isa/g17-free-bits.jsonl` in 672 bits across 108 forms
(`isa/g17-free-bits-known-drift.json`). The drift is pinned, not adopted, because a verdict feeds the
closed-form encoder (`agxforge.g17.encode.encode`): an invisible bit whose verdict is `constant` is
emitted at its value and every other invisible bit is emitted as 0. So adopting the drift changes
EMITTED BYTES wherever a bit's emitted value moves - `undetermined -> constant 1` starts emitting a 1,
`constant 1 -> varies` stops. Most of the 672 bits move between two verdicts that both emit 0 and
change nothing the encoder writes.

WHAT THIS MEASURES. For every form whose emitted bytes change, a PAIR of programs identical except
at exactly those bits: arm A carries the value the encoder emits today, arm B the value adoption
would emit. The bits are invisible to Apple's decoder, so the prediction for every pair is the SAME
OUTPUT on every case. A pair whose outputs differ means the bit is not free on hardware, and
adopting the verdict would miscompile that form - which is the finding this exists to catch.

HOW EACH ARM IS BUILT. The template is the form's most frequent encoding in Apple's corpora (so
arm B at those bits is, in most cases, exactly what Apple wrote). `spike/accel/re/oracle.py` builds
it with `author: "assembler"`, which writes the allocator's registers into the template AT ITS OWN
WIDTH and keeps every other bit Apple wrote; the scaffold loads each case into registers, runs the
one instruction, stores its result to the output buffer, and writes a canary last. Before anything
is dispatched both programs are built here and their emitted code is compared: the whole programs
must be the same length and differ ONLY at the adopted bit positions of the instruction under test,
and both copies of that instruction must decode, by Apple's decoder, to the same (opcode, length).
A pair that fails any of that is REFUSED with its reason, never dispatched.

WHAT IS NOT DISPATCHED. Forms whose opcode may load, store or is atomic are refused by
`tools/g17safe.py` (a store with a foreign address is how this GPU was hung); forms whose register
fields the assembler cannot write at this width, or whose destination cannot be stored, fail the
build and are listed with the reason. They stay unadopted: this check cannot clear them.

CONTROLS.
    ctl.op10279.mandatory   op10279 returns src + 4 (4100, 4101, 4356, 8196). The batch is void
                            without it.
    ctl.null.op10279        a pair of byte-IDENTICAL programs: must agree (harness determinism).
    ctl.positive.op612      op612 at lo = 16 vs lo = 0, w = 8: the arms differ in ONE operand bit
                            (operand 4, value bit 4), and the outputs MUST differ (7, 0, 15, 8 vs
                            0, 0, 0, 0; w612.lo16 already measured the first). If this pair agrees
                            the harness cannot see a one-bit difference and the batch is void.
    ctl.knowngood.*         a form with NO drift, flipping an invisible bit Apple's own instances
                            VARY: a bit known free, so the pair must agree.

    python3 tools/g17adoptcheck.py              list the pairs and predictions (pure; no GPU)
    python3 tools/g17adoptcheck.py --plan       re-derive isa/g17-adoption-check-plan.json from the
                                                pinned drift and both corpora (~2 min, CPU)
    python3 tools/g17adoptcheck.py --check-plan fail if the committed plan is not what they give
    python3 tools/g17adoptcheck.py --build      build every pair offline and check it; no dispatch
    python3 tools/g17adoptcheck.py --run        dispatch through spike/accel/re/oracle.py (one
                                                process per run, 3 runs, canary, fault stop) and
                                                write isa/g17-adoption-check-results.json
"""
import argparse, collections, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "spike", "accel", "re"))

ISA = os.path.join(ROOT, "isa")
DRIFT = os.path.join(ISA, "g17-free-bits-known-drift.json")
FREE = os.path.join(ISA, "g17-free-bits.jsonl")
PLAN = os.path.join(ISA, "g17-adoption-check-plan.json")
OUT = os.path.join(ISA, "g17-adoption-check-results.json")
ORACLE = os.path.join(ROOT, "spike", "accel", "re", "oracle.py")
SCRATCH = os.path.join(ROOT, "results", "g17-adoption-check")
M32 = 0xFFFFFFFF

# ADOPTED AFTER THE RUN (isa/g17-adoption-check-results.json): the 13 pairs whose arms agreed on an
# output that varies across cases, through `g17freebits.py --write --accept-forms`. NOT adopted:
# op11482/14 (the arms DIFFER - the bit is live), the 6 whose output was one constant
# (op5072/10, op9748/10, op11063/14, op11322/10, op11392/10, op11392/14 - the check could not
# see the bit), and the 11 refused offline. The plan file is FROZEN at the batch it describes, so
# `--check-plan` now reports it stale by exactly these forms; that is the record, not a defect.
ADOPTED = [(727, 14), (3293, 14), (9749, 10), (11456, 14), (11483, 10), (11487, 10), (11655, 12),
           (14393, 14), (16817, 12), (17015, 14), (17774, 10), (17775, 10), (17782, 10)]
# THE 252 DRIFT BITS OVER 77 FORMS THAT CHANGE NO EMITTED BYTE (0 -> 0 verdict changes, e.g.
# undetermined -> constant 0). No GPU check can clear them - both arms would be the same program - so
# adopting them was a decision, not a measurement, and it was taken: they stay pinned, unadopted.
NO_BYTE_DECISION = ("no emitted-byte change: no observable consequence (they change no emitted byte), "
                    "deliberately not adopted - Spencer, 2026-09-24")

NOT_ADOPTED = {
    (11482, 14): "LIVE ON HARDWARE: case 4 returns 0 with bit 11.6 = 1 and 1 with it = 0; adopting "
                 "'varies' would emit 0 and change the result",
    (5072, 10): "uninformative: both arms returned 0 on every case",
    (9748, 10): "uninformative: both arms returned 0 on every case",
    (11063, 14): "uninformative: both arms returned 0 on every case",
    (11322, 10): "uninformative: both arms returned 0 on every case",
    (11392, 10): "uninformative: both arms returned 0 on every case",
    (11392, 14): "uninformative: both arms returned 255 on every case",
    (445, 10): "refused offline: safety gate (MayStore)",
    (9328, 10): "refused offline: safety gate (MayStore)",
    (10022, 12): "refused offline: safety gate (MayLoad/MayStore/atomic)",
    (10090, 10): "refused offline: safety gate (MayLoad/MayStore/atomic)",
    (11769, 12): "refused offline: safety gate (MayLoad/MayStore/atomic)",
    (9700, 8): "refused offline: no register field authorable at this width",
    (9710, 8): "refused offline: no register field authorable at this width",
    (11375, 8): "refused offline: no register field authorable at this width",
    (9808, 8): "refused offline: Apple's decoder reads back other registers than written",
    (11393, 10): "refused offline: Apple's decoder reads back other registers than written",
    (13511, 10): "refused offline: no destination register, so no stored result",
}

# The inputs. Four cases, each source gets a different word so no two sources are equal; the low
# halves are ordinary halves (1.5, 3.0, 5.0 as fp16 bits 0x3E00/0x4200/0x4500, and -1.0 = 0xBC00) so
# a GPR16 source reads a real number, and the high halves are ordinary floats' high halves.
WORDS = [0x3F803E00, 0x40004200, 0x40A04500, 0xC000BC00, 0x00070005]
NCASES = 4


def emitted(verdict):
    """What the closed-form encoder writes for an invisible bit under this verdict, with no hints.

    agxforge.g17.encode.encode: `constant` emits its value; any other verdict emits 0."""
    return 1 if verdict and verdict[0] == "constant" and verdict[1] else 0


def drift_forms(drift=None):
    """{(op, length): [(bit "b.i", emitted today, emitted after adoption)]} where they differ."""
    if drift is None:
        drift = json.load(open(DRIFT))["drift"]
    out = collections.defaultdict(list)
    for op, ln, bit, before, after in drift:
        a, b = emitted(before), emitted(after)
        if a != b:
            out[(int(op), int(ln))].append((bit, a, b))
    return {k: sorted(v) for k, v in sorted(out.items())}


def cases(nsrc):
    return [[WORDS[(i + j) % len(WORDS)] for j in range(nsrc)] for i in range(NCASES)]


def range4(src, lo, w):
    return sum(1 << j for j in range(4) if lo <= ((src + j) & M32) < lo + w)


def _set_bits(hexbytes, bits):
    u = bytearray(bytes.fromhex(hexbytes))
    for key, v in bits:
        b, i = (int(x) for x in key.split("."))
        u[b] = (u[b] & ~(1 << i)) | ((v & 1) << i)
    return u.hex()


# ------------------------------------------------------------------------------------------------
# the plan: the only step that reads the corpora
def build_plan():
    """Re-derive the plan from the pinned drift and both corpora. Deterministic."""
    import g17formspecgen as G
    import g17safe, g17auth
    enc, msg = G.verified_corpus_encodings()
    if enc is None:
        raise SystemExit("REFUSED: " + msg)
    forms = drift_forms()
    rows = []
    for (op, ln), bits in forms.items():
        encs = enc.get((op, ln)) or {}
        ranked = sorted(encs.items(), key=lambda kv: (-kv[1], kv[0]))
        _d, srcs = g17auth.register_operands(op)
        rows.append(dict(op=op, length=ln, bits=[list(b) for b in bits],
                         template=ranked[0][0] if ranked else None,
                         apple_instances=sum(encs.values()), distinct_encodings=len(encs),
                         template_bits=({k: (bytes.fromhex(ranked[0][0])[int(k.split(".")[0])]
                                             >> int(k.split(".")[1])) & 1 for k, _a, _b in bits}
                                        if ranked else {}),
                         nsrc=len(srcs),
                         fieldmap_width=(g17auth.length(op) if op in g17auth.load() else None),
                         unsafe=g17safe.why_unsafe(op, require_apple_witness=False)))
    # THE KNOWN-GOOD CONTROL: a form with NO drift whose committed free-bits row has a bit Apple
    # VARIES, on an opcode the safety gate passes, with a destination and at least one register
    # source. Chosen by rule (most Apple instances first), not by hand.
    drifted = {(op, ln) for op, ln, *_ in json.load(open(DRIFT))["drift"]}
    free = {}
    for line in open(FREE):
        d = json.loads(line)
        free[(d["opcode"], d["length"])] = d.get("bits") or {}
    cands = []
    for (op, ln), bits in free.items():
        if (op, ln) in drifted or op not in g17auth.load():
            continue
        varies = sorted(k for k, v in bits.items()
                        if isinstance(v, dict) and v.get("verdict") == "varies")
        encs = enc.get((op, ln)) or {}
        if not varies or not encs:
            continue
        dsts, srcs = g17auth.register_operands(op)
        if not dsts or not srcs or g17safe.why_unsafe(op, require_apple_witness=False):
            continue
        cands.append((-sum(encs.values()), op, ln, varies[0], encs))
    cands.sort(key=lambda c: c[:4])
    known = []
    for neg, op, ln, bit, encs in cands[:12]:
        tmpl = sorted(encs.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        _d, srcs = g17auth.register_operands(op)
        known.append(dict(op=op, length=ln, bit=bit, template=tmpl, apple_instances=-neg,
                          nsrc=len(srcs), fieldmap_width=g17auth.length(op)))
    return dict(means=("Pairs for the free-bits adoption check (tools/g17adoptcheck.py). `bits` "
                       "is [bit, emitted today, emitted after adoption] for every bit of the form "
                       "whose EMITTED value adoption changes; `template` is the form's most frequent "
                       "Apple encoding over both corpora. Derived from "
                       "isa/g17-free-bits-known-drift.json."),
                corpus=msg, drift_bits=len(json.load(open(DRIFT))["drift"]),
                drift_forms=len(drifted), forms=rows, knowngood_candidates=known)


def load_plan():
    return json.load(open(PLAN))


# ------------------------------------------------------------------------------------------------
# the arms: pure over the committed plan
def arms(plan=None):
    """Every pair with its prediction, in dispatch order. Pure: no compiler, no device."""
    plan = plan or load_plan()
    out = []
    out.append(dict(id="ctl.op10279.mandatory", kind="single", control=True, op=10279,
                    records=[dict(id="ctl.op10279.mandatory", op=10279, author="fieldmap",
                                  witness_required=False, cases=[[4096], [4097], [4352], [8192]],
                                  expect=[4100, 4101, 4356, 8196])],
                    predict="values 4100, 4101, 4356, 8196 (src + 4)"))
    c10279 = [[4096], [4097], [4352], [8192]]
    out.append(dict(id="ctl.null.op10279", kind="pair", control=True, op=10279, predict="agree",
                    adopted_bits=[],
                    records=[dict(id="ctl.null.op10279.%s" % s, op=10279, author="fieldmap",
                                  witness_required=False, cases=c10279,
                                  expect=[4100, 4101, 4356, 8196]) for s in "AB"]))
    c612 = [[21], [24], [16], [13]]
    out.append(dict(id="ctl.positive.op612", kind="pair", control=True, op=612, predict="differ",
                    operand_bit=dict(operand=4, value_bit=4),
                    records=[dict(id="ctl.positive.op612.%s" % s, op=612, author="fieldmap",
                                  witness_required=False, imms={"4": lo, "5": 8}, cases=c612,
                                  expect=[range4(c[0], lo, 8) for c in c612])
                             for s, lo in (("A", 16), ("B", 0))]))
    for kg in plan.get("knowngood_candidates") or []:
        t = kg["template"]
        b, i = (int(x) for x in kg["bit"].split("."))
        v = (bytes.fromhex(t)[b] >> i) & 1
        pid = "ctl.knowngood.op%d.l%d" % (kg["op"], kg["length"])
        out.append(dict(id=pid, kind="pair", control=True, op=kg["op"], length=kg["length"],
                        predict="agree", adopted_bits=[kg["bit"]], candidate=True,
                        why="bit %s VARIES across Apple's %d instances of this form: known free"
                            % (kg["bit"], kg["apple_instances"]),
                        routes=_routes(pid, kg, [[(kg["bit"], v)], [(kg["bit"], 1 - v)]])))
    for f in plan["forms"]:
        pid = "adopt.op%d.l%d" % (f["op"], f["length"])
        row = dict(id=pid, kind="pair", control=False, op=f["op"], length=f["length"],
                   predict="agree", adopted_bits=[b[0] for b in f["bits"]], bits=f["bits"],
                   apple_instances=f["apple_instances"])
        if f.get("unsafe"):
            row.update(refused="the safety gate refuses op%d: %s - it may touch memory, so it is "
                               "not dispatched and its verdicts stay unadopted"
                               % (f["op"], "/".join(f["unsafe"])), records=[])
        elif not f.get("template"):
            row.update(refused="no Apple encoding of this form in either corpus", records=[])
        else:
            row["routes"] = _routes(pid, f, [[(b[0], b[1]) for b in f["bits"]],
                                             [(b[0], b[2]) for b in f["bits"]]])
        out.append(row)
    return out


def _routes(pid, f, arm_bits):
    """The ways to author this pair, in preference order; --build dispatches the first that builds.

    `assembler` writes registers at the form's OWN width from maps keyed (op, length, operand), so
    it is tried first. `fieldmap` authors through isa/g17-authoring.jsonl's map, which describes ONE
    width per opcode, so it is offered only where that width is this form's.
    """
    t = f["template"]
    authors = ["assembler"] + (["fieldmap"] if f.get("fieldmap_width") == f["length"] else [])
    return [dict(author=au, records=[dict(id="%s.%s" % (pid, s), op=f["op"], length=f["length"],
                                          author=au, witness_required=False,
                                          bytes=_set_bits(t, bits), cases=cases(f["nsrc"]))
                                     for s, bits in zip("AB", arm_bits)])
            for au in authors]


def adopted_mask(length, bits):
    m = bytearray(length)
    for key in bits:
        b, i = (int(x) for x in key.split("."))
        m[b] |= 1 << i
    return bytes(m)


def pair_diff(code_a, code_b, enc_a, enc_b, length, bits, where_a):
    """Why two built programs are NOT a clean pair, or None.

    Clean means: same length; every differing bit of the whole program lies inside the instruction
    under test and at an adopted position; and the instruction differs by exactly the adopted mask.
    Pure over bytes, so the test exercises it without building anything.
    """
    if len(code_a) != len(code_b):
        return "the two programs are %d and %d bytes" % (len(code_a), len(code_b))
    mask = adopted_mask(length, bits)
    if len(enc_a) != len(enc_b) or not enc_a:
        return "the programs hold %d and %d instances of the form" % (len(enc_a), len(enc_b))
    for ea, eb in zip(enc_a, enc_b):
        x = bytes(p ^ q for p, q in zip(bytes.fromhex(ea), bytes.fromhex(eb)))
        if x != mask:
            return "the instruction differs by %s, not the adopted mask %s" % (x.hex(), mask.hex())
    allowed = set()
    for at in where_a:
        for k in range(length):
            for i in range(8):
                if mask[k] >> i & 1:
                    allowed.add((at + k, i))
    for k, (p, q) in enumerate(zip(code_a, code_b)):
        d = p ^ q
        for i in range(8):
            if d >> i & 1 and (k, i) not in allowed:
                return "the programs differ at byte %d bit %d, outside the adopted bits" % (k, i)
    return None


def build_pair(arm):
    """Build a pair offline through its first route that yields a clean pair.

    Returns `ready`, the `records` to dispatch, the route's `author`, and `why` (every route's
    refusal when none is clean)."""
    if not arm.get("routes"):
        return _build_route(arm, arm["records"])
    whys = []
    for route in arm["routes"]:
        r = _build_route(arm, route["records"])
        if r["ready"]:
            return dict(r, author=route["author"], records=route["records"])
        whys.append("%s: %s" % (route["author"], r["why"]))
    return dict(id=arm["id"], ready=False, why=" | ".join(whys))


def _build_route(arm, records):
    import g17oracle
    from agxforge.g17 import ref as g17ref
    out = dict(id=arm["id"])
    built = []
    for rec in records:
        rec = json.loads(json.dumps(rec))
        why = g17oracle.refusal(rec)
        if why:
            return dict(out, ready=False, why="%s refused: %s" % (rec["id"], why))
        try:
            P, check = g17oracle.program(rec)
        except BaseException as e:           # SystemExit included: the oracle raises it
            return dict(out, ready=False, why="%s did not build: %s: %s"
                                              % (rec["id"], type(e).__name__, e))
        if not check["ok"]:
            return dict(out, ready=False, why="%s: %s" % (rec["id"], (check.get("operands") or {})
                                                            .get("why") or check))
        code = P.text
        where = [at for at, ln, op in g17ref.walk(code, g17oracle.ENTRY)
                 if op == rec["op"] and (rec.get("length") is None or ln == rec["length"])]
        forms = sorted({(op, ln) for at, ln, op in g17ref.walk(code, g17oracle.ENTRY)
                        if op == rec["op"]})
        built.append(dict(code=code, enc=check["encoded"], where=where, forms=forms))
    if arm["kind"] == "single":
        return dict(out, ready=True, why=None, records=records)
    a, b = built
    if a["forms"] != b["forms"] or len(a["forms"]) != 1:
        return dict(out, ready=False, why="the arms decode to %s and %s" % (a["forms"], b["forms"]))
    length = a["forms"][0][1]
    if arm["id"] == "ctl.positive.op612":
        x = [bytes(p ^ q for p, q in zip(bytes.fromhex(ea), bytes.fromhex(eb)))
             for ea, eb in zip(a["enc"], b["enc"])]
        nbits = sum(bin(c).count("1") for c in x[0]) if x else 0
        if not x or any(v != x[0] for v in x) or nbits != 1:
            return dict(out, ready=False, why="the positive control's arms differ in %d bits, "
                                              "not one operand bit" % nbits)
        return dict(out, ready=True, why=None, form=a["forms"][0], differing_bits=1,
                    records=records)
    why = pair_diff(a["code"], b["code"], a["enc"], b["enc"], length, arm["adopted_bits"],
                    a["where"])
    return dict(out, ready=why is None, why=why, form=a["forms"][0], records=records,
                encoded=dict(A=a["enc"][0], B=b["enc"][0]))


def build_all(verbose=True):
    rows = []
    for arm in arms():
        if arm.get("refused"):
            r = dict(id=arm["id"], ready=False, why=arm["refused"], gated=True)
        else:
            r = build_pair(arm)
        rows.append(r)
        if verbose:
            print("%-28s %-7s %-9s %s" % (r["id"], "READY" if r["ready"] else "refused",
                                          r.get("author") or "", r.get("why") or r.get("encoded") or ""))
    return rows


# ------------------------------------------------------------------------------------------------
# dispatch
def score(arm, results):
    """The verdict for one arm, from the oracle's per-record results. Pure."""
    got = [results.get(r["id"]) or {} for r in arm["records"]]
    if any(g.get("status") != "ok" for g in got):
        return dict(verdict="no-result", statuses=[g.get("status") for g in got],
                    why=[g.get("why") for g in got])
    vals = [g["values"] for g in got]
    if arm["kind"] == "single":
        ok = vals[0] == arm["records"][0]["expect"]
        return dict(verdict="as-predicted" if ok else "CONTROL-FAILED", values=vals[0])
    same = vals[0] == vals[1]
    predicted = (arm["predict"] == "agree") == same
    # AN AGREEMENT IS ONLY AS GOOD AS THE OBSERVABLE. Two arms returning one constant on every
    # case agree whatever the bit does, so a pair whose output does not vary across its cases is
    # reported as UNINFORMATIVE rather than cleared.
    varies = len(set(vals[0])) > 1 or len(set(vals[1])) > 1
    verdict = ("CONTROL-FAILED" if arm["control"] and not predicted else
               "ADOPTION-WOULD-MISCOMPILE" if not predicted else
               "agree-but-uninformative" if same and not varies and not arm["control"] else
               "as-predicted")
    return dict(values=dict(A=vals[0], B=vals[1]), outputs_agree=same,
                output_varies_across_cases=varies, verdict=verdict)


def run(replace):
    if os.path.exists(OUT) and not replace:
        print("REFUSING: %s holds retained execution results; pass --replace-results" % OUT)
        return 2
    built = build_all(verbose=False)
    ready = {r["id"]: r for r in built if r["ready"]}
    # the records dispatched are the ones the build CHECKED, route included
    todo = [dict(a, records=ready[a["id"]]["records"], author=ready[a["id"]].get("author"))
            for a in arms() if a["id"] in ready]
    missing = [c for c in ("ctl.op10279.mandatory", "ctl.null.op10279", "ctl.positive.op612")
               if c not in ready]
    if missing:
        print("REFUSING: control(s) %s do not build; the batch would be void" % missing)
        return 2
    # ONE known-good control is dispatched: the first candidate that built clean.
    kg = [a for a in todo if a["id"].startswith("ctl.knowngood.")]
    todo = [a for a in todo if not a["id"].startswith("ctl.knowngood.")] + kg[:1]
    order = sorted(todo, key=lambda a: (not a["control"], a["id"] != "ctl.op10279.mandatory"))
    os.makedirs(SCRATCH, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    plan = os.path.join(SCRATCH, "plan-%s.json" % stamp)
    res = os.path.join(SCRATCH, "oracle-%s.json" % stamp)
    json.dump([r for a in order for r in a["records"]], open(plan, "w"), indent=1)
    subprocess.run([sys.executable, ORACLE, plan, "--out", res], cwd=ROOT)
    got = {r["id"]: r for r in (json.load(open(res)) if os.path.exists(res) else [])}
    rows = []
    for a in order:
        rows.append(dict(id=a["id"], control=a["control"], op=a["op"], length=a.get("length"),
                         author=a.get("author"),
                         predict=a["predict"], adopted_bits=a.get("adopted_bits"),
                         bits=a.get("bits"), records=a["records"],
                         oracle=[got.get(r["id"]) for r in a["records"]], **score(a, got)))
    controls_ok = all(r["verdict"] == "as-predicted" for r in rows if r["control"])
    doc = dict(means=("The adoption check section 25.88 requires. Each non-control pair differs only "
                      "at the invisible bits whose emitted value adoption would change; the "
                      "prediction, committed before dispatch, is that both arms return the same "
                      "values. `ADOPTION-WOULD-MISCOMPILE` marks a pair that did not."),
               controls_ok=controls_ok, dispatched_pairs=len(rows),
               not_dispatched=[dict(id=r["id"], why=r["why"]) for r in built if not r["ready"]],
               rows=rows)
    json.dump(doc, open(OUT, "w"), indent=1)
    bad = [r["id"] for r in rows if r["verdict"] == "ADOPTION-WOULD-MISCOMPILE"]
    print("wrote %s: %d arms, controls %s, %d would miscompile %s"
          % (os.path.relpath(OUT, ROOT), len(rows), "ok" if controls_ok else "NOT ok - batch void",
             len(bad), bad))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--check-plan", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--replace-results", action="store_true")
    a = ap.parse_args(argv)
    if a.plan or a.check_plan:
        doc = build_plan()
        text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
        if a.check_plan:
            same = os.path.exists(PLAN) and open(PLAN).read() == text
            print("plan is current" if same else "STALE: re-run --plan")
            return 0 if same else 1
        open(PLAN, "w").write(text)
        print("wrote %s: %d forms whose emitted bytes change, of %d drifting"
              % (os.path.relpath(PLAN, ROOT), len(doc["forms"]), doc["drift_forms"]))
        return 0
    if a.build:
        rows = build_all()
        pairs = [r for r in rows if not r["id"].startswith("ctl.")]
        print("\n%d adoption pairs: %d dispatch-ready, %d refused; controls ready: %s"
              % (len(pairs), sum(r["ready"] for r in pairs), sum(not r["ready"] for r in pairs),
                 [r["id"] for r in rows if r["id"].startswith("ctl.") and r["ready"]]))
        return 0
    if a.run:
        return run(a.replace_results)
    for x in arms():
        print("%-28s %-6s predict %-6s bits %s %s" % (x["id"], x["kind"], x["predict"],
                                                     x.get("adopted_bits"), x.get("refused") or ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
