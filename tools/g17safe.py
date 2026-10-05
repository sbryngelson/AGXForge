#!/usr/bin/env python3
"""WHICH OPCODES ARE SAFE TO DISPATCH, as a table lookup rather than a judgement.

Two substitutions hung this machine's GPU on 2026-09-05 - MTLCommandBufferErrorDomain Code=2,
"Caused GPU Hang" - and both were instructions nobody had filtered. The reasoning is simple and does
not need to be right about the instruction: an opcode that touches MEMORY can address something the
host never set up, an opcode that BRANCHES can leave the program counter somewhere with no end
instruction, and either is a hang rather than a wrong answer. A hang on this machine has killed
WindowServer once already (ledger/g17-hang-poisons-the-run.toml,
memory agx-mutation-gpu-hang-hazard).

So: refuse by property, from Apple's own two flag words, before anything is authored.

    memory, atomic, texture at +16      the semantic word
    MayLoad, MayStore, Branch, Call,    the generic word at +8
    Return, Terminator, side effects

A probe that WANTS one of these - the threadgroup pair, the indexed store, mov.a - passes it in
`allow`, which is a deliberate act naming the opcode rather than a blanket exemption.

AND A WALKED WITNESS IS NOT RUNNABLE. 717 of 6,718 opcodes have an encoding Apple actually wrote;
the rest have only a repair walk, and an instruction authored from one of those can ignore its own
operands entirely - class 51's forty do exactly that, returning a constant whatever they are given.
`runnable` in isa/g17-contract.jsonl says which is which, and this module refuses the others by
default, because a dispatch on a walked witness measures the walk.
"""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
_GEN = None
_CONTRACT = None

MEMORY_BIT, ATOMIC_BIT, TEXTURE_BIT = 22, 25, 1          # the semantic word at +16
GENERIC = {19: "MayLoad", 20: "MayStore", 10: "Branch", 7: "Call", 5: "Return",
           9: "Terminator", 24: "UnmodeledSideEffects", 11: "IndirectBranch"}


# WHY THIS GATE FAILS CLOSED NOW. The table below is produced by shelling out to `agx3meta`, a
# binary built from tracked source and not checked in. When it is missing, unbuilt, or silent, the
# subprocess returns an empty stdout, the table parses to {}, and every opcode reads back as having
# no generic flags - so `why_unsafe` returned [] and ALLOWED op578, op579 and op450, the loop and
# call opcodes that must never be dispatched. Dispatching Apple's looping kernels with foreign
# buffer contents wedged the GPU and rebooted the machine (2026-09-05); this gate is what stands
# between that and a run.
#
# It was not hypothetical: root's broad gate caught the execution-guard case accepting op578 on a
# tree whose agx3meta had not been built, and rebuilding it made the case pass again. The case was
# right and the gate underneath it was fail-open, so the rebuild fixed the symptom.
#
# An unreadable table is now a REFUSAL REASON rather than an empty answer. `allow=` still
# short-circuits, because that is a caller naming a specific opcode it takes responsibility for,
# which is a different act from the table silently saying nothing about any of them.
_GEN_UNAVAILABLE = None


def _generic():
    global _GEN, _GEN_UNAVAILABLE
    if _GEN is None:
        _GEN = {}
        tool = os.path.join(HERE, "agx3meta")
        try:
            done = subprocess.run([tool, "instrs"], capture_output=True, text=True)
            out = done.stdout
            # A NONZERO EXIT WITH PARTIAL STDOUT IS THE WORST CASE, because the table parses and
            # looks ordinary while silently missing every row after the failure.
            if done.returncode != 0:
                _GEN_UNAVAILABLE = ("%s exited %d; its output is partial and a row missing from a "
                                    "truncated table is indistinguishable from a row with no flags"
                                    % (tool, done.returncode))
        except OSError as exc:
            out = ""
            _GEN_UNAVAILABLE = "%s could not be run (%s)" % (tool, exc)
        for line in out.splitlines():
            if line.startswith("#"): continue
            p = line.split()
            _GEN[int(p[0])] = int(p[5], 16)
        if not _GEN and _GEN_UNAVAILABLE is None:
            _GEN_UNAVAILABLE = ("%s produced no instruction rows - it is missing or unbuilt, and "
                                "an empty table would silently clear every generic flag" % tool)
    return _GEN


def _contract():
    global _CONTRACT
    if _CONTRACT is None:
        _CONTRACT = {}
        path = os.path.join(HERE, "..", "isa", "g17-contract.jsonl")
        if os.path.exists(path):
            for line in open(path):
                line = line.strip()
                if not line or line.startswith("#"): continue
                try: r = json.loads(line)
                except Exception: continue
                if "opcode" in r: _CONTRACT[int(r["opcode"])] = r
    return _CONTRACT


def why_unsafe(opcode, allow=(), require_apple_witness=True):
    """The reasons this opcode should not be dispatched, or [] if there are none."""
    if opcode in allow: return []
    out = []
    table = _generic()
    if _GEN_UNAVAILABLE:
        # FAIL CLOSED. Without the table this function cannot tell a loop opcode from an add, and
        # returning [] would read as "safe to dispatch".
        return ["opcode table unavailable: %s" % _GEN_UNAVAILABLE]
    # A MISSING ROW IS NOT A ROW OF ZEROES, and the first cut of this fix missed that: it guarded
    # the wholly-empty table and left `.get(opcode, 0)` aliasing "this opcode is absent" with "this
    # opcode has no generic flags". Root demonstrated it in a fresh process - a table holding one
    # unrelated row still allowed op578. A default that fills unknowns is the defect, not the empty
    # table that made it visible.
    if opcode not in table:
        return ["opcode %d is absent from the %d-row instruction table, so its flags are unknown"
                % (opcode, len(table))]
    v = table[opcode]
    for bit, name in sorted(GENERIC.items()):
        if (v >> bit) & 1: out.append(name)
    try:
        sys.path.insert(0, HERE)
        import g17auth
        ts = int(g17auth.record(opcode).get("tsflags") or 0)
        for bit, name in ((MEMORY_BIT, "memory"), (ATOMIC_BIT, "atomic"), (TEXTURE_BIT, "texture")):
            if (ts >> bit) & 1: out.append(name)
    except Exception:
        out.append("no authoring record")
    if require_apple_witness:
        r = (_contract().get(opcode) or {}).get("encoding") or {}
        # the contract stores its values as strings, so compare as one
        if r and str(r.get("runnable")).lower() != "true":
            out.append("no Apple witness - authored from a repair walk")
    return out


def check(opcodes, allow=(), require_apple_witness=True):
    """Refuse loudly. Returns the safe list; raises with the reasons if any are not."""
    bad = {o: why_unsafe(o, allow, require_apple_witness) for o in opcodes}
    bad = {o: w for o, w in bad.items() if w}
    if bad:
        raise SystemExit("REFUSING TO DISPATCH:\n" + "\n".join(
            "   op%-7d %s" % (o, ", ".join(w)) for o, w in sorted(bad.items())))
    return list(opcodes)


if __name__ == "__main__":
    ops = [int(x.replace("op", "")) for x in sys.argv[1:]]
    for o in ops:
        w = why_unsafe(o)
        print("op%-7d %s" % (o, ", ".join(w) if w else "safe to dispatch"))


def apple_witness(opcode):
    """The bytes of an instance Apple actually wrote, or None. An opcode authored from anything else
    can ignore its own operands - class 51's forty return a constant whatever they are given."""
    r = (_contract().get(opcode) or {}).get("encoding") or {}
    w = r.get("apple_witness")
    return bytes.fromhex(w) if w else None
