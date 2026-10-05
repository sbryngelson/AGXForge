#!/usr/bin/env python3
"""DOES EACH DISPATCHED RECORD STILL BUILD THE BYTES ITS RESULT MEASURED?

Every isa/g17-execution-*.json record whose result is `ok` retains the instruction bytes that ran
(`decoded.encoded`). g17oracle.program() rebuilds the record from its plan today. The allocation
pool's comment says "every previously dispatched record allocates exactly as it did"; Piece A
measured on 2026-09-22 that 537 of 1,688 do not, on a clean c5d3945f as well as on their branch.
A result stays a valid measurement of what RAN - the retained bytes are the evidence - but a plan
that no longer reproduces them means re-running it measures something else, and nothing said so.

This classifies every difference by what changed, operand by operand, through the disassembler:

    reg                 register numbers only: the allocator chose differently, the function is the same
    imm / expr (+reg)   an immediate or an address expression changed value - a lifetime carrier or a
                        modifier the liveness pass now writes; the instruction is NOT the one measured
    operand kinds       the operand list changed shape
    opcode or length    a different instruction
    instance count      a different number of instructions under test

    python3 tools/g17rebuilddrift.py            rebuild every record and print the classes
    python3 tools/g17rebuilddrift.py --write    and write isa/g17-execution-rebuild-drift.json
    python3 tools/g17rebuilddrift.py --check    re-derive the summary from the written rows; refuse
                                                any SEMANTIC class (opcode, length, operand kind, count)

WHY --check DOES NOT REBUILD. A rebuild moves whenever the allocator or liveness pass improves, and a
gate that went red on that would fire hardest when the compiler got better. The written rows are the
measurement; the check keeps the summary honest to them and refuses the classes that would mean a
retained result now names the wrong instruction. Nothing here is dispatched.
"""
import argparse
import ast
import collections
import glob
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEST = os.path.join(ROOT, "isa", "g17-execution-rebuild-drift.json")
SEMANTIC = ("opcode or length", "operand kinds", "instance count differs")


def plan_digest(rec):
    """The identity of a PLAN record, by content: ids repeat across batches (u1272 is in three)."""
    import hashlib
    return hashlib.sha256(json.dumps(rec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# WHAT A SAMPLE OF THE DRIFT MEASURED, 2026-09-23. isa/g17-execution-redispatch re-ran 20 drifted
# records on the GPU - 14 register-only, 6 with a changed lifetime or destination modifier - and all 20
# returned their retained values word for word (test_g17redispatch). The refusal below stays because
# the bytes differ and a sample is not the population, but the question "did the drift change what was
# measured" has a measured answer for those 20: no.
# THE BLIND SPOT, NAMED. This compares the instruction-under-test bytes, and a retained result does
# not record which buffers its image bound - so a record whose instruction is identical but whose
# binding moved reads `identical` here. Piece A found one on 2026-09-22: u12682.5 now loads buffer
# A's pattern (0xEE00EE00) where its retained run loaded B's (0xC0FFEE00), after the oracle began
# deriving bindings from the compiled function. Such records are refused by id+batch here, with the
# evidence, until results record their bindings.
KNOWN_BINDING_DRIFT = {
    ("g17-execution-unsafe.json", "u12682.5"): "binding moved: loads buffer A (0xEE00EE00) where the retained run loaded B (0xC0FFEE00) - Piece A, 2026-09-22",
}


def not_reproducible(path=DEST):
    """{plan digest: row} for every record whose rebuild changes an operand VALUE - re-dispatching
    one today would run a different instruction than the one its result measured."""
    try:
        with open(path) as fh:
            rows = json.load(fh)["rows"]
    except (OSError, ValueError, KeyError):
        return {}
    out = {r["plan_sha256"]: r for r in rows
           if r.get("plan_sha256") and r["cls"] not in SEMANTIC + ("identical", "reg", "unbuildable")}
    for r in rows:
        why = KNOWN_BINDING_DRIFT.get((r["batch"], r["id"]))
        if why and r.get("plan_sha256"):
            out[r["plan_sha256"]] = dict(r, cls=why, rebuilt="the same instruction bytes", retained="a different binding")
    return out


def _dis(hexes):
    sys.path.insert(0, HERE)
    import g17fields as FL
    import g17ref
    g17ref.binary()
    blob = b"".join(bytes.fromhex(h) for h in hexes)
    with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
        fh.write(blob)
        fh.flush()
        r = subprocess.run([FL.DIS, fh.name, "0", str(len(blob)), "--pc", "0"], capture_output=True, text=True)
    return [line.split()[1:] for line in r.stdout.splitlines() if line.strip()]


def classify_pair(old, new, dis=_dis):
    """The class of one record's difference, from the retained and the rebuilt instruction bytes."""
    if new == old:
        return "identical"
    if len(new) != len(old):
        return "instance count differs"
    kinds = set()
    for x, y in zip(dis(old), dis(new)):
        if x == y:
            continue
        if x[:2] != y[:2]:
            kinds.add("opcode or length")
            continue
        ox, oy = x[2:], y[2:]
        if [t.split(":")[0] for t in ox] != [t.split(":")[0] for t in oy]:
            kinds.add("operand kinds")
            continue
        for p, q in zip(ox, oy):
            if p != q:
                kinds.add("reg" if p.startswith("reg:") else p.split(":")[0])
    return "+".join(sorted(kinds)) or "bytes differ, decode identical"


def rebuild():
    sys.path.insert(0, HERE)
    import g17oracle as O
    rows = []
    for plan in sorted(glob.glob(os.path.join(ROOT, "isa", "g17-execution-*.json"))):
        if plan.endswith("-results.json"):
            continue
        res = plan[:-5] + "-results.json"
        if not os.path.exists(res):
            continue
        try:
            records = json.load(open(plan))
            results = {r["id"]: r for r in json.load(open(res)) if isinstance(r, dict) and "id" in r}
        except (ValueError, TypeError):
            continue
        if not isinstance(records, list):
            continue
        for rec in records:
            r = results.get(rec.get("id"))
            if not r or r.get("status") != "ok" or not r.get("decoded"):
                continue
            try:
                decoded = ast.literal_eval(r["decoded"]) if isinstance(r["decoded"], str) else r["decoded"]
            except (ValueError, SyntaxError):
                continue
            old = decoded.get("encoded")
            if not old:
                continue
            row = dict(batch=os.path.basename(plan), id=rec["id"], op=rec.get("op"), plan_sha256=plan_digest(rec))
            try:
                _program, chk = O.program(rec)
                new = chk.get("encoded") or []
                row.update(cls=classify_pair(old, new), retained=old[0], rebuilt=new[0] if new else None)
            except (Exception, SystemExit) as why:
                row.update(cls="unbuildable", why=str(why).splitlines()[0][:160])
            rows.append(row)
    return rows


class Refused(ValueError):
    pass


def summarise(rows):
    counts = collections.Counter(r["cls"] for r in rows)
    semantic = sorted("%s/%s" % (r["batch"], r["id"]) for r in rows if r["cls"] in SEMANTIC)
    value = sorted("%s/%s" % (r["batch"], r["id"]) for r in rows
                   if r["cls"] not in SEMANTIC and r["cls"] not in ("identical", "reg", "unbuildable"))
    return dict(examined=len(rows), classes=dict(sorted(counts.items())),
                records_whose_rebuild_is_a_different_instruction=semantic,
                records_whose_rebuild_changes_an_operand_value=value)


def check(path=DEST):
    with open(path) as fh:
        doc = json.load(fh)
    if summarise(doc["rows"]) != doc["summary"]:
        raise Refused("the written summary is not what its rows give")
    if doc["summary"]["records_whose_rebuild_is_a_different_instruction"]:
        raise Refused("%d retained records rebuild to a DIFFERENT instruction: %s"
                      % (len(doc["summary"]["records_whose_rebuild_is_a_different_instruction"]),
                         doc["summary"]["records_whose_rebuild_is_a_different_instruction"][:5]))
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.check:
            doc = check()
            print("rebuild drift valid: %d examined, %s" % (doc["summary"]["examined"], doc["summary"]["classes"]))
            return 0
        rows = rebuild()
        doc = dict(generated_by="tools/g17rebuilddrift.py --write", gpu_dispatched=False,
                   scope="every ok record of isa/g17-execution-*.json that retains decoded.encoded, rebuilt by g17oracle.program() on this tree",
                   limitations="compares the instruction-under-test bytes only, not the surrounding program; a rebuild is a function of the compiler at the commit it ran on",
                   summary=summarise(rows), rows=rows)
        print(json.dumps(doc["summary"]["classes"]))
        if args.write:
            with open(DEST, "w") as fh:
                json.dump(doc, fh, indent=1, sort_keys=True)
                fh.write("\n")
            print("wrote", os.path.relpath(DEST, ROOT))
        check_doc = doc
        if check_doc["summary"]["records_whose_rebuild_is_a_different_instruction"]:
            print("REFUSED: records rebuild to a different instruction")
            return 2
    except Refused as why:
        print("REFUSED: %s" % why)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
