#!/usr/bin/env python3
"""The narrow tex2d-read class: every fact determined and now structurally emittable.

    python3 tools/g17texnarrow.py build results/g17-texture-narrow-class-v1
    python3 tools/g17texnarrow.py check results/g17-texture-narrow-class-v1

The 2026-09-12f dispatch asks to take one retained source-owned `tex2d-read` contract through the
common image path, adding only the measured class, with a negative control for an omitted or
conflicting fact and an explicit refusal for anything unstated.

THE REFUSAL LIST IS EMPTY FOR THIS CONTRACT - the first texture contract for which it is. lane31
refuses eight facts; this one refuses none, and the difference is the program rather than any
loosening of the author:

    slot 27          DERIVED  [48] - the uniformly read resources with 44 removed
    slot 38, 42      DERIVED  8 and 8
    kind-9 record    DERIVED  absent, by the no-pool-content rule
    kind-6 split     DERIVED  field2 + field3 = slot 1
    slot 10          DERIVED  empty - NEW, see below
    slot 32          DERIVED  absent - NEW, see below
    pool form        STATED   the contract says which empty pool it means

Only the last is a statement rather than a derivation, and it is the decision that has been on the
boundary from the start. Everything else the author works out.

TWO NEW DERIVATIONS, BOTH FROM RULES MEASURED THIS SESSION:

  * slot 10 is empty exactly when no dynamic threadgroup buffer is declared. Measured by compiling
    both arms of both axes - a texture read and a device-buffer read, each with and without a
    dynamic threadgroup - and only the threadgroup arms carry a record. A contract that DOES
    declare one still refuses, because the rule for what a non-empty vector holds is not measured.

  * slot 32 is absent when main's instruction count is at or below the measured boundary.
    `g17mdgen.slot32_for` puts no slot 32 below 31, and four freshly compiled kernels confirm the
    absence arm at main counts of 6 to 9. A contract above the boundary still refuses, because the
    VALUE is not predicted by any readable fact.

AND I NEARLY PUBLISHED THAT LAW AS REFUTED. Counting the WHOLE __text rather than main gives 34 to
37 on those same four kernels, which predicts 1 where the sections carry none - four clean
counterexamples to a law used by a shipped author. The law is right; the count was the prologue plus
main. The same wrong input on the four cooperative witnesses happens to agree, because 79 against 51
and 397 against 369 do not cross a threshold - so the defect was invisible exactly where it had been
exercised.

A DEFECT THIS FOUND, which an empty refusal list would otherwise have hidden: `placed["slot 13"]`
was built from the contract's raw `constant_pool`, which is `[]` for BOTH empty states, so stating
`eight_byte_zero_vector` resolved the derived record facts and left the vector this side would emit
at zero length. The plan read as fully determined while the bytes it planned differed from the
witness in the eight bytes the statement was about. Found by checking a placed VALUE rather than
trusting an empty list.

WHAT THIS DOES NOT CLAIM. The structural emitter writes this one measured class; it does not make
other texture classes authorable. Loader eligibility is unchanged and unmet - no slot set carrying
38 or 42 has a dispatched receipt. No GPU dispatch or hardware claim is made here.
"""
from __future__ import annotations

import json
import os
import sys

from . import gpumd as GM
from . import teximage as T

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WITNESS = "results/g17-texture-surface-census-v1/measured/tex2d-read.metadata.bin"
LANE31 = "results/g17-texture-contract-v1/lane31/abi.json"
# Measured by disassembling the witness's own recompiled program from its entry, not the whole
# __text. The whole text is 36; the prologue is the difference and it crosses the boundary.
MAIN_INSTRUCTIONS = 8
# From the source: uint2 thread_position_in_grid is two coordinates, so two system registers.
SYSTEM_REGISTERS = (160, 161)


def contract():
    """A contract for the tex2d-read source, built from that source and the surface census.

    Nothing here is read out of the witness's metadata except the two COMPILER inputs a contract
    always states - the register count and the argument bytes - which is the same standing the
    cooperative reproduction took.
    """
    fields = GM.fields(open(os.path.join(ROOT, WITNESS), "rb").read())
    abi = json.loads(open(os.path.join(ROOT, LANE31)).read())
    abi["bindings"] = [{"element_bytes": 4, "element_type": "uint", "index": 0,
                        "offset": 4, "written": True}]
    abi["resources"]["access"] = [
        {"kind": "internal", "read": None, "record": 44, "uniform": None, "written": False},
        {"kind": "internal", "read": None, "record": 48, "uniform": None, "written": False},
        {"kind": "user", "read": False, "record": 0, "uniform": True, "written": True}]
    abi["resources"]["internal"] = [{"apple_index": 44, "kind": "texture_internal", "rank": 0},
                                    {"apple_index": 48, "kind": "texture_internal", "rank": 1}]
    abi["resources"]["textures"] = [{"access": "read", "coordinates": "publish.coord.x/y",
                                     "dense_index": 0, "dimension": "2d", "element": "uint32",
                                     "rank": None}]
    abi["resources"]["samplers"] = []
    abi["resources"]["not_stated"] = []
    abi["resources"]["constant_pool_emptiness"] = "eight_byte_zero_vector"
    abi["resources"]["argument_bytes"] = fields.get(1)
    abi["resources"]["spill_bytes"] = 0
    abi["register_count"] = fields.get(0)
    abi["spill_bytes"] = 0
    abi["main_instruction_count"] = MAIN_INSTRUCTIONS
    abi["has_back_edge"] = False
    # TWO SYSTEM REGISTERS, from the source: `uint2 g [[thread_position_in_grid]]` needs x and y.
    # The first version of this contract inherited lane31's single-register list and planned a
    # one-entry slot-29 vector against a witness carrying two - and `against_the_witness` did not
    # notice, because it compared five fields and slot 29 was not one of them. "No disagreements"
    # over a comparison set that small is the same defect as a rate over a population that cannot
    # refute it.
    abi["system_registers"] = list(SYSTEM_REGISTERS)
    return abi


def the_plan():
    plan = T.plan(contract())
    return {"not_determined": [e.split(" [")[0] for e in plan["not_determined"]],
            "placed": {k: plan["placed"][k] for k in sorted(plan["placed"])},
            "derived here for the first time": {
                "slot 10": plan["placed"].get("slot 10"),
                "slot 32": plan["placed"].get("slot 32")}}


def against_the_witness():
    """Every planned field that the witness also states, compared."""
    md = open(os.path.join(ROOT, WITNESS), "rb").read()
    fields = GM.fields(md)
    plan = T.plan(contract())["placed"]
    import struct
    pk = GM.kernel_table(md)
    slots = GM.table_at(md, pk)[0]
    at = pk + slots[13]
    vector = at + struct.unpack_from("<I", md, at)[0]
    pool_bytes = struct.unpack_from("<I", md, vector)[0]
    at = pk + slots[27]
    vector = at + struct.unpack_from("<I", md, at)[0]
    count = struct.unpack_from("<I", md, vector)[0]
    slot27 = [struct.unpack_from("<I", md, vector + 4 + 4 * i)[0] for i in range(count)]
    at = pk + slots[29]
    vector = at + struct.unpack_from("<I", md, at)[0]
    count = struct.unpack_from("<I", md, vector)[0]
    slot29 = [struct.unpack_from("<I", md, vector + 4 + 4 * i)[0] for i in range(count)]
    # EVERY SCALAR AND VECTOR THE PLAN PLACES THAT THE WITNESS ALSO STATES. The first version
    # compared five fields, reported "disagreements: none", and did not include slot 29 - where the
    # contract's inherited single system register planned one entry against a witness carrying two.
    # A comparison set that cannot disagree is the same defect as a population that cannot refute.
    rows = {
        "slot 0": {"planned": plan.get("slot 0"), "witness": fields.get(0)},
        "slot 1": {"planned": plan.get("slot 1"), "witness": fields.get(1)},
        "slot 3": {"planned": plan.get("slot 3"), "witness": fields.get(3)},
        "slot 27": {"planned": plan.get("slot 27"), "witness": slot27},
        "slot 29": {"planned": plan.get("slot 29"), "witness": slot29},
        "slot 38": {"planned": plan.get("slot 38"), "witness": fields.get(38)},
        "slot 42": {"planned": plan.get("slot 42"), "witness": fields.get(42)},
        "slot 13 byte count": {"planned": len(plan.get("slot 13") or []), "witness": pool_bytes},
        "slot 32": {"planned": (plan.get("slot 32") or {}).get("value"),
                    "witness": fields.get(32)},
    }
    return {"by_field": rows,
            "disagreements": sorted(k for k, v in rows.items() if v["planned"] != v["witness"]),
            "what this is not": ("an authored section. Nothing is emitted; these are planned "
                                 "values compared with a witness the plan never reads.")}


def negative_controls():
    """A contract that omits or conflicts with a fact must refuse the right one."""
    out = {}
    base = contract()
    cases = {
        "a dynamic threadgroup is declared": ({"uses_threadgroup": True}, "slot10_contents"),
        "main exceeds the slot-32 boundary": ({"main_instruction_count": 400}, "slot32_value"),
        "main's instruction count is omitted": ({"main_instruction_count": None}, "slot32_value"),
        "the pool form is not stated": (None, "constant_pool_emptiness"),
    }
    for label, (patch, expected) in cases.items():
        abi = json.loads(json.dumps(base))
        if patch is None:
            abi["resources"].pop("constant_pool_emptiness", None)
        else:
            abi.update(patch)
        try:
            refused = [e.split(" [")[0] for e in T.plan(abi)["not_determined"]]
            out[label] = {"refuses": refused, "refuses the expected fact": expected in refused,
                          "expected": expected}
        except Exception as error:
            out[label] = {"raised": type(error).__name__, "first words": str(error)[:80],
                          "refuses the expected fact": False, "expected": expected}
    return out


def the_status():
    plan = T.plan(contract())
    try:
        section = T.emit(contract(), None, {})
        emitted = True
        why = "the contract-driven structural emitter authored the measured class"
    except Exception as error:
        section = None
        emitted = False
        why = str(error)[:140]
    return {"measured": True,
            "every fact determined": not plan["not_determined"],
            "authorable": emitted,
            "bytes emitted": len(section) if section is not None else 0,
            "why not": why,
            "loader eligible": False,
            "loader because": ("no slot set carrying 38 or 42 has a dispatched execution receipt; "
                               "five other slot sets do"),
            "no hardware claim": "a metadata result implies nothing about hardware behaviour"}


def build(destination):
    os.makedirs(destination, exist_ok=True)
    document = {"contract": contract(), "plan": the_plan(),
                "against_the_witness": against_the_witness(),
                "negative_controls": negative_controls(), "status": the_status(),
                "gpu_dispatched": False, "bytes_emitted": the_status()["bytes emitted"]}
    open(os.path.join(destination, "narrow.json"), "w").write(
        json.dumps(document, indent=1, sort_keys=True) + "\n")
    return document


def check(destination):
    document = json.load(open(os.path.join(destination, "narrow.json")))
    findings = []
    plan = the_plan()
    if plan["not_determined"]:
        findings.append("the narrow contract now refuses %s; this delivery's result is that it "
                        "refuses nothing" % plan["not_determined"])
    for name in ("slot 10", "slot 32"):
        if not plan["derived here for the first time"].get(name):
            findings.append("%s is no longer derived, so the narrow class is determined by "
                            "something other than the rules this delivery rests on" % name)

    compared = against_the_witness()
    if compared["disagreements"]:
        findings.append("the plan disagrees with the witness on %s; a fully determined plan whose "
                        "values differ from the section is worse than a refusal"
                        % compared["disagreements"])

    controls = negative_controls()
    for label, row in sorted(controls.items()):
        if not row["refuses the expected fact"]:
            findings.append("the control %r no longer refuses %s (%s); a determined plan with no "
                            "refusing sibling does not show the author is reading the contract"
                            % (label, row["expected"], row.get("refuses") or row.get("raised")))

    status = the_status()
    if not status["authorable"]:
        findings.append("the determined narrow class is not authorable: %s" % status["why not"])
    if status["bytes emitted"] != 492:
        findings.append("the narrow class emitted %d bytes against its measured 492-byte section"
                        % status["bytes emitted"])
    if status["loader eligible"]:
        findings.append("loader eligibility has changed and this delivery says it is unmet")
    return findings


def main(argv):
    if len(argv) != 3 or argv[1] not in ("build", "check"):
        print(__doc__.strip().splitlines()[2].strip())
        print(__doc__.strip().splitlines()[3].strip())
        return 2
    if argv[1] == "build":
        document = build(argv[2])
        print("facts the narrow contract refuses: %s"
              % (document["plan"]["not_determined"] or "NONE"))
        for name, row in sorted(document["against_the_witness"]["by_field"].items()):
            print("   %-20s planned %-14s witness %s" % (name, row["planned"], row["witness"]))
        print("   disagreements: %s"
              % (document["against_the_witness"]["disagreements"] or "none"))
        print("controls refusing the expected fact: %d of %d"
              % (sum(1 for v in document["negative_controls"].values()
                     if v["refuses the expected fact"]), len(document["negative_controls"])))
        print("measured=%s determined=%s authorable=%s loader-eligible=%s"
              % (document["status"]["measured"], document["status"]["every fact determined"],
                 document["status"]["authorable"], document["status"]["loader eligible"]))
        return 0
    findings = check(argv[2])
    for finding in findings:
        print("FINDING: %s" % finding)
    print("%d finding(s)" % len(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
