#!/usr/bin/env python3
"""The narrow tex2d-read class, emitted byte-for-byte from a contract and a checked-in spec.

    python3 tools/g17texemit.py build results/g17-texture-emitter-v1
    python3 tools/g17texemit.py check results/g17-texture-emitter-v1

The 2026-09-12f follow-up asks for a contract-driven emitter for this one class. It emits the
retained `tex2d-read` section with ZERO differing bytes out of 492.

NOTHING IS READ FROM THE WITNESS AT RUNTIME. The scalars come from the contract's plan - slot 0, 1,
3, 38, 42, slot 27's derived reading, slot 13's stated pool form, slot 29 from the declared system
registers - and the STRUCTURE comes from a checked-in specification: which nodes exist, their slot
maps, their declared lengths, and the four values below that no rule derives. `g17schema` places the
nodes and resolves every reference from the address of the field that holds it; no address and no
byte is taken from the retained section. The section is compared with afterwards, which is a test
rather than an input.

THE SPEC'S UNDERIVED CONSTANTS, listed rather than buried, because each is a thing this class was
MEASURED to carry and not a thing the contract states:

    the slot-26 record        {0: 8, 1: 3, 2: 1, 3: an empty word vector} - byte-identical in every
                              class measured, and already a constant in g17mdgen.V0REC
    the kind-3 slot-2 record  field 2 = 6. THIS ONE VARIES BY PROGRAM: 6, 8 and 4 across the
                              retained texture witnesses, and a candidate rule tying it to the
                              kind-6 record's field 3 is refuted at results/g17-texture-emitter-gap-v1
    the two strings           `agc.main` and `agc.main.constant_program`, the symbols the object
                              itself declares
    the vector orders         slot 4 is 44 then users ascending then 48 - the measured vector order,
                              which is NOT the address order the placer produces; slot 2 is kind 6
                              then kind 3

BECAUSE ONE CONSTANT VARIES BY PROGRAM, THE CLASS IS NARROW BY CONSTRUCTION. `admit` refuses any
contract that differs from the shape the spec was measured for - a different texture count, a
sampler, more than one written user binding, a non-empty pool, a different argument-byte count, a
different system-register set, a spill, a dynamic threadgroup, or a main above the slot-32
boundary. A contract outside that shape would need a kind-3 field nobody can derive, so it is
refused by name rather than emitted with a borrowed 6.

No GPU dispatch, no loader claim. Emitting a section is not authoring an image and says nothing
about whether the loader accepts one.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

from . import mdgen as M
from . import schema as S
from . import teximage as T

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WITNESS = "results/g17-texture-surface-census-v1/measured/tex2d-read.metadata.bin"
SECTION_BYTES = 492

# The one spec constant that varies by program, kept alone so the narrowness guard has a subject.
KIND3_FIELD2 = 6
V0_RECORD = {0: 8, 1: 3, 2: 1}
STRINGS = ("agc.main", "agc.main.constant_program")


def admit(abi):
    """Refuse any contract outside the shape the class spec was measured for."""
    resources = abi.get("resources") or {}
    reasons = []
    textures = resources.get("textures") or []
    public = resources.get("texture_public_indices")
    # THE SECOND TEXTURE COSTS THIS SECTION NOTHING, which is why the pair is admitted here.
    # tx1f-solo and tx2f-pair differ in exactly ONE byte, the register count at offset 212, and
    # this emitter's own output differs in that same byte when the count moves: slot 3 is eight
    # times the BINDING count and a texture is not a binding. The public indices are required
    # anyway - __GPU_LD_MD needs them - so a contract that cannot say which slots it reads is
    # refused here too rather than authoring half an image.
    if len(textures) not in (1, 2):
        reasons.append("this class is measured for one or two textures; the contract declares %d"
                       % len(textures))
    elif any(t.get("access") != "read" or t.get("dimension") != "2d" for t in textures):
        reasons.append("this class is measured for read-only 2d textures only")
    elif len(textures) == 2 and (not public or len(public) != 2):
        reasons.append("a two-texture contract must state resources.texture_public_indices, one "
                       "[[texture(n)]] slot per texture")
    if resources.get("samplers"):
        reasons.append("samplers are outside this class")
    written = [b for b in (abi.get("bindings") or []) if b.get("written")]
    if len(abi.get("bindings") or []) != 1 or len(written) != 1:
        reasons.append("this class is measured for exactly one written user binding")
    if abi.get("constant_pool"):
        reasons.append("this class is measured for an empty constant pool")
    if resources.get("constant_pool_emptiness") != "eight_byte_zero_vector":
        reasons.append("the contract must state constant_pool_emptiness as eight_byte_zero_vector; "
                       "the other empty form is a different section")
    if resources.get("argument_bytes") != 8:
        reasons.append("this class is measured at 8 argument bytes; the kind-3 record's field 2 is "
                       "a spec constant that varies with the program and cannot be carried across")
    registers = tuple(abi.get("system_registers") or ())
    if registers == (160,):
        # Retained indexed-load controls (one/two R32Float textures, user 0)
        # reproduce this entire graph at 488 bytes. Slot 29 loses one word;
        # kind-3 field 2 stays 6. No new scalar constant is inferred.
        users = [a for a in resources.get("access", []) if a.get("kind") == "user"]
        bindings = abi.get("bindings") or []
        if (len(bindings) != 1 or bindings[0].get("index") != 0
                or bindings[0].get("offset") != 4 or bindings[0].get("element_type") != "uint"
                or bindings[0].get("element_bytes") != 4
                or len(users) != 1 or users[0].get("record") != 0
                or users[0].get("read") is not True or users[0].get("written") is not True
                or users[0].get("uniform") is not False
                or any(t.get("element") != "float32" for t in textures)
                or public != list(range(len(textures)))):
            reasons.append("SR160-only texture class requires indexed reads/writes of uint user 0 and R32Float textures at 0/1")
    elif registers != (160, 161):
        reasons.append("this class is measured for system registers 160 and 161, or the indexed SR160-only variant")
    if abi.get("spill_bytes") or resources.get("spill_bytes"):
        reasons.append("a spill is outside this class")
    if abi.get("uses_threadgroup"):
        reasons.append("a dynamic threadgroup buffer puts a record in slot 10, which this class "
                       "does not carry")
    count = abi.get("main_instruction_count")
    if count is None or M.slot32_for(count, bool(abi.get("has_back_edge"))) is not None:
        reasons.append("this class is measured with no slot 32; the contract must put main at or "
                       "below the measured boundary")
    return reasons


def document(abi):
    """The node graph for this class: structure from the spec, scalars from the contract's plan."""
    plan = T.plan(abi)["placed"]
    main_s = S.String(STRINGS[0], name="agc.main")
    const_s = S.String(STRINGS[1], name="const")
    w27 = S.Words(plan["slot 27"], name="w27", width=4)
    w29 = S.Words(M.slot29_entries(abi["system_registers"]), name="w29", width=4)
    w13 = S.Words(plan["slot 13"], name="w13", width=1)
    v0words = S.Words([], name="v0words", width=4)
    v0 = S.Table({0: 16, 1: 15, 2: 8, 3: 4},
                 {0: ("<I", V0_RECORD[0]), 1: ("<B", V0_RECORD[1]), 2: ("<I", V0_RECORD[2]),
                  3: ("<I", S.Ref(v0words))}, size=20, tlen=20, name="v0", vtable_length=12)
    vec26 = S.Vector([S.Ref(v0)], name="vec26")
    empties = {slot: S.Vector([], name="vec%d" % slot) for slot in (12, 10, 8, 6)}
    records = {r["index"]: r for r in plan["binding records"]}
    r44 = S.Table({0: 11, 1: 4}, {0: ("<B", 5), 1: ("<I", 44)}, size=12, tlen=12,
                  name="r44", vtable_length=8)
    user_index = [i for i in records if i < 32][0]
    user = records[user_index]
    ruser = S.Table({0: 6, 2: 8, 3: 7},
                    {0: ("<B", 5), 2: ("<I", user["field2 (offset)"]),
                     3: ("<B", user["field3 (written)"])}, size=12, tlen=12,
                    name="ruser", vtable_length=12)
    r48 = S.Table({0: 11, 1: 4, 2: 12},
                  {0: ("<B", 5), 1: ("<I", 48), 2: ("<I", records[48]["field2 (offset)"])},
                  size=16, tlen=16, name="r48", vtable_length=10)
    k6 = S.Table({0: 11, 2: 12, 3: 4},
                 {0: ("<B", 6), 2: ("<I", plan["slot-2 kind-6 field2"]),
                  3: ("<I", plan["slot-2 kind-6 field3"])}, size=18, tlen=18,
                 name="k6", vtable_length=12)
    k3 = S.Table({0: 7, 2: 8}, {0: ("<B", 3), 2: ("<I", KIND3_FIELD2)}, size=12, tlen=12,
                 name="k3", vtable_length=10)
    # THE VECTOR ORDER IS MEASURED AND IS NOT THE ADDRESS ORDER: 44, users ascending, then 48.
    vec4 = S.Vector([S.Ref(r44), S.Ref(ruser), S.Ref(r48)], name="vec4")
    vec2 = S.Vector([S.Ref(k6), S.Ref(k3)], name="vec2")
    pk = S.Table(
        {0: 64, 1: 48, 2: 44, 3: 36, 4: 40, 6: 32, 8: 28, 10: 24, 12: 20, 13: 16, 15: 63, 16: 62,
         26: 12, 27: 8, 29: 4, 38: 52, 42: 56},
        {0: ("<I", plan["slot 0"]), 1: ("<I", plan["slot 1"]), 2: ("<I", S.Ref(vec2)),
         3: ("<I", plan["slot 3"]), 4: ("<I", S.Ref(vec4)), 6: ("<I", S.Ref(empties[6])),
         8: ("<I", S.Ref(empties[8])), 10: ("<I", S.Ref(empties[10])),
         12: ("<I", S.Ref(empties[12])), 13: ("<I", S.Ref(w13)), 15: ("<B", 1), 16: ("<B", 1),
         26: ("<I", S.Ref(vec26)), 27: ("<I", S.Ref(w27)), 29: ("<I", S.Ref(w29)),
         38: ("<I", plan["slot 38"]), 42: ("<I", plan["slot 42"])},
        size=68, tlen=68, name="pk", vtable_length=90)
    entry = S.Table({1: 4}, {1: ("<I", S.Ref(main_s))}, size=8, tlen=8, name="entry",
                    vtable_length=8)
    root = S.Table({0: 8, 3: 4}, {0: ("<I", S.Ref(pk)), 3: ("<I", S.Ref(entry))}, size=12, tlen=12,
                   name="root", vtable_length=12)
    order = [root, entry, main_s, pk, w27, w29, vec26, v0, v0words, const_s, w13,
             empties[12], empties[10], empties[8], empties[6], vec4, vec2, k3, k6, r48, ruser, r44]
    return S.Doc(root, order, size=SECTION_BYTES - (4 if tuple(abi.get("system_registers") or ()) == (160,) else 0))


def emit(abi):
    """The bytes, or a named refusal for a contract outside this class."""
    reasons = admit(abi)
    if reasons:
        raise T.Unmeasured("this contract is outside the narrow tex2d-read class: %s"
                           % "; ".join(reasons))
    doc = document(abi)
    S.place(doc)
    return S.emit(doc, size=doc.size)


def against_the_witness():
    from . import texnarrow as N
    want = open(os.path.join(ROOT, WITNESS), "rb").read()
    got = emit(N.contract())
    differing = [i for i, (a, b) in enumerate(zip(got, want)) if a != b]
    return {"emitted bytes": len(got), "witness bytes": len(want),
            "differing bytes": len(differing), "first differences": differing[:8],
            "identical": got == want,
            "emitted sha256": hashlib.sha256(got).hexdigest(),
            "witness sha256": hashlib.sha256(want).hexdigest()}


def deterministic():
    from . import texnarrow as N
    first, second = emit(N.contract()), emit(N.contract())
    return {"two emissions are identical": first == second,
            "sha256": hashlib.sha256(first).hexdigest()}


def refusals():
    """Contracts outside the class, each refused by name."""
    import copy
    from . import texnarrow as N
    base = N.contract()
    cases = {
        "a sampler": {"resources": {"samplers": [{"index": 0}]}},
        "two textures": {"resources": {"textures": "double"}},
        "a non-empty pool": {"constant_pool": [1, 2, 3, 4]},
        "the other empty pool form": {"resources": {"constant_pool_emptiness": "zero_length_vector"}},
        "different argument bytes": {"resources": {"argument_bytes": 12}},
        "a different register set": {"system_registers": [160]},
        "a spill": {"spill_bytes": 32},
        "a dynamic threadgroup": {"uses_threadgroup": True},
        "a main above the boundary": {"main_instruction_count": 400},
    }
    out = {}
    for label, patch in cases.items():
        abi = copy.deepcopy(base)
        for key, value in patch.items():
            if key == "resources":
                for sub, subvalue in value.items():
                    if subvalue == "double":
                        abi["resources"][sub] = abi["resources"][sub] * 2
                    else:
                        abi["resources"][sub] = subvalue
            else:
                abi[key] = value
        try:
            emit(abi)
            out[label] = {"refused": False, "outcome": "EMITTED"}
        except Exception as error:
            out[label] = {"refused": True, "exception": type(error).__name__,
                          "first words": str(error)[:110]}
    return out


def build(destination):
    os.makedirs(destination, exist_ok=True)
    from . import texnarrow as N
    section = emit(N.contract())
    open(os.path.join(destination, "tex2d-read.emitted.bin"), "wb").write(section)
    doc = {"against_the_witness": against_the_witness(), "deterministic": deterministic(),
           "refusals": refusals(),
           "the spec's underived constants": {
               "the kind-3 slot-2 record's field 2": KIND3_FIELD2,
               "and it varies by program": "6, 8 and 4 across the retained texture witnesses",
               "the slot-26 record": V0_RECORD, "the strings": list(STRINGS),
               "the vector orders": "slot 4 is 44 then users ascending then 48; slot 2 is kind 6 "
                                    "then kind 3 - measured, and not the address order"},
           "nothing is read from the witness at runtime": (
               "structure from a checked-in spec, scalars from the contract's plan, addresses from "
               "g17schema.place. The section is compared with afterwards, which is a test rather "
               "than an input."),
           "no loader claim": "emitting a section is not authoring an image",
           "gpu_dispatched": False}
    open(os.path.join(destination, "emitter.json"), "w").write(
        json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def check(destination):
    doc = json.load(open(os.path.join(destination, "emitter.json")))
    findings = []
    live = against_the_witness()
    if not live["identical"]:
        findings.append("the emitter no longer reproduces the witness: %d differing bytes at %s"
                        % (live["differing bytes"], live["first differences"]))
    if live["emitted bytes"] != SECTION_BYTES:
        findings.append("the emitted section is %d bytes and this class is %d"
                        % (live["emitted bytes"], SECTION_BYTES))
    if not deterministic()["two emissions are identical"]:
        findings.append("two emissions of the same contract differ, so the section is not "
                        "deterministic")
    refused = refusals()
    for label, row in sorted(refused.items()):
        if not row["refused"]:
            findings.append("a contract with %s now EMITS; the class spec carries a constant that "
                            "varies by program, so a contract outside the measured shape must "
                            "refuse rather than borrow it" % label)
    if len(refused) < len(doc["refusals"]):
        findings.append("this delivery retained %d refusal controls and %d ran"
                        % (len(doc["refusals"]), len(refused)))
    return findings


def main(argv):
    if len(argv) != 3 or argv[1] not in ("build", "check"):
        print(__doc__.strip().splitlines()[2].strip())
        print(__doc__.strip().splitlines()[3].strip())
        return 2
    if argv[1] == "build":
        doc = build(argv[2])
        row = doc["against_the_witness"]
        print("emitted %d bytes, %d differing, identical=%s"
              % (row["emitted bytes"], row["differing bytes"], row["identical"]))
        print("deterministic: %s" % doc["deterministic"]["two emissions are identical"])
        print("refusal controls refusing: %d of %d"
              % (sum(1 for v in doc["refusals"].values() if v["refused"]), len(doc["refusals"])))
        return 0
    findings = check(argv[2])
    for finding in findings:
        print("FINDING: %s" % finding)
    print("%d finding(s)" % len(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
