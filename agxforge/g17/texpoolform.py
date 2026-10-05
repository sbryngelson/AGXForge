#!/usr/bin/env python3
"""Which empty-pool form the narrow class carries, and why no rule hands it to you.

    python3 tools/g17texpoolform.py build results/g17-texture-pool-form-v1
    python3 tools/g17texpoolform.py check results/g17-texture-pool-form-v1

`constant_pool_emptiness` is the last field a caller had to complete by hand before the narrow
texture class would author. The compiler owner declined to state it and was right to: a program
with no constants is emitted two ways, so the program does not determine it, and the compiler ABI
describes the program. The argument offered for moving it here was that it is a CLASS constant.

THAT ARGUMENT IS TOO BROAD AND THE CORPUS REFUTES IT. Over the 9,556 Apple `__GPU_METADATA`
sections, restricted to those whose slot-13 vector is ENTIRELY ZERO - a population that includes
every constant-free program and excludes every program whose pool has content - the vector's length
takes five values, and TEXTURE SECTIONS CARRY ALL FIVE:

    length   texture sections   other sections
         0                165              421
         4                235              336
         8                211              171
        12                232              406
        16                  3                4

So "a texture kernel gets the eight-byte form" is false, and a rule keyed on the texture route
would author the wrong vector for 635 of the 846 texture sections in that population. Texture-ness
does not decide it; nothing readable in the section does, which is what
`results/g17-texture-decisions-v1` already found by a different route - every readable candidate
overlaps between the two pool states.

WHAT IS TRUE IS NARROWER AND IS ENOUGH. The `tex2d-read` witness this class was measured on carries
an eight-byte all-zero slot-13 vector, and the class is one witness wide by construction - that is
what `g17texemit`'s guard says in nine other places. So the form is a constant OF THIS CLASS, not of
textures, and stating it is stating what the witness carries rather than applying a rule.

SO THE CLASS STATES IT AND THE CALLER STOPS COMPLETING IT. `resolve(abi)` returns the contract with
`constant_pool_emptiness` filled in when the contract is otherwise admissible and silent on it, and
REFUSES a contract that states the other form - stating `zero_length_vector` is asking for a
different section, not a variant of this one. The one thing it will not do is fill the field in for
a contract outside the class, because then the value would be travelling on the strength of a
census it has just been shown not to follow from.

No GPU dispatch, no image, no loader claim.
"""
from __future__ import annotations

import collections
import glob
import json
import os
import struct
import sys

from . import gpumd as GM

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WITNESS = "results/g17-texture-surface-census-v1/measured/tex2d-read.metadata.bin"
CORPUS = "~/.cache/agxforge/syslib/out/*.jsonl"
THE_CLASSS_FORM = "eight_byte_zero_vector"
THE_OTHER_FORM = "zero_length_vector"
_CACHE = {}


class Refused(ValueError):
    """A contract this class will not complete. Refusing beats carrying a value on a false census."""


def _slot13(md):
    """The slot-13 vector's bytes, or None when the section does not carry the slot."""
    pk = GM.kernel_table(md)
    if pk is None:
        return None
    slots = GM.table_at(md, pk)[0]
    if len(slots) <= 13 or not slots[13]:
        return None
    at = pk + slots[13]
    vector = at + struct.unpack_from("<I", md, at)[0]
    count = struct.unpack_from("<I", md, vector)[0]
    return bytes(md[vector + 4:vector + 4 + count])


def the_witness():
    with open(os.path.join(ROOT, WITNESS), "rb") as handle:
        md = handle.read()
    pool = _slot13(md)
    fields = GM.fields(md)
    return {
        "slot 13 length": len(pool) if pool is not None else None,
        "all zero": pool is not None and not any(pool),
        "carries the texture signature slots 38 and 42": 38 in fields and 42 in fields,
        "so the class's form": THE_CLASSS_FORM,
    }


def the_census():
    """Apple sections whose slot-13 vector is entirely zero, by length and by texture-ness."""
    if "census" in _CACHE:
        return _CACHE["census"]
    table = collections.Counter()
    sections = unreadable = 0
    for path in sorted(glob.glob(os.path.expanduser(CORPUS))):
        with open(path) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                for key, blob in (record.get("md") or {}).items():
                    if not key.startswith("__GPU_METADATA,"):
                        continue
                    md = bytes.fromhex(blob)
                    sections += 1
                    try:
                        pool = _slot13(md)
                        fields = GM.fields(md)
                    except Exception:
                        unreadable += 1
                        continue
                    if pool is None or any(pool):
                        continue
                    table[(38 in fields and 42 in fields, len(pool))] += 1
    rows = {}
    for length in sorted({length for _tex, length in table}):
        rows[str(length)] = {"texture": table[(True, length)], "other": table[(False, length)]}
    out = {
        "sections read": sections,
        "unreadable": unreadable,
        "population": "slot-13 vector present and entirely zero",
        "by length": rows,
        "texture sections in the population": sum(v["texture"] for v in rows.values()),
    }
    _CACHE["census"] = out
    return out


def the_broad_argument_is_refuted():
    census = the_census()
    rows = census["by length"]
    texture_total = census["texture sections in the population"]
    eight = rows.get("8", {}).get("texture", 0)
    return {
        "the argument": "the eight-byte form is what a texture kernel gets, so it is a class "
                        "constant of the texture class",
        "texture sections with an all-zero pool": texture_total,
        "of them carrying the eight-byte form": eight,
        "carrying something else": texture_total - eight,
        "lengths texture sections carry": sorted(int(k) for k in rows if rows[k]["texture"]),
        "so texture-ness decides it": len([k for k in rows if rows[k]["texture"]]) == 1,
        "what a rule keyed on the texture route would get wrong": texture_total - eight,
        "agreeing with": ("results/g17-texture-decisions-v1, which found by a different route that "
                          "every readable candidate overlaps between the two pool states"),
    }


def why_it_is_still_statable():
    return {
        "the class is one witness wide": ("g17texemit's guard refuses a different texture count, a "
                                          "sampler, more than one written user binding, a "
                                          "different argument-byte count, a different system "
                                          "register set, a spill, a dynamic threadgroup and a main "
                                          "above the slot-32 boundary"),
        "so stating the form": "states what the witness carries, rather than applying a rule",
        "and the compiler cannot state it": ("the program does not determine it - both forms appear "
                                             "among sections whose pool is entirely zero - and the "
                                             "compiler ABI describes the program"),
    }


def resolve(abi):
    """The contract with `constant_pool_emptiness` filled in, or a refusal naming why not."""
    from . import texemit as E
    resources = (abi.get("resources") or {})
    stated = resources.get("constant_pool_emptiness")
    if stated == THE_CLASSS_FORM:
        return json.loads(json.dumps(abi))
    if stated is not None:
        raise Refused(
            "the contract states constant_pool_emptiness=%r and this class carries %r. That is a "
            "different section, not a variant of this one: over the corpus population whose "
            "slot-13 vector is entirely zero both forms appear, so neither is a default for the "
            "other" % (stated, THE_CLASSS_FORM))
    filled = json.loads(json.dumps(abi))
    filled.setdefault("resources", {})["constant_pool_emptiness"] = THE_CLASSS_FORM
    remaining = [reason for reason in E.admit(filled)]
    if remaining:
        raise Refused(
            "this class will not state constant_pool_emptiness for a contract outside it - the "
            "value is the WITNESS's, and the census shows it does not follow from being a texture "
            "at all. The contract is refused for: %s" % "; ".join(remaining))
    return filled


def the_field_stops_being_a_hand_completion():
    """The narrow contract, resolved rather than completed at the call site."""
    from . import texemit as E
    from . import texnarrow as N
    contract = N.contract()
    silent = json.loads(json.dumps(contract))
    silent["resources"].pop("constant_pool_emptiness", None)
    resolved = resolve(silent)
    return {
        "the contract is silent on the field": "constant_pool_emptiness" not in
                                               silent.get("resources", {}),
        "and refused while silent": bool(E.admit(silent)),
        "resolved form": resolved["resources"]["constant_pool_emptiness"],
        "and admitted once resolved": E.admit(resolved) == [],
        "the emitted section is unchanged": E.emit(resolved) == E.emit(contract),
    }


def _document():
    return {
        "the_witness": the_witness(),
        "the_census": the_census(),
        "the_broad_argument_is_refuted": the_broad_argument_is_refuted(),
        "why_it_is_still_statable": why_it_is_still_statable(),
        "the_field_stops_being_a_hand_completion": the_field_stops_being_a_hand_completion(),
    }


def build(destination):
    os.makedirs(destination, exist_ok=True)
    path = os.path.join(destination, "pool-form.json")
    with open(path, "w") as handle:
        json.dump(_document(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def check(destination=None):
    findings = []
    witness = the_witness()
    if witness["slot 13 length"] != 8 or not witness["all zero"]:
        findings.append("the class's witness no longer carries an eight-byte all-zero slot-13 "
                        "vector, so the stated form has lost its subject")
    refuted = the_broad_argument_is_refuted()
    if refuted["so texture-ness decides it"]:
        findings.append("texture sections now carry one pool length only, so the broad class "
                        "argument this report refutes would hold and the report is stale")
    if refuted["carrying something else"] <= 0:
        findings.append("no texture section carries a length other than eight, so the refutation "
                        "has no counterexamples behind it")
    resolved = the_field_stops_being_a_hand_completion()
    if not resolved["and refused while silent"]:
        findings.append("a contract silent on the field is now admitted without it, so resolve() "
                        "is not what closes the gap")
    if not resolved["and admitted once resolved"]:
        findings.append("the resolved contract is still refused")
    if not resolved["the emitted section is unchanged"]:
        findings.append("resolving the field changed the emitted section")
    try:
        resolve({"resources": {"constant_pool_emptiness": THE_OTHER_FORM}})
    except Refused:
        pass
    else:
        findings.append("a contract stating the other pool form was completed rather than refused")
    try:
        resolve({"resources": {}})
    except Refused:
        pass
    else:
        findings.append("a contract outside the class had the field stated for it")
    if destination:
        path = os.path.join(destination, "pool-form.json")
        if not os.path.exists(path):
            findings.append("%s is not retained" % path)
    return findings


def main(argv):
    if len(argv) > 1 and argv[1] == "build":
        print(build(argv[2] if len(argv) > 2 else "results/g17-texture-pool-form-v1"))
        return 0
    if len(argv) > 1 and argv[1] == "check":
        findings = check(argv[2] if len(argv) > 2 else None)
        for finding in findings:
            print(finding)
        print("%d finding(s)" % len(findings))
        return 1 if findings else 0
    print(json.dumps(_document(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
