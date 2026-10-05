"""Contract-driven emitter for the measured, narrow tex2d-read metadata class.

The class specification contains node shapes and measured constants only.  It contains no
metadata bytes and this module never reads the retained witness.  Values that vary with the
program come from the ABI and are checked against the narrow class before serialization.
"""
from __future__ import annotations

import json
import os
import sys

from . import schema as S
from . import mdgen as M
from . import teximage as T

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPEC_PATH = os.path.join(ROOT, "isa", "g17-texture-narrow-structure.json")


def _spec():
    with open(SPEC_PATH) as handle:
        return json.load(handle)


def _narrow_shape(abi):
    resources = abi.get("resources") or {}
    textures = resources.get("textures") or []
    internals = resources.get("internal") or []
    bindings = abi.get("bindings") or []
    access = resources.get("access") or []
    if abi.get("abi_version") not in (6, 7):
        return False
    if len(textures) != 1 or textures[0] != {
            "access": "read", "coordinates": "publish.coord.x/y", "dense_index": 0,
            "dimension": "2d", "element": "uint32", "rank": None}:
        return False
    if resources.get("samplers") != [] or resources.get("not_stated") != []:
        return False
    if len(bindings) != 1 or bindings[0].get("index") != 0 \
            or bindings[0].get("offset") != 4 or not bindings[0].get("written") \
            or bindings[0].get("element_bytes") != 4 \
            or bindings[0].get("element_type") != "uint":
        return False
    if [(r.get("apple_index"), r.get("rank")) for r in internals] != [(44, 0), (48, 1)]:
        return False
    expected_access = [
        ("internal", 44, False), ("internal", 48, False), ("user", 0, True)]
    if [(a.get("kind"), a.get("record"), bool(a.get("written"))) for a in access] != expected_access:
        return False
    if abi.get("system_registers") != [160, 161]:
        return False
    if abi.get("register_count") != 3 or resources.get("argument_bytes") != 8:
        return False
    if abi.get("main_instruction_count") is None or abi["main_instruction_count"] > 30 \
            or abi.get("has_back_edge"):
        return False
    if resources.get("constant_pool_emptiness") != "eight_byte_zero_vector":
        return False
    if resources.get("constant_pool") not in (None, []):
        return False
    publications = resources.get("coordinate_publications") or []
    return [p.get("target_constant") for p in publications] == [0, 2]


def _field_format(width):
    if width == 1:
        return "<B"
    if width == 4:
        return "<I"
    raise T.Unmeasured("narrow texture structure has an unsupported scalar width %r" % width)


def emit(abi, led=None):
    """Return the 492-byte metadata section for exactly one measured contract."""
    if not _narrow_shape(abi):
        raise T.Unmeasured(
            "the narrow tex2d-read emitter only covers one uint2 2d-read class: bindings "
            "[(0,4,w)], internals [44,48], SR160/161, empty eight-byte pool, and no back edge")
    report = T.plan(abi)
    if report["not_determined"]:
        raise T.Unmeasured("the narrow texture contract still has unresolved facts: %s"
                           % report["not_determined"])
    spec = _spec()
    if spec.get("class") != "tex2d-read-empty-pool" or spec.get("size") != 492:
        raise T.Unmeasured("the narrow texture structural specification is not the measured class")
    dynamic = {
        "register_count": abi["register_count"],
        "argument_bytes": abi["resources"]["argument_bytes"],
        "slot3": report["placed"]["slot 3"],
        "slot27": report["placed"]["slot 27"],
        "slot29": report["placed"]["slot 29"],
        "slot13": report["placed"]["slot 13"],
        "slot38": report["placed"]["slot 38"],
        "slot42": report["placed"]["slot 42"],
    }
    nodes = {}
    pending = []
    for entry in spec["nodes"]:
        kind, name = entry["kind"], entry["name"]
        if kind == "string":
            nodes[name] = S.String(entry["text"], name=name)
        elif kind == "words":
            values = dynamic[entry["source"]] if "source" in entry else entry.get("values", [])
            nodes[name] = S.Words(values, name=name, width=entry["width"])
        elif kind == "vector":
            nodes[name] = S.Vector([], name=name)
            pending.append((name, entry.get("items", [])))
        elif kind == "table":
            fields = {}
            for slot, field in entry.get("fields", {}).items():
                slot = int(slot)
                if "ref" in field:
                    fields[slot] = (_field_format(field["width"]), None)
                elif "source" in field:
                    fields[slot] = (_field_format(field["width"]), dynamic[field["source"]])
                else:
                    fields[slot] = (_field_format(field["width"]), field["value"])
            table = S.Table({int(k): int(v) for k, v in entry["slots"].items()}, fields,
                            entry["tlen"], entry["tlen"], name=name,
                            vtable_length=entry["vlen"])
            nodes[name] = table
            for slot, field in entry.get("fields", {}).items():
                if "ref" in field:
                    pending.append((name, int(slot), field["ref"]))
        else:
            raise T.Unmeasured("unknown node kind %r in narrow texture specification" % kind)
    for item in pending:
        if len(item) == 2:
            name, refs = item
            nodes[name].items = [S.Ref(nodes[ref]) for ref in refs]
        else:
            name, slot, ref = item
            fmt, _ = nodes[name].fields[slot]
            nodes[name].fields[slot] = (fmt, S.Ref(nodes[ref]))
    order = [nodes[name] for name in spec["order"]]
    doc = S.Doc(nodes["root"], order)
    S.place(doc)
    # g17schema.place returns a conventional 16-byte rounded allocation.  The measured metadata
    # section is only four-byte aligned, so check the actual last node rather than rejecting its
    # harmless trailing alignment.
    end = max(n.addr + (n.size if isinstance(n, S.Table) else n.total) for n in order)
    if end > spec["size"]:
        raise T.Unmeasured("narrow texture graph needs %d bytes against measured %d" %
                           (end, spec["size"]))
    section = S.emit(doc, size=spec["size"])
    check = T.verify(section)
    if check["slot 1"] != dynamic["argument_bytes"] or not check["sum holds"] \
            or check["slot 38"] != dynamic["slot38"] or check["slot 42"] != dynamic["slot42"]:
        raise T.Unmeasured("narrow texture emitter postconditions failed: %s" % check)
    if led is not None:
        led["metadata class"] = (
            "MEASURED narrow tex2d-read class: contract-driven 492-byte structural graph")
        led["metadata specification"] = "isa/g17-texture-narrow-structure.json"
        led["metadata bytes"] = "serialized from g17schema nodes; no witness bytes read"
    return section
