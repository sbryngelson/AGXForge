#!/usr/bin/env python3
"""THE UNKNOWN-COVERAGE FEATURES, ASKED OF APPLE'S COMPILER: what does each one lower to?

Eight compiler capabilities and one linker capability were recorded as `unknown_coverage` - a
claim about our own ignorance with no number on it: "semantics is unknown; encoding is unknown;
lowering is missing", and a next action of "bounded census first". Apple's compiler is a free
oracle for the first half of that census. This compiles one Metal kernel per feature (and a
baseline without it), decodes the program, and records the forms the feature adds: how many,
how many Apple's table names, how many vendor instances each carries (isa/g17-vendor-corpus-
forms.json, 9,556 decoded vendor functions) and whether an isolated execution record exists for
the opcode. Nothing is dispatched: ac_archive builds a pipeline archive, no command buffer.

    python3 tools/g17featurecensus.py            rebuild and print
    python3 tools/g17featurecensus.py --write    rebuild and write isa/g17-feature-census.json
    python3 tools/g17featurecensus.py --check    re-derive every verdict from the written rows

WHAT A ROW DOES NOT SAY. An isolated record for a CROSS-LANE opcode is not semantics: the harness
gives every lane the same value, so a reduction or shuffle reads back its input or an identity
(isa/g17-execution-fits.json cross_lane_forms_this_harness_cannot_reach). Such opcodes are counted
apart. A new form is new against THIS baseline only - the feature's surrounding arithmetic lands
in it too - which is why each feature also names its SIGNATURE forms, and --check refuses if a
signature form is no longer in the arm (the list would otherwise go stale silently).

EACH ARM IS BUILT IN ITS OWN PROCESS: a crash in one archive call must not lose the rest, and
LD_MD carries a per-process build counter (see tools/g17aliasgrid.py).
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEST = os.path.join(ROOT, "isa", "g17-feature-census.json")
VENDOR = os.path.join(ROOT, "isa", "g17-vendor-corpus-forms.json")

_H = "#include <metal_stdlib>\nusing namespace metal;\n"
_IO = "device const float *a [[buffer(0)]], device float *o [[buffer(1)]]"

# arm -> (capability id, source, signature forms "opcode/length", role)
ARMS = {
    "baseline": (None, _H + "kernel void k(%s,\n              uint t [[thread_position_in_grid]]) {\n"
                             "  o[t] = a[t] * 2.0f;\n}\n" % _IO, [], "baseline"),
    "simd.reductions": ("compiler.simd.reductions", _H +
        "kernel void k(%s,\n              uint t [[thread_position_in_grid]]) {\n  float v = a[t] * 2.0f;\n"
        "  o[t] = simd_sum(v) + simd_max(v) + simd_shuffle(v, 3u) + simd_shuffle_xor(v, 1u) + simd_prefix_exclusive_sum(v);\n}\n" % _IO,
        ["16842/10", "16858/10", "14157/10", "14169/10", "16841/10"], "feature"),
    "conv.fp64": ("compiler.conv.fp64", _H +
        "kernel void k(%s,\n              uint t [[thread_position_in_grid]]) {\n"
        "  double d = (double)a[t]; o[t] = (float)(d * 2.0);\n}\n" % _IO, [], "refusal"),
    "cf.calls.switch": ("compiler.cf.calls", _H +
        "kernel void k(%s,\n              device const uint *s [[buffer(2)]], uint t [[thread_position_in_grid]]) {\n"
        "  float v = a[t];\n  switch (s[t] & 15u) {\n"
        "    case 0: v *= 2.0f; break; case 1: v += 3.0f; break; case 2: v = sqrt(v); break; case 3: v = exp2(v); break;\n"
        "    case 4: v -= 7.0f; break; case 5: v = v * v; break; case 6: v = log2(v); break; case 7: v = -v; break;\n"
        "    case 8: v = fabs(v); break; case 9: v = floor(v); break; case 10: v = ceil(v); break; default: v = 0.0f;\n"
        "  }\n  o[t] = v;\n}\n" % _IO, ["462/10"], "feature"),
    "cf.calls.noinline": ("compiler.cf.calls", _H +
        "__attribute__((noinline)) float f(float x, uint n) { float y = x; for (uint i = 0; i < n; ++i) y = y * 1.5f + 1.0f; return y; }\n"
        "kernel void k(%s,\n              device const uint *s [[buffer(2)]], uint t [[thread_position_in_grid]]) {\n"
        "  o[t] = f(a[t], s[0]) + f(a[t] * 2.0f, s[1]);\n}\n" % _IO, ["450/10"], "feature"),
    "cf.calls.visible_table": ("compiler.cf.calls", _H +
        "[[visible]] float g(float x) { return x * 3.0f; }\n"
        "kernel void k(%s,\n              visible_function_table<float(float)> tab [[buffer(2)]], uint t [[thread_position_in_grid]]) {\n"
        "  o[t] = tab[0](a[t]);\n}\n" % _IO, [], "feature"),
    "mem.address_spaces.constant": ("compiler.mem.address_spaces", _H +
        "kernel void k(%s,\n              constant float *c [[buffer(2)]], uint t [[thread_position_in_grid]]) {\n"
        "  o[t] = a[t] * c[t & 63u];\n}\n" % _IO, ["12682/8"], "feature"),
    "mem.address_spaces.pointer64": ("compiler.mem.address_spaces", _H +
        "kernel void k(%s,\n              device const ulong *p [[buffer(2)]], uint t [[thread_position_in_grid]]) {\n"
        "  device const float *q = (device const float *)(p[0] + (ulong)t * 4ul);\n  o[t] = a[t] * q[0];\n}\n" % _IO,
        ["12682/8"], "feature"),
    "mem.sampler": ("compiler.mem.sampler", _H +
        "kernel void k(texture2d<float> tx [[texture(0)]], sampler sm [[sampler(0)]], device float *o [[buffer(1)]],\n"
        "              uint t [[thread_position_in_grid]]) {\n  o[t] = tx.sample(sm, float2(t, 1) * 0.01f, level(0)).x;\n}\n",
        ["14661/22"], "feature"),
    "mem.sparse_resources.sparse": ("compiler.mem.sparse_resources", _H +
        "kernel void k(texture2d<float> tx [[texture(0)]], sampler sm [[sampler(0)]], device float *o [[buffer(1)]],\n"
        "              uint t [[thread_position_in_grid]]) {\n"
        "  sparse_color<float4> c = tx.sparse_sample(sm, float2(t, 1) * 0.01f, level(0));\n"
        "  o[t] = c.resident() ? c.value().x : -1.0f;\n}\n", ["14665/22", "14666/22"], "feature"),
    "mem.sparse_resources.icb": ("compiler.mem.sparse_resources", _H +
        "struct Args { command_buffer cb; };\n"
        "kernel void k(device Args &a [[buffer(0)]], device float *o [[buffer(1)]], uint t [[thread_position_in_grid]]) {\n"
        "  compute_command c(a.cb, t); c.reset(); o[t] = 1.0f;\n}\n", [], "feature"),
    "graphics.raytracing": ("compiler.graphics.raytracing", _H + "using namespace metal::raytracing;\n"
        "kernel void k(primitive_acceleration_structure as [[buffer(0)]], device float *o [[buffer(1)]],\n"
        "              uint t [[thread_position_in_grid]]) {\n"
        "  ray r(float3(0, 0, -1), float3(0, 0, 1), 0.0f, 100.0f);\n  intersector<triangle_data> it;\n"
        "  intersection_result<triangle_data> res = it.intersect(r, as);\n"
        "  o[t] = res.type == intersection_type::triangle ? res.distance : -1.0f;\n}\n", [], "feature"),
    "graphics.stages.vertex": ("compiler.graphics.stages", _H +
        "vertex void vs(uint vid [[vertex_id]], device const float4 *pos [[buffer(0)]], device float4 *o [[buffer(1)]]) {"
        " o[vid] = pos[vid] * 2.0f; }\n", [], "stage"),
    "graphics.stages.compute_twin": ("compiler.graphics.stages", _H +
        "kernel void k(uint vid [[thread_position_in_grid]], device const float4 *pos [[buffer(0)]], device float4 *o [[buffer(1)]]) {"
        " o[vid] = pos[vid] * 2.0f; }\n", [], "stage_twin"),
    # THE STAGES ac_archive_vertex CANNOT REACH: a rasterizing pipeline (ac_archive_render in
    # spike/accel/accel.mm) with a vertex function that HAS outputs and a fragment function that
    # reads an interpolated varying. One object per stage.
    "graphics.stages.raster": ("compiler.graphics.stages", _H +
        "struct V { float4 p [[position]]; float c; };\n"
        "vertex V vs(uint vid [[vertex_id]], device const float4 *pos [[buffer(0)]]) {\n"
        "  V v; v.p = pos[vid]; v.c = pos[vid].x * 0.5f; return v;\n}\n"
        "fragment float4 fs(V in [[stage_in]]) { return float4(in.c, 0.0f, 0.0f, 1.0f); }\n", [], "raster"),
    # THE SLOT-44 QUESTION ASKED OF APPLE'S COMPILER. linker.gap.tensor_descriptor_state called
    # slot 44 "a field this compiler emits and Apple's compiler never emits", from 0 of 9,547
    # VENDOR compute sections - a population with no tensor kernel in it. This is
    # results/g17-tensor-shape-witness-v1/base.metal, a Metal tensor_ops matmul, compiled by Apple.
    "tensor.matmul.apple": ("linker.resources.tensor_descriptors", '#include <metal_stdlib>\n#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\nusing namespace mpp;\nusing namespace mpp::tensor_ops;\nkernel void k(device half *a [[buffer(1)]], device half *b [[buffer(2)]],\n              device float *out [[buffer(3)]]) {\n  tensor<device half, dextents<int,2>, tensor_inline>\n    tA(a, dextents<int,2>(64,32), array<int,2>{1,64});\n  tensor<device half, dextents<int,2>, tensor_inline>\n    tB(b, dextents<int,2>(32,64), array<int,2>{1,32});\n  tensor<device float, dextents<int,2>, tensor_inline>\n    tC(out, dextents<int,2>(32,32), array<int,2>{1,32});\n  constexpr auto desc = matmul2d_descriptor(32,32,64,false,false,false,\n                                           matmul2d_descriptor::mode::multiply);\n  matmul2d<desc, execution_simdgroups<1>> op;\n  auto sA = tA.slice<64,32>(0,0);\n  auto sB = tB.slice<32,64>(0,0);\n  auto sC = tC.slice<32,32>(0,0);\n  op.run(sA,sB,sC);\n}\n', [], "slots"),
}


def _tag(src):
    return "featcensus-" + hashlib.sha256(src.encode()).hexdigest()[:16]


def _one(src):
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(ROOT, "spike", "accel", "re"))
    import tempfile
    import g17corpus as C
    state = C.build(_tag(src), src)
    if state not in ("built", "cached"):
        return dict(status=state)
    import agxdis
    import g17fields as FL
    import g17gpumd as GM
    import g17mdgen as M
    import g17obj
    import g17ref
    import machobj
    g17ref.binary()
    d = os.path.join(C.CACHE, _tag(src))
    if not os.path.exists(d + "/out/object/0-0"):
        return dict(status=state, objects=_objects(d))
    with open(d + "/out/object/0-0", "rb") as fh:
        obj = fh.read()
    loc = machobj.locate(d + "/s.arc.metallib", d + "/out/object/0-0")
    first, size = agxdis.sections(loc["obj"])
    text, entry = bytes(loc["obj"][first:first + size]), loc["syms"]["_agc.main"]
    with tempfile.NamedTemporaryFile(suffix=".bin") as fh:
        fh.write(text)
        fh.flush()
        r = subprocess.run([FL.DIS, fh.name, str(entry), str(len(text) - entry), "--pc", str(entry)],
                           capture_output=True, text=True)
    forms, undecoded = [], 0
    for line in r.stdout.splitlines():
        p = line.split()
        if len(p) >= 3 and p[1] != "bad":
            forms.append("%d/%d" % (int(p[2]), int(p[1])))
        elif len(p) >= 2 and p[1] == "bad":
            undecoded += 1
    sections, _ = g17obj.sections_of(obj)
    row = dict(status=state, forms=forms, undecoded=undecoded, symbols=sorted(loc["syms"]),
               sections={k: n for k, (_o, n) in sections.items()})
    for k, (o, n) in sections.items():
        if k.startswith("__GPU_METADATA,"):
            sec = obj[o:o + n]
            table = M.describe(sec)["tables"].get(GM.kernel_table(sec)) or {}
            row["metadata"] = dict(section=k, bytes=n, sha256=hashlib.sha256(sec).hexdigest(), hex=sec.hex(),
                                   per_kernel_slots={str(s): table["fields"][s] for s in sorted(table.get("slots") or {}, key=int)})
    return row


def _objects(d):
    """Every object of a multi-stage archive: its stage, forms, per-kernel slots and sections."""
    import g17gpumd as GM
    import g17mdgen as M
    import g17obj
    sys.path.insert(0, ROOT)
    from agxforge.g17 import model
    out = {}
    for name in sorted(os.listdir(os.path.join(d, "out", "object"))):
        with open(os.path.join(d, "out", "object", name), "rb") as fh:
            obj = fh.read()
        sections, _syms = g17obj.sections_of(obj)
        stage = sorted({k.split(",")[1] for k in sections if k.startswith("__GPU_METADATA")})
        o, n = sections["__TEXT,__text"]
        forms = []
        for i in model.decode(obj[o:o + n]):
            forms.append("%d/%d" % (i.opcode.id, len(i.raw)))
        row = dict(stage=stage[0] if stage else None, forms=forms,
                   sections={k: n for k, (_o, n) in sections.items()}, undescribed={},
                   metadata_hex={k: obj[o:o + n].hex() for k, (o, n) in sections.items() if k.startswith("__GPU_METADATA")})
        for k, (o, n) in sections.items():
            if not k.startswith("__GPU_METADATA"):
                continue
            sec = obj[o:o + n]
            try:
                table = M.describe(sec)["tables"].get(GM.kernel_table(sec)) or {}
                row["per_kernel_slots"] = sorted(int(s) for s in table.get("slots") or {})
            except Exception as why:
                row["undescribed"][k] = dict(bytes=n, hex=sec.hex(), why=str(why)[:120])
        out[name] = row
    return out


def rebuild():
    rows = {}
    for arm, (cap, src, sig, role) in ARMS.items():
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--one"], input=src,
                           capture_output=True, text=True)
        rows[arm] = (json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip()
                     else dict(status="worker failed: " + (r.stderr.strip().splitlines() or ["?"])[-1][:160]))
        if rows[arm]["status"].startswith(("COMPILE FAILED", "ac_archive")) and not r.returncode:
            # the compiler's own words are the finding for a refusal
            p = os.path.join(os.path.expanduser("~/.cache/agxforge/agx"), _tag(src), "s.metal")
            e = subprocess.run(["xcrun", "metal", "-c", "-o", os.devnull, p], capture_output=True, text=True)
            rows[arm]["compiler_errors"] = sorted({l.split("error: ", 1)[1] for l in e.stderr.splitlines() if "error: " in l})
        rows[arm].update(capability=cap, role=role, signature=sig,
                         source_sha256=hashlib.sha256(src.encode()).hexdigest())
    return rows


def _facts(root=ROOT):
    names = {}
    with open(os.path.join(root, "isa", "g17-authoring.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            names[r["opcode"]] = r.get("name")
    isolated = set()
    for path in sorted(glob.glob(os.path.join(root, "isa", "g17-execution-*-results.json"))):
        with open(path) as fh:
            try:
                data = json.load(fh)
            except ValueError:
                continue
        for r in data if isinstance(data, list) else []:
            if isinstance(r, dict) and r.get("status") == "ok" and isinstance(r.get("op"), int):
                isolated.add(r["op"])
    with open(VENDOR if root == ROOT else os.path.join(root, "isa", "g17-vendor-corpus-forms.json")) as fh:
        vendor = json.load(fh)["sources"]["vendor_shader_corpus"]
    return names, isolated, vendor


CROSS_LANE = ("simd.", "quad.")


# THE TABLES ARE SNAPSHOTTED, BECAUSE THEY MOVE WHEN OTHER WORK SUCCEEDS. The verdict counts forms
# with an isolated record; the first version recomputed it from the LIVE execution records at every
# --check, so the day another lane dispatched a record for a census opcode, check-ledgers went red
# for that lane's success. The facts each verdict used are now kept beside it, --check compares
# them with the live tables and names what moved, and --refresh re-derives the facts and the
# verdict from the retained rows - CPU only, no recompile - which is the whole of the repair.
def _snapshot(rows, facts):
    names, isolated, vendor = facts
    forms = sorted({f for r in rows.values() for f in (r.get("forms") or []) + list(r.get("signature") or [])
                    + [g for o in (r.get("objects") or {}).values() for g in o["forms"]]},
                   key=lambda f: tuple(int(x) for x in f.split("/")))
    ops = sorted({int(f.split("/")[0]) for f in forms})
    return dict(names={str(o): names.get(o) for o in ops}, isolated=sorted(o for o in ops if o in isolated),
                vendor_instances={f: vendor["forms"].get(f, 0) for f in forms})


def _from_snapshot(snap):
    return ({int(o): n for o, n in snap["names"].items()}, set(snap["isolated"]),
            {"forms": snap["vendor_instances"]})


class Refused(ValueError):
    pass


def classify(rows, facts):
    names, isolated, vendor = facts
    base = set((rows.get("baseline") or {}).get("forms") or [])
    if not base:
        raise Refused("the baseline did not build, so no form can be called new")
    out = {}
    for arm, row in rows.items():
        if row["role"] in ("baseline", "slots"):
            continue
        if row["role"] == "raster":
            if not row.get("objects"):
                raise Refused("%s built no stage objects: %s" % (arm, row["status"]))
            buffer_only = set((rows.get("graphics.stages.vertex") or {}).get("metadata", {}).get("per_kernel_slots", {}))
            stages = {}
            for o in row["objects"].values():
                slots = {str(x) for x in o.get("per_kernel_slots", [])}
                nontrivial = [f for f in o["forms"] if f not in base and not f.startswith(("13483/", "684/"))]
                stages[o["stage"]] = dict(
                    forms=len(o["forms"]), new_forms=sorted(set(nontrivial)),
                    unnamed=sorted({f for f in nontrivial if not names.get(int(f.split("/")[0]))}),
                    per_kernel_slots=sorted(int(x) for x in slots),
                    slots_added_over_buffer_only_vertex=sorted(int(x) for x in slots - buffer_only),
                    slots_dropped_from_buffer_only_vertex=sorted(int(x) for x in buffer_only - slots),
                    sections_not_described=sorted(o["undescribed"]))
            out[arm] = dict(capability=row["capability"], stages=stages)
            continue
        if row["role"] == "refusal":
            if row["status"].startswith("COMPILE FAILED") and row.get("compiler_errors"):
                out[arm] = dict(capability=row["capability"], verdict="refused_by_the_language",
                                compiler_errors=row["compiler_errors"])
                continue
            raise Refused("%s was expected to be refused by the compiler and was not: %s" % (arm, row["status"]))
        if row["status"] not in ("built", "cached"):
            raise Refused("%s did not build: %s" % (arm, row["status"]))
        forms = collections.Counter(row["forms"])
        new = sorted(f for f in forms if f not in base)
        missing_sig = [f for f in row["signature"] if f not in new]
        if missing_sig:
            raise Refused("%s: signature form(s) %s are no longer new in the arm; the signature is stale" % (arm, missing_sig))
        def op(f):
            return int(f.split("/")[0])
        named = [f for f in new if names.get(op(f))]
        cross = [f for f in new if str(names.get(op(f)) or "").startswith(CROSS_LANE)]
        out[arm] = dict(
            capability=row["capability"], instructions=len(row["forms"]), undecoded=row["undecoded"],
            new_forms=len(new), named=len(named), unnamed=sorted(set(new) - set(named)),
            with_isolated_record=len([f for f in new if op(f) in isolated and f not in cross]),
            cross_lane_with_uninformative_isolated_record=sorted(f for f in cross if op(f) in isolated),
            signature={f: dict(name=names.get(op(f)), vendor_instances=vendor["forms"].get(f, 0),
                               isolated_record=op(f) in isolated) for f in row["signature"]},
            new_forms_absent_from_every_vendor_function=sorted(f for f in new if not vendor["forms"].get(f)),
            separate_functions=sorted(s for s in row["symbols"] if not s.startswith("_agc.main")))
    t = rows.get("tensor.matmul.apple") or {}
    if t.get("metadata"):
        slots = sorted(t["metadata"]["per_kernel_slots"], key=int)
        out["tensor.matmul.apple"] = dict(capability=t["capability"], per_kernel_slots=[int(x) for x in slots],
                                          apple_compiler_emits_slot_44="44" in slots)
    v, c = rows.get("graphics.stages.vertex") or {}, rows.get("graphics.stages.compute_twin") or {}
    if v.get("metadata") and c.get("metadata"):
        vs, cs = v["metadata"]["per_kernel_slots"], c["metadata"]["per_kernel_slots"]
        out["graphics.stages.vertex"]["against_its_compute_twin"] = dict(
            metadata_section=v["metadata"]["section"], metadata_bytes=[v["metadata"]["bytes"], c["metadata"]["bytes"]],
            per_kernel_slots_only_in_vertex=sorted(set(vs) - set(cs), key=int),
            per_kernel_slots_only_in_compute=sorted(set(cs) - set(vs), key=int),
            per_kernel_slots_with_different_values=sorted((s for s in set(vs) & set(cs) if vs[s] != cs[s]), key=int),
            same_form_sequence=v["forms"] == c["forms"],
            sections=[v["sections"], c["sections"]])
    return out


def document(rows):
    return dict(generated_by="tools/g17featurecensus.py --write", gpu_dispatched=False,
                status="compiled_and_decoded_not_dispatched",
                vendor_population="isa/g17-vendor-corpus-forms.json vendor_shader_corpus (9,556 functions)",
                limitations="one kernel per feature against one baseline; forms are what Apple's compiler emits "
                            "for these spellings, not every lowering of the feature; an isolated record for a "
                            "cross-lane opcode is counted apart because the harness gives every lane one value",
                rows=rows, facts=_snapshot(rows, _facts()), verdict=classify(rows, _facts()))


def check(path=DEST):
    with open(path) as fh:
        doc = json.load(fh)
    if set(doc["rows"]) != set(ARMS):
        raise Refused("rows and arms differ")
    for arm, (cap, src, sig, role) in ARMS.items():
        row = doc["rows"][arm]
        if (row["source_sha256"], row["signature"], row["role"], row["capability"]) != \
                (hashlib.sha256(src.encode()).hexdigest(), sig, role, cap):
            raise Refused("%s: the written row is not for this tool's arm" % arm)
    if classify(doc["rows"], _from_snapshot(doc["facts"])) != doc["verdict"]:
        raise Refused("the written verdict is not what its own rows and facts give")
    live = _snapshot(doc["rows"], _facts())
    moved = [k for k in ("names", "isolated", "vendor_instances") if live[k] != doc["facts"][k]]
    if moved:
        raise Refused("the tables this census read have moved (%s) - most likely another lane landed "
                      "records; run python3 tools/g17featurecensus.py --refresh (CPU only) and rebuild "
                      "the capability inventories" % ", ".join(moved))
    return doc


def refresh(path=DEST):
    """Re-derive facts and verdict from the retained rows, without recompiling."""
    with open(path) as fh:
        doc = json.load(fh)
    doc["facts"] = _snapshot(doc["rows"], _facts())
    doc["verdict"] = classify(doc["rows"], _facts())
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return doc


def load(path=DEST):
    with open(path) as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="re-derive facts and verdict from the retained rows (no compile)")
    ap.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.one:
        print(json.dumps(_one(sys.stdin.read())))
        return 0
    try:
        if args.check:
            doc = check()
            print("feature census valid: %d arms" % len(doc["verdict"]))
            return 0
        if args.refresh:
            refresh()
            print("refreshed", os.path.relpath(DEST, ROOT))
            return 0
        doc = document(rebuild())
    except Refused as why:
        print("REFUSED: %s" % why)
        return 2
    print(json.dumps(doc["verdict"], indent=1))
    if args.write:
        with open(DEST, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print("wrote", os.path.relpath(DEST, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
