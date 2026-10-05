#!/usr/bin/env python3
"""The FORM REGISTRY: one canonical template per G17 form, and an explicit account of every bit
the compiler still inherits rather than authors.

WHY THIS EXISTS. Until now every encoder took a template taken from the instruction AT THE SITE
BEING OVERWRITTEN (spike/accel/re/kern1.py: ADD_SITES, STORE_SITES). That makes emission
impossible anywhere Apple did not already put an instruction of the same form, so the program's
layout, instruction count and ordering are Apple's, not ours. A compiler cannot be built on it.

Replacing the per-SITE template with one canonical per-FORM template does three things:
  1. the compiler can emit a form anywhere, so layout becomes ours;
  2. the inherited bits stop being "whatever was at that address" and become ONE tracked,
     auditable dependency per form;
  3. those bits become COUNTABLE, which gives Track A the metric the mission actually asks for -
     inherited bits per emitted instruction, driven to zero - in place of a corpus byte-exact
     percentage that says nothing about whether we can author a program.

HOW THE CANONICAL TEMPLATE IS CHOSEN. For each form, every corpus instance is reduced to its
RESIDUE: the instruction with all owned bits cleared. The modal residue wins, and the template is
a real instance carrying it. So the inherited bits are the most typical ones for the form rather
than one site's accident, and any site-specific oddity is excluded by construction.

This module is DATA, not policy: it does not decide what is emitted, only what the emitter starts
from and what it still owes an explanation for.
"""
import os, sys, glob, collections, json
_T = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_T))
from agxforge.g17 import dis as g17dis, asm as g17asm, cover as g17cover, machobj, agxdis

CACHE = os.path.expanduser("~/.cache/agxforge/agx")

# name -> (g17cover family, length, OWNED mask, INERT bit set, encoder, decoder)
# Only forms with a field model appear. A form with no model has nothing to inherit BECAUSE it has
# nothing authored, and listing it here would imply a capability that does not exist.
# Bits measured INERT at two independent sites: flipping one at a working authored site left the
# program's output unchanged. They are variable across the corpus and unowned, so the old
# accounting called them UNRESOLVED - but a backend does not have to reproduce them correctly,
# which makes them a fourth category, not a debt. Only alu and load have been bit-labelled;
# the other forms have no inert list, and an absent list means UNMEASURED, not "none inert".
INERT = {"alu.12": g17asm.INERT_ALU, "alu.14": g17asm.INERT_ALU,
         "load.14": g17asm.INERT_LOAD, "load.10": g17asm.INERT_LOAD,
         "load.8": g17asm.INERT_LOAD}

SPEC = {
    "alu.12":     ("alu.12",     12, g17asm.OWNED_ALU,     g17asm.INERT_ALU,
                   "encode_alu",    "decode_alu"),
    "load.10":    ("load.10",    10, g17asm.OWNED_LOAD,    g17asm.INERT_LOAD,
                   "encode_load",   "decode_load"),
    "load.14":    ("load.14",    14, g17asm.OWNED_LOAD,    g17asm.INERT_LOAD,
                   "encode_load",   "decode_load"),
    "store.14":   ("store.14",   14, g17asm.OWNED_STORE,   set(),
                   "encode_store",  "decode_store"),
    "store.8":    ("store.8",     8, g17asm.OWNED_STORE,   set(),
                   "encode_store",  "decode_store"),
    "read_sr.4":  ("read_sr.4",   4, g17asm.OWNED_SR,      set(),
                   "encode_sr",     "decode_sr"),
    "classb.4":   ("classb.4",    4, g17asm.OWNED_CLASSB,  set(),
                   "encode_classb", "decode_classb"),
    "movimm.8":   ("movimm.8",    8, g17asm.OWNED_MOVIMM,  set(),
                   "encode_movimm", "decode_movimm"),
}

# RETARGETING PROBES. A canonical template is only useful if the encoder can actually move its
# fields; the most COMMON residue is not automatically a usable one. store.8's modal residue is
# the byte5 = 0x04 single-component sub-form, which refuses a source change outright, so choosing
# by frequency alone produced a registry whose first compile raised ValueError.
#
# Each probe is a set of semantic field values; a template passes if every probe encodes and then
# decodes back to exactly what was asked for. That is the same byte-exact round-trip discipline
# used everywhere else in the project, applied to template SELECTION.
PROBES = {
    "alu.12": [dict(dest=4, src1=5, mode=0, src2=6, op=3, src1_w=1, srcb_w=1, dest_w=1),
               dict(dest=9, src1=2, mode=1, imm=200, op=2, src1_w=1, srcb_w=1, dest_w=1),
               dict(dest=1, src1=31, mode=1, imm=0, op=1, src1_w=0, srcb_w=0, dest_w=0)],
    "movimm.8": [dict(dest=3, imm=0xDEADBEEF), dict(dest=15, imm=0), dict(dest=0, imm=200)],
    "read_sr.4": [dict(dest=2, sr=0xa0, seq=0), dict(dest=15, sr=0x9e, seq=3)],
    "classb.4": [dict(dest=3, base=4), dict(dest=15, base=0)],
    "load.10": [dict(dest=4, base=4, offset=0, index_reg=5),
                dict(dest=15, base=8, offset=32, index_reg=2)],
    "load.14": [dict(dest=4, base=4, offset=0, index_reg=5),
                dict(dest=15, base=8, offset=130, index_reg=2)],
    "store.14": [dict(src=4, n=1, slot=112), dict(src=15, n=4, slot=130)],
    # The 8-byte store has NO byte13, so its slot must fit in the low 6 bits: probing it at slot
    # 112 asks for a field the form does not have and rejects every template in the corpus.
    "store.8":  [dict(src=4, n=1, slot=0), dict(src=15, n=2, slot=63)],
}
# Fields the decoder reports but the probe does not set are not compared - a template legitimately
# carries its own values for them, and demanding they match would reject every template.
def retargetable(name, u):
    enc = getattr(g17asm, SPEC[name][4]); dec = getattr(g17asm, SPEC[name][5])
    for want in PROBES.get(name, []):
        try:
            got = dec(enc(template=u, **want))
        except Exception:
            return False
        for k, v in want.items():
            if k in got and got[k] != v: return False
    return True

def _popcount_owned(owned, length):
    return sum(bin(owned.get(i, 0)).count("1") for i in range(length))

def harvest(apple_only=True):
    """Collect every corpus instance of each modelled form, grouped by residue."""
    seen = {k: collections.Counter() for k in SPEC}
    examples = {k: {} for k in SPEC}
    fam2form = {v[0]: k for k, v in SPEC.items()}
    for d in sorted(glob.glob(CACHE + "/*")):
        base = os.path.basename(d)
        if apple_only and not base.startswith("ds_"): continue
        arc, obj = d + "/s.arc.metallib", d + "/out/object/0-0"
        if not (os.path.exists(arc) and os.path.exists(obj)): continue
        try:
            loc = machobj.locate(arc, obj)
            f, sz = agxdis.sections(loc["obj"]); text = loc["obj"][f:f+sz]
            walked = list(g17dis.walk(text, loc["syms"]["_agc.main"]))
        except Exception:
            continue
        for off, n, kind in walked:
            if kind == "tensor.mac": continue
            fam = g17cover.family(text, off, n, kind)
            form = fam2form.get(fam)
            if form is None: continue
            u = bytes(text[off:off+n])
            owned = SPEC[form][2]
            res = bytes(b & ~owned.get(i, 0) for i, b in enumerate(u))
            seen[form][res] += 1
            examples[form].setdefault(res, (base, off, u))
    return seen, examples

def structural_bits(members, owned):
    """Bits the model does not own that are CONSTANT across every corpus instance of the form.

    A bit that never varies is not information taken from a template - it is a structural constant
    the model can write from a table, so the template is not needed for it. That is a real
    reduction in the Apple dependency and it costs no experiment.

    IT IS NOT UNDERSTANDING. These bits remain UNEXPLAINED: we can emit them because Apple always
    emits them, not because we know what they do. So they count against the template dependency
    and NOT towards bits-explained, and a form whose remaining variable bits are few is a form
    whose template is nearly retired rather than one that is nearly understood.
    """
    n = len(members[0]); first = members[0]; varying = 0
    for u in members:
        for i in range(n): varying |= (u[i] ^ first[i]) << (8 * i)
    return [(i, b) for i in range(n) for b in range(8)
            if not (owned.get(i, 0) >> b) & 1 and not (varying >> (8 * i + b)) & 1]

# THE SIX OPCODES THAT ADDRESS THREADGROUP MEMORY. From the peer's isolation sweep: a kernel whose
# only memory traffic is threadgroup emits op12361/12364/12367/12376 to read and op13285/13288 to
# write, and no other probe of 750 emits any of them. They live in the SAME families as the device
# load and store - same length, same family name, same encoder - so a registry that picks the modal
# member of a family can hand a device IR store an instruction that writes threadgroup memory, and
# nothing downstream would say so: the program compiles, the decoder accepts it, and the buffer the
# host reads is simply never written.
#
# This registry did exactly that. Its store.8, store.14 and load.14 were 13285, 13288 and 12364 -
# threadgroup, all three - while every probe that harvests its own templates from the same hosts
# (endtoend.py, whose 32 answers are read back out of a device buffer) got 17262 and 12682. So the
# executed path was device and the DEFAULT path was not, which is why no test caught it.
TG_MEMORY_OPCODES = {12361, 12364, 12367, 12376, 13285, 13288}

_OPCODE_CACHE = {}

# WHICH OPCODE A TEMPLATE IS, CHECKED IN. g17cc._memory_gate asks this of every load and store it
# emits, to refuse a form that turns out to address threadgroup memory - a good check, and one that
# was forking Apple's decoder in the middle of a compile. The answer is a pure function of the
# template's bytes and the backend's templates are a fixed, small set, so the table lives in the
# checkout and `python3 tools/g17authtables.py` re-derives it.
_TEMPLATE_OPCODES = os.path.join(_ROOT, "isa",
                                 "g17-template-opcodes.json")
_TO_LOADED = [False]


def _to_load():
    if _TO_LOADED[0]:
        return
    _TO_LOADED[0] = True
    if not os.path.exists(_TEMPLATE_OPCODES):
        return
    import json as _json
    try:
        doc = _json.load(open(_TEMPLATE_OPCODES))
    except ValueError:
        return
    for h, o in (doc.get("opcodes") or {}).items():
        _OPCODE_CACHE.setdefault(bytes.fromhex(h), o)


def template_opcodes_dump():
    """Write every template this process resolved, merged over the checked-in table."""
    import json as _json
    _to_load()
    out = {k.hex(): v for k, v in sorted(_OPCODE_CACHE.items()) if v is not None}
    tmp = _TEMPLATE_OPCODES + ".%d" % os.getpid()
    with open(tmp, "w") as fh:
        _json.dump({"source": "tools/g17authtables.py - one decode per template the backend emits",
                    "templates": len(out), "opcodes": out}, fh, indent=0)
    os.replace(tmp, _TEMPLATE_OPCODES)
    return len(out)


def _opcode_of(u):
    """The Apple opcode a template decodes to, or None if the decoder will not take it.

    Reads the checked-in table; reaches the decoder only under G17_AUTH_REFRESH=1, which only
    tools/g17authtables.py sets. A miss outside a refresh raises rather than forking, because a
    silent fork is the dependency this was moved to remove.
    """
    key = bytes(u)
    _to_load()
    if key not in _OPCODE_CACHE:
        if os.environ.get("G17_AUTH_REFRESH") != "1":
            raise KeyError(
                "template %s is not in isa/g17-template-opcodes.json, and resolving it needs "
                "Apple's decoder. Run `python3 tools/g17authtables.py` to re-derive the table."
                % key.hex())
        try:
            from agxforge.g17 import ref as g17ref
            r = list(g17ref.walk(key, 0))
            _OPCODE_CACHE[key] = r[0][2] if len(r) == 1 and r[0][1] == len(key) else None
        except Exception:
            _OPCODE_CACHE[key] = None
    return _OPCODE_CACHE[key]

def build(apple_only=True):
    seen, examples = harvest(apple_only)
    forms = {}
    for name, (fam, length, owned, inert, enc, dec) in SPEC.items():
        if not seen[name]: continue
        # Most frequent residue that the encoder can actually retarget - and, for a memory family,
        # not one of the threadgroup opcodes. The IR's load and store are DEVICE accesses; there is
        # no threadgroup address space in this compiler to select one deliberately, so a family
        # member that addresses threadgroup memory is never the right answer here.
        # Three tiers, most frequent first within each: a member whose opcode is known and is not
        # a threadgroup one; then a member the decoder will not read standalone, whose opcode is
        # unknown; and only then a known threadgroup member. Ranking the unknown ABOVE the known-bad
        # would be the same mistake in a quieter form, so it sits between them.
        # ONLY the memory families are re-ranked. Every other family keeps the modal member it
        # always had: the threadgroup evidence is about load and store, and quietly changing which
        # instruction `alu.12` means on the strength of it would be a second, unmeasured change
        # riding along with the fix.
        memory = name.startswith(("load", "store"))
        def _tier(u):
            if not memory: return 0
            o = _opcode_of(u)
            return 2 if o in TG_MEMORY_OPCODES else (1 if o is None else 0)
        pick = None
        for tier in (0, 1, 2):
            for res, count in seen[name].most_common():
                src, off, u = examples[name][res]
                if not retargetable(name, u) or _tier(u) != tier: continue
                pick = (res, count, src, off, u); break
            if pick is not None: break
        if pick is None: continue
        res, count, src, off, u = pick
        total = length * 8
        own = _popcount_owned(owned, length)
        # A bit measured inert at two independent sites is metadata a backend need not reproduce,
        # so it is reported separately - but it is still INHERITED until it is authored, and the
        # headline number does not quietly forgive it.
        inert_unowned = sum(1 for (by, bi) in inert
                            if by < length and not (owned.get(by, 0) >> bi) & 1)
        members = [ex for res2 in seen[name] for ex in [examples[name][res2][2]]]
        structural = structural_bits(members, owned)
        inert = [(i, b) for (i, b) in INERT.get(name, set())
                 if i < length and not (owned.get(i, 0) >> b) & 1
                 and (i, b) not in structural]
        # WHICH INSTRUCTION IS THIS TEMPLATE? The registry harvests by FAMILY NAME and picks the
        # most frequent retargetable residue, so a family that spans several Apple opcodes yields
        # whichever one the scan met first - and store.8 does exactly that, giving opcode 17244
        # from one host and 13285 from another. Selecting by prefix is the same error as alu.12
        # holding 25 opcodes. ledger/g17-frozen-bytes-gate.toml
        #
        # Not fixed by choosing differently - there is no evidence yet for WHICH opcode a given IR
        # op should select. Fixed by making it VISIBLE: the opcode of the chosen template, and
        # whether every member of the family agrees with it.
        opcode = _opcode_of(u)
        member_ops = {_opcode_of(m) for m in members}
        member_ops.discard(None)
        forms[name] = dict(name=name, family=fam, length=length, template=u,
                           opcode=opcode, opcodes_in_family=sorted(member_ops),
                           address_space=(None if not name.startswith(("load", "store")) else
                                          ("threadgroup" if opcode in TG_MEMORY_OPCODES else "device")),
                           opcode_homogeneous=(len(member_ops) <= 1),
                           structural=structural, structural_bits=len(structural),
                           inert_bits=len(inert), bit_labelled=name in INERT,
                           unresolved=total - own - len(structural) - len(inert),
                           owned_bits=own, total_bits=total, inherited=total - own,
                           inherited_excl_inert=total - own - inert_unowned,
                           instances=sum(seen[name].values()), modal_share=count,
                           source="%s+0x%x" % (src, off), encoder=enc, decoder=dec)
    return forms

def main():
    apple = "--probes" not in sys.argv
    forms = build(apple)
    print("=== G17 FORM REGISTRY  (%s corpus) ===" % ("Apple driver shaders" if apple else "all"))
    print("%-12s %4s %7s %11s %7s %11s  %s" %
          ("form", "len", "authored", "structural", "inert", "UNRESOLVED", "bit-labelled?"))
    ti = to = ts = tn = 0
    for n, f in sorted(forms.items()):
        ti += f["unresolved"]; to += f["owned_bits"]; ts += f["structural_bits"]
        tn += f["inert_bits"]
        print("%-12s %4d %6db %10db %6db %10db  %s" %
              (n, f["length"], f["owned_bits"], f["structural_bits"], f["inert_bits"],
               f["unresolved"], "yes" if f["bit_labelled"] else "NOT MEASURED"))
    tot = to + ts + ti + tn
    print("\n%d forms, %d bits:" % (len(forms), tot))
    print("  authored    %4d  %5.1f%%   written from semantics" % (to, 100.0 * to / max(tot, 1)))
    print("  structural  %4d  %5.1f%%   constant corpus-wide; writable from a table, NOT understood"
          % (ts, 100.0 * ts / max(tot, 1)))
    print("  inert       %4d  %5.1f%%   measured to have no effect at two sites; any value works"
          % (tn, 100.0 * tn / max(tot, 1)))
    print("  UNRESOLVED  %4d  %5.1f%%   genuinely unknown and genuinely needed"
          % (ti, 100.0 * ti / max(tot, 1)))
    print("\nA backend must get authored + structural right and may write anything for inert, so the\n"
          "real debt is UNRESOLVED. bits-EXPLAINED is unchanged by any of this: structural and\n"
          "inert bits are emitted without being understood.")
    print("Forms marked NOT MEASURED have no inert list, so their UNRESOLVED count is an UPPER\n"
          "bound - some of it is probably inert and nobody has checked.")
    if "--write" in sys.argv:
        out = os.path.join(_T, "..", "isa", "g17-forms.toml")
        with open(out, "w") as fh:
            fh.write("# GENERATED by tools/g17forms.py --write. The compiler's canonical template\n"
                     "# per form, and the bits it still inherits rather than authors.\n"
                     "# Regenerate after any change to an OWNED mask; never hand-edit.\n")
            for n, f in sorted(forms.items()):
                fh.write("\n[[form]]\nname = \"%s\"\nfamily = \"%s\"\nlength = %d\n"
                         "template = \"%s\"\nauthored_bits = %d\nstructural_bits = %d\n"
                         "unresolved_bits = %d\ninherited_excl_inert = %d\ninstances = %d\n"
                         "modal_share = %d\ncanonical_source = \"%s\"\nencoder = \"%s\"\n"
                         "# structural: constant across every corpus instance, so writable from a\n"
                         "# table rather than copied - but NOT understood. Not counted as explained.\n"
                         "structural = %s\n"
                         % (n, f["family"], f["length"], f["template"].hex(), f["owned_bits"],
                            f["structural_bits"], f["unresolved"], f["inherited_excl_inert"],
                            f["instances"], f["modal_share"], f["source"], f["encoder"],
                            "[" + ", ".join("[%d,%d]" % b for b in f["structural"]) + "]"))
        print("wrote %s" % os.path.normpath(out))

if __name__ == "__main__":
    main()
