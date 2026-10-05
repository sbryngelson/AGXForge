#!/usr/bin/env python3
"""Corpus-wide ENCODE coverage: of every instruction g17dis walks, how many can g17asm
re-encode byte-exactly from a FOREIGN template, and how many have no field model at all?

This is the mission's ENCODE number. Families with no model are reported as UNMODELLED rather
than skipped, so the denominator is every instruction in the corpus.
"""
import os, sys, glob, collections
from agxforge.g17 import dis as g17dis, asm as g17asm, machobj, agxdis

def family(t, off, ln, kind):
    b = t[off:off+ln]
    # A four-byte class-e that matches a recovered branch form in every non-displacement bit.
    # Split out of classe.4 (3288 instructions in Apple's corpus) because the displacement IS a
    # recovered field and the rest of that family is not modelled at all - folding them together
    # would score a real model against instructions it cannot reach.
    if kind == "class e" and ln == 4 and g17asm.is_branch(b): return "branch.4"
    if ln == 2 and g17asm.is_mov_short(b): return "movshort.2"
    if kind == "class c" and ln == 4: return "read_sr.4"
    # mov.wide.imm: the 32-bit constant form. byte2 == 0x02 identifies it among 8-byte class-c;
    # the other byte2 values (0x29, 0x00, 0x22, 0x37 ...) are different forms and stay unmodelled
    # rather than being swept in to pad the family. ledger/g17-mov-wide-immediate.toml
    if kind == "class c" and ln == 8 and b[2] == 0x02: return "movimm.8"
    if kind == "class b" and ln == 4: return "classb.4"
    if kind in ("class 7", "class f"):
        if ln == 12 and (b[6] & 0xF0) == 0xA0: return "alu.12"
        # The 14-byte ALU (ledger/g17-alu-fourteen-bytes.toml) is the SAME opcode shape with two
        # more bytes. It is named apart rather than folded into alu.12 because encode_alu models
        # 12 bytes and has no field model for byte12/13, so counting it as alu.12 would inflate
        # the family total against a model that cannot reach it.
        if ln == 14 and (b[6] & 0xF0) == 0xA0 and (b[2] & 0x3F) != 0x03: return "alu.14"
        if ln == 12: return "wide.12.op%02x" % (b[6] & 0xF0)
        # Direction is byte4[0], measured causally (ledger/g17-loadstore-direction.toml).
        # byte1 is the BASE REGISTER and must NOT gate this - keying on it silently dropped 31%
        # of memory instructions (ledger/g17-loadstore-classifier-defect.toml).
        if ln in (8, 12, 14) and (b[2] & 0x3F) == 0x03:
            return ("store" if (b[4] & 1) else "load") + ".%d" % ln
        # THE BARRIER, and byte1 alone is not enough to recognise it. byte1 is the SCOPE
        # (0x51 threadgroup, 0x69 device - g17asm.BARRIER_SCOPE), but in op998, op999 and op2190
        # byte1 is a REGISTER field that sometimes holds 0x51, and this rule was classifying
        # fifteen of those as barriers. Apple's own decoder calls the barrier op447 and every
        # instance has byte0 = 0x27 or 0x0f with bytes 2..5 = 00 06 00 00; requiring the tail
        # separates it from an ALU whose register happens to collide.
        # Defect reported by the ISA agent, who found it while cataloguing synchronisation.
        if ln == 6 and b[1] in (0x51, 0x69) and b[2:6] == b"\x00\x06\x00\x00":
            return "barrier"
        return "cls7f.%d" % ln
    return "%s.%d" % (kind.replace(" ", ""), ln)

# alu.14 was in `ownmap` - so the bits-EXPLAINED metric counted its 36 owned bits per instruction
# as understood - while being absent from MODELLED, so try_encode was never asked to produce one.
# The honest metric was claiming bits for a family the model could not emit. Adding it here makes
# the claim testable: bytes 12-13 are unowned, so they land in the KEY and the model must still
# reproduce bytes 0-11 from semantics. ledger/g17-alu-fourteen-bytes.toml establishes the form
# causally and records byte13 as causal-but-unmodelled.
MODELLED = {"branch.4", "movshort.2", "alu.12", "alu.14", "store.8", "store.12", "store.14", "read_sr.4", "load.8", "load.12", "load.14", "classb.4", "movimm.8"}
def try_encode(fam, u, T):
    if fam == "branch.4": return g17asm.encode_branch(template=T, **g17asm.decode_branch(u))
    if fam == "movshort.2": return g17asm.encode_mov_short(template=T, **g17asm.decode_mov_short(u))
    if fam in ("alu.12", "alu.14"): return g17asm.encode_alu(template=T, **g17asm.decode_alu(u))
    if fam == "read_sr.4": return g17asm.encode_sr(template=T, **g17asm.decode_sr(u))
    if fam == "movimm.8": return g17asm.encode_movimm(template=T, **g17asm.decode_movimm(u))
    if fam.startswith("load"): return g17asm.encode_load(template=T, **g17asm.decode_load(u))
    if fam == "classb.4": return g17asm.encode_classb(template=T, **g17asm.decode_classb(u))
    if fam.startswith("store"):
        # STALE GUARD REMOVED. This used to be `if not T[5] & 0x08: return None`, the ORIGINAL
        # over-broad reading of the store sub-form rule. encode_store's own condition was
        # narrowed twice - first to fire only when src actually moves, then to byte5 == 0x04
        # exactly - because a byte5 = 0x06 store had its source changed r1 -> r6 and stored the
        # right value from the right register (ledger/g17-stage3-full-chain.toml). The encoder
        # was corrected; this copy of the rule was not, so every store whose template lacked
        # bit3 was skipped rather than attempted. Let the encoder apply its own rule and raise.
        return g17asm.encode_store(template=T, **g17asm.decode_store(u))
    return None

# The ENCODE number has the same contamination the framing number had: ~/.cache/agxforge/agx is
# dominated by probe kernels I wrote to exercise forms already modelled, so a mixed-corpus
# percentage flatters the result. ledger/g17-corpus-contaminated-by-probes.toml
APPLE = lambda name: name.startswith("ds_")

def main(only=None, quiet=False):
    tmpl = {}; vacuous = set(); stats = collections.defaultdict(lambda: [0, 0, 0, 0, 0])   # [total, exact, attempted, inert-tol, NULL]
    for d in sorted(glob.glob(os.path.expanduser("~/.cache/agxforge/agx/*"))):
        if only and not only(os.path.basename(d)): continue
        arc, obj = d+"/s.arc.metallib", d+"/out/object/0-0"
        if not (os.path.exists(arc) and os.path.exists(obj)): continue
        try:
            loc = machobj.locate(arc, obj); f, sz = agxdis.sections(loc["obj"]); t = loc["obj"][f:f+sz]
            walk = list(g17dis.walk(t, loc["syms"]["_agc.main"], limit=len(t)))
        except Exception: continue
        for off, ln, kind in walk:
            u = t[off:off+ln]
            if u == b"\x06\x00" * (ln // 2): continue        # compiler filler
            fam = family(t, off, ln, kind)
            s = stats[fam]; s[0] += 1
            if fam not in MODELLED: continue
            # THE KEY MAY ONLY HOLD BYTES THE MODEL NEVER WRITES. Derived per family from g17asm's
            # OWNED maps: a byte with any owned bit, held fixed, hides exactly the gap the gate exists
            # to measure. See ledger/g17-gate-key-inflation.toml.
            owned = (g17asm.OWNED_MOVIMM if fam == "movimm.8" else
                 g17asm.OWNED_BRANCH if fam == "branch.4" else
                 g17asm.OWNED_MOVSHORT if fam == "movshort.2" else
                 g17asm.OWNED_ALU if fam in ("alu.12", "alu.14") else
                     g17asm.OWNED_STORE if fam.startswith("store") else
                     g17asm.OWNED_LOAD if fam.startswith("load") else
                     g17asm.OWNED_CLASSB if fam == "classb.4" else
                     g17asm.OWNED_SR)
            key = (fam, ln) + tuple(u[i] for i in range(ln) if not owned.get(i, 0))
            # A family whose every byte is either OWNED or in the KEY cannot fail the round trip -
            # the test carries no information. Detect it rather than reporting 100%.
            free = [i for i in range(ln) if owned.get(i, 0) and (owned[i] & 0xFF) != 0xFF]
            keyed = [i for i in range(ln) if not owned.get(i, 0)]
            if not any(i not in keyed and (owned.get(i, 0) & 0xFF) != 0xFF and
                       (~owned.get(i, 0) & 0xFF) & ~0x0F for i in range(ln)):
                vacuous.add(fam)
            tmpl.setdefault(key, u); T = tmpl[key]
            try: enc = try_encode(fam, u, T)
            except Exception: enc = None
            if enc is None: continue
            s[2] += 1
            if enc == bytes(u): s[1] += 1
            # THE NULL CONTROL, added 2026-09-04. A byte-exact re-encode is only evidence for the
            # MODEL if the model beats copying the foreign template unchanged. It often does not:
            # store.14 scores 100% byte-exact and 94.1% of that is the template already being
            # identical, because its key holds 8 of its 14 bytes. movimm.8's model adds exactly
            # nothing over the null. Without this column the metric credits the model for the
            # homogeneity of the family. ledger/g17-encode-null-control.toml
            if T == bytes(u): s[4] += 1
            inert = (g17asm.INERT_ALU if fam in ("alu.12", "alu.14") else
                     g17asm.INERT_LOAD if fam.startswith("load") else None)
            if inert is not None:                    # second score: ignore measured-inert bits
                diff = [i for i in range(ln) if enc[i] != u[i]]
                if all(all(((enc[i] ^ u[i]) >> b & 1) == 0 or (i, b) in inert
                           for b in range(8)) for i in diff):
                    s.append(0) if len(s) < 4 else None
                    s[3] = s[3] + 1 if len(s) > 3 else 1

    # WORK-ORDER (c) MEASURED DIRECTLY: what fraction of the corpus's BITS does the model
    # explain, rather than inherit from a template? This is the honest complement to the
    # byte-exact score, which is structurally weak on short instructions - 40.7% of Apple's
    # corpus is 2-byte, where a foreign template has 16 bits to differ in and 4 of them are the
    # class tag. ledger/g17-encode-metric-ceiling.toml
    ownmap = {"branch.4": g17asm.OWNED_BRANCH, "movshort.2": g17asm.OWNED_MOVSHORT,
              "alu.12": g17asm.OWNED_ALU, "alu.14": g17asm.OWNED_ALU,
              "read_sr.4": g17asm.OWNED_SR, "classb.4": g17asm.OWNED_CLASSB,
              "movimm.8": g17asm.OWNED_MOVIMM}
    ebits = ibits = 0
    for fam, v in stats.items():
        tail = fam.rsplit(".", 1)[-1]
        if not tail.isdigit(): continue            # e.g. "barrier", "wide.12.op70"
        ln = int(tail)
        om = ownmap.get(fam) or (g17asm.OWNED_STORE if fam.startswith("store")
                                 else g17asm.OWNED_LOAD if fam.startswith("load") else None)
        n = v[0]
        if om is None: ibits += n * ln * 8; continue
        owned = sum(bin(om.get(i, 0)).count("1") for i in range(ln))
        ebits += n * owned; ibits += n * (ln*8 - owned)
    print("bits EXPLAINED by named fields: %d / %d  (%.1f%%)   inherited: %.1f%%"
          % (ebits, ebits+ibits, 100.0*ebits/max(ebits+ibits,1), 100.0*ibits/max(ebits+ibits,1)))
    tot = sum(v[0] for v in stats.values())
    exact = sum(v[1] for k, v in stats.items() if k not in vacuous)
    att = sum(v[2] for k, v in stats.items() if k not in vacuous)
    print("corpus instructions walked (filler excluded): %d" % tot)
    null = sum(v[4] for k, v in stats.items() if k not in vacuous)
    print("byte-exact re-encode from a foreign template: %d  (%.1f%% of all, %.1f%% of attempted)"
          % (exact, 100.0*exact/tot, 100.0*exact/max(att,1)))
    print("NULL control - foreign template copied VERBATIM:  %d  (%.1f%% of attempted)"
          % (null, 100.0*null/max(att,1)))
    print("MODEL GAIN OVER NULL: %+.1f points  <- the number that is evidence for the model"
          % (100.0*(exact-null)/max(att,1)))
    print("\n%-18s %8s %8s %8s %7s   %s"
          % ("family", "count", "attempt", "exact", "null", "status"))
    # Show every MODELLED family plus the largest unmodelled ones. Ranking by count alone hid
    # movimm.8 (63 instructions) entirely, and a family the model claims to cover must always be
    # visible - that is where a regression would show up first.
    ranked = sorted(stats.items(), key=lambda x: -x[1][0])
    shown = [kv for kv in ranked if kv[0] in MODELLED] + \
            [kv for kv in ranked if kv[0] not in MODELLED][:18]
    for fam, v in sorted(shown, key=lambda x: -x[1][0]):
        n, ex, at = v[0], v[1], v[2]
        # A family in MODELLED with zero attempts is a DEFECT, not an absence: the model claims
        # to cover it and every attempt threw, silently swallowed by the except around
        # try_encode. load.8 sat like that for 545 instructions because decode_load indexes past
        # the end of an 8-byte form. Displaying it as UNMODELLED hid a broken claim behind an
        # honest-looking label. ledger/g17-claimed-but-never-attempted.toml
        st = ("VACUOUS - every byte owned or keyed" if fam in vacuous else
              "%.1f%%" % (100.0*ex/at) if at else
              "CLAIMED BUT NEVER ATTEMPTED - model raises" if fam in MODELLED else "UNMODELLED")
        extra = ""
        if len(stats[fam]) > 3 and stats[fam][3] and at:
            extra = "   ignoring measured-inert bits: %.1f%%" % (100.0*stats[fam][3]/at)
        nl = stats[fam][4] if len(stats[fam]) > 4 else 0
        if at and nl == ex and ex:
            extra += "   NULL-EQUAL: the model adds nothing over copying the template"
        elif at and ex and nl:
            extra += "   (null %.1f%%, gain %+.1f)" % (100.0*nl/at, 100.0*(ex-nl)/at)
        print("%-18s %8d %8d %8d %7d   %s%s" % (fam, n, at, ex, nl, st, extra))


if __name__ == "__main__":
    import sys as _s
    if "--apple" in _s.argv: main(APPLE)
    elif "--both" in _s.argv:
        print("=== APPLE DRIVER SHADERS ONLY (the corpus the mission names) ===")
        main(APPLE)
        print("\n=== EVERYTHING, including my own probe kernels (flatters the result) ===")
        main()
    else: main()
