#!/usr/bin/env python3
"""Apple's decoder renumbered its opcodes and registers; translate them back to the ids this project uses.

Every opcode id in this repository (op684 end, op11842 movimm, op12674 the tensor load ...) is an index
into the instruction enum of the AGX3 LLVM inside GPUCompiler.framework, and every `reg:N` operand is an
MCRegister number from the same build. Those are TableGen artefacts, not encodings: rebuilding the LLVM
re-sorts the enum. macOS 27.0.1 (26A434) shipped such a rebuild. The instruction table grew from 17,796
to 17,905 entries and the register table from 3,568 to 3,686, and op684 `end` now decodes as 706.

The ENCODING DID NOT MOVE. Re-decoding all 6,594 programs of isa/g17-corpus-programs.jsonl (184,349
instructions) with the new decoder gives every instruction the offset and length recorded under the old
one, and each of the 557 old opcodes the corpus uses lands on one new opcode. So the repair is a
translation, applied inside the decoders (tools/agx3renumber.h, included by agx3dis.c and agx3dislib.c),
and every reader keeps the ids the machine model, the ledgers and the technical reference cite.

HOW THE TABLE IS BUILT, from committed old tables, a live dump of the new ones, and decoded records:

  registers  by NAME. Register names survived Apple's name stripping, every one of the 3,567 old names
             is present in the new table, and no name repeats. Exact. Register classes likewise by name
             (the new build dropped 11 synthesized classes and added 96).
  anchors    3,244 old opcodes whose bytes were decoded under the old build and are decoded again here:
             the corpus (isa/g17-corpus-programs.jsonl), every form witness in isa/g17-form-spec.jsonl
             and every opcode-bit flip of one, and the full decodes retained under results/. No old id
             decodes to two new ones (one merge, below), and every anchor's descriptor is unchanged.
  alignment  between consecutive anchors the two MCInstrDesc tables (isa/g17-agx3meta-instrs.txt and
             `agx3meta instrs`) are aligned as sequences: TableGen sorts the enum by record name, so the
             renumbering is monotone. Signatures are operand count, def count, flag words, implicit
             uses/defs and per-operand register class, translated by name, with the `_shifted` and
             `_alignedrc` suffixes dropped (the new build stopped distinguishing them in operand
             descriptors). Only pairs that the earliest- and the latest-matching LCS alignments both make
             are kept: inside a periodic run of identical descriptors a deletion has no fixed position, and
             old 11704-11706 aligned 32 slots off until that rule left them to their witnesses. Built from
             the corpus alone, this mapped 2,684 of 2,687 held-out form anchors right, none wrong, and left
             3 unmapped. 11,583 opcodes are mapped this way and 934 left unmapped as ambiguous.
  scheduling classes through the mapped opcodes, which pair them one-to-one (refused otherwise).

IMMEDIATE TAGS. The new decoder ORs a two-bit field into bits 29-30 of some immediates (memory and
tensor forms). Per (old opcode, operand, field value) a fitted XOR restores the old value: usually a
constant bit, but one family moves old bit 16 into the field. Fitted on the witnesses and the retained
decodes, tested on the operand-bit flips (35,441 match, 13 differ - field values the fit had not seen,
none contradicting it), then refitted on all. A field value with no entry keeps its bits, which that
slot's old values never have, so it reads as a mismatch rather than a plausible wrong value.

ONE MERGE. Old 447 and old 14156 (no glossary entry, used by nothing here) both decode as new 466,
which has a third operand, always 0. Its second immediate carries 0x1000 exactly for old 447. The
decoders read that bit, clear it and drop the extra operand.

WHAT HAS NO OLD ID is printed as UNMAPPED_BASE + the new id (1,000,000 + n) - opcodes, registers and
classes alike: numeric, so every parser still reads it, outside every old range, so nothing matches it
by accident. 919 old opcodes have no new counterpart at all (whole families were removed).

THE DECODERS DECIDE AT START-UP which table applies, by decoding canaries: corpus instructions whose
old and new ids differ. All old -> the original build, no translation. All new -> translate. Anything
else -> refuse, naming the canary. agx3meta decides by table size and prints its tables in old-id
order; the 26A434 build's translated tables are committed as isa/g17-agx3meta-<mode>-26A434.txt,
because some descriptors really changed (classes), and test_g17ccguards holds each build to its own.

    python3 tools/g17renumber.py --write     regenerate tools/agx3renumber.h and the JSON record
    python3 tools/g17renumber.py --check     header matches the inputs; the corpus, the form record and
                                             the retained decodes read back exactly through the decoder
"""
import argparse, collections, difflib, functools, hashlib, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ISA = os.path.join(ROOT, "isa")
HEADER = os.path.join(HERE, "agx3renumber.h")
RECORD = os.path.join(ISA, "g17-agx3-renumber.json")
CORPUS = os.path.join(ISA, "g17-corpus-programs.jsonl")
FORMS = os.path.join(ISA, "g17-form-spec.jsonl")
META = os.path.join(HERE, "agx3meta")
DIS = os.path.join(HERE, "agx3dis")
UNMAPPED_BASE = 1000000
MERGED_NEW, MERGED_HIGH, MERGED_LOW, MERGED_SPLIT = 466, 447, 14156, 0x1000
CANARIES = 8    # `end` and the most frequent corpus opcodes whose id moved


# THE 26A434 DECODER ALSO REJECTS THREE REPAIR-WALK WITNESSES: op9994 and op9995 (atomic.idx) and op9996
# (64-bit no-return atomic, max/min), authored by mutation. Each decodes as `bad` there, so on that build
# the decoder cannot address their forms and D1 counts three fewer opcodes. Apple's own op9996 still
# decodes. Tests that count D1 ask on_26a434() which population to expect.
REJECTED_BY_26A434 = [9994, 9995, 9996]


@functools.lru_cache(maxsize=1)
def on_26a434():
    """True when Apple's decoder answers in the macOS 27 (26A434) numbering."""
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        f.write(bytes.fromhex("0e000000"))      # op684, `end`, in the original numbering
        f.flush()
        out = subprocess.run([DIS, f.name, "0", "4"], capture_output=True, text=True, check=True,
                             env=dict(os.environ, AGX3_RAW="1")).stdout.split()
    return int(out[2]) != 684


def _names(lines):
    d = {}
    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.rstrip("\n").split(" ", 1)
        d[int(parts[0])] = parts[1].split(" ")[0] if len(parts) > 1 else ""
    return d


def _old(kind):
    with open(os.path.join(ISA, "g17-agx3meta-%s.txt" % kind)) as f:
        return f.read().splitlines()


def _new(kind):
    """The running build's own tables, in its own numbering."""
    return subprocess.run([META, kind], check=True, capture_output=True, text=True,
                          env=dict(os.environ, AGX3_RAW="1")).stdout.splitlines()


def _signatures(lines, classes, regs):
    out = []
    for line in lines:
        if line.startswith("#"):
            continue
        p = line.split()
        rn = lambda s: "-" if s == "-" else ",".join(regs.get(int(x), "?" + x) for x in s.split(","))
        ops = []
        for t in p[8:]:
            rc, ty, fl = t.split(":")
            name = classes.get(int(rc), rc) if int(rc) >= 0 else "-1"
            ops.append("%s:%s:%s" % (name.replace("_shifted", "").replace("_alignedrc", ""), ty, fl))
        out.append((int(p[0]), " ".join(p[1:3] + p[4:6] + ["u=" + rn(p[6].split("=")[1]),
                                                          "d=" + rn(p[7].split("=")[1])] + ops)))
    return out


def _batch_decode(items, raw=True):
    """Decode many (bytes, start, length) with one agx3dis process; one row list per item."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "manifest"), "w") as m:
            for i, (code, start, length) in enumerate(items):
                path = os.path.join(d, "%d.bin" % i)
                with open(path, "wb") as f:
                    f.write(code)
                m.write("%s %d %d %d\n" % (path, start, length, start))
        env = dict(os.environ, AGX3_RAW="1") if raw else {k: v for k, v in os.environ.items() if k != "AGX3_RAW"}
        out = subprocess.run([DIS, "--batch", os.path.join(d, "manifest")], check=True,
                             capture_output=True, text=True, env=env).stdout
    rows, cur = [[] for _ in items], None
    for line in out.splitlines():
        if line.startswith("=== end"):
            cur = None
        elif line.startswith("==="):
            cur = int(line.split()[1]) - 1
        elif cur is not None:
            p = line.split()
            if len(p) >= 3 and p[1] != "bad":
                rows[cur].append((int(p[0], 16), int(p[1]), int(p[2]), p[3:]))
    return rows


def _corpus():
    recs = [json.loads(line) for line in open(CORPUS)]
    items = []
    for r in recs:
        spans = r["spans"]
        items.append((bytes.fromhex(r["text"]), spans[0][0], spans[-1][0] + spans[-1][1] - spans[0][0]))
    return recs, items


def _form_pairs():
    """(bytes, old opcode, {operand: old value}) from isa/g17-form-spec.jsonl: each form's witness, with
    the operand values its bit records start from; each witness with one opcode-class bit flipped, which
    the record says decoded as `flips_to`; and each witness with one operand-class bit flipped, with the
    operand values the record says that flip produced."""
    pairs = []
    for line in open(FORMS):
        r = json.loads(line)
        if r.get("error"):
            continue
        w = bytes.fromhex(r["witness"])
        start = {}
        for b in r["bits"].values():
            for idx, _, frm, _ in b.get("moved") or []:
                start.setdefault(idx, frm)
        pairs.append((w, r["opcode"], start, "witness"))
        for k, b in sorted(r["bits"].items()):
            i, j = map(int, k.split("."))
            x = bytearray(w)
            x[i] ^= 1 << j
            if b.get("class") == "opcode" and isinstance(b.get("flips_to"), int):
                pairs.append((bytes(x), b["flips_to"], {}, "flip"))
            elif b.get("class") == "operand" and b.get("moved"):
                pairs.append((bytes(x), r["opcode"], {idx: to for idx, _, _, to in b["moved"]}, "flip"))
    return pairs


def _result_pairs():
    """(instruction bytes, old opcode, {operand: old immediate}, "result") from the full decodes retained
    beside Apple-compiled programs in results/ (*.instructions.json next to *.bin, extracted from the
    committed evidence archives by `make evidence`). These carry every operand token, so they test the
    translation end to end; --check compares them token for token."""
    out = []
    for f, b in _result_files():
        code = open(b, "rb").read()
        for r in _result_rows(f):
            want = {i: int(t[4:]) for i, t in enumerate(r["fields"]) if t.startswith("imm:")}
            out.append((code[r["offset"]:r["offset"] + r["length"]], r["opcode"], want, "result"))
    return out


def _result_files():
    import glob
    files = [(f, f[:-len(".instructions.json")] + ".bin")
             for f in sorted(glob.glob(os.path.join(ROOT, "results", "**", "*.instructions.json"), recursive=True))]
    files = [(f, b) for f, b in files if os.path.exists(b)]
    if not files:
        raise SystemExit("no retained decodes under results/: run `make evidence` first")
    return files


def _result_rows(f):
    rows = json.load(open(f))
    if rows and isinstance(rows[0], list):
        rows = [dict(offset=a, length=b, opcode=c, fields=d) for a, b, c, d in rows]
    return rows


def _lcs_greedy(A, B):
    """One maximum (LCS) alignment of A to B, matching as EARLY as possible: {i: j}."""
    m, n = len(A), len(B)
    L = [[0] * (n + 1) for _ in range(m + 1)]          # L[i][j] = LCS of A[i:], B[j:]
    for i in range(m - 1, -1, -1):
        Li, Ln, a = L[i], L[i + 1], A[i]
        for j in range(n - 1, -1, -1):
            Li[j] = Ln[j + 1] + 1 if a == B[j] else (Ln[j] if Ln[j] >= Li[j + 1] else Li[j + 1])
    out, i, j = {}, 0, 0
    while i < m and j < n:
        if A[i] == B[j] and L[i][j] == L[i + 1][j + 1] + 1:
            out[i] = j
            i, j = i + 1, j + 1
        elif L[i + 1][j] == L[i][j]:
            i += 1
        else:
            j += 1
    return out


def _align(sig_o, sig_n, anchors):
    """Align the descriptor tables between consecutive anchors, keeping only what EVERY maximum
    alignment agrees on. The earliest-matching and the latest-matching LCS alignments are computed
    (the latter on the reversed sequences); inside a periodic run of identical descriptors they put a
    deletion at opposite ends, so a pair both place identically is fixed and the rest is left unmapped,
    not guessed. Old 11704-11706 are why: a 32-entry deletion inside an 8-periodic run of atomic forms
    aligned 32 slots off, and only a decoded witness showed it."""
    aligned, ambiguous = {}, 0
    for (o1, n1), (o2, n2) in zip([(-1, -1)] + anchors, anchors + [(len(sig_o), len(sig_n))]):
        A, B = sig_o[o1 + 1:o2], sig_n[n1 + 1:n2]
        if not A or not B:
            continue
        early = _lcs_greedy(A, B)
        late = {len(A) - 1 - i: len(B) - 1 - j for i, j in _lcs_greedy(A[::-1], B[::-1]).items()}
        for k, v in early.items():
            if late.get(k) == v:
                aligned[o1 + 1 + k] = n1 + 1 + v
            else:
                ambiguous += 1
    return aligned, ambiguous


def _translate(op, opcode, toks):
    """An opcode and operand tokens from a RAW decode, in this repository's numbering (opcode only;
    the operand values are compared raw, so a tag shows as a difference)."""
    if opcode == MERGED_NEW:
        return MERGED_HIGH if len(toks) > 1 and toks[1].startswith("imm:") and int(toks[1][4:]) & MERGED_SPLIT \
            else MERGED_LOW
    return op[opcode] if opcode < len(op) else UNMAPPED_BASE + opcode


TAG_SHIFT, TAG_FIELD = 29, 3 << 29
CONFIRMED_TAGS = {(17258, 2, 1): 0, (17263, 2, 1): 0, (12343, 2, 1): 0}
SUSPECTED_TAGS = [(10023, 1), (10095, 2), (9996, 1)]


def _imm_samples(op, pairs, rows, kinds):
    """{(old opcode, operand): [(new value, old value)]} for immediates the record gives an old value for."""
    out = collections.defaultdict(list)
    for (code, old, want, kind), r in zip(pairs, rows):
        if kind not in kinds or not want or not r or r[0][1] != len(code) or _translate(op, r[0][2], r[0][3]) != old:
            continue
        toks = r[0][3]
        for idx, value in want.items():
            if idx < len(toks) and toks[idx].startswith("imm:"):
                out[(old, idx)].append((int(toks[idx][4:]), value))
    return out


def _fit_tags(samples):
    """Slots whose new value always carries bits 29-30 that the old value never has, and for each value
    of that two-bit field the XOR that restores the old value: {(opcode, operand, field): delta}.
    Most slots carry a constant bit (delta 0); one family moves old bit 16 into the field (bit 30 when
    it was clear, bit 29 when set). A field value mapping to two deltas is refused, not averaged."""
    table, conflicts = {}, []
    for slot, xs in samples.items():
        if not all(new & TAG_FIELD and not old & TAG_FIELD for new, old in xs):
            continue
        for new, old in xs:
            key = slot + ((new & TAG_FIELD) >> TAG_SHIFT,)
            delta = old ^ (new & ~TAG_FIELD)
            if table.setdefault(key, delta) != delta:
                conflicts.append((key, table[key], delta))
    return table, conflicts


def _untag(table, old, idx, new):
    if old == MERGED_HIGH and idx == 1:
        return new & ~MERGED_SPLIT
    d = table.get((old, idx, (new & TAG_FIELD) >> TAG_SHIFT))
    return new if d is None else (new & ~TAG_FIELD) ^ d


def _operand_tags(op, pairs, rows):
    """Fit the tag table on the witnesses and the retained full decodes, test it on the operand-bit flips
    it never saw, then refit on all. Every immediate a record gives an old value for must reappear once
    the tags are undone."""
    wit = _imm_samples(op, pairs, rows, ("witness", "result"))
    flips = _imm_samples(op, pairs, rows, ("flip",))
    fitted, c1 = _fit_tags(wit)
    held = dict(match=0, differ=0)
    for (old, idx), xs in flips.items():
        for new, value in xs:
            held["match" if _untag(fitted, old, idx, new) == value else "differ"] += 1
    both = collections.defaultdict(list)
    for d in (wit, flips):
        for k, xs in d.items():
            both[k] += xs
    table, c2 = _fit_tags(both)
    # ONE SLOT NO RECORD WITH OLD OPERAND VALUES COVERS, confirmed another way: op17258 operand 2
    # (a store; its sibling op17257's operand 2 is a fitted bit-29 tag on the same value 0x8f2).
    # tools/g17fieldmax.py's retained report was computed by the ORIGINAL decoder over the corpus;
    # with this slot untagged it rebuilds 900 bits recovered for the retained 898, and with it tagged
    # all three of its sections reproduce exactly. Three more corpus slots carry bit 29 in every
    # instance with no record either way - op17263 operand 2, op10023 operand 1, op10095 operand 2 -
    # and that report does not depend on them, so they are NOT tagged (SUSPECTED_TAGS, recorded).
    # TWO MORE FROM isa/g17-form-bases.json, the field bases g17bases harvested by decoding each form's
    # witness under the original decoder: op17263 and op12343 re-derive operand 2 as the checked-in
    # base plus exactly 0x20000000 (242 vs 0x200000f2, 2161 vs 0x20000871), and with the tag undone all
    # 2,384 forms agree. op17258 agrees there too. op10023 operand 1, op10095 operand 2 and op9996
    # (Apple emits it for 64-bit atomic_max: operand 1 reads 0x20040206) have no base entry, so they
    # stay suspected.
    for key, delta in CONFIRMED_TAGS.items():
        table.setdefault(key, delta)
    if c1 or c2:
        raise SystemExit("a tag field maps to two deltas: %s" % (c1 + c2)[:5])
    final = sum(1 for (o, i), xs in both.items() for n, v in xs if _untag(table, o, i, n) != v)
    return table, dict(tag_entries=len(table),
                       flipped_immediates_matching_with_witness_fit=held["match"],
                       flipped_immediates_differing_with_witness_fit=held["differ"],
                       immediates_differing_after_full_fit=final)


def build(sources=("corpus", "forms", "results")):
    oc, nc = _names(_old("classes")), _names(_new("classes"))
    orr, nr = _names(_old("regs")), _names(_new("regs"))
    by_name = {v: k for k, v in orr.items()}
    if len(by_name) != len(orr) or len(set(nr.values())) != len(nr):
        raise SystemExit("register names repeat; the by-name register map is not defined")
    missing = sorted(set(by_name) - set(nr.values()))
    if missing:
        raise SystemExit("old registers absent from the new table: %s" % missing[:10])
    reg = {n: by_name.get(name, UNMAPPED_BASE + n) for n, name in nr.items()}

    O = _signatures(_old("instrs"), oc, orr)
    N = _signatures(_new("instrs"), nc, nr)

    # THE CORPUS: Apple-compiled programs with their recorded decode. Framing must be unchanged.
    recs, items = _corpus()
    decoded = collections.defaultdict(collections.Counter)
    counts = collections.Counter()
    framing = total = 0
    samples = {}
    for r, (code, _, _), rows in zip(recs, items, _batch_decode(items)):
        got = {o: (s, op) for o, s, op, _ in rows}
        for o, s, op in r["spans"]:
            total += 1
            if o not in got or got[o][0] != s:
                framing += 1
                continue
            counts[op] += 1
            if "corpus" in sources:
                decoded[op][got[o][1]] += 1
            samples.setdefault(op, code[o:o + s])
    if framing:
        raise SystemExit("%d of %d corpus instructions changed offset or length: the encoding moved, "
                         "and no renumbering repairs that" % (framing, total))
    from_corpus = {o: decoded[o].most_common(1)[0][0] for o in counts if "corpus" in sources}
    # THE FORM RECORD: witnesses and opcode-bit flips, decoded as recorded (first instruction).
    form_ids = set()
    pairs = _form_pairs() + _result_pairs()
    form_rows = _batch_decode([(c, 0, len(c)) for c, _, _, _ in pairs])
    for (code, old, _, kind), rows in zip(pairs, form_rows):
        if rows and rows[0][0] == 0 and ("forms" if kind != "result" else "results") in sources:
            decoded[old][rows[0][2]] += 1
            form_ids.add(old)
    split = {o: dict(c) for o, c in decoded.items() if len(c) > 1}
    if split:
        raise SystemExit("old opcodes decoding to several new ones: %s" % list(split.items())[:10])
    witnessed = {o: c.most_common(1)[0][0] for o, c in decoded.items()}

    # ANCHORED ALIGNMENT. Each decoded pair is a fixed point; the descriptor tables are aligned only
    # BETWEEN consecutive anchors, so an alignment can never cross a decoded fact. The merged pair is
    # not an anchor: old 14156 moved out of order, to 466.
    anchors = sorted((o, n) for o, n in witnessed.items() if o not in (MERGED_HIGH, MERGED_LOW))
    if any(b[1] <= a[1] for a, b in zip(anchors, anchors[1:])):
        raise SystemExit("the decoded renumbering is not monotone outside the known merge")
    sig_o, sig_n = [s for _, s in O], [s for _, s in N]
    sigdiff = [o for o, n in anchors if sig_o[o] != sig_n[n]]
    aligned, ambiguous = _align(sig_o, sig_n, anchors)

    op_old = dict(aligned)
    op_old.update(witnessed)
    new2old = {}
    for o, n in op_old.items():
        new2old.setdefault(n, set()).add(o)
    merged = {n: sorted(s) for n, s in new2old.items() if len(s) > 1}
    if merged and merged != {MERGED_NEW: [MERGED_HIGH, MERGED_LOW]}:
        raise SystemExit("unexpected opcode merges %s; only %d <- (%d, %d) has a rule"
                         % (merged, MERGED_NEW, MERGED_HIGH, MERGED_LOW))
    n_new = len(N)
    op = [UNMAPPED_BASE + n for n in range(n_new)]
    for n, s in new2old.items():
        op[n] = min(s)                      # 466 is resolved by operand in the decoders
    tags, tag_check = _operand_tags(op, pairs, form_rows)
    # REGISTER CLASSES by name (the new build dropped some synthesized classes and added others);
    # SCHEDULING CLASSES through the mapped opcodes, which pair them one-to-one or are refused.
    cls_by_name = {v: k for k, v in oc.items()}
    cls = [cls_by_name.get(nc[n], UNMAPPED_BASE + n) for n in range(len(nc))]
    so = {int(l.split()[0]): int(l.split()[3]) for l in _old("instrs") if not l.startswith("#")}
    sn = {int(l.split()[0]): int(l.split()[3]) for l in _new("instrs") if not l.startswith("#")}
    spairs = collections.defaultdict(set)
    for n, o in enumerate(op):
        if o < UNMAPPED_BASE:
            spairs[sn[n]].add(so[o])
    sinv = collections.defaultdict(set)
    for k, v in spairs.items():
        for x in v:
            sinv[x].add(k)
    if any(len(v) > 1 for v in spairs.values()) or any(len(v) > 1 for v in sinv.values()):
        raise SystemExit("scheduling classes do not pair one-to-one through the mapped opcodes")
    n_sched = max(sn.values()) + 1
    sched = [next(iter(spairs[k])) if k in spairs else UNMAPPED_BASE + k for k in range(n_sched)]
    canaries = []
    if "corpus" in sources:
        moved = sorted((o for o, n in from_corpus.items() if o != n and o not in (MERGED_HIGH, MERGED_LOW)),
                       key=lambda o: (-counts[o], o))
        picks = [684] + [o for o in moved if o != 684][:CANARIES - 1]
        canaries = [dict(old=o, new=from_corpus[o], hex=samples[o].hex()) for o in picks]
    return dict(op=op, reg=[reg[n] for n in range(len(nr))], cls=cls, sched=sched, old_counts=(len(O), len(orr)),
                canaries=canaries, witnessed=witnessed, tags=tags,
                stats=dict(old_opcodes=len(O), new_opcodes=n_new, old_registers=len(orr),
                           new_registers=len(nr), corpus_instructions=total,
                           witnessed_opcodes=len(witnessed), from_corpus=len(from_corpus),
                           from_forms_only=len(form_ids - set(from_corpus)),
                           anchors_whose_descriptor_changed=len(sigdiff),
                           aligned_by_descriptor=len(set(aligned) - set(witnessed)),
                           left_unmapped_as_ambiguous=ambiguous,
                           old_without_new=len(O) - len(op_old),
                           new_without_old=sum(1 for v in op if v >= UNMAPPED_BASE),
                           registers_without_old=sum(1 for v in reg.values() if v >= UNMAPPED_BASE),
                           classes_without_old=sum(1 for v in cls if v >= UNMAPPED_BASE),
                           old_classes_absent=len(oc) - sum(1 for v in cls if v < UNMAPPED_BASE),
                           sched_classes_without_old=sum(1 for v in sched if v >= UNMAPPED_BASE),
                           **tag_check))


def header(t):
    rows = lambda xs: ",\n".join("  " + ", ".join(str(v) for v in xs[i:i + 16]) for i in range(0, len(xs), 16))
    can = ",\n".join('  {"%s", %d, %d}' % (c["hex"], c["old"], c["new"]) for c in t["canaries"])
    return """// GENERATED by tools/g17renumber.py --write; do not edit. See that file for how this was built.
// Translates the opcode and register numbering of the macOS 27 (26A434) AGX3 decoder back to the ids
// this repository uses (the original 17,796-opcode build). UNMAPPED_BASE + n marks a new id with no old one.
#define AGX3_UNMAPPED_BASE %d
#define AGX3_NEW_OPCODES %d
#define AGX3_NEW_REGISTERS %d
#define AGX3_MERGED_NEW %d
#define AGX3_MERGED_HIGH %d
#define AGX3_MERGED_LOW %d
#define AGX3_MERGED_SPLIT 0x%x
static const int AGX3_OP_NEW2OLD[AGX3_NEW_OPCODES] = {
%s
};
static const int AGX3_REG_NEW2OLD[AGX3_NEW_REGISTERS] = {
%s
};
#define AGX3_OLD_OPCODES %d
#define AGX3_OLD_REGISTERS %d
#define AGX3_NEW_CLASSES %d
#define AGX3_NEW_SCHED %d
static const int AGX3_CLASS_NEW2OLD[AGX3_NEW_CLASSES] = {
%s
};
static const int AGX3_SCHED_NEW2OLD[AGX3_NEW_SCHED] = {
%s
};
struct agx3_canary { const char *hex; int old_id; int new_id; };
static const struct agx3_canary AGX3_CANARIES[] = {
%s
};
// immediates the new decoder tags in bits 29-30: (old opcode, operand, field value) -> XOR restoring the old value
#define AGX3_TAG_SHIFT %d
#define AGX3_TAG_FIELD 0x%xLL
struct agx3_tag { int op; int idx; int field; long long delta; };
static const struct agx3_tag AGX3_TAGS[] = {
%s
};
""" % (UNMAPPED_BASE, len(t["op"]), len(t["reg"]), MERGED_NEW, MERGED_HIGH, MERGED_LOW, MERGED_SPLIT,
       rows(t["op"]), rows(t["reg"]), t["old_counts"][0], t["old_counts"][1], len(t["cls"]), len(t["sched"]),
       rows(t["cls"]), rows(t["sched"]), can, TAG_SHIFT, TAG_FIELD,
       ",\n".join("  {%d, %d, %d, 0x%x}" % (o, i, f, d) for (o, i, f), d in sorted(t["tags"].items())))


def check(t):
    problems = []
    with open(HEADER) as f:
        if f.read() != header(t):
            problems.append("tools/agx3renumber.h is stale: run tools/g17renumber.py --write")
    # THE TRANSLATED DECODER MUST REPRODUCE THE RECORDED CORPUS, opcode for opcode
    bad = total = 0
    for line in open(CORPUS):
        r = json.loads(line)
        code, spans = bytes.fromhex(r["text"]), r["spans"]
        a, b = spans[0][0], spans[-1][0] + spans[-1][1]
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            f.write(code)
            f.flush()
            out = subprocess.run([DIS, f.name, str(a), str(b - a), "--pc", str(a)],
                                 capture_output=True, text=True).stdout
        got = [(int(p[0], 16), int(p[1]), int(p[2])) for p in (l.split() for l in out.splitlines())
               if len(p) >= 3 and p[1] != "bad"]
        total += len(spans)
        bad += sum(1 for s, g in zip(spans, got) if tuple(s) != g) + abs(len(spans) - len(got))
    if bad:
        problems.append("%d of %d corpus instructions decode differently from the record" % (bad, total))
    print("corpus: %d instructions, %d differ from the recorded decode" % (total, bad))
    # EVERY OPERAND VALUE THE FORM RECORD GIVES, through the TRANSLATING decoder: opcode, registers and
    # immediates must all read as recorded (expressions are allocator addresses without --expr).
    pairs = _form_pairs()
    rows = _batch_decode([(c, 0, len(c)) for c, _, _, _ in pairs], raw=False)
    stat = collections.Counter()
    for (code, old, want, kind), r in zip(pairs, rows):
        if not r or r[0][1] != len(code):
            stat["length changed by the flip"] += 1
            continue
        if r[0][2] != old:
            stat["opcode differs"] += 1
            continue
        for idx, value in want.items():
            toks = r[0][3]
            k, _, v = toks[idx].partition(":") if idx < len(toks) else ("missing", "", "")
            if k in ("reg", "imm"):
                stat["%s %s" % (k, "match" if int(v) == value else "DIFFER")] += 1
    print("form record through the translating decoder:", dict(sorted(stat.items())))
    # THE RETAINED FULL DECODES, token for token with --expr (expression constants included)
    same = differ = 0
    for f, b in _result_files():
        rec = _result_rows(f)
        if not rec:
            continue
        start, end = rec[0]["offset"], rec[-1]["offset"] + rec[-1]["length"]
        out = subprocess.run([DIS, b, str(start), str(end - start), "--pc", str(start), "--expr"],
                             capture_output=True, text=True, env={k: v for k, v in os.environ.items()
                                                                  if k != "AGX3_RAW"}).stdout
        got = {int(p[0], 16): (int(p[1]), int(p[2]), p[3:]) for p in (l.split() for l in out.splitlines())
               if len(p) >= 3 and p[1] != "bad"}
        for r in rec:
            if got.get(r["offset"]) == (r["length"], r["opcode"], r["fields"]):
                same += 1
            else:
                differ += 1
    print("retained full decodes: %d instructions identical, %d differ" % (same, differ))
    if differ:
        problems.append("%d retained instructions under results/ decode differently from their record" % differ)
    if stat["opcode differs"] or stat["reg DIFFER"] or stat["imm DIFFER"]:
        problems.append("the translating decoder disagrees with isa/g17-form-spec.jsonl: %s" % dict(stat))
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    t = build()
    print(json.dumps(t["stats"]))
    if a.write:
        with open(HEADER, "w") as f:
            f.write(header(t))
        rec = dict(stats=t["stats"], canaries=t["canaries"], unmapped_base=UNMAPPED_BASE,
                   confirmed_tags_from_retained_reports=[list(k) for k in CONFIRMED_TAGS],
                   suspected_tags_not_applied=[list(k) for k in SUSPECTED_TAGS],
                   merged=dict(new=MERGED_NEW, high=MERGED_HIGH, low=MERGED_LOW, split=MERGED_SPLIT),
                   header_sha256=hashlib.sha256(header(t).encode()).hexdigest(),
                   platform=subprocess.run(["sw_vers", "-buildVersion"], capture_output=True,
                                           text=True).stdout.strip())
        with open(RECORD, "w") as f:
            json.dump(rec, f, indent=1)
            f.write("\n")
        print("wrote", os.path.relpath(HEADER, ROOT), "and", os.path.relpath(RECORD, ROOT))
    if a.check:
        problems = check(t)
        for p in problems:
            print("FAIL:", p)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
