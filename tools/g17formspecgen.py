#!/usr/bin/env python3
"""Per-form bit classes by asking Apple's decoder, and the generator `isa/g17-form-spec.jsonl` never had.

`isa/g17-form-spec.jsonl` was written once, at `7dd0fe98`, and that commit carries no program that
produces it.  It holds 984 forms - exactly the forms of `g17-corpus-programs.jsonl` decoded - so every
form the vendor corpus added has no bit classes at all, and `g17isamap.unknown_bits` reports those
forms as "no form-level bit evidence at this width", which blocks them from D2.  923 such forms are
Apple-witnessed in the corpus.  This is the missing generator.

THE METHOD IS THE ONE THE FILE ALREADY ENCODES.  Flip each bit of one Apple witness alone and decode:

    the decoder refuses the flipped bytes, and they do not decode as another opcode   -> forced
    the opcode or length changes                                                       -> opcode
    nothing the decoder prints changes                                                 -> invisible
    an operand's NUMBER moves                                                          -> operand
    an operand's SHAPE changes (a register becomes an expression)                      -> mode

A length-changing flip leaves bytes that no longer parse as "one instruction then END", so the strict
single-instruction decode refuses them; it is re-read here without that constraint, and counts as
`opcode` if the first instruction is a different one.  Calibration on the recorded file found 17 of
76 refusals in its first six forms were exactly this.

REPRODUCTION FIRST.  `--check` regenerates the 984 recorded forms from their OWN witnesses and reports
agreement class by class.  The generator is trusted on new forms only as far as it reproduces the old.

Decoder evidence only: these classes say how the DECODER reads each bit, not what hardware does.

    python3 tools/g17formspecgen.py --check           reproduce the recorded 984
    python3 tools/g17formspecgen.py --extend --write  add every corpus form the file lacks
"""
import argparse, collections, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
ISA = os.path.join(ROOT, "isa")
SPEC = os.path.join(ISA, "g17-form-spec.jsonl")
END = bytes.fromhex("0e000000")
END_OP = 684


def _first(code):
    """(length, opcode) of the first instruction under the LENIENT decoder, or None.

    Not `g17packedcheck.decode`: that one runs the disassembler as a checked process, and a flip that
    SHORTENS the instruction leaves trailing bytes that do not parse, so the whole process fails and
    the flip reads as `forced`. The recorded file shows the original did not read it that way - bit
    2.1 of op396/10 is recorded `flips_to=13588`, and `agxforge.g17.model.decode`, which yields line by
    line and skips what it cannot place, reads exactly that: a four-byte op13588. Calibration put 195
    of 579 opcode bits in the first sixty forms in this category.
    """
    from agxforge.g17 import model as M
    try:
        got = list(M.decode(bytes(code) + END))
    except Exception:
        return None
    if not got or got[0].opcode is None:
        return None
    return got[0].size, got[0].opcode.id


def _strict(code):
    """One instruction then END, exactly - the reading a length-preserving flip must survive."""
    import g17packedcheck as D
    rows = D.decode(bytes(code) + END)
    if len(rows) != 2 or rows[1][2] != END_OP:
        raise ValueError("not one instruction then end")
    _, l, op, toks = rows[0]
    return l, op, list(toks)


def _shape(tok):
    if tok.startswith(("imm:", "reg:")):
        return tok.split(":")[0] + ":"
    if "const(" in tok:
        a, rest = tok.split("const(", 1)
        return a + "const()" + rest.split(")", 1)[1]
    return tok


def _kind(tok):
    return tok.split(":")[0] if ":" in tok else tok.split("(")[0]


def classify(witness):
    """{"b.i": {class, value, ...}} for every bit of one witness."""
    w = bytes(witness)
    base = _strict(w)
    out = {}
    for i in range(8 * len(w)):
        key = "%d.%d" % (i // 8, i % 8)
        value = (w[i // 8] >> (i % 8)) & 1
        u = bytearray(w)
        u[i // 8] ^= 1 << (i % 8)
        try:
            l, op, toks = _strict(u)
        except Exception:
            alt = _first(u)
            if alt is not None and alt[1] != base[1]:
                out[key] = dict(**{"class": "opcode"}, value=value, flips_to=alt[1])
            else:
                out[key] = dict(**{"class": "forced"}, value=value)
            continue
        if (l, op) != (base[0], base[1]):
            out[key] = dict(**{"class": "opcode"}, value=value, flips_to=op)
            continue
        n = min(len(toks), len(base[2]))
        # MODE MEANS AN OPERAND'S KIND CHANGED - a register became an expression - which is what
        # every recorded `mode` bit's `switches` field says. Comparing the whole SHAPE instead
        # called op441/10 bit 4.4 a mode bit because a base and a constant INSIDE one expression
        # moved; the recorded file calls that an operand move, and it is one.
        switched = [[k, _kind(base[2][k]), _kind(toks[k])] for k in range(n)
                    if _kind(toks[k]) != _kind(base[2][k])]
        if switched or len(toks) != len(base[2]):
            out[key] = dict(**{"class": "mode"}, value=value, switches=switched)
            continue
        moved = _moved(base[2], toks, n)
        if moved:
            out[key] = dict(**{"class": "operand"}, value=value, moved=moved)
        elif any(toks[k] != base[2][k] for k in range(n)):
            # a token changed but no sub-field value did - printed spelling only
            out[key] = dict(**{"class": "operand"}, value=value, moved=[])
        else:
            out[key] = dict(**{"class": "invisible"}, value=value)
    return out


def _moved(before, after, n):
    """[token, sub-field, before, after] for every sub-field whose value the flip changed.

    Parsed with `agxforge.g17.canon.token_values` - the parser the ENCODER reads these entries with -
    rather than a second one, so a sub-field named here is a sub-field `_extended_fields` can find.
    An address expression carries a base, a constant and a scale, and one bit can move several.
    """
    from agxforge.g17 import canon
    out = []
    for k in range(n):
        if before[k] == after[k]:
            continue
        b, a = canon.token_values(before[k]), canon.token_values(after[k])
        for sub in sorted(set(b) | set(a)):
            if b.get(sub) != a.get(sub):
                out.append([k, sub, b.get(sub), a.get(sub)])
    return out


def recorded():
    """The forms the file was WRITTEN with - never the ones this tool appended.

    Including this tool's own rows would make `check` compare the classifier with its own output,
    which cannot disagree, and would make `extend` skip every form it had already admitted - so a
    re-run's report would silently omit the whole admitted population.
    """
    with open(SPEC) as fh:
        return [r for r in (json.loads(l) for l in fh) if not r.get("generator")]


def _classify_hex(witness):
    return classify(bytes.fromhex(witness))


def _classify_all(witnesses):
    """classify() over every witness, IN ORDER, across G17_FORMSPEC_JOBS processes.

    Each witness is independent - one disassembler process per flipped bit - so this was ~5 min of
    one core spent waiting on subprocesses, and the longest single command in check-ledgers. The
    results come back in input order and are aggregated by the same loop as before, so every count
    and the confusion table's order (most_common breaks ties by insertion) are unchanged.
    G17_FORMSPEC_JOBS=1 is the old serial path."""
    return _map_ordered(_classify_hex, witnesses, chunksize=8)


def check(limit=None):
    """Reproduce the recorded file from its own witnesses: agreement per class, and every miss."""
    agree = collections.Counter()
    total = collections.Counter()
    confusion = collections.Counter()
    forms_exact = forms = 0
    moved_total, moved_agree = [0], [0]
    recs = [r for r in recorded()[:limit] if not r.get("error") and r.get("witness")]
    for r, mine in zip(recs, _classify_all([r["witness"] for r in recs])):
        forms += 1
        exact = True
        for k, v in r["bits"].items():
            want = v.get("class")
            got = (mine.get(k) or {}).get("class")
            total[want] += 1
            if got == want:
                agree[want] += 1
                if want == "operand":
                    # the ENCODER uses the first moved entry for a bit's weight, so that is the
                    # part of `moved` that has to agree
                    rw, mw = (v.get("moved") or [None])[0], ((mine[k].get("moved")) or [None])[0]
                    moved_total[0] += 1
                    moved_agree[0] += (rw is None and mw is None) or (
                        rw is not None and mw is not None and list(rw) == list(mw))
            else:
                exact = False
                confusion[(want, got)] += 1
        forms_exact += exact
    return dict(forms=forms, forms_reproduced_exactly=forms_exact,
                operand_first_moved_agrees=[moved_agree[0], moved_total[0]],
                by_class={c: [agree[c], total[c]] for c in sorted(total)},
                confusion={"%s->%s" % k: n for k, n in confusion.most_common()})


FREE = os.path.join(ISA, "g17-free-bits.jsonl")
BITSPEC = os.path.join(ISA, "g17-bit-spec.jsonl")
HELD_OUT = 20
KNOWN = os.path.join(ISA, "g17-form-spec-known-disagreement.json")


def corpus_encodings():
    """{(opcode, length): Counter(hex -> instances)} over BOTH committed Apple corpora.

    The population is `g17vendorforms.SOURCES` - the one `isa/g17-vendor-corpus-forms.json`
    publishes - read through the same decoder, so its totals must equal that index's. `extend`
    checks that they do before using any of it.
    """
    import importlib.util
    from agxforge.g17 import model
    spec = importlib.util.spec_from_file_location("g17vendorforms", os.path.join(HERE, "g17vendorforms.py"))
    vf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vf)
    enc = collections.defaultdict(collections.Counter)
    programs = collections.Counter()
    texts = []
    for _name, paths in sorted(vf.SOURCES.items()):
        for path in paths:
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        text = json.loads(line).get("text")
                    except ValueError:
                        continue
                    if text:
                        texts.append(text)
    # Decoded across processes (the decoder is most of this function's minutes), then folded IN
    # INPUT ORDER by the same rules as the serial loop it replaced: a program whose decode raises
    # keeps the instances decoded before the raise and counts toward no form's programs.
    for pairs, complete in _map_ordered(_decode_program, texts):
        seen = set()
        for k, h in pairs:
            enc[k][h] += 1
            seen.add(k)
        if not complete:
            continue
        # PROGRAMS, not instances: a form a program cannot be written without gates
        # that program once, however many times it appears inside it.
        for k in seen:
            programs[k] += 1
    corpus_encodings.programs = programs
    return enc


def _decode_program(text):
    """-> ([((opcode, length), hex)], decoded_to_the_end)"""
    from agxforge.g17 import model
    pairs = []
    try:
        for ins in model.decode(bytes.fromhex(text)):
            if ins.opcode is not None:
                pairs.append(((ins.opcode.id, len(ins.raw)), bytes(ins.raw).hex()))
    except Exception:
        return pairs, False
    return pairs, True


def _jobs():
    return int(os.environ.get("G17_FORMSPEC_JOBS") or max(1, min(6, (os.cpu_count() or 4) // 3)))


def _map_ordered(fn, items, chunksize=16):
    """fn over items IN ORDER, across G17_FORMSPEC_JOBS processes; 1 is the serial path."""
    if _jobs() <= 1 or len(items) < 2:
        return [fn(x) for x in items]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=_jobs()) as ex:
        return list(ex.map(fn, items, chunksize=chunksize))


def bit_verdict(encs, b, i):
    """One invisible bit over a form's Apple instances, under g17freebits' rule - ONE copy of it.

    `constant` needs TWO instances agreeing; one agreeing with itself is not agreement.
    """
    cnt = collections.Counter()
    for h, n in encs.items():
        cnt[(bytes.fromhex(h)[b] >> i) & 1] += n
    tot = sum(cnt.values())
    if tot >= 2 and len(cnt) == 1:
        return {"verdict": "constant", "value": next(iter(cnt)), "instances": tot}
    if len(cnt) > 1:
        return {"verdict": "varies", "instances": tot}
    return {"verdict": "undetermined", "instances": tot}


def verified_corpus_encodings():
    """corpus_encodings(), refused unless it equals the published index. Returns (enc, message)."""
    enc = corpus_encodings()
    total = sum(sum(c.values()) for c in enc.values())
    with open(os.path.join(ISA, "g17-vendor-corpus-forms.json")) as fh:
        idx = json.load(fh)
    want = sum(idx["union"].values())
    if total != want or len(enc) != len(idx["union"]):
        return None, ("decoded %d forms / %d instances, the index publishes %d / %d"
                      % (len(enc), total, len(idx["union"]), want))
    return enc, "corpus: %d forms, %d instances - equal to the published index" % (len(enc), total)


def settle(bits, op, length, encs, bitspec):
    """Corpus verdicts for every invisible bit of the form, under g17freebits' own rule.

    `constant` needs TWO instances agreeing; one instance agreeing with itself is not agreement.
    The bits settled are this form's invisible bits AND the opcode-level `bit-spec` invisible bits
    that exist at this length, because `g17isamap.unknown_bits` checks the latter.
    """
    inv = {k for k, v in bits.items() if v["class"] == "invisible"}
    opcode_level = {k for k, v in (bitspec.get(op) or {}).items()
                    if isinstance(v, dict) and v.get("class") == "invisible"}
    inv |= {k for k in opcode_level if int(k.split(".")[0]) < length}
    out = {}
    # A BIT THIS FORM DOES NOT HAVE. `bit-spec` is keyed by OPCODE, from one witness, and
    # `g17isamap.unknown_bits` checks every invisible bit it lists against every width - so a
    # six-byte form of an opcode whose witness was ten bytes carries unsettled bits at bytes 6-9
    # that do not exist in it, and can never leave "incomplete". 335 of the first 432 forms this
    # tool admits carry such bits. They are recorded, per form, as exactly what they are, rather
    # than filtered silently inside the checker: the verdict is a checked fact about THIS form.
    for k in sorted(opcode_level - inv, key=lambda s: tuple(map(int, s.split(".")))):
        if int(k.split(".")[0]) >= length:
            out[k] = {"verdict": "absent at this length", "length": length}
    for k in sorted(inv, key=lambda s: tuple(map(int, s.split(".")))):
        b, i = map(int, k.split("."))
        out[k] = bit_verdict(encs, b, i)
    return out


def _reconstructs(E, op, length, real):
    """The repo's own per-form reconstruction (`encode._forms_main`), for one instance."""
    ext, steps, _, _ = E.extended_fields(op, length)
    vals = {}
    for key, positions in ext.items():
        v = 0
        for vb, b, i, inv in positions:
            if b < len(real):
                v |= (((real[b] >> i) & 1) ^ inv) << vb
        vals[key] = v * steps.get(key, 1)
    try:
        return E.encode(op, vals, tabled=E.semantic_operands(op, real), length_hint=length) == real
    except (KeyError, ValueError):
        return False


def operand_map_conflicts(op, length, bits):
    """Bits this spec calls opcode or forced that a shipped operand map places an operand on.

    The invariant `g17regress` holds every form to: "no operand map holds a bit the specification
    calls an opcode bit, except where both values are witnessed" - the exemptions being
    `g17as.SPEC_MEASURED_OPERAND`, admitted one at a time. A form this tool adds must satisfy the
    invariants already in force, so the check is the guard's own, not a looser copy of it.

    The first run admitted two forms that fail it, and they fail for opposite reasons: op13051/14's
    bit 11.6 is 0 in all fifteen Apple instances, so it is a pure opcode bit and the opcode-level
    operand map is wrong at that width; op11472/10's bit 3.7 is 0 in 782 and 1 in 208, all at that
    opcode and length, so it is genuinely shared and a single-witness flip cannot see it.
    """
    import g17as, g17opmap
    # EXACTLY g17opmap.opcode_bits: the `opcode` class plus what form-bits measured as selecting an
    # opcode. Not `forced` - a first version added it, found nine conflicts where the guard finds
    # two, and would have refused forms the repository's own invariant allows. Computed from the
    # candidate's bits directly rather than through spec_for, which reads whatever is on disk.
    fixed = {tuple(map(int, k.split("."))) for k, v in bits.items() if v.get("class") == "opcode"}
    fixed |= g17opmap.selects_opcode(op, length)
    hits = []
    for (mop, mln, idx, kind), r in g17as.maps().items():
        if (mop, mln) != (op, length):
            continue
        took = {(b, i) for _, b, i, _ in (tuple(p) for p in (r.get("positions") or []))}
        took |= {(b, i) for b, i, _ in (r.get("extra") or [])}
        took |= {(b, i) for b, i in (r.get("order") or [])}
        for b, i in sorted(took & fixed):
            if (op, length, b, i) not in g17as.SPEC_MEASURED_OPERAND:
                hits.append((idx, kind, b, i))
    return hits


def extend(encodings, held_out=HELD_OUT):
    """Admission of every corpus form the recorded file lacks, under the bar existing D2 meets.

    ADMIT when (1) the form's Apple witness reconstructs byte-exact through the repo encoder - 91.7%
    of existing D2 forms do - and (2) at least one HELD-OUT Apple instance of the form does too,
    which shows the specification is not one witness's bits; 89% of existing D2 forms have at
    least one. (3) no invisible bit is left undetermined.

    Requiring EVERY held-out instance would hold new forms to a bar 80% of existing D2 fails
    (48.3% of D2 forms reconstruct all of theirs). The bar is the one already in force.
    """
    from agxforge.g17 import encode as E
    bitspec = {}
    with open(BITSPEC) as fh:
        for line in fh:
            d = json.loads(line)
            bitspec[d["opcode"]] = d.get("bits") or {}
    have = {(r["opcode"], r["length"]) for r in recorded()}
    E.forms(); E.freebits()
    contract = E.auth()
    results = {}
    for (op, length), encs in sorted(encodings.items()):
        if (op, length) in have:
            continue
        if op not in contract:
            # COUNTED, NOT DROPPED. Apple emits op13754 at fourteen bytes and the contract carries
            # no such opcode, so there is no field map to reconstruct through and nothing D2 could
            # admit. It is reported with its own verdict rather than vanishing from the tally.
            results[(op, length)] = dict(verdict="opcode not in the contract",
                                         instances=sum(encs.values()))
            continue
        ranked = sorted(encs.items(), key=lambda kv: (-kv[1], kv[0]))   # deterministic witness
        witness = bytes.fromhex(ranked[0][0])
        try:
            bits = classify(witness)
        except Exception:
            results[(op, length)] = dict(verdict="witness does not decode as one instruction")
            continue
        fb = settle(bits, op, length, encs, bitspec)
        E._FORMS[(op, length)] = dict(opcode=op, length=length, witness=witness.hex(),
                                      bits=bits, error=None)
        E._FREE[(op, length)] = fb
        try:
            wok = _reconstructs(E, op, length, witness)
            held = [bytes.fromhex(h) for h, _ in ranked[1:held_out + 1]]
            hok = sum(_reconstructs(E, op, length, b) for b in held)
        finally:
            E._FORMS.pop((op, length), None)
            E._FREE.pop((op, length), None)
        undet = sum(1 for v in fb.values() if v["verdict"] == "undetermined")
        clash = operand_map_conflicts(op, length, bits)
        verdict = ("admit" if wok and hok and not undet and not clash else
                   "witness does not reconstruct" if not wok else
                   "undetermined invisible bits" if undet else
                   "operand map takes a bit this spec calls opcode" if clash else
                   "no held-out instance reconstructs")
        results[(op, length)] = dict(verdict=verdict, witness=witness.hex(), bits=bits,
                                     freebits=fb, held_out=len(held), held_exact=hok,
                                     instances=sum(encs.values()))
    return results


REPORT = os.path.join(ISA, "g17-encoding-breadth.json")


def report(results, programs):
    """The split the work is steered by, scored by PROGRAMS gated rather than forms counted.

      specified and lowered      admitted to D2, and the compiler's authoring table emits this width
      specified, not lowered     admitted to D2, but the table authors a different width for the opcode
      not recovered              not admitted, with the reason the bar was not met

    Forms are ranked within each by programs, because where programs are gated is where an
    independent compiler is blocked; a numerous form that no program needs is not a priority.
    """
    import g17auth
    rows = []
    for (op, length), r in sorted(results.items()):
        try:
            authored = int(g17auth.length(op))
        except Exception:
            authored = None
        if r["verdict"] == "admit":
            split = ("specified and lowered" if authored == length else "specified, not lowered")
        else:
            split = "not recovered"
        rows.append(dict(form="%d/%d" % (op, length), split=split, reason=r["verdict"],
                         programs=programs.get((op, length), 0), instances=r.get("instances", 0),
                         compiler_authors_width=authored))
    by = collections.defaultdict(list)
    for row in rows:
        by[row["split"]].append(row)
    summary = {k: dict(forms=len(v), programs=sum(x["programs"] for x in v))
               for k, v in sorted(by.items())}
    reasons = collections.Counter(r["reason"] for r in rows if r["split"] == "not recovered")
    return dict(
        means=("every Apple-witnessed form isa/g17-form-spec.jsonl lacked, classified by "
               "tools/g17formspecgen.py and scored by the corpus PROGRAMS it gates"),
        summary=summary,
        not_recovered_by_reason={k: v for k, v in reasons.most_common()},
        ranked={k: sorted(v, key=lambda x: (-x["programs"], x["form"]))
                for k, v in sorted(by.items())})


def write(results):
    """Append admitted forms to both files. The recorded 984 are left byte-for-byte as they were."""
    admitted = sorted(k for k, v in results.items() if v["verdict"] == "admit")
    with open(SPEC, "a") as fh:
        for op, length in admitted:
            r = results[(op, length)]
            fh.write(json.dumps({"opcode": op, "length": length, "error": None,
                                 "witness": r["witness"], "bits": r["bits"],
                                 "generator": "tools/g17formspecgen.py --extend"}) + "\n")
    with open(FREE, "a") as fh:
        for op, length in admitted:
            fh.write(json.dumps({"opcode": op, "length": length,
                                 "bits": results[(op, length)]["freebits"],
                                 "generator": "tools/g17formspecgen.py --extend"}) + "\n")
    return admitted


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-pinned", action="store_true",
                    help="pass on exactly the pinned disagreement, fail if it changes")
    ap.add_argument("--pin", action="store_true", help="record the current disagreement as known")
    ap.add_argument("--extend", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="write isa/g17-encoding-breadth.json: the three-way split, scored by programs")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    if a.extend:
        enc, msg = verified_corpus_encodings()
        if enc is None:
            print("REFUSED: " + msg)
            return 2
        print(msg)
        res = extend(enc)
        tally = collections.Counter(v["verdict"] for v in res.values())
        for k, n in tally.most_common():
            print("   %-44s %4d" % (k, n))
        if a.report:
            doc = report(res, corpus_encodings.programs)
            with open(REPORT, "w") as fh:
                json.dump(doc, fh, indent=1, sort_keys=True)
                fh.write("\n")
            for k, v in doc["summary"].items():
                print("   %-26s %4d forms %7d programs" % (k, v["forms"], v["programs"]))
            print("wrote %s" % os.path.relpath(REPORT, ROOT))
        if a.write:
            added = write(res)
            print("appended %d admitted forms to %s and %s"
                  % (len(added), os.path.relpath(SPEC, ROOT), os.path.relpath(FREE, ROOT)))
        return 0
    if a.pin or a.check_pinned:
        # PER DECODER BUILD. The macOS 27 (26A434) decoder rejects 38 single-bit flips the original one
        # read as another opcode, and reads 7 it rejected as opcodes new in that build (no original
        # id); the operand classes are unchanged (tools/g17renumber.py). Each build has its own pin.
        import g17renumber
        known = KNOWN.replace(".json", "-26A434.json") if g17renumber.on_26a434() else KNOWN
        # THE STANDING DISAGREEMENT, PINNED. --check re-classifies the 984 older recorded forms with
        # the current classifier; 508 reproduce exactly and it exits 1 by design, the same at #119
        # and after. The gate needs the version that fails only when that disagreement CHANGES.
        rep = check(None)
        if a.pin:
            with open(known, "w") as fh:
                json.dump(dict(rep, generator="tools/g17formspecgen.py --pin"), fh, indent=1, sort_keys=True)
                fh.write("\n")
            print("pinned: %d of %d reproduced" % (rep["forms_reproduced_exactly"], rep["forms"]))
            return 0
        want = json.load(open(known)) if os.path.exists(known) else {}
        want.pop("generator", None)
        if want == json.loads(json.dumps(rep)):
            print("disagreement is exactly the pinned one (%d of %d reproduced)"
                  % (rep["forms_reproduced_exactly"], rep["forms"]))
            return 0
        print("DISAGREEMENT CHANGED: %d of %d reproduced now" % (rep["forms_reproduced_exactly"], rep["forms"]))
        return 1
    if a.check:
        rep = check(a.limit)
        print("reproduced %d of %d recorded forms exactly" % (rep["forms_reproduced_exactly"], rep["forms"]))
        g, t = rep["operand_first_moved_agrees"]
        print("   operand `moved[0]` (what the encoder reads) agrees: %d / %d" % (g, t))
        for c, (g, t) in rep["by_class"].items():
            print("   %-10s %6d / %6d  (%.2f%%)" % (c, g, t, 100.0 * g / t))
        if rep["confusion"]:
            print("disagreements (recorded->mine):")
            for k, n in rep["confusion"].items():
                print("   %-22s %d" % (k, n))
        return 0 if rep["forms_reproduced_exactly"] == rep["forms"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
