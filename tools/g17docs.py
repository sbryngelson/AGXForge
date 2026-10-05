#!/usr/bin/env python3
"""The docs/ directory: a short list of living documents at the top, everything else archived.

docs/ had grown to 293 markdown files, most written for one moment - lane handoffs, root
assignments, preregistrations and batch plans for finished work, dated dispatch notes - beside the
dozen a reader actually needs. The living ones are listed, with what each is for, in
docs/README.md; the rest are in docs/archive/, moved with `git mv` so their history follows them.
Nothing was deleted: the machine model cites several as provenance.

THE CHECK, because a rule would not hold: every top-level docs/*.md must be listed in
docs/README.md. A new one-off note goes to docs/archive/ directly, or is listed deliberately; one
that is neither fails `make check-ledgers` on the commit that adds it.

    python3 tools/g17docs.py            the plan: what stays, what would move (no changes)
    python3 tools/g17docs.py --apply    git mv the rest into docs/archive/ and rewrite every reference
    python3 tools/g17docs.py --check    exit 1 if a top-level doc is not listed in docs/README.md, a
                                        designated document is missing or unlisted, or a machine-model
                                        section cited by an entry point does not exist

THE DESIGNATED DOCUMENTS (authorized 2026-10-01). Three documents carry the account, each with one job:
the showcase is the results document, the technical reference explains how the system works, and the
machine model is the evidence record. The first two cite the third by section ("MM 25.190"), and so do
the two README files, so a renumbered or removed section would silently break every citation. --check
therefore resolves each cited section number against the machine model's headings.

THE TECHNICAL REFERENCE IS LATEX (docs/tex/, built to docs/g17-technical-reference.pdf). It cites the
machine model as \\mm{25.190}. docs/tex/mm-anchors.tex is generated from the machine model's headings on every
build (--write-mm-anchors; not committed), so rewording a heading cannot break a citation; a number that heads two
sections is cited with its tag (\\mm{25.4/R6}). docs/tex/mm-aliases.tex may name a finding (\\mmalias{name}{25.91})
so that a citation can follow it to a later section by one edit. --check verifies every \\mm key is a section or
an alias, every alias points at a section, and every \\ref or \\cref names a \\label.

OPCODES NAME THEMSELVES. \\op{5107} prints "tensor MMA, 16-bit x 16-bit, no accumulator (op5107)" from
docs/tex/opnames.tex, which `--write-opnames` generates from isa/g17-opcode-glossary.json (\\Op capitalises the
name for a sentence start; \\opn{5107} prints the number alone, for a table cell beside a name column or a
deliberately shortened name). So prose edits cannot separate an opcode from its name and a glossary rename
reaches every mention. --check refuses a stale opnames.tex, an opcode with no glossary entry that is not marked
"no glossary entry" in its paragraph, and an opN in a code listing whose name is not nearby; a number-only
\\opn whose paragraph does not name the opcode is reported as a note, not refused.
"""
import os, re, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
ARCHIVE = os.path.join(DOCS, "archive")
INDEX = os.path.join(DOCS, "README.md")

# THE LIVING DOCUMENTS - updated as work continues. Everything else is a record of a moment.
KEEP = {
    "README.md",
    # the designated documents: results, technical reference (the machine model is listed below)
    "showcase.md",
    # the project
    "architecture.md", "guide.md", "rules.md", "reproducing.md", "findings.md",
    "execution-validation.md", "porting-the-recovery.md",
    # the ISA and the machine
    "agx-isa-status.md", "agx3-oracle.md", "g17-isa-reference.md", "g17-isa-cartography.md",
    "g17-tensorops-machine-model.md", "g17-tensorops-accelerator-recon.md",
    # the state of the work
    "g17-project-index.md", "g17-capability-frontier.md", "g17-first-dispatch-trials.md",
    "g17-seta-status.md", "g17-setb-status.md", "g17-set-c-status.md",
}


def tracked(pattern):
    out = subprocess.run(["git", "-C", ROOT, "ls-files", pattern], capture_output=True, text=True, check=True)
    return [l for l in out.stdout.splitlines() if l]


def top_level():
    return sorted(os.path.basename(p) for p in tracked("docs/*.md") if p.count("/") == 1)


def plan():
    names = top_level()
    return [n for n in names if n in KEEP], [n for n in names if n not in KEEP]


def listed(text, name):
    return re.search(r"(\(|`|\s|^)%s(\)|`|\s|$)" % re.escape(name), text, re.M) is not None


# The designated documents and their jobs; docs/README.md lists them under "The three principal documents".
DESIGNATED = {
    "showcase.md": "results",
    "tex/g17-technical-reference.tex": "technical reference",
    "g17-tensorops-machine-model.md": "evidence record",
}
MACHINE_MODEL = os.path.join(DOCS, "g17-tensorops-machine-model.md")
# The entry points whose machine-model citations must resolve (paths from the repository root).
CITERS = ["README.md", "docs/README.md", "docs/showcase.md"]
# The technical reference's LaTeX source (checked by tex_problems, below).
TEX = os.path.join(DOCS, "tex")
OPNAMES = os.path.join(TEX, "opnames.tex")
VALUES_TEX = os.path.join(TEX, "values.tex")
OPINDEX = os.path.join(TEX, "opindex.tex")
MM_ANCHORS = os.path.join(TEX, "mm-anchors.tex")
MM_ALIASES = os.path.join(TEX, "mm-aliases.tex")

# NUMBERS THE DOCUMENT QUOTES FROM GENERATED ARTIFACTS. Each is printed by \\val{key} from docs/tex/values.tex,
# which --write-values regenerates, so a regenerated artifact moves the prose with it. Historical constants
# (the 25G83 table size, an earlier count in appendix A) stay literal: they are facts about a moment.
def _coverage(*path):
    def get():
        import json
        v = json.load(open(os.path.join(ROOT, "isa", "g17-coverage.json")))
        for k in path:
            v = v[k]
        return v
    return get


def _renumber(key):
    def get():
        import json
        return json.load(open(os.path.join(ROOT, "isa", "g17-agx3-renumber.json")))["stats"][key]
    return get


def _glossary(rule):
    def get():
        import json
        return sum(1 for e in json.load(open(GLOSSARY))["opcodes"] if rule(e))
    return get


VALUES = [
    ("d1", "D1 structural forms", _coverage("forms", "totals", "d1_structural")),
    ("d2", "D2 encoded forms", _coverage("forms", "totals", "d2_encoded")),
    ("d3-any", "D3 forms, any own instrument", _coverage("forms", "totals", "d3_any")),
    ("d3-verified", "D3 verified class", _coverage("forms", "totals", "d3_semantics_checked")),
    ("dispatched-unverified", "dispatched forms outside the verified class", _coverage("forms", "totals", "executed_unchecked")),
    ("dispatched-other-evidence", "of those, with other semantic evidence",
     _coverage("forms", "totals", "executed_unchecked_with_other_evidence")),
    ("dispatched-no-evidence", "dispatched forms with no semantic evidence",
     _coverage("forms", "totals", "executed_with_no_semantic_column")),
    ("admitted-opcodes", "decoder-admitted opcode records", _coverage("width_coverage", "admitted_opcodes")),
    ("named-instructions", "glossary opcodes that are instructions (decoder tags excluded)",
     _glossary(lambda e: "decoder tag" not in e["name"])),
    ("executed-instructions", "glossary opcodes confirmed by execution", _glossary(lambda e: e["confidence"] == "executed")),
    ("renumber-new-opcodes", "26A434 instruction table entries", _renumber("new_opcodes")),
    ("renumber-old-registers", "25G83 register table entries", _renumber("old_registers")),
    ("renumber-new-registers", "26A434 register table entries", _renumber("new_registers")),
    ("renumber-corpus-instructions", "corpus instructions re-decoded", _renumber("corpus_instructions")),
    ("renumber-witnessed", "opcodes anchored by decoded encodings", _renumber("witnessed_opcodes")),
    ("renumber-aligned", "opcodes mapped by descriptor alignment", _renumber("aligned_by_descriptor")),
    ("renumber-ambiguous", "opcodes left unmapped as ambiguous", _renumber("left_unmapped_as_ambiguous")),
    ("renumber-tags", "fitted immediate-tag entries", _renumber("tag_entries")),
]


def values_tex():
    """docs/tex/values.tex: \\valdef{key}{number} for each VALUES entry, thousands separated as the prose writes."""
    lines = ["% GENERATED by tools/g17docs.py --write-values from the artifacts named in its VALUES; do not edit."]
    for key, what, get in VALUES:
        v = get()
        if not isinstance(v, int):
            raise ValueError("value %s (%s) is not an integer: %r" % (key, what, v))
        lines.append("\\valdef{%s}{%s}%% %s" % (key, format(v, ","), what))
    return "\n".join(lines) + "\n"

_NUM = r"\d+(?:\.\d+)*(?:\.?[A-Za-z]\b)?"
# "MM 25.190", "MM 25.141.14 and 25.141.13", "MM 25.149, 25.149.1", "MM 25.98 to 25.111", "MM 0"
_MM = re.compile(r"\bMM (%s(?:(?:,| and| to|, and) %s)*)" % (_NUM, _NUM))
# "section 25.210", "sections 25.98 to 25.111", "section 25.143 of the machine model": dotted 25.x only,
# because a bare "section 3" in the showcase names one of its own sections.
_SECTION = re.compile(r"\bsections? (25\.%s(?:(?:,| and| to|, and) 25\.%s)*)" % (_NUM, _NUM))


def mm_heading_counts(text):
    """{section number: how many machine-model headings define it} ("## 25.", "### 25.190 ", "#### 25.210.9a")."""
    out = {}
    for m in re.finditer(r"^#{2,5} (\d+(?:\.\d+)*(?:\.?[A-Za-z])?)[ .]", text, re.M):
        out[m.group(1)] = out.get(m.group(1), 0) + 1
    return out


def mm_sections(text):
    return set(mm_heading_counts(text))


_NUMLINK = re.compile(r"\[(\d[\w.]*)\]\([^)]*\)")


def cited_sections(text):
    """(number, line) for every machine-model section the text cites, linked ("MM [25.190](...#...)") or not."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        line = _NUMLINK.sub(r"\1", line)
        for rx in (_MM, _SECTION):
            for m in rx.finditer(line):
                for n in re.findall(_NUM, m.group(1)):
                    out.append((n, i))
    return out


# A few early numbers head two different sections (25.4, 25.5 and 25.7). They are not renumbered, since the numbers
# are anchors; a citation of one must say which, by its title and a link to the intended heading on the same line.
_LINKED = re.compile(r"\]\([^)]*g17-tensorops-machine-model\.md#([^)]+)\)")


def slug(heading):
    """GitHub's anchor for a heading's text: lower case, punctuation other than '-' and '_' dropped, spaces to '-'."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def mm_anchors(text):
    """{section number: the anchors of the headings that define it}."""
    out = {}
    for m in re.finditer(r"^#{2,5} ((\d+(?:\.\d+)*(?:\.?[A-Za-z])?)[ .].*)$", text, re.M):
        out.setdefault(m.group(2), set()).add(slug(m.group(1)))
    return out


def github_anchors(text):
    """Every anchor GitHub gives the document's headings, code blocks skipped and repeats suffixed -1, -2, ..."""
    seen, out, code = {}, set(), False
    for line in text.splitlines():
        if line.startswith("```"):
            code = not code
            continue
        m = None if code else re.match(r"^#{1,6} (.*)$", line)
        if m:
            s = slug(m.group(1))
            n = seen.get(s, 0)
            seen[s] = n + 1
            out.add(s if n == 0 else "%s-%d" % (s, n))
    return out


_ANYLINK = re.compile(r"g17-tensorops-machine-model\.md#([^)\s]+)\)")


def dead_links(citer_texts, mm_text):
    """[(path, line, anchor)] for every link into the machine model whose anchor no heading has."""
    have = github_anchors(mm_text)
    return [(path, i, a) for path, text in citer_texts for i, line in enumerate(text.splitlines(), 1)
            for a in _ANYLINK.findall(line) if a not in have]


# NAME FIRST, NUMBER SECOND, in the designated documents a reader meets first (the machine model has its own check,
# tools/g17docnames.py). An opcode number is checked against isa/g17-opcode-glossary.json: its name, or the part of the
# name before the first comma, must appear in the same paragraph, table row or annotated listing. A code block counts
# with the paragraph that introduces it.
NAMED_DOCS = ["docs/showcase.md"]
GLOSSARY = os.path.join(ROOT, "isa", "g17-opcode-glossary.json")


def _paragraphs(text):
    """[(first line, unit text, is_code)]: a unit is a blank-line paragraph, a list item, a table row or a fenced
    block. Link targets are dropped, so an anchor's 'op9320' is not a mention."""
    units, cur, start, code = [], [], 1, False

    def flush(is_code=False):
        if cur:
            units.append((start, re.sub(r"\]\([^)]*\)", "]", "\n".join(cur)), is_code))

    for i, line in enumerate(text.splitlines(), 1):
        if line.startswith("```"):
            if not code:
                flush()
                cur, start = [], i
            cur.append(line)
            code = not code
            if not code:
                flush(True)
                cur = []
            continue
        if code:
            cur.append(line)
            continue
        if not line.strip():
            flush()
            cur = []
            continue
        if line.startswith("|") or re.match(r"\s*(?:[-*]|\d+\.) ", line):
            flush()
            cur, start = [line], i
            if line.startswith("|"):
                flush()
                cur = []
            continue
        if not cur:
            start = i
        cur.append(line)
    flush()
    return units


def unnamed_opcodes(text, glossary):
    """[(line, opcode)] for every opN whose glossary name (or the part before its first comma) is absent from its
    unit; a fenced block may take its names from the paragraph that introduces it."""
    out, prev = [], ""
    for n, unit, is_code in _paragraphs(text):
        low = " ".join(unit.lower().split())
        ctx = low + " " + prev if is_code else low
        for m in re.finditer(r"\bop(\d+)\b", unit):
            name = glossary.get(int(m.group(1)))
            if name is None:
                # an opcode the glossary does not name must say so where it is mentioned
                if "no glossary entr" not in ctx:
                    out.append((n, "op" + m.group(1)))
            elif name.lower() not in ctx and name.split(",")[0].lower() not in ctx:
                out.append((n, "op" + m.group(1)))
        prev = low
    return out


def unresolved(citer_texts, mm_text):
    """[(path, line, number)] cited in an entry point but defined by no machine-model heading, or by several
    when the citing line carries no link to the intended one."""
    counts, anchors = mm_heading_counts(mm_text), mm_anchors(mm_text)
    out = []
    for path, text in citer_texts:
        lines = text.splitlines()
        for n, i in cited_sections(text):
            linked = set(_LINKED.findall(lines[i - 1]))
            if n not in counts or (counts[n] > 1 and not linked & anchors[n]):
                out.append((path, i, n))
    return out


def tex_sources():
    """[(path from the repository root, text)] for the technical reference's LaTeX source."""
    paths = [os.path.join(TEX, "g17-technical-reference.tex")]
    for sub in ("ch", "fig"):
        d = os.path.join(TEX, sub)
        if os.path.isdir(d):
            paths += [os.path.join(d, n) for n in sorted(os.listdir(d)) if n.endswith(".tex")]
    return [(os.path.relpath(p, ROOT), open(p).read()) for p in paths if os.path.exists(p)]


def mm_anchors_tex(mm_text):
    """docs/tex/mm-anchors.tex, generated from the machine model's headings on every build: \\mmdef{number}{anchor}
    for each numbered section, with GitHub's anchor (repeats suffixed as GitHub does). Where one number heads two
    sections the key adds the heading's tag (25.4/R6) and the bare number is not defined, so a citation must say
    which. A reworded heading changes only this generated file; the document's citations do not move."""
    seen, code, rows = {}, False, []
    for line in mm_text.splitlines():
        if line.startswith("```"):
            code = not code
            continue
        m = None if code else re.match(r"^#{1,6} (.*)$", line)
        if not m:
            continue
        s = slug(m.group(1))
        n = seen.get(s, 0)
        seen[s] = n + 1
        anchor = s if n == 0 else "%s-%d" % (s, n)
        num = re.match(r"(\d+(?:\.\d+)*(?:\.?[A-Za-z])?)[ .](.*)$", m.group(1)) if line.startswith(("##", "###", "####", "#####")) else None
        if num:
            rows.append((num.group(1), num.group(2), anchor))
    counts = {}
    for key, _, _ in rows:
        counts[key] = counts.get(key, 0) + 1
    out = ["% GENERATED by tools/g17docs.py --write-mm-anchors from docs/g17-tensorops-machine-model.md; do not edit.",
           "% \\mmdef{section}{anchor}; a number that heads two sections is keyed with its tag (25.4/R6)."]
    done = set()
    for key, rest, anchor in rows:
        if counts[key] > 1:
            tag = re.match(r"\s*([A-Z]+\d+)\b", rest)
            if not tag:
                raise ValueError("machine-model section %s is repeated and one heading has no tag: %r" % (key, rest[:60]))
            key = "%s/%s" % (key, tag.group(1))
        if key in done:
            raise ValueError("machine-model key %s would be defined twice" % key)
        done.add(key)
        out.append("\\mmdef{%s}{%s}" % (key, anchor))
    return "\n".join(out) + "\n"


def mm_aliases(text):
    """{alias: key} from mm-aliases.tex's \\mmalias{alias}{key} lines."""
    return dict(re.findall(r"^\\mmalias\{([^}]+)\}\{([^}]+)\}", text, re.M))


def mm_definitions(mm_text):
    """The \\mmdef text --check validates citations against: the generated anchors, plus each alias resolved to its
    target's anchor (an alias whose target is undefined resolves to nothing and is refused as undefined)."""
    gen = mm_anchors_tex(mm_text)
    defs = mm_defs(gen)
    extra = []
    if os.path.exists(MM_ALIASES):
        for alias, key in sorted(mm_aliases(open(MM_ALIASES).read()).items()):
            if key in defs:
                extra.append("\\mmdef{%s}{%s}" % (alias, defs[key]))
    return gen + "\n".join(extra) + ("\n" if extra else "")


def mm_defs(text):
    """{key: anchor} from mm-anchors.tex's \\mmdef{key}{anchor} lines."""
    return dict(re.findall(r"^\\mmdef\{([^}]+)\}\{([^}]+)\}", text, re.M))


def _tex_plain(s):
    """LaTeX to comparable text: commands dropped (arguments kept), braces dropped, x for the times sign."""
    s = s.replace("\u00d7", "x").replace("\\_", "_").replace("\\&", "&").replace("\\#", "#").replace("\\%", "%")
    s = re.sub(r"\\textasciitilde\{\}", "~", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    return " ".join(re.sub(r"[{}]", "", s).lower().split())


def _tex_units(text):
    """[(first line, unit text, is_code)]: a paragraph, a list item, a table row or a listing."""
    units, cur, start, code = [], [], 1, False

    def flush(is_code=False):
        if cur:
            units.append((start, "\n".join(cur), is_code))

    for i, line in enumerate(text.splitlines(), 1):
        if line.startswith("\\begin{lstlisting}"):
            flush()
            cur, start, code = [line], i, True
            continue
        if code:
            cur.append(line)
            if line.startswith("\\end{lstlisting}"):
                flush(True)
                cur, code = [], False
            continue
        if not line.strip() or line.lstrip().startswith("\\item") or line.startswith("\\begin{") or line.startswith("\\end{"):
            flush()
            cur, start = ([line], i) if line.strip() else ([], i + 1)
            continue
        if not cur:
            start = i
        cur.append(line)
        if line.rstrip().endswith("\\\\"):     # a table row ends
            flush()
            cur = []
    flush()
    return units


def opname_display(name):
    """A glossary name as the document prints it: x between operand sizes becomes a multiplication sign."""
    name = re.sub(r"(?<=\d)x(?=\d)", "\u00d7", name)
    return re.sub(r"(?<=[\w)]) x (?=[\w(])", " \u00d7 ", name)


def opname_tex(glossary):
    """docs/tex/opnames.tex: one name per opcode, read by \\op, \\Op and \\opn (preamble.tex)."""
    lines = ["% GENERATED by tools/g17docs.py --write-opnames from isa/g17-opcode-glossary.json; do not edit.",
             "% \\op{N} prints the name and the linked number; an opcode with no entry prints the number only."]
    for op in sorted(glossary, key=int):
        name = opname_display(glossary[op])
        if re.search(r"[\\{}$&#^_%~]", name):
            raise ValueError("glossary name for op%s needs LaTeX escaping: %r" % (op, name))
        lines.append("\\opnamedef{%s}{%s}{%s}" % (op, name, name[:1].upper() + name[1:]))
    return "\n".join(lines) + "\n"


# THE OPCODE INDEX (appendix C), GENERATED ON EVERY BUILD by the docs Makefile and not committed: names and
# standings come from the glossary; "discussed in" from where the document mentions each opcode (up to two
# sections; the appendices and the open-questions index only when no main chapter mentions it; most mentions
# first, then first mention); the
# machine-model column from the glossary's sections that have a citable anchor (chapter-level numbers and the
# 24.x inventory only when nothing else is cited; chapter 25 first, then numerically; up to four).
OPINDEX_MM_OVERRIDES = {
    # the glossary's sections for these rows are numbers of an older document, not machine-model sections
    1016: ["6", "13"], 2862: ["0.3", "9"], 14169: ["12"],
    **{op: ["0.3", "2"] for op in (5098, 5099, 5100, 5101, 5104, 5105, 5106, 5107, 10384, 10385)},
}


def _mm_sort_key(sec):
    return (not sec.startswith("25."), tuple(int(x) if x.isdigit() else 0 for x in re.sub(r"[a-z]$", "", sec).split(".")))


def opcode_mentions(sources):
    """{op: {label: count}}, {(op, label): first position}, {appendix labels}, over the chapters in book order."""
    main = open(os.path.join(TEX, "g17-technical-reference.tex")).read()
    order = [m + ("" if m.endswith(".tex") else ".tex") for m in re.findall(r"^\\(?:include|input)\{(ch/[^}]+)\}", main, re.M)]
    by_rel = {os.path.relpath(os.path.join(ROOT, p), TEX): t for p, t in sources}     # paths are ROOT-relative
    mentions, first, appendix, pos = {}, {}, set(), 0
    for rel in order:
        if rel.endswith("c-opcodes.tex") or rel not in by_rel:
            continue
        label = None
        for line in by_rel[rel].splitlines():
            m = re.search(r"\\(?:chapter|addchap|section)\*?\{.*\}\\label\{([^}]*)\}", line)
            if m:
                label = m.group(1)
            for op in re.findall(r"\\(?:op|Op|opn)\{(\d+)\}", line):
                op = int(op)
                mentions.setdefault(op, {})
                mentions[op][label] = mentions[op].get(label, 0) + 1
                pos += 1
                first.setdefault((op, label), pos)
                if re.match(r"ch/([a-d]-|14-)", rel):      # appendices and the open-questions index
                    appendix.add(label)
    return mentions, first, appendix


def opcode_index_tex(glossary, sources, anchors_text):
    entries = {int(e["op"]): e for e in __import__("json").load(open(GLOSSARY))["opcodes"]}
    keys = set(re.findall(r"\\mmdef\{([^}/]*)\}", anchors_text))
    mentions, first, appendix = opcode_mentions(sources)
    rows = []
    for op in sorted(mentions):
        ranked = sorted(mentions[op], key=lambda k: (-mentions[op][k], first[(op, k)]))
        discussed = ([k for k in ranked if k not in appendix] or ranked)[:2]
        refs = ", ".join("\\ref{%s}" % k for k in discussed)
        e = entries.get(op)
        if e is None:
            rows.append("\\opdef{%d} & (no glossary entry) & & %s & \\\\" % (op, refs))
            continue
        secs = e["sections"]
        secs = __import__("ast").literal_eval(secs) if isinstance(secs, str) else list(secs)
        cited = [x for x in secs if x in keys]
        mm = OPINDEX_MM_OVERRIDES.get(op) or sorted(
            [x for x in cited if "." in x and not x.startswith("24.")] or cited, key=_mm_sort_key)[:4]
        mmcol = ", ".join("\\mm{%s}" % x for x in mm) if mm else "the glossary entry"
        rows.append("\\opdef{%d} & %s & %s & %s & %s \\\\" % (op, opname_display(e["name"]), e["confidence"], refs, mmcol))
    if not rows:
        raise SystemExit("opcode index: no opcode mentions found in the chapters; refusing to build an empty index")
    head = ("Opcode & Name & Standing & Discussed in & Machine-model sections \\\\\n")
    return ("% GENERATED by tools/g17docs.py --write-opindex on every build (docs/tex/Makefile); do not edit.\n"
            "\\begin{xltabular}{\\linewidth}{@{}l>{\\hsize=1.3\\hsize}Xll>{\\hsize=0.7\\hsize}X@{}}\n"
            "\\caption{Opcode index: glossary name, standing, and where each opcode is discussed.}\\label{tab:opcode-index}\\\\\n"
            "\\toprule\n" + head + "\\midrule\n\\endfirsthead\n\\tablecontinued{5}\n\\toprule\n" + head +
            "\\midrule\n\\endhead\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{xltabular}\n")


def tex_unnamed_opcodes(text, glossary):
    """([(line, opcode)] refused, [(line, opcode)] noted).

    Refused: an \\op/\\Op/\\opn with no glossary entry whose paragraph does not say "no glossary entry", an
    \\opdef whose row does not carry the name, and an opN in a code listing with no name nearby. Noted: a
    number-only \\opn whose paragraph neither names the opcode nor mentions it with \\op. A named \\op prints
    its own name, so it is never refused."""
    hard, soft, prev, prev_named = [], [], "", set()
    for n, unit, is_code in _tex_units(text):
        low = _tex_plain(unit)
        ctx = low + " " + prev if is_code else low
        named_here = set(re.findall(r"\\[oO]p\{(\d+)\}", unit))
        if is_code:      # a listing is named by the paragraph that introduces it, where \op prints the name
            named_here |= prev_named
        def has_name(num):
            name = glossary.get(int(num))
            return name is not None and any(
                _tex_plain(v) in ctx for v in (name, name.split(",")[0], opname_display(name),
                                                opname_display(name).split(",")[0]))
        for kind, num in re.findall(r"\\(op|Op|opn|opdef)\{(\d+)\}", unit):
            if glossary.get(int(num)) is None:
                if "no glossary entr" not in ctx:
                    hard.append((n, "op" + num))
            elif kind == "opdef" and not has_name(num):
                hard.append((n, "op" + num))
            elif kind == "opn" and num not in named_here and not has_name(num):
                soft.append((n, "op" + num))
        if is_code:
            for num in re.findall(r"\bop(\d+)\b", unit):
                if glossary.get(int(num)) is None and "no glossary entr" not in ctx:
                    hard.append((n, "op" + num))
                elif glossary.get(int(num)) is not None and num not in named_here and not has_name(num):
                    hard.append((n, "op" + num))
        prev, prev_named = low, set(re.findall(r"\\[oO]p\{(\d+)\}", unit))
    return hard, soft


def tex_problems(sources, defs_text, mm_text, glossary):
    """Messages for every broken citation, reference or bare opcode in the LaTeX technical reference."""
    defs, out = mm_defs(defs_text), []
    have, counts, anchors = github_anchors(mm_text), mm_heading_counts(mm_text), mm_anchors(mm_text)
    for key, a in sorted(defs.items()):
        num = key.split("/")[0]
        if num not in counts:
            out.append("mm-anchors.tex defines %s, which no machine-model heading defines" % key)
        elif a not in have or a not in anchors[num]:
            out.append("mm-anchors.tex gives %s the anchor #%s, which is not a heading of that section" % (key, a))
        elif counts[num] > 1 and "/" not in key:
            out.append("mm-anchors.tex key %s is ambiguous: %d headings use that number; add the tag (%s/R6)"
                       % (key, counts[num], num))
    labels = {"tab:opcode-index"}      # defined by the generated opcode index (opcode_index_tex)
    for _, text in sources:
        labels |= set(re.findall(r"\\label\{([^}]+)\}", text))
    for path, text in sources:
        for i, line in enumerate(text.splitlines(), 1):
            for key in re.findall(r"\\mm\{([^}]+)\}", line):
                if key not in defs:
                    out.append("%s:%d cites \\mm{%s}, which mm-anchors.tex does not define" % (path, i, key))
            refs = re.findall(r"\\hyperref\[([^\]]+)\]", line)
            for grp in re.findall(r"\\(?:[cC]ref|ref|pageref|[cC]refrange)\{([^}]+)\}(?:\{([^}]+)\})?", line):
                refs += [k.strip() for g in grp if g for k in g.split(",")]
            for lab in refs:
                if lab not in labels:
                    out.append("%s:%d refers to label %s, which no \\label defines" % (path, i, lab))
        # a citation written out without the macro must still resolve
        plain = re.sub(r"\\mm\{[^}]+\}", "#", text)
        out += ["%s:%d cites machine-model section %s, which no heading defines, or which several do" % (p, i, n)
                for p, i, n in unresolved([(path, plain)], mm_text)]
        out += ["%s:%d names %s by number only; use \\op, or mark it \"no glossary entry\"" % (path, i, op)
                for i, op in tex_unnamed_opcodes(text, glossary)[0]]
    return out


def tex_notes(sources, glossary):
    """Advisory messages: a number-only \\opn whose paragraph does not name the opcode."""
    return ["%s:%d gives %s by number only (\\opn) and its paragraph does not name it" % (path, i, op)
            for path, text in sources for i, op in tex_unnamed_opcodes(text, glossary)[1]]


def check():
    text = open(INDEX).read() if os.path.exists(INDEX) else ""
    bad = [n for n in top_level() if n != "README.md" and not listed(text, n)]
    if bad:
        print("REFUSED: top-level docs not listed in docs/README.md: %s" % ", ".join(bad))
        print("  list each with what it is for, or put a one-off note in docs/archive/")
        return 1
    absent = [n for n in DESIGNATED if not os.path.exists(os.path.join(DOCS, n))]
    if absent:
        print("REFUSED: designated documents missing from docs/: %s" % ", ".join(absent))
        return 1
    texts = [(p, open(os.path.join(ROOT, p)).read()) for p in CITERS if os.path.exists(os.path.join(ROOT, p))]
    dangling = unresolved(texts, open(MACHINE_MODEL).read())
    if dangling:
        for path, i, n in dangling:
            print("REFUSED: %s:%d cites machine-model section %s, which no heading defines, or which several do"
                  " and the line does not link the intended one" % (path, i, n))
        print("  machine-model section numbers are citation anchors: fix the citation, never renumber the section")
        return 1
    import json
    gloss = {e["op"]: e["name"] for e in json.load(open(GLOSSARY))["opcodes"]}
    bare = [(p, n, op) for p in NAMED_DOCS for n, op in unnamed_opcodes(open(os.path.join(ROOT, p)).read(), gloss)]
    if bare:
        for p, n, op in bare:
            print("REFUSED: %s:%d names %s by number only; give its glossary name in the same paragraph" % (p, n, op))
        return 1
    dead = dead_links(texts, open(MACHINE_MODEL).read())
    if dead:
        for path, i, a in dead:
            print("REFUSED: %s:%d links machine-model anchor #%s, which no heading has" % (path, i, a))
        return 1
    sources = tex_sources()
    mm_text = open(MACHINE_MODEL).read()
    defs_text = mm_definitions(mm_text)
    problems = tex_problems(sources, defs_text, mm_text, gloss)
    if os.path.exists(MM_ALIASES):
        known = mm_defs(mm_anchors_tex(mm_text))
        for alias, key in sorted(mm_aliases(open(MM_ALIASES).read()).items()):
            if key not in known:
                problems.append("mm-aliases.tex points %s at %s, which no machine-model heading defines" % (alias, key))
            elif alias in known:
                problems.append("mm-aliases.tex alias %s is also a section number" % alias)
    if not os.path.exists(VALUES_TEX) or open(VALUES_TEX).read() != values_tex():
        problems.append("docs/tex/values.tex is stale against its artifacts: run python3 tools/g17docs.py --write-values")
    keys = {k for k, _, _ in VALUES}
    for path, text in sources:
        for i, line in enumerate(text.splitlines(), 1):
            for k in re.findall(r"\\val\{([^}]*)\}", line):
                if k not in keys:
                    problems.append("%s:%d uses \\val{%s}, which tools/g17docs.py VALUES does not define" % (path, i, k))
    fresh = opname_tex(gloss)
    if not os.path.exists(OPNAMES) or open(OPNAMES).read() != fresh:
        problems.append("docs/tex/opnames.tex is stale against isa/g17-opcode-glossary.json: "
                        "run python3 tools/g17docs.py --write-opnames")
    if problems:
        for msg in problems:
            print("REFUSED: %s" % msg)
        return 1
    for msg in tex_notes(sources, gloss):
        print("note: %s" % msg)
    ntex = sum(len(re.findall(r"\\mm\{", t)) for _, t in sources)
    print("docs/: %d living documents, each listed in docs/README.md; %d archived; %d machine-model citations"
          " in %d entry points and %d in the technical reference resolve"
          % (len(top_level()), len(tracked("docs/archive/*.md")),
             sum(len(cited_sections(t)) for _, t in texts), len(texts), ntex))
    return 0


def pinned_sources():
    """Files an inventory pins by sha256 as evidence. Rewriting a path inside one changes its hash
    and the inventory refuses (the first run hit agxforge/g17/mdgen.py, tools/g17texformatlinker.py
    and tools/g17texture.py). Their owners refresh their own pins; a stale path in a comment there
    is the lesser harm, and --plan lists them."""
    import json, glob
    out = set()

    def walk(x):
        if isinstance(x, dict):
            if isinstance(x.get("path"), str) and isinstance(x.get("sha256"), str):
                out.add(x["path"])
            for y in x.values():
                walk(y)
        elif isinstance(x, list):
            for y in x:
                walk(y)
    # only the inventories the rebuild does NOT regenerate: the compiler, integration and unified
    # ones are rewritten from the current files, so their pins follow any edit; the linker's and
    # the runtime's are evidence their owners refresh
    for name in ("linker", "runtime"):
        walk(json.load(open(os.path.join(ROOT, "isa", "g17-capabilities-%s.json" % name))))
    return {p for p in out if os.path.isfile(os.path.join(ROOT, p))}


def rewrite(moved):
    """Every reference to a moved document, in the same commit as the move."""
    mv = set(moved)
    alt = "|".join(re.escape(n) for n in sorted(mv, key=len, reverse=True))
    repo_path = re.compile(r"(?<![/\w])docs/(%s)" % alt)                         # docs/NAME.md
    join_dq = re.compile(r'"docs"(\s*,\s*)"(%s)"' % alt)                         # "docs", "NAME.md"
    join_sq = re.compile(r"'docs'(\s*/\s*)'(%s)'" % alt)                         # 'docs' / 'NAME.md'
    changed = []
    pinned = pinned_sources()
    for rel in tracked("."):
        p = os.path.join(ROOT, rel)
        if rel.startswith(("results/", "evidence/")) or not os.path.isfile(p) or rel in pinned:
            continue
        try:
            s = open(p, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        # A LINE MARKED `g17docs: keep` keeps an old path on purpose (a path on another git ref)
        t = "".join(line if "g17docs: keep" in line else
                    join_sq.sub(r"'docs'\1'archive'\1'\2'",
                                join_dq.sub(r'"docs"\1"archive"\1"\2"',
                                            repo_path.sub(r"docs/archive/\1", line)))
                    for line in s.splitlines(True))
        if rel.startswith("docs/"):
            here = os.path.basename(rel)
            archived_now = rel.startswith("docs/archive/") or here in mv
            # ONLY a document moving IN THIS RUN gets its ../ links deepened: one already in the
            # archive has them deepened already, and a second pass broke three of them
            moving_now = rel == "docs/" + here and here in mv
            def link(m):
                target = m.group(2)
                if target in mv and not archived_now:
                    return m.group(1) + "archive/" + target        # living -> archived
                if archived_now and target not in mv and os.path.exists(os.path.join(DOCS, target)):
                    return m.group(1) + "../" + target             # archived -> living
                return m.group(0)
            if moving_now:
                t = re.sub(r"\]\(\.\./", "](../../", t)            # one level deeper now (first, so the
                                                                   # ../ link() adds below is not deepened)
            t = re.sub(r"(\]\()([A-Za-z0-9_.-]+\.md)", link, t)
        if t != s:
            open(p, "w", encoding="utf-8").write(t)
            changed.append(rel)
    return changed


def apply():
    keep, moved = plan()
    os.makedirs(ARCHIVE, exist_ok=True)
    changed = rewrite(moved)
    for n in moved:
        subprocess.run(["git", "-C", ROOT, "mv", "docs/" + n, "docs/archive/" + n], check=True)
    print("moved %d documents to docs/archive/; rewrote references in %d files" % (len(moved), len(changed)))
    return 0


def main(argv):
    if "--check" in argv:
        return check()
    if "--check-generated" in argv:
        import json
        gloss = {e["op"]: e["name"] for e in json.load(open(GLOSSARY))["opcodes"]}
        stale = [os.path.relpath(p, ROOT) for p, fresh in ((VALUES_TEX, values_tex()), (OPNAMES, opname_tex(gloss)))
                 if not os.path.exists(p) or open(p).read() != fresh]
        for p in stale:
            print("STALE: %s (run tools/g17docs.py --write-values / --write-opnames)" % p)
        if not stale:
            print("fresh: docs/tex/values.tex and docs/tex/opnames.tex match their artifacts")
        return 1 if stale else 0
    if "--write-mm-anchors" in argv:
        text = mm_anchors_tex(open(MACHINE_MODEL).read())
        if not os.path.exists(MM_ANCHORS) or open(MM_ANCHORS).read() != text:
            with open(MM_ANCHORS, "w") as fh:
                fh.write(text)
        print("machine-model anchors: %d sections" % text.count("\\mmdef{"))
        return 0
    if "--write-opindex" in argv:
        import json
        gloss = {e["op"]: e["name"] for e in json.load(open(GLOSSARY))["opcodes"]}
        text = opcode_index_tex(gloss, tex_sources(), mm_definitions(open(MACHINE_MODEL).read()))
        if not os.path.exists(OPINDEX) or open(OPINDEX).read() != text:
            with open(OPINDEX, "w") as fh:
                fh.write(text)
        print("opcode index: %d rows" % text.count("\\opdef{"))
        return 0
    if "--write-values" in argv:
        with open(VALUES_TEX, "w") as fh:
            fh.write(values_tex())
        print("wrote %s: %d values" % (os.path.relpath(VALUES_TEX, ROOT), len(VALUES)))
        return 0
    if "--write-opnames" in argv:
        import json
        gloss = {e["op"]: e["name"] for e in json.load(open(GLOSSARY))["opcodes"]}
        with open(OPNAMES, "w") as fh:
            fh.write(opname_tex(gloss))
        print("wrote %s: %d opcode names" % (os.path.relpath(OPNAMES, ROOT), len(gloss)))
        return 0
    if "--apply" in argv:
        return apply()
    keep, moved = plan()
    print("stays (%d):" % len(keep))
    for n in keep:
        print("   ", n)
    print("would move to docs/archive/ (%d)" % len(moved))
    missing = sorted(KEEP - set(keep))
    if missing:
        print("KEEP names absent from docs/: %s" % missing)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
