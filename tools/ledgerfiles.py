#!/usr/bin/env python3
"""Where the claims ledger's entries live, and how to read and place them - storage only.

Since 2026-09-26 an entry is a TOML table keyed by its id - [<id>] for the fields, [<id>.<section>]
for the prose - in one of six area files under ledger/, sorted by id. Until then each entry was its
own file, 645 of them; they were merged verbatim and every entry proved to parse exactly as its file
did (tag PRE_MERGE_TAG). An entry is still CITED as ledger/<id>.toml, and resolve() maps that name to
it. A new entry may still be written as its own file ("loose"); tools/ledger.py fold places it.

THIS IS ITS OWN MODULE so that a reader of entries (the ISA map's opcode index, the tree catalog) does
not depend on the checks and the page tools/ledger.py builds on top: the fast gate follows imports,
and reading an entry should not select the tests of everything the ledger's page reads.
"""
import glob, os, re, tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AREA_FILES = ("claims", "isa", "compiler", "notes", "codec", "other")
PRE_MERGE_TAG = "archive/pre-ledger-merge-2026-09-26"
AREA_PREAMBLE = ("# The claims ledger, area: %s. One TOML table per entry, keyed by its id: [<id>] holds the\n"
                 "# fields, [<id>.<section>] the prose. Sorted by id. A new entry may be written as ledger/<id>.toml;\n"
                 "# `python3 tools/ledger.py fold` moves it here. ledger/README.md says what the fields mean.\n")
_HEADER = re.compile(r"""^(\[\[?)\s*([A-Za-z0-9_.\-"' ]+?)\s*(\]\]?)(\s*(?:#.*)?)$""")
_TRIPLES = ('"' * 3, "'" * 3)


def area_path(area, root=None):
  return os.path.join(root or ROOT, "ledger", area + ".toml")


def loose_paths(root=None):
  """ledger/<id>.toml files that are single entries, not area files."""
  areas = {a + ".toml" for a in AREA_FILES}
  return sorted(p for p in glob.glob(os.path.join(root or ROOT, "ledger", "*.toml")) if os.path.basename(p) not in areas)


def code_lines(lines):
  """Indices of the lines that start outside a multi-line string (prose opens lines with [[links]])."""
  inside, out = None, set()
  for i, ln in enumerate(lines):
    if inside is None: out.add(i)
    for q in _TRIPLES:
      if inside in (None, q) and ln.count(q) % 2 == 1:
        inside = None if inside == q else q
  return out


def header(ln):
  """The match of a TOML table header line, or None."""
  m = _HEADER.match(ln)
  return m if m and (m.group(1) == "[") == (m.group(3) == "]") else None


def blocks(text):
  """[(id, first_line, end_line)] of the entries in an area file, in file order."""
  lines, starts, cur = text.split("\n"), [], None
  for i in sorted(code_lines(lines)):
    m = header(lines[i])
    if m is None: continue
    k = m.group(2).split(".")[0].strip().strip("\"'")
    if k != cur: starts.append((k, i)); cur = k
  return [(k, a, starts[j + 1][1] if j + 1 < len(starts) else len(lines)) for j, (k, a) in enumerate(starts)]


def to_block(stem, text):
  """A single-entry file's text as its table in an area file: [<id>] first, each [section] -> [<id>.section]."""
  lines = text.rstrip("\n").split("\n")
  code, out = code_lines(lines), ["[%s]" % stem]
  for i, ln in enumerate(lines):
    m = header(ln) if i in code else None
    if m: ln = "%s%s.%s%s%s" % (m.group(1), stem, m.group(2).strip(), m.group(3), m.group(4))
    out.append(ln)
  return "\n".join(out).rstrip("\n")


def from_block(stem, block):
  """to_block's inverse: an entry's table as the text of its own file - what a reader that searches
  an entry's prose must see, since the id in every header would otherwise read as prose (a stem like
  g17-op590-tracks-loads names an opcode)."""
  lines = block.rstrip("\n").split("\n")
  if lines and lines[0].strip() == "[%s]" % stem: lines = lines[1:]
  code, pre = code_lines(lines), stem + "."
  for i, ln in enumerate(lines):
    m = header(ln) if i in code else None
    if m and m.group(2).startswith(pre):
      lines[i] = "%s%s%s%s" % (m.group(1), m.group(2)[len(pre):], m.group(3), m.group(4))
  return "\n".join(lines).strip("\n") + "\n"


def read_area(path):
  """(text, {id: entry}) of an area file; ("", {}) when it does not exist."""
  if not os.path.exists(path): return "", {}
  with open(path, encoding="utf-8") as f: text = f.read()
  return text, tomllib.loads(text)


def iter_raw(root=None):
  """(relpath, id, entry | None, error | None) for every entry: the area files, then loose files."""
  root = root or ROOT
  for a in AREA_FILES:
    p = area_path(a, root)
    if not os.path.exists(p): continue
    rel = os.path.relpath(p, root)
    try:
      _, doc = read_area(p)
    except Exception as e:
      yield rel, None, None, "not valid TOML: %s" % str(e)[:60]; continue
    for stem, e in doc.items():
      if isinstance(e, dict): yield rel, stem, e, None
      else: yield rel, stem, None, "top-level key %r is not an entry table" % stem
  for p in loose_paths(root):
    rel = os.path.relpath(p, root)
    try:
      with open(p, "rb") as f: e = tomllib.load(f)
    except Exception as ex:
      yield rel, os.path.basename(p)[:-5], None, "not valid TOML: %s" % str(ex)[:60]; continue
    yield rel, os.path.basename(p)[:-5], e, None


def entries(root=None, with_text=False):
  """{id: entry} with `_path` (the file holding it), `_loose`, `_duplicate` (a second file holding the
  same id) and, with_text, `_text` - the entry's own text as its old file read."""
  root, out, texts = root or ROOT, {}, {}
  if with_text:
    for a in AREA_FILES:
      p = area_path(a, root)
      if not os.path.exists(p): continue
      with open(p, encoding="utf-8") as f: t = f.read()
      lines = t.split("\n")
      for k, i, j in blocks(t): texts[(os.path.relpath(p, root), k)] = from_block(k, "\n".join(lines[i:j]))
  for rel, stem, e, err in iter_raw(root):
    if e is None: continue
    loose = os.path.basename(rel)[:-5] not in AREA_FILES
    e["_path"], e["_loose"] = rel, loose
    if with_text:
      if loose:
        with open(os.path.join(root, rel), encoding="utf-8") as f: e["_text"] = f.read()
      else:
        e["_text"] = texts.get((rel, stem), "")
    if stem in out: out[stem]["_duplicate"] = rel; continue
    out[stem] = e
  return out


def resolve(name, root=None):
  """'ledger/<id>.toml', '<id>.toml' or '<id>' -> (file relpath, id), or None if no entry has that id."""
  stem = os.path.basename(str(name))
  stem = stem[:-5] if stem.endswith(".toml") else stem
  for rel, s, e, err in iter_raw(root):
    if s == stem and e is not None: return rel, s
  return None
